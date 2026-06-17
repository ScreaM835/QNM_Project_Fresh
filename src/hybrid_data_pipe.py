"""Build input/target tensors for the hybrid FNO from a raw hybrid dataset.

The raw dataset (src/hybrid_dataset.py) stores Phi_coarse on its own coarse
grid. The FNO operates on the fine grid; this module performs the upsampling
once (cubic for the legacy supervised path, quintic degree-5 tensor-product
spline for the Richardson path) so training-time costs only see contiguous
arrays of fine-grid tensors.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator

from .hybrid_fno import HYBRID_IN_CHANNELS


def upsample_to_fine(
    phi_c: np.ndarray,
    x_c: np.ndarray, t_c: np.ndarray,
    x_f: np.ndarray, t_f: np.ndarray,
    method: str = "quintic",
) -> np.ndarray:
    """Spline-upsample one coarse (Nt_c, Nx_c) field to the fine (Nt_f, Nx_f) grid.

    method="quintic" uses a degree-5 tensor-product spline (RectBivariateSpline,
    kx=ky=5): the interpolant validated for the Richardson prior, whose O(dx^4)
    interpolation error stays well below the 0.36% extrapolation signal.
    method="cubic" keeps the original degree-3 RegularGridInterpolator.
    """
    if method == "quintic":
        spl = RectBivariateSpline(t_c, x_c, phi_c, kx=5, ky=5)
        return spl(t_f, x_f)
    interp = RegularGridInterpolator(
        (t_c, x_c), phi_c, method="cubic", bounds_error=False, fill_value=0.0,
    )
    T, X = np.meshgrid(t_f, x_f, indexing="ij")
    return interp(np.stack([T.ravel(), X.ravel()], axis=1)).reshape(t_f.size, x_f.size)


def assemble_split(
    split: Dict[str, np.ndarray],
    grid_x_coarse: np.ndarray, grid_t_coarse: np.ndarray,
    grid_x_fine: np.ndarray,   grid_t_fine: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_in, Y_target, Phi_coarse_upsampled).

    Shapes:
        X_in                       (N, HYBRID_IN_CHANNELS, Nt_f, Nx_f)
        Y_target = Phi_fine - up   (N, 1, Nt_f, Nx_f)
        Phi_coarse_upsampled       (N, Nt_f, Nx_f)        kept for evaluation

    Channel layout:
        0 : upsample(Phi_coarse)         (t, x)
        1 : upsample(Phi_coarse)[0, :]   broadcast over t  — IC displacement
        2 : V_fine(x)                    broadcast over t
        3 : M                            broadcast over (t, x)
        4 : Phi_coarse_t_0(x)            broadcast over t  — IC velocity (FD from coarse)
    """
    Phi_f = split["Phi_fine"]          # (N, Nt_f, Nx_f)
    Phi_c = split["Phi_coarse"]        # (N, Nt_c, Nx_c)
    V_f   = split["V_fine"]            # (N, Nx_f)
    P     = split["P"]                 # (N, 3)  -- (M, x0, sigma)

    N, Nt_f, Nx_f = Phi_f.shape
    dt_c = float(grid_t_coarse[1] - grid_t_coarse[0])

    X_in = np.empty((N, HYBRID_IN_CHANNELS, Nt_f, Nx_f), dtype=np.float32)
    Y    = np.empty((N, 1, Nt_f, Nx_f), dtype=np.float32)
    up_all = np.empty((N, Nt_f, Nx_f), dtype=np.float32)

    for i in range(N):
        up = upsample_to_fine(
            Phi_c[i], grid_x_coarse, grid_t_coarse, grid_x_fine, grid_t_fine,
            method="cubic",
        ).astype(np.float32)
        Phi0 = up[0, :]                                # IC displacement at t=0
        Pi0_c = (Phi_c[i, 1, :] - Phi_c[i, 0, :]) / dt_c
        # Resample Pi0_c onto the fine x-grid (1D linear interpolation)
        Pi0 = np.interp(grid_x_fine, grid_x_coarse, Pi0_c).astype(np.float32)
        M_i = float(P[i, 0])

        X_in[i, 0] = up
        X_in[i, 1] = np.broadcast_to(Phi0, (Nt_f, Nx_f))
        X_in[i, 2] = np.broadcast_to(V_f[i], (Nt_f, Nx_f))
        X_in[i, 3] = np.full((Nt_f, Nx_f), M_i, dtype=np.float32)
        X_in[i, 4] = np.broadcast_to(Pi0, (Nt_f, Nx_f))

        Y[i, 0]    = Phi_f[i] - up
        up_all[i]  = up

    return X_in, Y, up_all


def assemble_richardson(
    split_k4: Dict[str, np.ndarray],
    split_k2: Dict[str, np.ndarray],
    gx_c4: np.ndarray, gt_c4: np.ndarray,
    gx_c2: np.ndarray, gt_c2: np.ndarray,
    gx_f: np.ndarray,  gt_f: np.ndarray,
    target_mode: str = "richardson",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble Richardson-target training tensors from sample-aligned k2 + k4 splits.

    The FNO input prior (channel 0) is the *single* cheapest coarse solve (k4),
    quintic-upsampled. The training target is label-free:

        Phi_R = (4 * up2 - up4) / 3          (a-priori order p=2 Richardson)
        Y     = Phi_R - up4                  (target_mode="richardson")

    so the network never sees the fine FD field in its loss. With
    target_mode="supervised" the target is Phi_fine - up4 (fine FD label), kept
    for the A/B baseline only. Phi_fine is returned regardless, as an eval-only
    metric.

    k2 and k4 splits are sample-aligned (shared Sobol params); Phi_fine is taken
    from the k4 split and asserted to match the k2 split's Phi_fine.

    Returns (X_in, Y, up4_all, Phi_fine).
    """
    if target_mode not in ("richardson", "supervised"):
        raise ValueError(
            f"target_mode must be 'richardson' or 'supervised', got {target_mode!r}"
        )

    Phi_f  = split_k4["Phi_fine"]      # (N, Nt_f, Nx_f) -- identical across k
    Phi_c4 = split_k4["Phi_coarse"]    # (N, Nt_c4, Nx_c4)
    Phi_c2 = split_k2["Phi_coarse"]    # (N, Nt_c2, Nx_c2)
    V_f    = split_k4["V_fine"]        # (N, Nx_f)
    P      = split_k4["P"]             # (N, 3)

    if not np.allclose(Phi_f, split_k2["Phi_fine"], atol=1e-6, rtol=0.0):
        raise ValueError(
            "k2 and k4 Phi_fine differ: splits are not sample-aligned "
            "(check shared Sobol seed / sweep / split sizes)."
        )

    N, Nt_f, Nx_f = Phi_f.shape
    dt_c4 = float(gt_c4[1] - gt_c4[0])

    X_in   = np.empty((N, HYBRID_IN_CHANNELS, Nt_f, Nx_f), dtype=np.float32)
    Y      = np.empty((N, 1, Nt_f, Nx_f), dtype=np.float32)
    up_all = np.empty((N, Nt_f, Nx_f), dtype=np.float32)

    for i in range(N):
        up4 = upsample_to_fine(
            Phi_c4[i], gx_c4, gt_c4, gx_f, gt_f, method="quintic",
        ).astype(np.float32)
        up2 = upsample_to_fine(
            Phi_c2[i], gx_c2, gt_c2, gx_f, gt_f, method="quintic",
        ).astype(np.float32)
        Phi_R = (4.0 * up2 - up4) / 3.0

        Phi0  = up4[0, :]                                   # IC displacement at t=0
        Pi0_c = (Phi_c4[i, 1, :] - Phi_c4[i, 0, :]) / dt_c4
        Pi0   = np.interp(gx_f, gx_c4, Pi0_c).astype(np.float32)
        M_i   = float(P[i, 0])

        X_in[i, 0] = up4
        X_in[i, 1] = np.broadcast_to(Phi0, (Nt_f, Nx_f))
        X_in[i, 2] = np.broadcast_to(V_f[i], (Nt_f, Nx_f))
        X_in[i, 3] = np.full((Nt_f, Nx_f), M_i, dtype=np.float32)
        X_in[i, 4] = np.broadcast_to(Pi0, (Nt_f, Nx_f))

        if target_mode == "richardson":
            Y[i, 0] = (Phi_R - up4).astype(np.float32)
        else:
            Y[i, 0] = (Phi_f[i] - up4).astype(np.float32)
        up_all[i] = up4

    return X_in, Y, up_all, Phi_f


def to_torch(arr: np.ndarray, device: str = "cpu") -> torch.Tensor:
    return torch.from_numpy(arr).to(device)

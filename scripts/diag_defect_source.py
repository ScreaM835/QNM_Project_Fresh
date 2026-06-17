"""Diagnostic: does the coarse-truncation source S correlate with the residual r?

For one (or a few) val samples of the hybrid k=2 dataset, computes
    r = psi_fine - U(psi_coarse)
    S_coarse  = -(d_tt - d_xx + V)(psi_coarse)         on the COARSE grid (clean stencil),
                then upsampled to the fine grid via the same cubic upsampler
                used in src/hybrid_data_pipe.py.
    S_fine    = -(d_tt - d_xx + V_fine)(U(psi_coarse)) on the fine grid (polluted at
                coarse-node locations because cubic-convolution upsampler is only C^1).

We report ||S||/||r||, Pearson correlation of |S| and |r|, and save a 3-panel
side-by-side PNG (r, S_coarse_up, S_fine) for the first sample.

Pure CPU, runs on the login node in ~minutes. No training, no GPU.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.hybrid_dataset import load_dataset
from src.hybrid_data_pipe import upsample_to_fine
from src.fd_solver import _second_derivative
from src.potentials import V_of_x


DATASET = os.path.join(ROOT, "outputs/hybrid/dataset_sw_k2.npz")
OUT_DIR = os.path.join(ROOT, "outputs/hybrid/diag_defect")
N_SAMPLES = 4


def d2_t_centred(phi: np.ndarray, dt: float) -> np.ndarray:
    """Centred 2nd-order d^2/dt^2 along axis 0. Edges use one-sided 4-point."""
    out = np.empty_like(phi)
    out[1:-1] = (phi[2:] - 2.0 * phi[1:-1] + phi[:-2]) / (dt ** 2)
    out[0]  = (2.0 * phi[0]  - 5.0 * phi[1]  + 4.0 * phi[2]  - 1.0 * phi[3])  / (dt ** 2)
    out[-1] = (2.0 * phi[-1] - 5.0 * phi[-2] + 4.0 * phi[-3] - 1.0 * phi[-4]) / (dt ** 2)
    return out


def d2_x(phi: np.ndarray, dx: float) -> np.ndarray:
    """Apply 1D _second_derivative along axis -1 for every time slice."""
    out = np.empty_like(phi)
    for n in range(phi.shape[0]):
        out[n] = _second_derivative(phi[n], dx)
    return out


def wave_residual(phi: np.ndarray, V: np.ndarray, dx: float, dt: float) -> np.ndarray:
    """S = -(d_tt - d_xx + V) phi, broadcasting V over time axis."""
    return -(d2_t_centred(phi, dt) - d2_x(phi, dx) + V[None, :] * phi)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel(); b = b.ravel()
    a = a - a.mean(); b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[DIAG] loading {DATASET}", flush=True)
    splits, grid, meta = load_dataset(DATASET)
    val = splits["val"]
    print(f"[DIAG] val shapes: Phi_fine {val['Phi_fine'].shape}, "
          f"Phi_coarse {val['Phi_coarse'].shape}", flush=True)
    dx_f = float(grid.x_fine[1] - grid.x_fine[0])
    dt_f = float(grid.t_fine[1] - grid.t_fine[0])
    dx_c = float(grid.x_coarse[1] - grid.x_coarse[0])
    dt_c = float(grid.t_coarse[1] - grid.t_coarse[0])
    print(f"[DIAG] fine dx={dx_f} dt={dt_f}  coarse dx={dx_c} dt={dt_c}", flush=True)

    Phi_f_all = val["Phi_fine"]
    Phi_c_all = val["Phi_coarse"]
    V_f_all   = val["V_fine"]
    P_all     = val["P"]
    l         = int(meta.get("l", 2)) if isinstance(meta, dict) else 2

    rows = []
    for i in range(min(N_SAMPLES, Phi_f_all.shape[0])):
        Phi_f = Phi_f_all[i].astype(np.float64)
        Phi_c = Phi_c_all[i].astype(np.float64)
        V_f   = V_f_all[i].astype(np.float64)
        M_i, x0_i, sigma_i = (float(v) for v in P_all[i])

        V_c = V_of_x(grid.x_coarse.astype(np.float64), M=M_i, l=l, potential="zerilli")

        up = upsample_to_fine(
            Phi_c, grid.x_coarse, grid.t_coarse, grid.x_fine, grid.t_fine,
        ).astype(np.float64)

        r = Phi_f - up

        # Path A: S on coarse grid, then upsample S to fine grid.
        S_c = wave_residual(Phi_c, V_c, dx_c, dt_c)
        S_c_up = upsample_to_fine(
            S_c, grid.x_coarse, grid.t_coarse, grid.x_fine, grid.t_fine,
        ).astype(np.float64)

        # Path B: S on fine grid directly from upsampled coarse field (polluted).
        S_f = wave_residual(up, V_f, dx_f, dt_f)

        norm_r   = float(np.linalg.norm(r))
        ratio_A  = float(np.linalg.norm(S_c_up) / norm_r) if norm_r > 0 else float("nan")
        ratio_B  = float(np.linalg.norm(S_f)    / norm_r) if norm_r > 0 else float("nan")
        corr_A   = pearson(np.abs(S_c_up), np.abs(r))
        corr_B   = pearson(np.abs(S_f),    np.abs(r))
        # also signed correlation (does S match r in sign?)
        scorr_A  = pearson(S_c_up, r)
        scorr_B  = pearson(S_f,    r)

        rows.append(dict(
            i=i, M=M_i, x0=x0_i, sigma=sigma_i,
            norm_r=norm_r,
            ratio_coarse_up=ratio_A, corr_abs_coarse_up=corr_A, corr_signed_coarse_up=scorr_A,
            ratio_fine    =ratio_B, corr_abs_fine    =corr_B, corr_signed_fine    =scorr_B,
        ))

        print(f"[DIAG] sample {i}  M={M_i:.3f} x0={x0_i:+.2f} sigma={sigma_i:.2f}", flush=True)
        print(f"        ||r||={norm_r:.3e}", flush=True)
        print(f"        path A (S coarse -> upsample): ||S||/||r|| = {ratio_A:.3e}"
              f"   corr|.| = {corr_A:+.3f}   corr signed = {scorr_A:+.3f}", flush=True)
        print(f"        path B (S on fine, polluted) : ||S||/||r|| = {ratio_B:.3e}"
              f"   corr|.| = {corr_B:+.3f}   corr signed = {scorr_B:+.3f}", flush=True)

        if i == 0:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
            extent = [float(grid.x_fine[0]), float(grid.x_fine[-1]),
                      float(grid.t_fine[-1]), float(grid.t_fine[0])]
            vmax_r = float(np.percentile(np.abs(r), 99.5)) or 1.0
            vmax_S = float(np.percentile(np.abs(S_c_up), 99.5)) or 1.0
            vmax_F = float(np.percentile(np.abs(S_f), 99.5)) or 1.0
            for ax, dat, title, vm in [
                (axes[0], r,      "r = phi_f - U(phi_c)",   vmax_r),
                (axes[1], S_c_up, "S (coarse stencil, upsampled)", vmax_S),
                (axes[2], S_f,    "S (fine stencil on upsampled, polluted)", vmax_F),
            ]:
                im = ax.imshow(dat, extent=extent, aspect="auto",
                               cmap="RdBu_r", vmin=-vm, vmax=vm)
                ax.set_title(title)
                ax.set_xlabel("x")
                fig.colorbar(im, ax=ax, fraction=0.046)
            axes[0].set_ylabel("t")
            fig.suptitle(f"defect-source diagnostic, val sample 0  "
                         f"(M={M_i:.3f}, x0={x0_i:+.2f}, sigma={sigma_i:.2f})")
            fig.tight_layout()
            out_png = os.path.join(OUT_DIR, "diag_defect_sample0.png")
            fig.savefig(out_png, dpi=120)
            print(f"[DIAG] wrote {out_png}", flush=True)

    # summary
    print("\n[DIAG] summary across samples:", flush=True)
    keys = ["ratio_coarse_up", "corr_abs_coarse_up", "corr_signed_coarse_up",
            "ratio_fine",     "corr_abs_fine",     "corr_signed_fine"]
    for k in keys:
        vals = np.array([r[k] for r in rows])
        print(f"  {k:25s}  mean={vals.mean():+.3e}  min={vals.min():+.3e}  max={vals.max():+.3e}",
              flush=True)


if __name__ == "__main__":
    main()

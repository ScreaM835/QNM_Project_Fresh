"""Generate / load the multi-parameter Zerilli dataset for FNO training.

Each sample is one full FD solve with a different (M, x0, sigma).  The
operator we learn is

    G : (Phi0(x), Pi0(x), V(x; M, l)) -> Phi(x, t).

This module is OPT-IN: it lives next to the existing FD/PINN code but is only
imported by the FNO scripts and never modifies the existing pipelines.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

from .fd_solver import solve_fd


# Channel layout used everywhere in the FNO pipeline (kept in one place):
#   ch 0 : Phi(x,0)        (broadcast over t)
#   ch 1 : Phi_t(x,0)      (broadcast over t)
#   ch 2 : V(x; M, l)      (broadcast over t)
#   ch 3 : M (scalar)      (broadcast over t and x)
IN_CHANNELS = 4
OUT_CHANNELS = 1


@dataclass
class GridSpec:
    x: np.ndarray   # (Nx,)
    t: np.ndarray   # (Nt,)


def _make_sample_config(base_cfg: Dict, M: float, x0: float, sigma: float) -> Dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["physics"]["M"] = float(M)
    cfg["initial_data"]["x0"] = float(x0)
    cfg["initial_data"]["sigma"] = float(sigma)
    return cfg


def _channels_from_solution(sol: Dict[str, np.ndarray], M: float) -> np.ndarray:
    """Build (IN_CHANNELS, Nt, Nx) input tensor from one FD solve.

    Phi0 and Pi0 are read from the FD output itself (so they are exactly the
    initial condition the FD solver actually used, regardless of any
    velocity-profile convention).
    """
    phi = sol["phi"]                      # (Nt, Nx)
    Vx = sol["V"]                         # (Nx,)
    dt = float(sol["dt"])
    Nt, Nx = phi.shape

    Phi0 = phi[0]                         # (Nx,)
    # Two-step forward difference is fine here; only used as input feature.
    Pi0 = (phi[1] - phi[0]) / dt          # (Nx,)

    chans = np.empty((IN_CHANNELS, Nt, Nx), dtype=np.float32)
    chans[0] = np.broadcast_to(Phi0, (Nt, Nx))
    chans[1] = np.broadcast_to(Pi0, (Nt, Nx))
    chans[2] = np.broadcast_to(Vx, (Nt, Nx))
    chans[3] = np.full((Nt, Nx), float(M), dtype=np.float32)
    return chans


def sample_params(rng: np.random.Generator, sweep: Dict, n: int) -> Iterable[Tuple[float, float, float]]:
    M_lo, M_hi = sweep["M_range"]
    x0_lo, x0_hi = sweep["x0_range"]
    s_lo, s_hi = sweep["sigma_range"]
    Ms = rng.uniform(M_lo, M_hi, size=n)
    x0s = rng.uniform(x0_lo, x0_hi, size=n)
    sigmas = rng.uniform(s_lo, s_hi, size=n)
    return list(zip(Ms.tolist(), x0s.tolist(), sigmas.tolist()))


def generate_split(
    base_cfg: Dict,
    params: Iterable[Tuple[float, float, float]],
    stride_t: int = 1,
    stride_x: int = 1,
    progress_prefix: str = "",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, GridSpec, np.ndarray]:
    """Run FD for each (M, x0, sigma) and stack arrays.

    Returns
    -------
    X : (N, IN_CHANNELS, Nt, Nx) float32 — input channels
    Y : (N, OUT_CHANNELS, Nt, Nx) float32 — target Phi(x,t)
    P : (N, 3) float32 — (M, x0, sigma) for each sample
    grid : GridSpec — common (x, t) grid (after striding)
    V_per_sample : (N, Nx) float32 — per-sample potential, useful for PINO loss
    """
    params = list(params)
    if not params:
        raise ValueError("Empty parameter list.")

    X_list, Y_list, V_list = [], [], []
    grid: GridSpec | None = None

    for i, (M, x0, sigma) in enumerate(params):
        cfg_i = _make_sample_config(base_cfg, M, x0, sigma)
        sol = solve_fd(cfg_i)
        x = sol["x"][::stride_x]
        t = sol["t"][::stride_t]
        if grid is None:
            grid = GridSpec(x=x.astype(np.float32), t=t.astype(np.float32))

        chans = _channels_from_solution(sol, M=M)             # (C, Nt, Nx)
        chans = chans[:, ::stride_t, ::stride_x]
        phi = sol["phi"][::stride_t, ::stride_x][None, ...]    # (1, Nt, Nx)
        Vx = sol["V"][::stride_x]

        X_list.append(chans.astype(np.float32))
        Y_list.append(phi.astype(np.float32))
        V_list.append(Vx.astype(np.float32))

        if progress_prefix:
            print(f"  {progress_prefix} sample {i+1}/{len(params)}: "
                  f"M={M:.3f} x0={x0:+.2f} sigma={sigma:.2f}", flush=True)

    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    V = np.stack(V_list, axis=0)
    P = np.array(params, dtype=np.float32)
    assert grid is not None
    return X, Y, P, grid, V


def save_dataset(path: str, splits: Dict[str, Dict[str, np.ndarray]], grid: GridSpec, meta: Dict) -> None:
    """Save a multi-split dataset to a single .npz.

    Layout in the .npz:
      x, t                 -- common grid (after stride)
      <split>_X, <split>_Y, <split>_P, <split>_V   for split in splits
      meta_*               -- scalar metadata
    """
    import os, json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload: Dict[str, np.ndarray] = {"x": grid.x, "t": grid.t}
    for name, d in splits.items():
        payload[f"{name}_X"] = d["X"]
        payload[f"{name}_Y"] = d["Y"]
        payload[f"{name}_P"] = d["P"]
        payload[f"{name}_V"] = d["V"]
    payload["meta_json"] = np.array(json.dumps(meta), dtype=object)
    np.savez_compressed(path, **payload)


def load_dataset(path: str):
    """Inverse of save_dataset.  Returns (splits_dict, grid, meta)."""
    import json
    npz = np.load(path, allow_pickle=True)
    grid = GridSpec(x=npz["x"], t=npz["t"])
    splits: Dict[str, Dict[str, np.ndarray]] = {}
    for key in ["train", "val", "test"]:
        if f"{key}_X" in npz.files:
            splits[key] = {
                "X": npz[f"{key}_X"],
                "Y": npz[f"{key}_Y"],
                "P": npz[f"{key}_P"],
                "V": npz[f"{key}_V"],
            }
    meta = json.loads(str(npz["meta_json"])) if "meta_json" in npz.files else {}
    return splits, grid, meta

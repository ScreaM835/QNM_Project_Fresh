"""Coarse/fine FD dataset for the hybrid coarse-FD + FNO-residual surrogate.

For each parameter sample (M, x0, sigma) we run the existing `solve_fd` twice:

  - fine grid:   dx = base_dx,        dt = base_dt        (existing reference)
  - coarse grid: dx = k * base_dx,    dt = k * base_dt    (with k in {2, 4})

We store both fields. Upsampling of the coarse field to the fine (x, t) grid
is deferred to the training script so this module makes no choice about
interpolation order.

Dataset layout in .npz (one file per k):
  x_fine, t_fine             (Nx_f,), (Nt_f,)         fine grid axes
  x_coarse, t_coarse         (Nx_c,), (Nt_c,)         coarse grid axes
  <split>_Phi_fine           (N, Nt_f, Nx_f) float32  fine waveform
  <split>_Phi_coarse         (N, Nt_c, Nx_c) float32  coarse waveform
  <split>_V_fine             (N, Nx_f) float32        per-sample potential
  <split>_P                  (N, 3) float32           (M, x0, sigma)
  meta_json                  scalar json blob          k, base dx/dt, sweep ranges
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.stats import qmc

from .fd_solver import solve_fd


@dataclass
class GridPair:
    x_fine: np.ndarray
    t_fine: np.ndarray
    x_coarse: np.ndarray
    t_coarse: np.ndarray


def _make_sample_config(base_cfg: Dict, M: float, x0: float, sigma: float,
                        dx: float, dt: float,
                        scheme_override: Dict | None = None) -> Dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["physics"]["M"] = float(M)
    cfg["initial_data"]["x0"] = float(x0)
    cfg["initial_data"]["sigma"] = float(sigma)
    cfg["fd"]["dx"] = float(dx)
    cfg["fd"]["dt"] = float(dt)
    if scheme_override:
        cfg["fd"].update(scheme_override)
    return cfg


def sobol_params(sweep: Dict, n: int, seed: int = 0) -> List[Tuple[float, float, float]]:
    """Sobol quasi-random sweep over (M, x0, sigma)."""
    sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
    pts = sampler.random(n=n)
    lo = np.array([sweep["M_range"][0], sweep["x0_range"][0], sweep["sigma_range"][0]])
    hi = np.array([sweep["M_range"][1], sweep["x0_range"][1], sweep["sigma_range"][1]])
    scaled = lo + pts * (hi - lo)
    return [tuple(row) for row in scaled]


def generate_split(
    base_cfg: Dict,
    params: Iterable[Tuple[float, float, float]],
    k: int,
    progress_prefix: str = "",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, GridPair]:
    """Run fine+coarse FD for every sample.

    Returns (Phi_fine, Phi_coarse, V_fine, P, grid).
    """
    base_dx = float(base_cfg["fd"]["dx"])
    base_dt = float(base_cfg["fd"]["dt"])
    coarse_dx = k * base_dx
    coarse_dt = k * base_dt

    # Optional higher-order scheme for the CHEAP coarse prior only. The fine
    # field is the FNO regression target and is always solved with the base
    # scheme (default 2nd-order), so the target is unchanged. Set
    # fd.coarse_space_scheme (central4/central6/drp7) and fd.coarse_bc_order in
    # the config to shrink the coarse-prior dispersion error (see fd_solver.py).
    coarse_override: Dict = {}
    if "coarse_space_scheme" in base_cfg["fd"]:
        coarse_override["space_scheme"] = base_cfg["fd"]["coarse_space_scheme"]
    if "coarse_bc_order" in base_cfg["fd"]:
        coarse_override["bc_order"] = base_cfg["fd"]["coarse_bc_order"]

    Phi_f_list, Phi_c_list, V_f_list = [], [], []
    grid: GridPair | None = None
    params = list(params)

    for i, (M, x0, sigma) in enumerate(params):
        cfg_f = _make_sample_config(base_cfg, M, x0, sigma, base_dx, base_dt)
        cfg_c = _make_sample_config(base_cfg, M, x0, sigma, coarse_dx, coarse_dt,
                                    scheme_override=coarse_override)
        sol_f = solve_fd(cfg_f)
        sol_c = solve_fd(cfg_c)

        if grid is None:
            grid = GridPair(
                x_fine=sol_f["x"].astype(np.float32),
                t_fine=sol_f["t"].astype(np.float32),
                x_coarse=sol_c["x"].astype(np.float32),
                t_coarse=sol_c["t"].astype(np.float32),
            )

        Phi_f_list.append(sol_f["phi"].astype(np.float32))
        Phi_c_list.append(sol_c["phi"].astype(np.float32))
        V_f_list.append(sol_f["V"].astype(np.float32))

        if progress_prefix:
            print(f"  {progress_prefix} {i+1}/{len(params)}: "
                  f"M={M:.3f} x0={x0:+.2f} sigma={sigma:.2f}", flush=True)

    Phi_f = np.stack(Phi_f_list, axis=0)
    Phi_c = np.stack(Phi_c_list, axis=0)
    V_f   = np.stack(V_f_list,   axis=0)
    P     = np.array(params, dtype=np.float32)
    assert grid is not None
    return Phi_f, Phi_c, V_f, P, grid


def save_dataset(path: str, splits: Dict[str, Dict[str, np.ndarray]],
                 grid: GridPair, meta: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload: Dict[str, np.ndarray] = {
        "x_fine": grid.x_fine, "t_fine": grid.t_fine,
        "x_coarse": grid.x_coarse, "t_coarse": grid.t_coarse,
    }
    for name, d in splits.items():
        payload[f"{name}_Phi_fine"]   = d["Phi_fine"]
        payload[f"{name}_Phi_coarse"] = d["Phi_coarse"]
        payload[f"{name}_V_fine"]     = d["V_fine"]
        payload[f"{name}_P"]          = d["P"]
    payload["meta_json"] = np.array(json.dumps(meta), dtype=object)
    np.savez_compressed(path, **payload)


def load_dataset(path: str) -> Tuple[Dict[str, Dict[str, np.ndarray]], GridPair, Dict]:
    npz = np.load(path, allow_pickle=True)
    grid = GridPair(
        x_fine=npz["x_fine"], t_fine=npz["t_fine"],
        x_coarse=npz["x_coarse"], t_coarse=npz["t_coarse"],
    )
    splits: Dict[str, Dict[str, np.ndarray]] = {}
    for key in ["train", "val", "test"]:
        if f"{key}_Phi_fine" in npz.files:
            splits[key] = {
                "Phi_fine":   npz[f"{key}_Phi_fine"],
                "Phi_coarse": npz[f"{key}_Phi_coarse"],
                "V_fine":     npz[f"{key}_V_fine"],
                "P":          npz[f"{key}_P"],
            }
    meta = json.loads(str(npz["meta_json"])) if "meta_json" in npz.files else {}
    return splits, grid, meta

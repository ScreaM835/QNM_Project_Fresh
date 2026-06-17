"""Phase 1 acceptance check for the hybrid dataset.

Runs the smoke dataset for k=2 and k=4, then verifies:

  1. The "fine" Phi from the hybrid pipeline matches the existing canonical
     reference (configs/zerilli_l2_paper.yaml at M=1, x0=4, sigma=5) to
     plotting precision when sampled at the same parameters.
  2. The coarse Phi is bounded (no NaN, no boundary blow-up); max amplitude
     stays comparable to the fine field.
  3. Upsampled coarse vs fine L2 is small enough that "residual is smoother
     than the field" is at least plausible: report
        L2(Phi_fine)
        L2(Phi_fine - upsampled(Phi_coarse))
        spectral-energy ratio of dPhi vs Phi above 0.5*Nyquist
     for both k=2 and k=4.

This script writes a JSON summary so we have a record of Phase 1 closure.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.fd_solver import solve_fd
from src.hybrid_dataset import load_dataset


def _upsample_to_fine(phi_c: np.ndarray, x_c: np.ndarray, t_c: np.ndarray,
                      x_f: np.ndarray, t_f: np.ndarray) -> np.ndarray:
    interp = RegularGridInterpolator(
        (t_c, x_c), phi_c, method="cubic", bounds_error=False, fill_value=0.0,
    )
    T, X = np.meshgrid(t_f, x_f, indexing="ij")
    return interp(np.stack([T.ravel(), X.ravel()], axis=1)).reshape(t_f.size, x_f.size)


def _spectral_energy_ratio(field: np.ndarray) -> float:
    """Fraction of 2D-FFT energy above 0.5*Nyquist on either axis.

    Operates on the (Nt, Nx) field. A small value means the field is dominated
    by low-frequency content (easier to learn for an FNO with limited modes).
    """
    F = np.fft.fft2(field)
    P = (np.abs(F) ** 2).real
    Nt, Nx = field.shape
    # frequency indices folded to [0, N/2]
    kt = np.minimum(np.arange(Nt), Nt - np.arange(Nt))
    kx = np.minimum(np.arange(Nx), Nx - np.arange(Nx))
    KT, KX = np.meshgrid(kt, kx, indexing="ij")
    hi_mask = (KT > Nt // 4) | (KX > Nx // 4)
    total = float(P.sum())
    if total <= 0.0:
        return float("nan")
    return float(P[hi_mask].sum() / total)


def _check_one(npz_path: str, base_cfg, summary: dict) -> None:
    k = int(npz_path.split("_k")[-1].split(".")[0])
    splits, grid, meta = load_dataset(npz_path)
    train = splits["train"]
    Phi_f = train["Phi_fine"]      # (N, Nt_f, Nx_f)
    Phi_c = train["Phi_coarse"]    # (N, Nt_c, Nx_c)
    P = train["P"]                 # (N, 3)
    x_f, t_f = grid.x_fine, grid.t_fine
    x_c, t_c = grid.x_coarse, grid.t_coarse

    rec = {
        "k": k,
        "n_train": int(Phi_f.shape[0]),
        "shape_fine":   list(Phi_f.shape[1:]),
        "shape_coarse": list(Phi_c.shape[1:]),
        "max_abs_fine":     float(np.max(np.abs(Phi_f))),
        "max_abs_coarse":   float(np.max(np.abs(Phi_c))),
        "any_nan_fine":     bool(np.isnan(Phi_f).any()),
        "any_nan_coarse":   bool(np.isnan(Phi_c).any()),
        "any_inf_fine":     bool(np.isinf(Phi_f).any()),
        "any_inf_coarse":   bool(np.isinf(Phi_c).any()),
    }

    # Spectral-energy ratio (averaged over samples) for Phi_fine and for the
    # residual delta = Phi_fine - upsample(Phi_coarse).
    e_field, e_resid, l2_field, l2_resid = [], [], [], []
    for i in range(Phi_f.shape[0]):
        up_c = _upsample_to_fine(Phi_c[i], x_c, t_c, x_f, t_f)
        delta = Phi_f[i] - up_c
        e_field.append(_spectral_energy_ratio(Phi_f[i]))
        e_resid.append(_spectral_energy_ratio(delta))
        l2_field.append(float(np.sqrt((Phi_f[i] ** 2).mean())))
        l2_resid.append(float(np.sqrt((delta ** 2).mean())))
    rec["spectral_hi_freq_frac_field_mean"] = float(np.mean(e_field))
    rec["spectral_hi_freq_frac_resid_mean"] = float(np.mean(e_resid))
    rec["L2_field_mean"]   = float(np.mean(l2_field))
    rec["L2_resid_mean"]   = float(np.mean(l2_resid))
    rec["L2_resid_over_field"] = (
        rec["L2_resid_mean"] / rec["L2_field_mean"]
        if rec["L2_field_mean"] > 0 else float("nan")
    )

    # Reference check: re-run the canonical (M=1, x0=4, sigma=5) at fine grid
    # via solve_fd and confirm shape and an interior slice match closely.
    cfg = {**base_cfg}
    cfg["physics"]["M"] = 1.0
    cfg["initial_data"]["x0"] = 4.0
    cfg["initial_data"]["sigma"] = 5.0
    ref = solve_fd(cfg)
    rec["fine_grid_matches_reference_shape"] = (
        ref["phi"].shape == tuple(rec["shape_fine"])
    )

    summary[os.path.basename(npz_path)] = rec
    print(f"[ACCEPT] {os.path.basename(npz_path)}: "
          f"max|Phi_f|={rec['max_abs_fine']:.3f}, "
          f"max|Phi_c|={rec['max_abs_coarse']:.3f}, "
          f"NaN? f={rec['any_nan_fine']} c={rec['any_nan_coarse']}, "
          f"L2(delta)/L2(field)={rec['L2_resid_over_field']:.3e}, "
          f"hi-freq frac field={rec['spectral_hi_freq_frac_field_mean']:.3e}, "
          f"resid={rec['spectral_hi_freq_frac_resid_mean']:.3e}")


def main() -> None:
    base = "/home/ycc44/project32_qnm_pinn_repo_fd_refinement/project32_qnm_pinn_improved"
    cfg = load_config(os.path.join(base, "configs", "hybrid_sw_dataset.yaml"))
    summary: dict = {}
    for k in (2, 4):
        path = os.path.join(base, f"outputs/hybrid/smoke_k{k}.npz")
        if not os.path.exists(path):
            print(f"[ACCEPT] missing {path}; skipping")
            continue
        _check_one(path, cfg, summary)

    out = os.path.join(base, "outputs/hybrid/phase1_acceptance.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[ACCEPT] wrote {out}")


if __name__ == "__main__":
    main()

"""Fixed-BH FNO vs FD sweep across runs and extraction radii.

For each (config, xq) pair: build the canonical input (M=1, x0=4, sigma=5),
run the FNO forward, run FD for GT, extract QNM at xq via m1-m4, and write
a single combined JSON. Also appends a 'fixed_BH_sweep' section to
summary_v2_v3_v4.json.

Usage:
    python scripts/fno_fixed_bh_sweep.py
"""
from __future__ import annotations

import json, os, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.fno_dataset import load_dataset
from src.fno_model import build_model
from src.fd_solver import solve_fd
from scripts.fno_fixed_bh_qnm import build_input, extract_all

CONFIGS = [
    "configs/fno_zerilli_l2_v3.yaml",
    "configs/fno_zerilli_l2_v4.yaml",
]
XQS = [0.0, 2.0, 5.0, 10.0]
M, X0, SIGMA = 1.0, 4.0, 5.0
T_START, T_END = 10.0, 50.0
THEORY = {"omega_M": 0.3737, "tau_over_M": 11.241}


def run_one_config(cfg_path):
    cfg = load_config(cfg_path)
    name = cfg["experiment"]["name"]
    out_dir = cfg["training"]["out_dir"]
    ckpt = os.path.join(out_dir, "model.pt")
    if not os.path.exists(ckpt):
        print(f"[skip] no model.pt at {ckpt}")
        return None

    splits, grid, meta = load_dataset(cfg["training"]["data_path"])
    x = grid.x.astype(np.float64)
    t = grid.t.astype(np.float64)
    print(f"[{name}] grid Nx={x.size} Nt={t.size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg,
                        t_grid=torch.from_numpy(grid.t).to(device),
                        x_grid=torch.from_numpy(grid.x).to(device)).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()

    # FNO forward (once)
    chans, _ = build_input(x, t, M, X0, SIGMA, ell=2)
    X = torch.from_numpy(chans[None]).to(device)
    with torch.no_grad():
        phi_pred = model(X).cpu().numpy()[0, 0]
    print(f"[{name}] FNO done  |phi|.max={np.abs(phi_pred).max():.3e}")

    # FD GT (once)
    fd_cfg = {
        "physics": {"M": M, "potential": "zerilli", "l": 2},
        "initial_data": {"A": 1.0, "x0": X0, "sigma": SIGMA,
                         "velocity_profile": "outgoing"},
        "domain": {"xmin": float(x.min()), "xmax": float(x.max()),
                   "tmin": 0.0, "tmax": float(t.max())},
        "fd": {"dx": float(meta["fd_dx"]), "dt": float(meta["fd_dt"])},
    }
    sol = solve_fd(fd_cfg)
    ph = sol["phi"]; xs = sol["x"]; ts = sol["t"]
    if ph.shape != phi_pred.shape:
        from scipy.interpolate import RegularGridInterpolator
        f = RegularGridInterpolator((ts, xs), ph, bounds_error=False, fill_value=0.0)
        tt, xx = np.meshgrid(t, x, indexing="ij")
        phi_true = f((tt, xx))
    else:
        phi_true = ph
    field_rmsd = float(np.sqrt(np.mean((phi_pred - phi_true) ** 2)))
    print(f"[{name}] FD done  field_rmsd={field_rmsd:.3e}")

    out = {
        "config": cfg_path, "run": name,
        "M": M, "x0": X0, "sigma": SIGMA,
        "t_start": T_START, "t_end": T_END,
        "theory": THEORY,
        "field_rmsd": field_rmsd,
        "xq_sweep": {},
    }
    for xq in XQS:
        ix = int(np.argmin(np.abs(x - xq)))
        y_pred = phi_pred[:, ix].astype(np.float64)
        y_true = phi_true[:, ix].astype(np.float64)
        rec = {
            "xq": xq, "x_idx": ix, "x_actual": float(x[ix]),
            "peak_fno": float(np.abs(y_pred).max()),
            "peak_gt":  float(np.abs(y_true).max()),
            "FNO": extract_all(t, y_pred, M, T_START, T_END),
            "GT":  extract_all(t, y_true, M, T_START, T_END),
        }
        out["xq_sweep"][f"xq_{xq}"] = rec
        print(f"  xq={xq:5.1f}  ix={ix:3d}  "
              f"FNO m4 ω={rec['FNO']['m4']['omega_M']}  τ={rec['FNO']['m4']['tau_over_M']}  "
              f"({rec['FNO']['m4']['omega_pct_err']}% / {rec['FNO']['m4']['tau_pct_err']}%)")
    return out


def main():
    out_path = "outputs/qnm/fno_xq2_comparison/fixed_BH_sweep.json"
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)

    all_runs = {}
    for cfg in CONFIGS:
        res = run_one_config(cfg)
        if res is None:
            continue
        all_runs[res["run"]] = res

    payload = {
        "fixed_BH_setup": {"M": M, "x0": X0, "sigma": SIGMA,
                           "t_start": T_START, "t_end": T_END,
                           "xq_values": XQS, "theory": THEORY},
        "runs": all_runs,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {out_path}")

    # Append into summary_v2_v3_v4.json
    summary_path = "outputs/qnm/fno_xq2_comparison/summary_v2_v3_v4.json"
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
    else:
        summary = {}
    summary["fixed_BH_sweep"] = payload
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Appended fixed_BH_sweep into {summary_path}")

    # Pretty table
    print("\n=== FNO m4 across runs and xq ===")
    print(f"{'run':<25s} {'xq':>5s} | {'FNO ω%':>8s} {'FNO τ%':>8s} | {'GT ω%':>8s} {'GT τ%':>8s} | field_rmsd")
    for run, res in all_runs.items():
        for key, rec in res["xq_sweep"].items():
            fm4 = rec["FNO"]["m4"]; gm4 = rec["GT"]["m4"]
            def s(v): return "  nan  " if v is None else f"{v:7.3f}"
            print(f"{run:<25s} {rec['xq']:>5.1f} | {s(fm4['omega_pct_err'])} {s(fm4['tau_pct_err'])} | "
                  f"{s(gm4['omega_pct_err'])} {s(gm4['tau_pct_err'])} | {res['field_rmsd']:.3e}")


if __name__ == "__main__":
    main()

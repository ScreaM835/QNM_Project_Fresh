"""Dense xq scan for principled extraction-radius selection.

For each xq on a fine grid, extract m4 (two-mode window-scan) QNM from both
FNO prediction and FD ground truth at the canonical fixed BH (M=1, x0=4,
sigma=5). Score each xq by:
  - intrinsic plateau width: rolling std of m4 omega over a +/-k xq-neighborhood
    (smaller = better; truth-free metric)
  - method agreement: |omega_m4 - omega_m3| at the same xq
  - validation (uses truth): |omega - omega_theory|

Recommend xq* = argmin( rolling-std of omega ) restricted to xq where all
methods return valid finite values and tau in (0, 50].

Usage:
    python scripts/fno_xq_scan.py --config configs/fno_zerilli_l2_v4.yaml
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.fno_dataset import load_dataset
from src.fno_model import build_model
from src.fd_solver import solve_fd
from scripts.fno_fixed_bh_qnm import build_input, extract_all

M, X0, SIGMA = 1.0, 4.0, 5.0
T_START, T_END = 10.0, 100.0
THEORY_OMEGA = 0.3737
THEORY_TAU = 11.241


def _f(v):
    if v is None:
        return float("nan")
    try:
        return float(v)
    except Exception:
        return float("nan")


def rolling_std(arr, half_window):
    """std over [i-k, i+k] window. Returns nan if window has <3 finite values."""
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(n):
        lo, hi = max(0, i - half_window), min(n, i + half_window + 1)
        seg = arr[lo:hi]
        seg = seg[np.isfinite(seg)]
        if len(seg) >= 3:
            out[i] = float(np.std(seg))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fno_zerilli_l2_v4.yaml")
    ap.add_argument("--xq_min", type=float, default=-2.0)
    ap.add_argument("--xq_max", type=float, default=15.0)
    ap.add_argument("--n", type=int, default=35)
    ap.add_argument("--half_window", type=int, default=3,
                    help="rolling-window half width in xq steps")
    ap.add_argument("--out", default="outputs/qnm/fno_xq2_comparison/xq_scan_v4.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    name = cfg["experiment"]["name"]
    ckpt = os.path.join(cfg["training"]["out_dir"], "model.pt")
    splits, grid, meta = load_dataset(cfg["training"]["data_path"])
    x = grid.x.astype(np.float64)
    t = grid.t.astype(np.float64)
    print(f"[{name}] grid Nx={x.size} Nt={t.size}  x_range=[{x.min()},{x.max()}]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg,
                        t_grid=torch.from_numpy(grid.t).to(device),
                        x_grid=torch.from_numpy(grid.x).to(device)).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()

    # Single FNO forward + single FD solve
    chans, _ = build_input(x, t, M, X0, SIGMA, ell=2)
    with torch.no_grad():
        phi_pred = model(torch.from_numpy(chans[None]).to(device)).cpu().numpy()[0, 0]
    fd_cfg = {
        "physics": {"M": M, "potential": "zerilli", "l": 2},
        "initial_data": {"A": 1.0, "x0": X0, "sigma": SIGMA,
                         "velocity_profile": "outgoing"},
        "domain": {"xmin": float(x.min()), "xmax": float(x.max()),
                   "tmin": 0.0, "tmax": float(t.max())},
        "fd": {"dx": float(meta["fd_dx"]), "dt": float(meta["fd_dt"])},
    }
    sol = solve_fd(fd_cfg)
    if sol["phi"].shape != phi_pred.shape:
        from scipy.interpolate import RegularGridInterpolator
        f = RegularGridInterpolator((sol["t"], sol["x"]), sol["phi"],
                                    bounds_error=False, fill_value=0.0)
        tt, xx = np.meshgrid(t, x, indexing="ij")
        phi_true = f((tt, xx))
    else:
        phi_true = sol["phi"]
    print(f"[{name}] FNO done, FD done. Scanning xq...")

    xqs = np.linspace(args.xq_min, args.xq_max, args.n)
    records = []
    for xq in xqs:
        ix = int(np.argmin(np.abs(x - xq)))
        y_p = phi_pred[:, ix].astype(np.float64)
        y_t = phi_true[:, ix].astype(np.float64)
        fno = extract_all(t, y_p, M, T_START, T_END)
        gt  = extract_all(t, y_t, M, T_START, T_END)
        records.append({
            "xq": float(xq), "x_idx": ix, "x_actual": float(x[ix]),
            "peak_fno": float(np.abs(y_p).max()),
            "peak_gt": float(np.abs(y_t).max()),
            "FNO": fno, "GT": gt,
        })

    # Plateau metric on m4 omega
    fno_w = np.array([_f(r["FNO"]["m4"]["omega_M"]) for r in records])
    fno_tau = np.array([_f(r["FNO"]["m4"]["tau_over_M"]) for r in records])
    gt_w = np.array([_f(r["GT"]["m4"]["omega_M"]) for r in records])
    gt_tau = np.array([_f(r["GT"]["m4"]["tau_over_M"]) for r in records])
    fno_w3 = np.array([_f(r["FNO"]["m3"]["omega_M"]) for r in records])

    # validity mask
    valid_fno = (np.isfinite(fno_w) & np.isfinite(fno_tau)
                 & (fno_w > 0) & (fno_tau > 0) & (fno_tau < 50)
                 & (np.abs(fno_w) < 1.0))

    plateau_std = rolling_std(np.where(valid_fno, fno_w, np.nan), args.half_window)
    method_agreement = np.abs(fno_w - fno_w3)  # FNO m4 vs FNO m3

    # Recommended xq*: smallest plateau_std AND method_agreement < 0.05 AND valid
    score = plateau_std.copy()
    score[~valid_fno] = np.inf
    score[~np.isfinite(method_agreement)] = np.inf
    # combine: dominated by plateau std, tie-break by method agreement
    combined = np.where(np.isfinite(score), score + 0.1 * np.nan_to_num(method_agreement, nan=1.0), np.inf)
    best_i = int(np.argmin(combined))
    best_xq = float(xqs[best_i])

    # also report best by validation (uses truth)
    val_err = np.abs(fno_w - THEORY_OMEGA) + (1.0 / THEORY_TAU) * np.abs(fno_tau - THEORY_TAU)
    val_err = np.where(valid_fno, val_err, np.inf)
    best_val_i = int(np.argmin(val_err))

    print(f"\n[recommended] intrinsic best xq = {best_xq:+.3f}  "
          f"(idx={best_i}, plateau_std={plateau_std[best_i]:.2e}, "
          f"method_agree={method_agreement[best_i]:.2e})")
    print(f"  FNO m4: Mω={fno_w[best_i]:.6f}  τ/M={fno_tau[best_i]:.4f}  "
          f"(err: ω={(fno_w[best_i]-THEORY_OMEGA)/THEORY_OMEGA*100:+.3f}%  "
          f"τ={(fno_tau[best_i]-THEORY_TAU)/THEORY_TAU*100:+.3f}%)")
    print(f"\n[validation] truth-based best xq = {xqs[best_val_i]:+.3f}  "
          f"(FNO ω={fno_w[best_val_i]:.6f}  τ={fno_tau[best_val_i]:.4f})")

    print(f"\n{'xq':>6s} | {'FNO m4 ω':>9s} {'FNO m4 τ':>9s} | {'ω% err':>7s} {'τ% err':>7s} | "
          f"{'plateau_std':>11s} {'m4-m3':>7s} {'valid':>5s}")
    for i, xq in enumerate(xqs):
        ws = f"{fno_w[i]:9.5f}" if np.isfinite(fno_w[i]) else "    nan  "
        ts = f"{fno_tau[i]:9.4f}" if np.isfinite(fno_tau[i]) else "    nan  "
        ew = (fno_w[i] - THEORY_OMEGA) / THEORY_OMEGA * 100 if np.isfinite(fno_w[i]) else float("nan")
        et = (fno_tau[i] - THEORY_TAU) / THEORY_TAU * 100 if np.isfinite(fno_tau[i]) else float("nan")
        es = f"{ew:+7.3f}" if np.isfinite(ew) else "   nan "
        eq = f"{et:+7.3f}" if np.isfinite(et) else "   nan "
        ps = f"{plateau_std[i]:.3e}" if np.isfinite(plateau_std[i]) else "    nan   "
        ma = f"{method_agreement[i]:.4f}" if np.isfinite(method_agreement[i]) else "  nan  "
        mark = "*" if i == best_i else (" " if valid_fno[i] else "x")
        print(f"{xq:+6.2f} | {ws} {ts} | {es} {eq} | {ps} {ma}  {mark}")

    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    payload = {
        "config": args.config, "run": name,
        "setup": {"M": M, "x0": X0, "sigma": SIGMA,
                  "t_start": T_START, "t_end": T_END},
        "scan": {"xq_min": args.xq_min, "xq_max": args.xq_max, "n": args.n,
                 "half_window": args.half_window},
        "theory": {"omega_M": THEORY_OMEGA, "tau_over_M": THEORY_TAU},
        "recommended": {
            "method": "argmin(plateau_std + 0.1 * |m4-m3|)",
            "xq": best_xq, "idx": best_i,
            "plateau_std_omega": float(plateau_std[best_i]),
            "method_agreement": float(method_agreement[best_i]),
            "FNO_m4_omega_M": float(fno_w[best_i]),
            "FNO_m4_tau_over_M": float(fno_tau[best_i]),
        },
        "validation_best": {
            "xq": float(xqs[best_val_i]),
            "FNO_m4_omega_M": float(fno_w[best_val_i]),
            "FNO_m4_tau_over_M": float(fno_tau[best_val_i]),
        },
        "records": records,
        "diagnostics": {
            "xqs": xqs.tolist(),
            "fno_w_m4": fno_w.tolist(),
            "fno_tau_m4": fno_tau.tolist(),
            "gt_w_m4": gt_w.tolist(),
            "gt_tau_m4": gt_tau.tolist(),
            "plateau_std": plateau_std.tolist(),
            "method_agreement": method_agreement.tolist(),
            "valid_fno": valid_fno.tolist(),
        },
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

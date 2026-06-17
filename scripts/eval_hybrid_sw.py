"""Evaluate the hybrid (coarse FD + FNO residual) surrogate end-to-end.

Pipeline (matches the rest of the project):
  1. Build fine/coarse/upsampled fields for the test split.
  2. Predict residual delta with the trained model -> psi_hybrid.
  3. For each test sample and each xq in args.xq, run all FIVE QNM extractors
     (M1 FFT+log-linear, M2 NLS, M3 ESPRIT, M4 two-mode plateau, M5 2-D scan)
     on each of three time series:
       (a) hybrid prediction,
       (b) baseline coarse-up (does the FNO help at all?),
       (c) fine-FD truth (intrinsic floor for our discretisation).
     Theory window [10, 50] M -- project convention.
  4. Aggregate median percent-error across samples.
  5. Print a comparison table that also includes the previously reported
     best PINN-forward and pure-FNO numbers from
     outputs/reportable_results.json.

Usage:
    python scripts/eval_hybrid_sw.py --config configs/hybrid_sw_train_k2_h64.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from typing import Any, Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.hybrid_data_pipe import assemble_split
from src.hybrid_dataset import load_dataset
from src.hybrid_fno import build_hybrid_fno
from src.qnm import (
    qnm_method_1,
    qnm_method_2,
    qnm_method_3_esprit,
    qnm_method_4_window_scan,
    qnm_method_5_2d_scan,
    percentage_errors,
    theory_ref,
)

warnings.filterwarnings("ignore")  # curve_fit / overflow noise from many fits


# ---------------------------------------------------------------------------
# Reference baselines previously reported in the project (single-sample numbers
# pulled from outputs/reportable_results.json and outputs/qnm/*_summary.json).
# Printed alongside our table for context; NOT used in any pass/fail logic.
# ---------------------------------------------------------------------------
PROJECT_BASELINES: Dict[str, Dict[str, Dict[str, float]]] = {
    "PINN forward (zerilli_l2_greedy_f03_lbfgs30k)": {
        "M1":  {"omega": 0.6793,  "tau": 2.832},
        "M2":  {"omega": 1.5380,  "tau": 1.991},
        "M3":  {"omega": 29.226,  "tau": 30.279},   # ill-conditioned per JSON
        "M4":  {"omega": 0.0286,  "tau": 2.653},
        "M5":  {"omega": 0.2305,  "tau": 3.905},
    },
    "Pure FNO @ xq=10 (sample 0, M=0.878)": {
        "M1":  {"omega": 1.5048,  "tau": 10.012},
        "M2":  {"omega": 4.3763,  "tau": 38.505},
        "M3":  {"omega": 0.8133,  "tau": 5.531},
        "M4":  {"omega": 0.1187,  "tau": 0.408},
        "M5":  {"omega": 0.1675,  "tau": 0.622},
    },
    "Fine-FD GT @ xq=10 (sample 0, M=0.878)": {
        "M1":  {"omega": 1.5468,  "tau": 9.925},
        "M2":  {"omega": 4.4274,  "tau": 38.382},
        "M3":  {"omega": 0.8536,  "tau": 5.450},
        "M4":  {"omega": 0.0707,  "tau": 0.301},
        "M5":  {"omega": 0.1202,  "tau": 0.486},
    },
}


def _safe(fn, *args, **kwargs) -> Dict[str, float]:
    """Call a QNM extractor, return {omega, tau} or NaNs on failure."""
    try:
        r = fn(*args, **kwargs)
        return {"omega": float(r.get("omega", float("nan"))),
                "tau":   float(r.get("tau",   float("nan")))}
    except Exception:
        return {"omega": float("nan"), "tau": float("nan")}


def _all_methods(t: np.ndarray, y: np.ndarray, t_start: float, t_end: float,
                 potential: str, ell: int, M: float) -> Dict[str, Dict[str, float]]:
    """Run M1..M5 and return per-method dimensionless quantities + pct errors."""
    raw = {
        "M1": _safe(qnm_method_1, t, y, t_start, t_end),
        "M2": _safe(qnm_method_2, t, y, t_start, t_end),
        "M3": _safe(qnm_method_3_esprit, t, y, t_start, t_end, K=4),
        "M4": _safe(qnm_method_4_window_scan, t, y,
                    t_start_min=t_start, t_start_max=t_start + 8.0,
                    t_end=t_end, n_starts=12,
                    potential=potential, ell=ell),
        "M5": _safe(qnm_method_5_2d_scan, t, y,
                    t_start_min=t_start, t_start_max=t_start + 8.0,
                    t_end_min=t_end - 10.0, t_end_max=t_end,
                    n_starts=8, n_ends=5,
                    potential=potential, ell=ell),
    }
    out: Dict[str, Dict[str, float]] = {}
    for name, res in raw.items():
        err = percentage_errors(res, potential=potential, ell=ell, M=M)
        out[name] = {
            "omega_dim": err.get("omega_dim", float("nan")),
            "tau_dim":   err.get("tau_dim",   float("nan")),
            "omega_pct_err": err.get("omega_pct_err", float("nan")),
            "tau_pct_err":   err.get("tau_pct_err",   float("nan")),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None,
                    help="Override model checkpoint (default: <out_dir>/model_best.pt).")
    ap.add_argument("--n", type=int, default=0,
                    help="If >0, restrict to first n test samples.")
    ap.add_argument("--xq", type=float, nargs="+", default=[2.0, 10.0],
                    help="Extraction radii (default 2.0 10.0).")
    ap.add_argument("--t_start", type=float, default=10.0,
                    help="QNM fit window start (project convention 10M).")
    ap.add_argument("--t_end", type=float, default=50.0,
                    help="QNM fit window end (project convention 50M).")
    ap.add_argument("--ell", type=int, default=2)
    ap.add_argument("--potential", default="zerilli")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg["logging"]["out_dir"]
    ckpt_path = args.ckpt or os.path.join(out_dir, "model_best.pt")
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"No checkpoint at {ckpt_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[EVAL] device={device}  ckpt={ckpt_path}")

    # ---- data ----------------------------------------------------------------
    ds_path = cfg["dataset"]["path"]
    print(f"[EVAL] loading {ds_path}")
    splits, grid, meta = load_dataset(ds_path)
    test = splits["test"]
    n_total = test["Phi_fine"].shape[0]
    n_eval = n_total if args.n <= 0 else min(args.n, n_total)
    print(f"[EVAL] test: {n_total} samples, evaluating {n_eval}")

    t0 = time.time()
    X_te, Y_te, up_te = assemble_split(
        test, grid.x_coarse, grid.t_coarse, grid.x_fine, grid.t_fine,
    )
    print(f"[EVAL] assembled in {time.time()-t0:.1f}s  X_te {X_te.shape}")

    # ---- model ---------------------------------------------------------------
    model = build_hybrid_fno(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    # A raw state_dict saved via torch.save(model.state_dict(), ...) carries an
    # OrderedDict `_metadata` attribute that round-trips as an unexpected key on
    # a strict load; drop it before loading.
    if isinstance(state, dict):
        state.pop("_metadata", None)
    model.load_state_dict(state)
    model.eval()

    # ---- predict residual, reconstruct fields --------------------------------
    bs = 4
    delta_pred = np.empty_like(Y_te)
    t0 = time.time()
    with torch.no_grad():
        for i0 in range(0, n_eval, bs):
            i1 = min(i0 + bs, n_eval)
            xb = torch.from_numpy(X_te[i0:i1]).to(device, non_blocking=True)
            pb = model(xb).cpu().numpy()
            delta_pred[i0:i1] = pb
    print(f"[EVAL] inference {time.time()-t0:.1f}s for {n_eval} samples")

    psi_fine = test["Phi_fine"][:n_eval].astype(np.float64)
    psi_up   = up_te[:n_eval].astype(np.float64)
    psi_hyb  = psi_up + delta_pred[:n_eval, 0].astype(np.float64)

    def _mse(a, b):
        return float(np.mean((a - b) ** 2))
    def _rl2(a, b):
        diff = a - b
        num = np.sqrt(np.mean(diff ** 2, axis=(1, 2)))
        den = np.sqrt(np.mean(a ** 2, axis=(1, 2))) + 1e-30
        return float(np.mean(num / den))

    mse_hyb  = _mse(psi_fine, psi_hyb)
    mse_base = _mse(psi_fine, psi_up)
    rl2_hyb  = _rl2(psi_fine, psi_hyb)
    rl2_base = _rl2(psi_fine, psi_up)
    print(f"[EVAL] field MSE :  hybrid={mse_hyb:.3e}   baseline(coarse-up)={mse_base:.3e}"
          f"   ratio={mse_hyb/mse_base:.3e}")
    print(f"[EVAL] field rL2 :  hybrid={rl2_hyb:.3e}   baseline(coarse-up)={rl2_base:.3e}"
          f"   ratio={rl2_hyb/rl2_base:.3e}")

    # ---- QNM extraction ------------------------------------------------------
    t_fine = grid.t_fine.astype(np.float64)
    x_fine = grid.x_fine.astype(np.float64)
    t_start = float(args.t_start)
    t_end = float(args.t_end)

    ref = theory_ref(args.potential, args.ell)
    print(f"[EVAL] QNM theory ({args.potential} l={args.ell}): "
          f"M*omega={ref['omega']:.4f}  tau/M={ref['tau']:.4f}")
    print(f"[EVAL] QNM fit window: t in [{t_start:.1f}, {t_end:.1f}]  (project convention)")

    P = test["P"][:n_eval]
    per_sample: List[Dict[str, Any]] = []
    accum: Dict[str, List[float]] = {}

    t0 = time.time()
    for i in range(n_eval):
        M_i = float(P[i, 0])
        rec: Dict[str, Any] = {"i": i, "M": M_i, "x0": float(P[i, 1]),
                                "sigma": float(P[i, 2])}
        for xq in args.xq:
            ix = int(np.argmin(np.abs(x_fine - float(xq))))
            for tag, field in [
                ("hyb",  psi_hyb[i, :, ix]),
                ("base", psi_up[i, :, ix]),
                ("fine", psi_fine[i, :, ix]),
            ]:
                methods = _all_methods(t_fine, field, t_start, t_end,
                                       args.potential, args.ell, M_i)
                rec[f"xq{xq:g}_{tag}"] = methods
                for mname, mres in methods.items():
                    for k in ("omega_pct_err", "tau_pct_err"):
                        key = f"xq{xq:g}_{tag}_{mname}_{k}"
                        accum.setdefault(key, []).append(mres[k])
        per_sample.append(rec)
        if (i + 1) % 10 == 0 or i == n_eval - 1:
            print(f"[EVAL] QNM sample {i+1}/{n_eval} done ({time.time()-t0:.0f}s)")

    # aggregate
    summary: Dict[str, Any] = {
        "n_eval": n_eval,
        "field": {
            "mse_hybrid":   mse_hyb,
            "mse_baseline": mse_base,
            "rl2_hybrid":   rl2_hyb,
            "rl2_baseline": rl2_base,
        },
        "qnm_window": {"t_start": t_start, "t_end": t_end,
                       "xq_list": list(args.xq)},
        "qnm_theory": ref,
        "qnm_pct_err_median": {},
        "qnm_pct_err_mean":   {},
        "qnm_pct_err_max":    {},
    }
    for key, vals in accum.items():
        v = np.array([x for x in vals if np.isfinite(x)])
        if v.size == 0:
            summary["qnm_pct_err_median"][key] = float("nan")
            summary["qnm_pct_err_mean"][key]   = float("nan")
            summary["qnm_pct_err_max"][key]    = float("nan")
        else:
            summary["qnm_pct_err_median"][key] = float(np.median(v))
            summary["qnm_pct_err_mean"][key]   = float(np.mean(v))
            summary["qnm_pct_err_max"][key]    = float(np.max(v))

    eval_dir = os.path.join(out_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    with open(os.path.join(eval_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(eval_dir, "per_sample.json"), "w") as f:
        json.dump(per_sample, f, indent=2)
    print(f"\n[EVAL] wrote {eval_dir}/summary.json and per_sample.json")

    # ---- pretty print --------------------------------------------------------
    methods = ["M1", "M2", "M3", "M4", "M5"]
    for xq in args.xq:
        print("\n" + "=" * 92)
        print(f"  Median % error over {n_eval} test samples, xq = {xq:g}, "
              f"window [{t_start:g}, {t_end:g}] M")
        print("=" * 92)
        print(f"  {'source':<28}  " + "  ".join(f"{m:>11}" for m in methods))
        for tag, label in [("hyb",  "this hybrid h64"),
                           ("base", "coarse-up (no FNO)"),
                           ("fine", "fine-FD (truth)")]:
            row_o = [summary["qnm_pct_err_median"]
                     [f"xq{xq:g}_{tag}_{m}_omega_pct_err"] for m in methods]
            row_t = [summary["qnm_pct_err_median"]
                     [f"xq{xq:g}_{tag}_{m}_tau_pct_err"] for m in methods]
            print(f"  {label+' (omega %)':<28}  "
                  + "  ".join(f"{v:>11.4f}" for v in row_o))
            print(f"  {label+' (tau   %)':<28}  "
                  + "  ".join(f"{v:>11.4f}" for v in row_t))

    print("\n" + "=" * 92)
    print("  PREVIOUSLY REPORTED PROJECT BASELINES (single-sample, for context only)")
    print("=" * 92)
    print(f"  {'baseline':<50}  " + "  ".join(f"{m:>11}" for m in methods))
    for label, mdict in PROJECT_BASELINES.items():
        row_o = [mdict[m]["omega"] for m in methods]
        row_t = [mdict[m]["tau"]   for m in methods]
        print(f"  {label+' (omega %)':<50}  "
              + "  ".join(f"{v:>11.4f}" for v in row_o))
        print(f"  {label+' (tau   %)':<50}  "
              + "  ".join(f"{v:>11.4f}" for v in row_t))

    # acceptance criterion (M4 omega at xq=2)
    print("\n" + "=" * 92)
    print("  Acceptance: this-hybrid M4 omega %err <= 2 * this-fine M4 omega %err")
    print("=" * 92)
    for xq in args.xq:
        om_h = summary["qnm_pct_err_median"][f"xq{xq:g}_hyb_M4_omega_pct_err"]
        om_f = summary["qnm_pct_err_median"][f"xq{xq:g}_fine_M4_omega_pct_err"]
        verdict = ("PASS" if (np.isfinite(om_h) and np.isfinite(om_f)
                              and om_h <= 2.0 * om_f) else "FAIL")
        print(f"    xq={xq:g}:  hybrid M4 omega %err = {om_h:.4f}   "
              f"2 x fine M4 omega %err = {2*om_f:.4f}   -> {verdict}")


if __name__ == "__main__":
    main()

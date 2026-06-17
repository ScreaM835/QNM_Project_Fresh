"""
Aggregate full-100 QNM extraction for an FNO run, matching the v2 schema in
outputs/qnm/fno_xq2_comparison/full100_qnm_xq2.json.

For each of the 100 test predictions in
    outputs/fno/<run>/predictions/test/sample_NNN.npz
load the waveform at the slice x = xq (default 2.0) and run methods 1-4 on
both the GT (`phi_true`) and the FNO (`phi_pred`) trace. Save percentage
errors against Schwarzschild l=2 theory (Mω=0.3737, τ/M=11.241).

Usage:
    python scripts/aggregate_full100_qnm.py \
        --run fno_zerilli_l2_v3 \
        --out outputs/qnm/fno_xq2_comparison/full100_qnm_xq2_v3.json \
        --t_start 10 --t_end 100 --xq 2.0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.qnm import (
    qnm_method_1,
    qnm_method_2,
    qnm_method_3_esprit,
    qnm_method_4_window_scan,
    percentage_errors,
)


def _safe(result, M):
    """Run percentage_errors and merge with omega/tau; return safe dict."""
    try:
        pct = percentage_errors(result, potential="zerilli", ell=2, M=M)
        return {
            "omega_M": pct.get("omega_dim", float("nan")),
            "tau_over_M": pct.get("tau_dim", float("nan")),
            "omega_pct_err": pct.get("omega_pct_err", float("nan")),
            "tau_pct_err": pct.get("tau_pct_err", float("nan")),
        }
    except Exception:
        return {
            "omega_M": float("nan"),
            "tau_over_M": float("nan"),
            "omega_pct_err": float("nan"),
            "tau_pct_err": float("nan"),
        }


def _nan_to_none(x):
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return None
    return x


def _clean(d):
    return {k: _nan_to_none(v) for k, v in d.items()}


def extract_one(t, y, M, t_start, t_end):
    """Run M1-M4 on a single trace and return {m1,m2,m3,m4} dict."""
    try:
        r1 = qnm_method_1(t, y, t_start, t_end)
    except Exception:
        r1 = {"omega": float("nan"), "tau": float("nan")}
    try:
        r2 = qnm_method_2(t, y, t_start, t_end)
    except Exception:
        r2 = {"omega": float("nan"), "tau": float("nan")}
    try:
        r3 = qnm_method_3_esprit(t, y, t_start, t_end, K=4)
    except Exception:
        r3 = {"omega": float("nan"), "tau": float("nan")}
    try:
        r4 = qnm_method_4_window_scan(
            t, y, t_start_min=t_start, t_start_max=t_start + 10.0,
            t_end=t_end, n_starts=12,
        )
    except Exception:
        r4 = {"omega": float("nan"), "tau": float("nan")}

    return {
        "m1": _clean(_safe(r1, M)),
        "m2": _clean(_safe(r2, M)),
        "m3": _clean(_safe(r3, M)),
        "m4": _clean(_safe(r4, M)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True,
                    help="FNO run name, e.g. fno_zerilli_l2_v3")
    ap.add_argument("--root", default="outputs/fno",
                    help="root containing the run directory")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--xq", type=float, default=2.0)
    ap.add_argument("--t_start", type=float, default=10.0)
    ap.add_argument("--t_end", type=float, default=100.0)
    ap.add_argument("--n_max", type=int, default=100)
    args = ap.parse_args()

    pred_dir = Path(args.root) / args.run / "predictions" / "test"
    files = sorted(pred_dir.glob("sample_*.npz"))[: args.n_max]
    if not files:
        raise SystemExit(f"No sample files in {pred_dir}")

    print(f"[agg] {args.run}: {len(files)} samples, xq={args.xq}, "
          f"t=[{args.t_start},{args.t_end}]")

    records = []
    for k, fp in enumerate(files):
        d = np.load(fp)
        x = d["x"].astype(float)
        t = d["t"].astype(float)
        phi_gt = d["phi_true"].astype(float)     # (Nt, Nx)
        phi_fn = d["phi_pred"].astype(float)
        M = float(d["M"])
        x0 = float(d["x0"])
        sigma = float(d["sigma"])

        # Pick nearest x index to xq.
        ix = int(np.argmin(np.abs(x - args.xq)))

        # Determine layout: phi shape is (Nt, Nx) or (Nx, Nt). From the
        # inspection: shape (401,401). Use the dim matching t to be safe.
        if phi_gt.shape[0] == t.size:
            y_gt = phi_gt[:, ix]
            y_fn = phi_fn[:, ix]
        else:
            y_gt = phi_gt[ix, :]
            y_fn = phi_fn[ix, :]

        peak_gt = float(np.max(np.abs(y_gt)))
        peak_fn = float(np.max(np.abs(y_fn)))

        gt = extract_one(t, y_gt, M, args.t_start, args.t_end)
        fn = extract_one(t, y_fn, M, args.t_start, args.t_end)

        records.append({
            "i": k,
            "M": M,
            "x0": x0,
            "sigma": sigma,
            "peak_gt": peak_gt,
            "peak_fn": peak_fn,
            "GT": gt,
            "FNO": fn,
        })
        if (k + 1) % 10 == 0:
            print(f"  [{k+1}/{len(files)}]")

    out = {
        "xq": args.xq,
        "theory_omega_M": 0.3737,
        "theory_tau_over_M": 11.241,
        "n_samples": len(records),
        "t_start": args.t_start,
        "t_end": args.t_end,
        "run": args.run,
        "records": records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[agg] wrote {args.out}")


if __name__ == "__main__":
    main()

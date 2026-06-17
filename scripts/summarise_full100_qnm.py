"""Summarise full100_qnm_xq2_*.json files (v2/v3/v4) into a comparison table.

For each (run, source ∈ {GT, FNO}, method ∈ {m1..m4}) report:
  - n_valid
  - median omega_pct_err, median tau_pct_err
  - pass<5% rate (both omega and tau within 5%)
  - pass<1% rate (both within 1%)
  - tau-guard rejects: tau<=0 or tau/M>50 (M=1 normalized, so tau/M>50)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _is_finite(x):
    return x is not None and np.isfinite(float(x))


THEORY_OMEGA_M = 0.3737
THEORY_TAU_OVER_M = 11.241


def summarise_run(path, label):
    d = json.load(open(path))
    records = d["records"]
    out = {
        "label": label,
        "path": str(path),
        "n_samples": len(records),
        "t_end": d.get("t_end", "?"),
        "theory_omega_M": THEORY_OMEGA_M,
        "theory_tau_over_M": THEORY_TAU_OVER_M,
        "by": {},
    }

    for src in ("GT", "FNO"):
        for m in ("m1", "m2", "m3", "m4"):
            om_vals, ta_vals = [], []     # raw Mω and τ/M
            om_pct, ta_pct = [], []       # % errors
            n_rej = 0
            pass5 = pass1 = 0
            for r in records:
                rec = r[src][m]
                tau_over_M = rec.get("tau_over_M")
                # tau-guard
                if tau_over_M is not None and np.isfinite(tau_over_M):
                    if tau_over_M <= 0 or tau_over_M > 50:
                        n_rej += 1
                        continue
                om_M = rec.get("omega_M")
                if _is_finite(om_M):
                    om_vals.append(float(om_M))
                if _is_finite(tau_over_M):
                    ta_vals.append(float(tau_over_M))
                op = rec.get("omega_pct_err")
                tp = rec.get("tau_pct_err")
                if _is_finite(op):
                    om_pct.append(float(op))
                if _is_finite(tp):
                    ta_pct.append(float(tp))
                if _is_finite(op) and _is_finite(tp):
                    if op < 5 and tp < 5:
                        pass5 += 1
                    if op < 1 and tp < 1:
                        pass1 += 1

            n_total = len(records) - n_rej
            med_om = float(np.median(om_vals)) if om_vals else None
            med_ta = float(np.median(ta_vals)) if ta_vals else None
            abs_err_omega = (abs(med_om - THEORY_OMEGA_M)
                             if med_om is not None else None)
            abs_err_tau = (abs(med_ta - THEORY_TAU_OVER_M)
                           if med_ta is not None else None)

            out["by"][f"{src}/{m}"] = {
                "n_valid_omega": len(om_vals),
                "n_valid_tau": len(ta_vals),
                "n_rejected_tau_guard": n_rej,
                # raw extracted (dimensionless) medians
                "median_omega_M": med_om,
                "median_tau_over_M": med_ta,
                # absolute errors vs theory
                "abs_err_omega_M": abs_err_omega,
                "abs_err_tau_over_M": abs_err_tau,
                # percentage errors (median of per-sample %-errors)
                "median_omega_pct": (float(np.median(om_pct))
                                     if om_pct else None),
                "median_tau_pct": (float(np.median(ta_pct))
                                   if ta_pct else None),
                # pass rates over all 100 samples (post tau-guard denominator)
                "pass_lt_5pct": pass5 / max(1, n_total),
                "pass_lt_1pct": pass1 / max(1, n_total),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/qnm/fno_xq2_comparison/summary_v2_v3_v4.json")
    args = ap.parse_args()

    root = Path("outputs/qnm/fno_xq2_comparison")
    runs = [
        (root / "full100_qnm_xq2.json", "v2"),
        (root / "full100_qnm_xq2_v3.json", "v3"),
        (root / "full100_qnm_xq2_v4.json", "v4"),
    ]

    summary = {"runs": []}
    for path, label in runs:
        if not path.exists():
            print(f"[skip] {path}")
            continue
        s = summarise_run(path, label)
        summary["runs"].append(s)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"[wrote] {args.out}")

    # Pretty print compact table.
    print(f"\nTheory:  Mω = {THEORY_OMEGA_M},  τ/M = {THEORY_TAU_OVER_M}")
    print("=== Raw extracted medians and errors ===")
    methods = ("m1", "m2", "m3", "m4")
    for src in ("GT", "FNO"):
        print(f"\n--- {src} ---")
        print(f"{'run':<4} {'m':<3} {'n':>3}  "
              f"{'med Mω':>8} {'med τ/M':>8}   "
              f"{'|Δω|':>8} {'|Δτ|':>8}   "
              f"{'ω%':>6} {'τ%':>6}   {'p<5%':>5} {'p<1%':>5}")
        for r in summary["runs"]:
            for m in methods:
                b = r["by"].get(f"{src}/{m}", {})
                mo = b.get("median_omega_M")
                mt = b.get("median_tau_over_M")
                eo = b.get("abs_err_omega_M")
                et = b.get("abs_err_tau_over_M")
                pom = b.get("median_omega_pct")
                pta = b.get("median_tau_pct")
                p5 = b.get("pass_lt_5pct")
                p1 = b.get("pass_lt_1pct")
                n = b.get("n_valid_omega", 0)
                f = lambda x, w, p: (f"{x:>{w}.{p}f}" if x is not None
                                     else f"{'nan':>{w}}")
                print(f"{r['label']:<4} {m:<3} {n:>3}  "
                      f"{f(mo,8,4)} {f(mt,8,3)}   "
                      f"{f(eo,8,4)} {f(et,8,3)}   "
                      f"{f(pom,5,2)}% {f(pta,5,2)}%   "
                      f"{(100*p5):>4.0f}% {(100*p1):>4.0f}%")


if __name__ == "__main__":
    main()

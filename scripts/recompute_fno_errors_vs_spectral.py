"""Recompute FNO v4 QNM errors against the spectral reference.

Why this exists
---------------
The headline metric for FNO v4 ("M4-on-FD-GT floor: omega 0.055%, tau 0.275%")
compares the M4 fit on the FNO prediction against the M4 fit on the FD ground
truth. That conflates two distinct error sources:

  (a) The extractor itself is imperfect on finite-duration ringdown data.
  (b) The FNO prediction differs from the true solution.

This script disentangles them by introducing a high-accuracy spectral
reference value (computed via Chebyshev+PML, validated to 1e-7 on
M*omega = 0.3737 - 0.0890i, see scripts/validate_spectral_solver.py).

For each FNO test sample i it reports three percentage errors per (omega, tau):
  E_GT_vs_spec  = |M4(phi_true)  - spectral| / |spectral|   "extractor floor"
  E_FNO_vs_spec = |M4(phi_pred)  - spectral| / |spectral|   "true FNO error"
  E_FNO_vs_GT   = |M4(phi_pred)  - M4(phi_true)| / |M4(phi_true)| "current metric"

Aggregates (median / p90) are written to JSON.
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.potentials import V_of_x
from src.qnm import qnm_method_4_window_scan
from src.spectral_qnm_hyperboloidal import solve_schwarzschild_convergent


# ---------------------------------------------------------------------------
# Spectral reference for Schwarzschild Zerilli l=2 fundamental
# ---------------------------------------------------------------------------
def spectral_reference_M_times_omega(ell: int = 2) -> complex:
    """Compute M*omega for Schwarzschild Zerilli l=2 fundamental.

    Uses the hyperboloidal Chebyshev solver (JMS framework,
    src/spectral_qnm_hyperboloidal.py), which matches Leaver to ~1e-10
    and is the same reference used by src.qnm.theory_ref().
    Returns a single complex number; per-sample omega is then Mw / M.
    """
    res = solve_schwarzschild_convergent(
        ell=ell, N=80, dN=40, M=1.0, parity='Z', tol=1e-3, n_return=4,
    )
    pos = sorted([w for w in res.omegas if w.real > 0], key=lambda w: -w.imag)
    if not pos:
        raise RuntimeError("hyperboloidal solver returned no positive-Re modes")
    return complex(pos[0])


# ---------------------------------------------------------------------------
# QNM extraction at observer xq
# ---------------------------------------------------------------------------
def extract_m4(t: np.ndarray, x: np.ndarray, phi: np.ndarray, xq: float,
               t_start: float, t_end: float) -> tuple[float, float]:
    """Run Method 4 (window-scan two-mode fit) on phi[:, ix(xq)]."""
    ix = int(np.argmin(np.abs(x - xq)))
    y = phi[:, ix]
    res = qnm_method_4_window_scan(
        t, y, t_start_min=t_start, t_start_max=t_start + 15.0,
        t_end=t_end, n_starts=16, potential="zerilli", ell=2,
    )
    return float(res["omega"]), float(res["tau"])


def pct_err(a: float, b: float) -> float:
    if not (np.isfinite(a) and np.isfinite(b)) or b == 0:
        return float("nan")
    return 100.0 * abs(a - b) / abs(b)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default="outputs/fno/fno_zerilli_l2_v4/predictions/test",
                    help="Directory of FNO prediction sample_NNN.npz files.")
    ap.add_argument("--xq", type=float, default=10.0, help="Observer x location.")
    ap.add_argument("--t-start", type=float, default=10.0)
    ap.add_argument("--t-end", type=float, default=50.0)
    ap.add_argument("--out", default="outputs/fno/fno_zerilli_l2_v4/errors_vs_spectral.json")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Optional cap on number of samples (debug).")
    args = ap.parse_args()

    pred_dir = os.path.join(ROOT, args.pred_dir)
    files = sorted(f for f in os.listdir(pred_dir) if f.endswith(".npz"))
    if args.max_samples:
        files = files[: args.max_samples]
    print(f"[info] {len(files)} samples in {args.pred_dir}")

    print("[info] Computing spectral reference M*omega ...")
    Mw_spec = spectral_reference_M_times_omega(ell=2)
    omegaM_spec = float(Mw_spec.real)
    tauOverM_spec = float(-1.0 / Mw_spec.imag)
    print(f"[info] spectral M*omega = {omegaM_spec:.7f}   tau/M = {tauOverM_spec:.7f}")

    rows = []
    for fi, fname in enumerate(files):
        d = np.load(os.path.join(pred_dir, fname))
        x = np.asarray(d["x"], dtype=float)
        t = np.asarray(d["t"], dtype=float)
        phi_pred = np.asarray(d["phi_pred"], dtype=float)
        phi_true = np.asarray(d["phi_true"], dtype=float)
        M = float(d["M"])

        # raw (code-time) omega/tau
        try:
            w_pred, tau_pred = extract_m4(t, x, phi_pred, args.xq, args.t_start, args.t_end)
        except Exception as e:
            w_pred, tau_pred = float("nan"), float("nan")
        try:
            w_true, tau_true = extract_m4(t, x, phi_true, args.xq, args.t_start, args.t_end)
        except Exception:
            w_true, tau_true = float("nan"), float("nan")

        # Rescale to dimensionless M*omega, tau/M for comparison
        Mw_pred = M * w_pred
        Mw_true = M * w_true
        tauM_pred = tau_pred / M
        tauM_true = tau_true / M

        # Three percentage-error metrics
        row = {
            "i": fi,
            "file": fname,
            "M": M,
            "Mw_pred": Mw_pred,
            "Mw_true": Mw_true,
            "tauM_pred": tauM_pred,
            "tauM_true": tauM_true,
            # extractor-floor: GT extraction vs spectral truth
            "E_omega_GT_vs_spec": pct_err(Mw_true, omegaM_spec),
            "E_tau_GT_vs_spec":   pct_err(tauM_true, tauOverM_spec),
            # honest FNO error: FNO extraction vs spectral truth
            "E_omega_FNO_vs_spec": pct_err(Mw_pred, omegaM_spec),
            "E_tau_FNO_vs_spec":   pct_err(tauM_pred, tauOverM_spec),
            # current paper metric: FNO vs GT-extractor
            "E_omega_FNO_vs_GT": pct_err(Mw_pred, Mw_true),
            "E_tau_FNO_vs_GT":   pct_err(tauM_pred, tauM_true),
        }
        rows.append(row)
        if (fi + 1) % 10 == 0:
            print(f"  [{fi+1}/{len(files)}] sample done")

    # Aggregate
    def stats(key: str) -> dict:
        vals = np.array([r[key] for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return {"n": 0, "median": float("nan"), "p90": float("nan"),
                    "mean": float("nan"), "max": float("nan")}
        return {
            "n": int(vals.size),
            "median": float(np.median(vals)),
            "p90": float(np.percentile(vals, 90)),
            "mean": float(vals.mean()),
            "max": float(vals.max()),
        }

    metric_keys = [
        "E_omega_GT_vs_spec", "E_tau_GT_vs_spec",
        "E_omega_FNO_vs_spec", "E_tau_FNO_vs_spec",
        "E_omega_FNO_vs_GT", "E_tau_FNO_vs_GT",
    ]
    aggregates = {k: stats(k) for k in metric_keys}

    # Stratified: only samples where the GT extractor itself is meaningful.
    # "Meaningful" = GT extraction agrees with spectral truth to <5% on both
    # omega and tau. This is the subset on which the FNO test metric is
    # actually evaluable; the rest are degenerate ringdowns at xq.
    clean_rows = [r for r in rows
                  if (r["E_omega_GT_vs_spec"] < 5.0
                      and r["E_tau_GT_vs_spec"] < 5.0)]
    n_clean = len(clean_rows)

    def stats_subset(key: str, subset):
        vals = np.array([r[key] for r in subset], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return {"n": 0, "median": float("nan"), "p90": float("nan"),
                    "mean": float("nan"), "max": float("nan")}
        return {
            "n": int(vals.size),
            "median": float(np.median(vals)),
            "p90": float(np.percentile(vals, 90)),
            "mean": float(vals.mean()),
            "max": float(vals.max()),
        }

    aggregates_clean = {k: stats_subset(k, clean_rows) for k in metric_keys}

    out = {
        "spectral_reference": {
            "M_times_omega_real": omegaM_spec,
            "M_times_omega_imag": float(Mw_spec.imag),
            "tau_over_M": tauOverM_spec,
            "ell": 2,
            "potential": "zerilli",
            "method": "Hyperboloidal Chebyshev (JMS 2021), src.spectral_qnm_hyperboloidal.solve_schwarzschild_convergent",
            "validated_accuracy": "1e-10 on Leaver reference",
        },
        "observer": {"xq": args.xq, "t_start": args.t_start, "t_end": args.t_end},
        "n_total": len(rows),
        "n_clean_subset": n_clean,
        "clean_filter": "GT-vs-spectral < 5% on both omega and tau",
        "aggregates_all": aggregates,
        "aggregates_clean": aggregates_clean,
        "per_sample": rows,
    }

    out_path = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] wrote {out_path}")

    # Human-readable summary
    def print_block(title, agg):
        print()
        print("=" * 78)
        print(f"  {title}")
        print("=" * 78)
        fmt = ("  {label:<32}  omega: {om_med:>7.3f} / {om_p90:>7.3f}   "
               "tau: {tm_med:>7.3f} / {tm_p90:>7.3f}")
        for label, kw, kt in [
            ("Extractor floor (GT vs spec)", "E_omega_GT_vs_spec", "E_tau_GT_vs_spec"),
            ("FNO vs GT-extractor (paper)", "E_omega_FNO_vs_GT", "E_tau_FNO_vs_GT"),
            ("FNO vs spectral truth (honest)", "E_omega_FNO_vs_spec", "E_tau_FNO_vs_spec"),
        ]:
            print(fmt.format(
                label=label,
                om_med=agg[kw]["median"], om_p90=agg[kw]["p90"],
                tm_med=agg[kt]["median"], tm_p90=agg[kt]["p90"],
            ))
        print(f"  (median / p90, percent; n = {agg['E_omega_GT_vs_spec']['n']})")
        print("=" * 78)

    print_block("ALL samples", aggregates)
    print_block(f"CLEAN subset (GT-vs-spec < 5%, n={n_clean}/{len(rows)})",
                aggregates_clean)


if __name__ == "__main__":
    main()

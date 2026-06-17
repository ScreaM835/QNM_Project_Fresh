"""Competence-topology analysis of the hybrid (prior + FNO correction) field.

Question (user): the FNO degrades the smooth / high-t / high-x regions where the
coarse prior is already machine-exact, and under-corrects the high-gradient
wavefronts where only the FNO can help. Prior and FNO are therefore good on
*disjoint* regions. This script MAPS that topology and asks whether an
INFERENCE-SAFE rule can route each (x,t) point to the better predictor.

Everything is computed from a single canonical-BH field cache (no GPU, no fine
solve here): psi_fine (truth), psi_coarse_up (prior), psi_hybrid (prior+corr).

Definitions
-----------
    prior_err(x,t) = |fine - prior|
    fno_err(x,t)   = |fine - hybrid|
    corr(x,t)      = hybrid - prior          (the FNO output; available at deploy)
    ORACLE(x,t)    = prior if prior_err<fno_err else hybrid   (pointwise best)

The ORACLE is the lower bound on field error for ANY blend psi = prior + g*corr
with g in [0,1] restricted to {0,1}; it tells us the prize for getting the gate
right. We then test two *inference-safe* gates (computable with no truth):

    (A) |corr| gate   -- keep the FNO correction only where |corr| exceeds a
                         threshold, else trust the prior. Rationale: the FNO's
                         own correction magnitude is its estimate of how wrong
                         the prior is; the spurious smooth-region speckle sits at
                         a low |corr| floor, the genuine wavefront corrections
                         are orders of magnitude larger.
    (B) grad gate     -- keep the FNO correction only where the prior field is
                         locally steep (|grad prior| high = wavefront). Rationale:
                         FD discretisation error scales with high derivatives, so
                         |grad prior| flags exactly where the prior is deficient.

For each gate we sweep its threshold, report the field rL2 curve, and at the
field-optimal threshold report the QNM (M4/M5 at xq=2) so we see the BOTH-metrics
consequence, compared against prior / hybrid / oracle / fine.

Usage:
    venv_csd3/bin/python scripts/diag_competence_topology.py \
        [--run outputs/hybrid/fno_sw_obsloss_derisk]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.qnm import (  # noqa: E402
    qnm_method_4_window_scan,
    qnm_method_5_2d_scan,
    percentage_errors,
)

THEORY_W, THEORY_T = 0.3737, 11.241
XQ = 2.0


def rl2(fine, pred):
    return float(np.sqrt(np.mean((fine - pred) ** 2)) /
                 (np.sqrt(np.mean(fine ** 2)) + 1e-30))


def qnm_at_xq(x, t, field2d, xq=XQ):
    """M4 + M5 omega/tau %err at the extraction column (matches eval_hybrid_sw)."""
    ix = int(np.argmin(np.abs(x - xq)))
    y = field2d[:, ix].astype(np.float64)
    out = {}
    try:
        r4 = qnm_method_4_window_scan(
            t, y, t_start_min=10.0, t_start_max=18.0, t_end=50.0,
            n_starts=12, potential="zerilli", ell=2)
        e4 = percentage_errors({"omega": r4["omega"], "tau": r4["tau"]},
                               potential="zerilli", ell=2, M=1.0)
        out["M4_om"] = float(e4["omega_pct_err"])
        out["M4_ta"] = float(e4["tau_pct_err"])
    except Exception as e:
        out["M4_om"] = out["M4_ta"] = float("nan")
        out["M4_err"] = str(e)
    try:
        r5 = qnm_method_5_2d_scan(
            t, y, t_start_min=10.0, t_start_max=18.0,
            t_end_min=40.0, t_end_max=50.0, n_starts=8, n_ends=5,
            potential="zerilli", ell=2)
        e5 = percentage_errors({"omega": r5["omega"], "tau": r5["tau"]},
                               potential="zerilli", ell=2, M=1.0)
        out["M5_om"] = float(e5["omega_pct_err"])
        out["M5_ta"] = float(e5["tau_pct_err"])
    except Exception as e:
        out["M5_om"] = out["M5_ta"] = float("nan")
        out["M5_err"] = str(e)
    return out


def smoothstep(z):
    z = np.clip(z, 0.0, 1.0)
    return z * z * (3.0 - 2.0 * z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/hybrid/fno_sw_obsloss_derisk",
                    help="run dir holding field_cache/canonical.npz")
    args = ap.parse_args()

    cache = ROOT / args.run / "field_cache" / "canonical.npz"
    z = np.load(cache)
    x, t = z["x"], z["t"]
    fine = z["psi_fine"].astype(np.float64)
    prior = z["psi_coarse_up"].astype(np.float64)
    hyb = z["psi_hybrid"].astype(np.float64)
    corr = hyb - prior
    print(f"[load] {cache}")
    print(f"[grid] Nt={t.size} Nx={x.size}  x in [{x.min():.1f},{x.max():.1f}]  "
          f"t in [{t.min():.1f},{t.max():.1f}]")

    ep = np.abs(fine - prior)
    ef = np.abs(fine - hyb)
    oracle = np.where(ef < ep, hyb, prior)

    # ---- headline field numbers ------------------------------------------
    print("\n================ FIELD (2D rL2, canonical BH) ================")
    r_prior = rl2(fine, prior)
    r_hyb = rl2(fine, hyb)
    r_oracle = rl2(fine, oracle)
    print(f"  prior  rL2 = {r_prior*100:7.3f}%")
    print(f"  hybrid rL2 = {r_hyb*100:7.3f}%   ({r_prior/r_hyb:.1f}x better than prior)")
    print(f"  ORACLE rL2 = {r_oracle*100:7.3f}%   "
          f"({r_hyb/r_oracle:.1f}x better than hybrid; pointwise-best ceiling)")
    win = ef < ep
    print(f"  FNO beats prior on {100*win.mean():.1f}% of POINTS; "
          f"prior beats FNO on {100*(~win).mean():.1f}%")
    # energy-weighted: where does the field's L2 mass actually live?
    w2 = fine ** 2
    print(f"  FNO beats prior on {100*(w2*win).sum()/w2.sum():.1f}% of FIELD ENERGY")

    # ---- clean-region contamination (the speckle metric rL2 can't see) ---
    # Where the prior is already machine-accurate, how much error does the FNO
    # ADD? This is the user's core concern: speckle where the coarse already
    # fits the fine FD. rL2 is amplitude-weighted and blind to it.
    print("\n========== CLEAN-REGION CONTAMINATION (speckle) ==========")
    for thr in (1e-5, 1e-4):
        clean = ep < thr                       # prior already this good
        frac = 100 * clean.mean()
        pe = ep[clean].mean() if clean.any() else float("nan")
        fe = ef[clean].mean() if clean.any() else float("nan")
        print(f"  prior_err < {thr:.0e}  ({frac:5.1f}% of domain): "
              f"prior {pe:.2e} -> FNO {fe:.2e}  (FNO is {fe/(pe+1e-30):.0f}x worse here)")
    # causal exterior: far right where the field is physically ~0
    ext = x > 80.0
    pe_e = ep[:, ext].mean(); fe_e = ef[:, ext].mean()
    amp_e = np.abs(fine[:, ext]).mean()
    print(f"  causal exterior x>80 (|field|~{amp_e:.1e}): "
          f"prior {pe_e:.2e} -> FNO {fe_e:.2e}  (FNO hallucination {fe_e/(pe_e+1e-30):.0f}x)")

    # ---- competence vs local prior gradient (wavefront indicator) --------
    gt, gx = np.gradient(prior, t, x)
    gmag = np.sqrt(gx ** 2 + gt ** 2)
    print("\n========== competence by |grad prior| decile (wavefront) ==========")
    print("  decile     <|grad|>    prior_err    fno_err    FNO_win%   region")
    gflat = gmag.ravel()
    order = np.argsort(gflat)
    epf, eff = ep.ravel(), ef.ravel()
    n = gflat.size
    for d in range(10):
        idx = order[d * n // 10:(d + 1) * n // 10]
        pe, fe = epf[idx].mean(), eff[idx].mean()
        wr = (eff[idx] < epf[idx]).mean()
        tag = "smooth" if d < 5 else ("front" if d >= 7 else "mid")
        print(f"   {d*10:3d}-{(d+1)*10:3d}%  {gflat[idx].mean():.3e}   "
              f"{pe:.3e}   {fe:.3e}   {100*wr:5.1f}    {tag}")

    # ---- competence vs |corr| (the FNO's own deficiency estimate) --------
    print("\n========== competence by |corr| decile (FNO output magnitude) ==========")
    print("  decile     <|corr|>    prior_err    fno_err    FNO_win%")
    cflat = np.abs(corr).ravel()
    order_c = np.argsort(cflat)
    for d in range(10):
        idx = order_c[d * n // 10:(d + 1) * n // 10]
        pe, fe = epf[idx].mean(), eff[idx].mean()
        wr = (eff[idx] < epf[idx]).mean()
        print(f"   {d*10:3d}-{(d+1)*10:3d}%  {cflat[idx].mean():.3e}   "
              f"{pe:.3e}   {fe:.3e}   {100*wr:5.1f}")

    # ---- GATE A: |corr| hard threshold (inference-safe) ------------------
    print("\n================ GATE A: keep corr only where |corr| > lambda ================")
    print("  (lambda=0 -> pure hybrid; lambda=inf -> pure prior)")
    print("  lambda      field rL2   vs_hyb   M4_om  M4_ta  M5_om  M5_ta")
    ac = np.abs(corr)
    best_A = None
    for lam in [0.0, 1e-5, 3e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3]:
        psi = prior + np.where(ac > lam, corr, 0.0)
        r = rl2(fine, psi)
        q = qnm_at_xq(x, t, psi) if lam in (0.0, 5e-5, 1e-4, 2e-4) else None
        line = f"  {lam:.1e}   {r*100:7.3f}%   {r_hyb/r:5.2f}x"
        if q:
            line += (f"   {q['M4_om']:.3f} {q['M4_ta']:.3f} "
                     f"{q['M5_om']:.3f} {q['M5_ta']:.3f}")
        print(line)
        if best_A is None or r < best_A[1]:
            best_A = (lam, r)

    # ---- GATE B: gradient percentile gate (inference-safe) ---------------
    print("\n================ GATE B: keep corr only where |grad prior| > q-th pct ================")
    print("  (q=0 -> pure hybrid; q=100 -> pure prior)")
    print("  pct      thresh      field rL2   vs_hyb   M4_om  M4_ta  M5_om  M5_ta")
    best_B = None
    for q in [0, 30, 50, 60, 70, 80, 90, 95]:
        thr = np.percentile(gmag, q) if q > 0 else -1.0
        g = (gmag > thr).astype(np.float64)
        psi = prior + g * corr
        r = rl2(fine, psi)
        qm = qnm_at_xq(x, t, psi) if q in (0, 50, 70, 90) else None
        line = f"  {q:3d}%   {thr:.3e}   {r*100:7.3f}%   {r_hyb/r:5.2f}x"
        if qm:
            line += (f"   {qm['M4_om']:.3f} {qm['M4_ta']:.3f} "
                     f"{qm['M5_om']:.3f} {qm['M5_ta']:.3f}")
        print(line)
        if best_B is None or r < best_B[1]:
            best_B = (q, r)

    # ---- reference QNM rows ----------------------------------------------
    print("\n================ QNM reference (xq=2) ================")
    for tag, fld in [("prior", prior), ("hybrid", hyb),
                     ("oracle", oracle), ("fine", fine)]:
        q = qnm_at_xq(x, t, fld)
        print(f"  {tag:7s}  M4_om {q['M4_om']:.3f}  M4_ta {q['M4_ta']:.3f}   "
              f"M5_om {q['M5_om']:.3f}  M5_ta {q['M5_ta']:.3f}")

    print(f"\n[summary] best |corr|-gate lambda={best_A[0]:.1e} -> "
          f"rL2 {best_A[1]*100:.3f}% (hyb {r_hyb*100:.3f}%, oracle {r_oracle*100:.3f}%)")
    print(f"[summary] best grad-gate pct={best_B[0]} -> "
          f"rL2 {best_B[1]*100:.3f}%")


if __name__ == "__main__":
    main()

"""Measure the TRUE high-spin QNM headroom of the coarse k4 prior at ell=4.

'Headroom' = how much a hybrid could improve the QNM = (coarse k4 prior QNM
error) - (fine QNM error). The corpus is a Sobol sweep over a/M in [0,0.95],
r0 in [8,11], w in [1.0,1.5]; a single pulse can be lucky, so we sweep a grid of
representative pulses x spins. For each we evolve the SAME audited coarse (k4,
N=201) and fine (N=801) grids, extract the QNM at scri with the FIXED
mode-selective ensemble (omega_target = Leaver (4,2,0)), and report per-spin:

  prior_qnm  : coarse k4 QNM error vs Leaver (what the cheap solve delivers)
  fine_qnm   : fine QNM error vs Leaver (the trustworthy reference / ceiling)
  headroom   : prior_qnm - fine_qnm (max QNM the hybrid could recover)
  field_L2   : coarse-vs-fine scri rel-L2 (the FIELD headroom, for contrast)

The targeted extractor is given the Leaver target only to IDENTIFY the mode (12%
gate); it reports each grid's OWN frequency, so a large prior_qnm is a real
coarse-grid error, not a snap. sel_dist (printed) guards against false matches.
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

import kerr.src.kerr_dataset as kd
from kerr.src.teukolsky_minimal_gauge import scri_index
from kerr.src.qnm_kerr_reference import kerr_qnm
from kerr.src.qnm_ensemble_kerr import extract_qnm_kerr_ensemble

ELL, MM = 4, 2


def pct(v, r):
    return abs(v - r) / abs(r) * 100.0 if (np.isfinite(v) and abs(r) > 1e-30) else float("nan")


def rel_l2(p, t):
    return float(np.linalg.norm(p - t) / max(np.linalg.norm(t), 1e-30)) * 100.0


def qnm_err(tau, y, a, r):
    wt = complex(r.M_omega_R, r.M_omega_I)
    out = extract_qnm_kerr_ensemble(tau, y.astype(np.complex128), a_over_M=a,
                                    tau_ref=r.tau_over_M, tau_final=float(tau[-1]),
                                    omega_target=wt)
    om = float(out.get("omega", np.nan))
    return pct(om, r.M_omega_R), float(out.get("sel_dist", np.nan)), int(out.get("n_omega", 0))


def _work(task):
    """One (a, r0, w): evolve coarse+fine, return (a, mw, prior_qnm, fine_qnm, L2, seld)."""
    a, r0, w = task
    kd.ELL, kd.MM = ELL, MM
    r = kerr_qnm(a_over_M=a, ell=4, m=MM, n=0)
    tau, fine, opf, _ = kd.evolve_full_field(a, kd.FINE_N, r0, w)
    _, k4, op4, _ = kd.evolve_full_field(a, kd.COARSE_N[4], r0, w)
    yf = fine[:, scri_index(opf)]
    y4 = k4[:, scri_index(op4)]
    pe, sd1, _ = qnm_err(tau, y4, a, r)
    fe, sd2, _ = qnm_err(tau, yf, a, r)
    return (a, float(r.M_omega_R), pe, fe, rel_l2(y4, yf),
            float(np.nanmax([sd1, sd2])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spins", type=float, nargs="*",
                    default=[0.0, 0.5, 0.7, 0.9, 0.95])
    ap.add_argument("--r0s", type=float, nargs="*", default=[8.5, 9.5, 10.5])
    ap.add_argument("--ws", type=float, nargs="*", default=[1.1, 1.4])
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    kd.ELL, kd.MM = ELL, MM
    pulses = [(r0, w) for r0 in args.r0s for w in args.ws]
    tasks = [(a, r0, w) for a in args.spins for (r0, w) in pulses]

    print(f"=== coarse(k4 N={kd.COARSE_N[4]}) prior QNM headroom  ell=4 m=2 ===")
    print(f"pulses: {len(pulses)} (r0 in {args.r0s}, w in {args.ws}); "
          f"{len(tasks)} evals on {args.workers} workers; targeted ensemble")

    with Pool(processes=args.workers) as pool:
        res = pool.map(_work, tasks)

    by_a = {}
    for (a, mw, pe, fe, l2, sd) in res:
        by_a.setdefault(a, {"mw": mw, "pq": [], "fq": [], "fl": [], "sd": 0.0})
        by_a[a]["pq"].append(pe); by_a[a]["fq"].append(fe)
        by_a[a]["fl"].append(l2); by_a[a]["sd"] = max(by_a[a]["sd"], sd)

    print(f"{'a/M':>5} {'Mw_ref':>8} | {'prior_qnm%':>21} | {'fine_qnm%':>21} | "
          f"{'headroom%':>10} | {'field_L2%':>18}")
    print(f"{'':>5} {'':>8} | {'med  [min ,  max]':>21} | {'med  [min ,  max]':>21} | "
          f"{'median':>10} | {'med [min, max]':>18}")
    print("-" * 100)
    seld_max = 0.0
    for a in args.spins:
        d = by_a[a]
        pq = np.array(d["pq"]); fq = np.array(d["fq"]); fl = np.array(d["fl"])
        head = float(np.nanmedian(pq) - np.nanmedian(fq))
        seld_max = max(seld_max, d["sd"])
        print(f"{a:5.2f} {d['mw']:8.4f} | "
              f"{np.nanmedian(pq):5.2f} [{np.nanmin(pq):5.2f},{np.nanmax(pq):6.2f}] | "
              f"{np.nanmedian(fq):5.2f} [{np.nanmin(fq):5.2f},{np.nanmax(fq):6.2f}] | "
              f"{head:10.2f} | "
              f"{np.nanmedian(fl):5.1f} [{np.nanmin(fl):5.1f},{np.nanmax(fl):5.1f}]")
    print("-" * 100)
    print("headroom = median(prior_qnm) - median(fine_qnm); ~0 => QNM already at "
          "fine-grid floor (no QNM win available).")
    print(f"max sel_dist over all pulses/spins = {seld_max:.3f} "
          f"(small => poles genuinely AT (4,2,0)).")


if __name__ == "__main__":
    main()

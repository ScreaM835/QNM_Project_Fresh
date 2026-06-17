"""Validate the new omega_target path in extract_qnm_kerr_ensemble.

Evolves the fine field and checks:
  - l=4 a=0.5/0.9: targeted ensemble recovers (4,2,0) (<~5%) where the
    untargeted ensemble gave ~26-28%; reports spread (error bar), sel_dist
    (confidence guard) and how many single-mode members were gated out.
  - l=4 a=0.0: untargeted real path unchanged (sanity).
Times each targeted call so we know the per-field cost for the batch eval.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

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
R0, W = 9.5, 1.25


def pct(v, r):
    return abs(v - r) / abs(r) * 100.0 if (np.isfinite(v) and abs(r) > 1e-30) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spins", type=float, nargs="*", default=[0.0, 0.5, 0.9])
    args = ap.parse_args()
    kd.ELL, kd.MM = ELL, MM

    print(f"{'a/M':>5} {'Mw_ref':>8} | {'untarg':>8} {'err%':>6} | "
          f"{'TARGET':>8} {'err%':>6} {'spr%':>6} {'seld':>6} {'n':>2} {'drop':>4} {'t(s)':>5}")
    print("-" * 86)
    for a in args.spins:
        q = kerr_qnm(a_over_M=a, ell=4, m=MM, n=0)
        mw = float(q.M_omega_R)
        tau_ref = q.tau_over_M
        wt = complex(q.M_omega_R, q.M_omega_I)

        tau, fine, op, info = kd.evolve_full_field(a, kd.FINE_N, R0, W)
        y = fine[:, scri_index(op)].astype(np.complex128)

        u = extract_qnm_kerr_ensemble(tau, y, a_over_M=a, tau_ref=tau_ref,
                                      tau_final=float(tau[-1]))
        ou = u["omega"]

        t0 = time.time()
        if a == 0.0:
            tdict = u  # no target path at a=0; show same
            ot, spr, seld, n, drop = ou, u["omega_std"], float("nan"), u["n_omega"], 0
        else:
            tdict = extract_qnm_kerr_ensemble(
                tau, y, a_over_M=a, tau_ref=tau_ref, tau_final=float(tau[-1]),
                omega_target=wt)
            ot = tdict["omega"]
            spr = tdict["omega_std"] / abs(ot) * 100.0 if ot else float("nan")
            seld = tdict.get("sel_dist", float("nan"))
            n = tdict["n_omega"]
            drop = tdict.get("n_single_dropped", 0)
        dt = time.time() - t0

        spr_u = u["omega_std"] / abs(ou) * 100.0 if ou else float("nan")
        print(f"{a:5.2f} {mw:8.4f} | {ou:8.4f} {pct(ou, mw):6.2f} | "
              f"{ot:8.4f} {pct(ot, mw):6.2f} {spr:6.2f} {seld:6.3f} {n:2d} {drop:4d} {dt:5.1f}")


if __name__ == "__main__":
    main()

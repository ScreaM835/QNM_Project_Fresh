"""Quick label-free-ceiling probe at ell=4, m=2 (the queued corpus mode).

For a handful of spins it solves the SAME audited grids the corpus uses
(fine 801 / k2 401 / k4 201) on one fixed (r0, w) pulse and reports the QNM
M*omega error of:
  - prior      : the coarse k4 field at scri (what the cheap solve gives)
  - Richardson : (4*k2 - k4)/3 at scri (the LABEL-FREE training teacher)
  - fine       : the N=801 field at scri (the trusted reference solve)
against the Leaver QNM from the qnm package. Also reports the scri-waveform
rel-L2 of prior / Richardson vs fine, so we see how good the teacher is.

This bounds whether a label-free hybrid can rescue the ell=4 QNM: the hybrid
trained on Richardson cannot beat the Richardson QNM/field shown here.
"""
from __future__ import annotations

import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

import kerr.src.kerr_dataset as kd
from kerr.scripts.kv3_qnm import extract_fundamental
from kerr.src.teukolsky_minimal_gauge import scri_index
from kerr.scripts.kv3_qnm import kerr_qnm as _kerr_qnm  # the validated handle
from kerr.src.qnm_ensemble_kerr import extract_qnm_kerr_ensemble

ELL, MM = 4, 2
kd.ELL, kd.MM = ELL, MM

SPINS = [0.0, 0.5, 0.9, 0.95]
R0, W = 9.5, 1.25


def pct(val, ref):
    if not np.isfinite(val) or abs(ref) < 1e-30:
        return float("nan")
    return abs(val - ref) / abs(ref) * 100.0


def rel_l2(pred, tgt):
    return float(np.linalg.norm(pred - tgt) / max(np.linalg.norm(tgt), 1e-30))


def qnm_of(tau, y_scri, a, r):
    out = extract_fundamental(
        tau, y_scri.astype(np.complex128),
        is_real=(abs(a) < 1e-6),
        tau_ref=r.tau_over_M, tau_final=float(tau[-1]),
    )
    return float(out.get("omega", np.nan))


def qnm_ens(tau, y_scri, a, r):
    """Multi-method consensus omega_R + across-method spread (rel %).

    Passes the Leaver target so the ensemble uses the mode-selective path at
    a>0 (fixes the high-l correlated tail-lock; reduces to the real consensus
    at a=0).
    """
    wt = complex(r.M_omega_R, r.M_omega_I)
    out = extract_qnm_kerr_ensemble(
        tau, y_scri.astype(np.complex128), a_over_M=a, tau_ref=r.tau_over_M,
        tau_final=float(tau[-1]), omega_target=wt,
    )
    om = float(out.get("omega", np.nan))
    spread = float(out.get("omega_std", np.nan))
    spread_rel = (spread / abs(om) * 100.0) if (np.isfinite(om) and om != 0) else np.nan
    return om, spread_rel, int(out.get("n_omega", 0))


def main():
    print(f"=== Richardson ceiling probe  ell={ELL} m={MM}  r0={R0} w={W} ===")
    print("single = one complex-phase fit (current);  ens = m1-m5 consensus")
    print(f"{'a/M':>5} | {'Mw_ref':>8} | "
          f"{'sgl prior':>9} {'sgl Rich':>9} {'sgl fine':>9} | "
          f"{'ens prior':>9} {'ens Rich':>9} {'ens fine':>9} | "
          f"{'spread_f':>8} {'nf':>3} | {'fld prior':>9} {'fld Rich':>9}")
    print("-" * 130)
    for a in SPINS:
        r = _kerr_qnm(a_over_M=a, ell=ELL, m=MM, n=0)
        mw_ref = float(r.M_omega_R)

        tau, fine, op_f, _ = kd.evolve_full_field(a, kd.FINE_N, R0, W)
        _, k2, op_2, _ = kd.evolve_full_field(a, kd.COARSE_N[2], R0, W)
        _, k4, op_4, _ = kd.evolve_full_field(a, kd.COARSE_N[4], R0, W)

        sf = scri_index(op_f)
        s2 = scri_index(op_2)
        s4 = scri_index(op_4)

        fine_s = fine[:, sf]
        k2_s = k2[:, s2]
        k4_s = k4[:, s4]
        rich_s = (4.0 * k2_s - k4_s) / 3.0   # nested grids share the scri point

        # single-method (current path)
        sp = qnm_of(tau, k4_s, a, r)
        sr = qnm_of(tau, rich_s, a, r)
        sn = qnm_of(tau, fine_s, a, r)
        # multi-method consensus
        ep, ep_spr, ep_n = qnm_ens(tau, k4_s, a, r)
        er, er_spr, er_n = qnm_ens(tau, rich_s, a, r)
        en, en_spr, en_n = qnm_ens(tau, fine_s, a, r)

        print(f"{a:5.2f} | {mw_ref:8.5f} | "
              f"{pct(sp, mw_ref):8.2f}% {pct(sr, mw_ref):8.2f}% {pct(sn, mw_ref):8.2f}% | "
              f"{pct(ep, mw_ref):8.2f}% {pct(er, mw_ref):8.2f}% {pct(en, mw_ref):8.2f}% | "
              f"{en_spr:7.2f}% {en_n:3d} | "
              f"{rel_l2(k4_s, fine_s)*100:8.2f}% {rel_l2(rich_s, fine_s)*100:8.2f}%")
    print("-" * 130)
    print("sgl/ens cols = % err vs Leaver; spread_f/nf = ensemble across-method")
    print("spread & #methods on the FINE field; fld cols = scri rel-L2 vs fine.")
    print("Richardson is the label-free teacher: the hybrid cannot beat it.")


if __name__ == "__main__":
    main()

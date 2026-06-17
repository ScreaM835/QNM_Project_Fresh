"""Verify the l=4 QNM extraction fix recovers (4,2,0) on the FINE field.

Diagnosis (diag_l4_spectrum): the operator cleanly supports (4,2,0); the late
[10,14]*tau_ref window (tuned for l=2) is too late for the fast l=4 QNM, which
by then sits below the slowly-decaying late-time tail ladder (Mw_R~0.3-0.7,
Mw_I~-0.015). Single-mode fitters lock onto that tail (~0.67). Multi-mode ESPRIT
already resolves (4,2,0) as a sub-dominant pole.

This script evolves the fine field at the problem spins and compares:
  A) current ensemble (late window single-method consensus),
  B) ESPRIT mode-SELECTION: run complex ESPRIT over a scan of windows x model
     orders, collect every pole, keep the one nearest the known Leaver (4,2,0)
     in the complex omega plane; report median + spread (the honest error bar)
     and the selection distance (a guard against picking a spurious pole),
  C) an EARLY-window single-mode ensemble for contrast.

If B recovers ~0.907 (a=0.5) / ~1.084 (a=0.9) with a small spread and small
selection distance, mode-selective ESPRIT is the l>=4 extractor.
"""
from __future__ import annotations

import argparse
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

import kerr.src.kerr_dataset as kd
from kerr.src.teukolsky_minimal_gauge import scri_index
from kerr.src.qnm_kerr_reference import kerr_qnm
from kerr.src.qnm_ensemble_kerr import extract_qnm_kerr_ensemble
from kerr.src.extractor_m4 import qnm_complex_esprit

ELL, MM = 4, 2
R0, W = 9.5, 1.25


def pct(val, ref):
    if not np.isfinite(val) or abs(ref) < 1e-30:
        return float("nan")
    return abs(val - ref) / abs(ref) * 100.0


def esprit_select(tau, psi, tau_ref, tau_final, w_target):
    """Collect ESPRIT poles over a window x order scan; pick nearest target.

    w_target = complex Leaver (4,2,0) = omega_R - i*omega_I (omega_I>0 decay).
    Returns (omega_R_median, omega_R_spread_pct, median_selection_distance, n).
    """
    picks = []
    dists = []
    # scan start in [1.5,5]*tau_ref, end in [6,12]*tau_ref (early-to-mid ring),
    # several model orders so the QNM pole is resolved alongside the tail.
    t0s = np.linspace(1.5 * tau_ref, 5.0 * tau_ref, 5)
    t1s = np.linspace(6.0 * tau_ref, min(12.0 * tau_ref, tau_final), 4)
    for K in (3, 4, 5, 6):
        for t0 in t0s:
            for t1 in t1s:
                if t1 - t0 < 3.0 * tau_ref:
                    continue
                r = qnm_complex_esprit(tau, psi, float(t0), float(t1), K=K)
                modes = r.get("all_modes") or r.get("modes") or []
                cand = []
                for m in modes:
                    oR = m.get("omega_R", np.nan)
                    oI = m.get("omega_I", np.nan)
                    if not np.isfinite(oR) or oR <= 0:
                        continue
                    # ESPRIT omega_I may be inf for a non-decaying pole; clamp.
                    oI_s = oI if np.isfinite(oI) else 0.0
                    d = abs((oR - w_target.real) + 1j * (-oI_s - w_target.imag))
                    cand.append((d, oR))
                if cand:
                    d, oR = min(cand, key=lambda x: x[0])
                    picks.append(oR)
                    dists.append(d)
    if not picks:
        return float("nan"), float("nan"), float("nan"), 0
    picks = np.array(picks)
    med = float(np.median(picks))
    spread = float(np.std(picks)) / abs(med) * 100.0 if med else float("nan")
    return med, spread, float(np.median(dists)), int(picks.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spins", type=float, nargs="*", default=[0.5, 0.9])
    args = ap.parse_args()
    kd.ELL, kd.MM = ELL, MM

    for a in args.spins:
        q420 = kerr_qnm(a_over_M=a, ell=4, m=MM, n=0)
        mw_ref = float(q420.M_omega_R)
        tau_ref = q420.tau_over_M
        w_target = complex(q420.M_omega_R, q420.M_omega_I)   # omega_R - i omega_I

        tau, fine, op, info = kd.evolve_full_field(a, kd.FINE_N, R0, W)
        y = fine[:, scri_index(op)].astype(np.complex128)

        print("\n" + "=" * 70)
        print(f"ell=4 m=2  a/M={a:.3f}   Leaver (4,2,0) Mw={mw_ref:.5f} "
              f"tau_ref={tau_ref:.3f}")
        print("=" * 70)

        # A) current late-window ensemble
        outA = extract_qnm_kerr_ensemble(tau, y, a_over_M=a, tau_ref=tau_ref,
                                         tau_final=float(tau[-1]))
        omA = outA["omega"]
        print(f"  A) current ensemble (late win): Mw={omA:.4f}  "
              f"err={pct(omA, mw_ref):6.2f}%  spread={outA['omega_std']/abs(omA)*100:.2f}%  n={outA['n_omega']}")

        # B) ESPRIT mode-selection (the proposed l>=4 fix)
        omB, sprB, distB, nB = esprit_select(tau, y, tau_ref, float(tau[-1]), w_target)
        print(f"  B) ESPRIT mode-select        : Mw={omB:.4f}  "
              f"err={pct(omB, mw_ref):6.2f}%  spread={sprB:5.2f}%  "
              f"sel_dist={distB:.3f}  n={nB}")

        # C) early-window ensemble (single-method, early band)
        outC = extract_qnm_kerr_ensemble(tau, y, a_over_M=a,
                                         tau_ref=tau_ref * 0.35,  # shrink band x0.35 -> early
                                         tau_final=float(tau[-1]))
        omC = outC["omega"]
        print(f"  C) early-window ensemble     : Mw={omC:.4f}  "
              f"err={pct(omC, mw_ref):6.2f}%  spread={outC['omega_std']/abs(omC)*100:.2f}%  n={outC['n_omega']}")


if __name__ == "__main__":
    main()

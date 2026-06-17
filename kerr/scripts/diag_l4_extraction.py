"""Diagnose the ell=4, m=2 QNM extraction anomaly at intermediate spin.

The Richardson-ceiling probe showed the ENSEMBLE on the FINE (ground-truth)
field returns ~26% (a=0.5) / ~28% (a=0.9) error vs the Leaver (4,2,0) mode,
yet ~0.2% at a=0 and a=0.95. A wrong number on the *fine* field is not a
physics error of the solve -- it means the extraction is locking onto a
DIFFERENT mode than the (4,2,0) we compare against.

Hypothesis: the initial pulse is a *spherical* Y_{4,2}. For a>0 the QNM
eigenfunctions are *spheroidal*; a spherical pulse projects onto several
spheroidal harmonics, exciting (3,2,0)/(2,2,0) admixtures that damp SLOWER
than (4,2,0) and therefore dominate the late-time fit window. 26% below
0.907 is ~0.67, suspiciously close to the (3,2,0) frequency.

This script evolves ONLY the fine field at the problem spins and dumps:
  - Leaver (l,2,0) for l=2..6 and the (4,2,1) overtone, so we can see which
    physical mode the extracted frequency matches;
  - the full per-method ensemble table (omega_R, %err vs each candidate);
  - an independent FFT dominant frequency in the late fit window;
  - the same with an EARLY window, to test whether (4,2,0) is recoverable
    before the slower lower-l admixture takes over.
No corpus, no training -- a focused physics check.
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

ELL, MM = 4, 2
R0, W = 9.5, 1.25
CAND_L = [2, 3, 4, 5, 6]   # candidate spheroidal l for the same m=2, n=0


def pct(val, ref):
    if not np.isfinite(val) or abs(ref) < 1e-30:
        return float("nan")
    return abs(val - ref) / abs(ref) * 100.0


def fft_peak(tau, y, t0, t1):
    """Dominant positive frequency of Re(y) over [t0, t1] (M*omega units)."""
    m = (tau >= t0) & (tau <= t1)
    if m.sum() < 16:
        return float("nan")
    tt = tau[m]
    yy = np.real(y[m]) - np.mean(np.real(y[m]))
    dt = float(np.median(np.diff(tt)))
    Y = np.fft.rfft(yy * np.hanning(yy.size))
    f = np.fft.rfftfreq(yy.size, d=dt)
    k = int(np.argmax(np.abs(Y)))
    return float(2.0 * np.pi * f[k])   # angular frequency = M*omega (M=1)


def nearest_mode(omega, cands):
    """Return (l, mw) of the candidate (l,2,0) closest to omega."""
    best = min(cands, key=lambda lc: abs(lc[1] - omega))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spins", type=float, nargs="*", default=[0.5, 0.9, 0.95])
    args = ap.parse_args()

    kd.ELL, kd.MM = ELL, MM

    for a in args.spins:
        print("\n" + "=" * 78)
        print(f"ell={ELL} m={MM}  a/M={a:.3f}   r0={R0} w={W}")
        print("=" * 78)

        # Leaver candidates for the same (m, n=0) across l, plus (4,2,1) overtone.
        cands = []
        for l in CAND_L:
            try:
                q = kerr_qnm(a_over_M=a, ell=l, m=MM, n=0)
                cands.append((l, float(q.M_omega_R), float(q.tau_over_M)))
            except Exception as e:
                print(f"  (l={l} ref failed: {e})")
        ref420 = next((c for c in cands if c[0] == 4), None)
        try:
            q421 = kerr_qnm(a_over_M=a, ell=4, m=MM, n=1)
            ot = (float(q421.M_omega_R), float(q421.tau_over_M))
        except Exception:
            ot = (float("nan"), float("nan"))
        print("  Leaver (l,2,0):  " + "   ".join(
            f"l={l}:Mw={mw:.4f},tau={tu:.1f}" for (l, mw, tu) in cands))
        print(f"  Leaver (4,2,1) overtone: Mw={ot[0]:.4f} tau={ot[1]:.1f}")
        mw_ref = ref420[1]
        tau_ref = ref420[2]

        # --- evolve fine field only ---
        tau, fine, op, info = kd.evolve_full_field(a, kd.FINE_N, R0, W)
        sf = scri_index(op)
        y = fine[:, sf].astype(np.complex128)
        imagfrac = np.max(np.abs(y.imag)) / max(np.max(np.abs(y.real)), 1e-30)
        print(f"  fine evolved: tau_ref={tau_ref:.3f}  imag/real={imagfrac:.2e}  "
              f"tau_final={tau[-1]:.0f}")

        cl = [(l, mw) for (l, mw, _) in cands]

        # --- ensemble with full per-method detail (nominal late window) ---
        out = extract_qnm_kerr_ensemble(
            tau, y, a_over_M=a, tau_ref=tau_ref, tau_final=float(tau[-1]),
            return_detail=True)
        om = out["omega"]; spr = out["omega_std"]
        print(f"\n  ENSEMBLE consensus: Mw={om:.4f}  spread={spr:.4f} "
              f"({spr/abs(om)*100:.2f}%)  n={out['n_omega']}  "
              f"te_cap={out['te_cap']:.0f}")
        nl, nmw = nearest_mode(om, cl)
        print(f"    -> consensus is {pct(om, mw_ref):.2f}% from (4,2,0); "
              f"NEAREST Leaver mode = (l={nl},2,0) Mw={nmw:.4f} "
              f"[{pct(om, nmw):.2f}% from it]")
        print(f"  {'method':>10} | {'omega_R':>9} | {'%err(4,2,0)':>11} | nearest-l")
        print("  " + "-" * 50)
        for (label, o, t) in out["methods"]:
            if not np.isfinite(o):
                print(f"  {label:>10} | {'nan':>9} |")
                continue
            ll, lmw = nearest_mode(o, cl)
            print(f"  {label:>10} | {o:9.4f} | {pct(o, mw_ref):11.2f} | l={ll}")

        # --- independent FFT in late vs early window ---
        t0_late, t1_late = 10.0 * tau_ref, min(14.0 * tau_ref, tau[-1])
        t0_early, t1_early = 1.5 * tau_ref, 4.0 * tau_ref
        f_late = fft_peak(tau, y, t0_late, t1_late)
        f_early = fft_peak(tau, y, t0_early, t1_early)
        ll_late = nearest_mode(f_late, cl)
        ll_early = nearest_mode(f_early, cl)
        print(f"\n  FFT peak LATE  [{t0_late:.0f},{t1_late:.0f}]: Mw={f_late:.4f} "
              f"-> nearest (l={ll_late[0]},2,0); {pct(f_late, mw_ref):.1f}% from (4,2,0)")
        print(f"  FFT peak EARLY [{t0_early:.0f},{t1_early:.0f}]: Mw={f_early:.4f} "
              f"-> nearest (l={ll_early[0]},2,0); {pct(f_early, mw_ref):.1f}% from (4,2,0)")


if __name__ == "__main__":
    main()

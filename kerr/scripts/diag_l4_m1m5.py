"""Answer 'why didn't m1-m5 work at l=4?' by running EACH member explicitly.

The Kerr ensemble currently wires m2, m3, m5 (on Re/Im) + cphase + cesprit, and
takes a robust median (MAD outlier rejection). At l=4 a=0.5/0.9 it returned ~26%
(the slow late-time tail Mw~0.67), NOT the (4,2,0) QNM (~0.907).

This script evolves the fine field and runs every m1-m5 member on ONE mid-ring
window, printing each method's pole(s) and contrasting:
  - the NAIVE pick (dominant amplitude / longest tau / single-mode) = what the
    ensemble uses now,
  - the TARGET pick = the pole nearest the known Leaver (l,2,0) in the complex
    omega plane (legitimate: the corpus is labelled by (l,m,n)).

Expected story: single-mode members (m1, m2, cphase) and dominant-amplitude m3
all lock onto the TAIL (correlated failure). The MULTI-mode members (m4 two-mode,
m3/cesprit pole list) CONTAIN (4,2,0) -- they only mis-report it because the
default selection is by amplitude / longest tau, which the slow tail wins. So the
ensemble didn't fail for lack of m1-m5; it failed because (a) the single-mode
majority shares one failure mode, breaking the independence the consensus
assumes, and (b) the multi-mode members select the wrong pole.
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
from kerr.src.extractor_m4 import (
    qnm_method_1, qnm_method_2, qnm_method_3_esprit,
    qnm_method_4_two_mode, qnm_complex_esprit,
)

ELL, MM = 4, 2
R0, W = 9.5, 1.25
CAND_L = [2, 3, 4, 5, 6]


def pct(val, ref):
    if not np.isfinite(val) or abs(ref) < 1e-30:
        return float("nan")
    return abs(val - ref) / abs(ref) * 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spins", type=float, nargs="*", default=[0.5, 0.9])
    args = ap.parse_args()
    kd.ELL, kd.MM = ELL, MM

    for a in args.spins:
        q420 = kerr_qnm(a_over_M=a, ell=4, m=MM, n=0)
        mw_ref = float(q420.M_omega_R)
        tau_ref = q420.tau_over_M
        wR, wI = q420.M_omega_R, q420.M_omega_I   # omega_R, omega_I(<0)
        cands = []
        for l in CAND_L:
            try:
                cands.append((l, float(kerr_qnm(a_over_M=a, ell=l, m=MM, n=0).M_omega_R)))
            except Exception:
                pass

        def nearest_l(o):
            return min(cands, key=lambda c: abs(c[1] - o))[0]

        tau, fine, op, info = kd.evolve_full_field(a, kd.FINE_N, R0, W)
        y = fine[:, scri_index(op)].astype(np.complex128)
        yr = y.real

        # one representative MID-ring window: QNM still alive, tail already present
        t0, t1 = 3.0 * tau_ref, 9.0 * tau_ref

        print("\n" + "=" * 74)
        print(f"ell=4 m=2  a/M={a:.3f}   Leaver (4,2,0) Mw={mw_ref:.4f}  "
              f"window=[{t0:.0f},{t1:.0f}] (tau_ref={tau_ref:.2f})")
        print(f"  Leaver ladder: " + "  ".join(f"l={l}:{mw:.3f}" for l, mw in cands))
        print("=" * 74)
        print(f"  {'method':>16} {'kind':>10} | {'NAIVE Mw':>9} {'err':>6} l | "
              f"{'TARGET Mw':>9} {'err':>6} l")
        print("  " + "-" * 70)

        def row(name, kind, naive, target):
            nl = nearest_l(naive) if np.isfinite(naive) else 0
            tl = nearest_l(target) if np.isfinite(target) else 0
            print(f"  {name:>16} {kind:>10} | {naive:9.4f} {pct(naive, mw_ref):5.1f}% {nl} | "
                  f"{target:9.4f} {pct(target, mw_ref):5.1f}% {tl}")

        # m1 FFT+envelope (single-mode): naive==target (no pole list)
        r1 = qnm_method_1(tau, yr, t0, t1)
        row("m1 FFT+env", "single", r1.get("omega", np.nan), r1.get("omega", np.nan))

        # m2 damped-cosine NLS (single-mode)
        try:
            r2 = qnm_method_2(tau, yr, t0, t1)
            o2 = r2.get("omega", np.nan)
        except Exception:
            o2 = np.nan
        row("m2 dampedcos", "single", o2, o2)

        # m4 two-mode NLS: fits {fundamental(longer tau), overtone}. NAIVE =
        # 'fundamental' (longer tau = the TAIL here); TARGET = nearer (4,2,0).
        r4 = qnm_method_4_two_mode(tau, yr, t0, t1)
        o4a, o4b = r4.get("omega", np.nan), r4.get("omega1", np.nan)
        naive4 = o4a   # the 'fundamental' (longer tau) it reports
        cand4 = [o for o in (o4a, o4b) if np.isfinite(o) and o > 0]
        tgt4 = min(cand4, key=lambda o: abs(o - wR)) if cand4 else np.nan
        row("m4 two-mode", "multi(2)", naive4, tgt4)

        # m3 real ESPRIT (K=6): naive = dominant amplitude; target = nearest (4,2,0)
        r3 = qnm_method_3_esprit(tau, yr, t0, t1, K=6, use_analytic=True)
        o3_all = np.array(r3.get("all_omegas", []), float)
        t3_all = np.array(r3.get("all_taus", []), float)
        pos = o3_all[(o3_all > 0) & np.isfinite(t3_all) & (t3_all > 0)]
        naive3 = r3.get("omega", np.nan)
        tgt3 = pos[np.argmin(np.abs(pos - wR))] if pos.size else np.nan
        row("m3 ESPRIT", "multi(6)", naive3, tgt3)

        # complex ESPRIT (K=6): naive = modes[0] (longest tau = TAIL); target=nearest
        ce = qnm_complex_esprit(tau, y, t0, t1, K=6)
        allm = ce.get("all_modes", ce.get("modes", []))
        physm = [m for m in allm if np.isfinite(m.get("omega_R", np.nan))
                 and m["omega_R"] > 0]
        naive_ce = (ce.get("modes") or [{}])[0].get("omega_R", np.nan) if ce.get("modes") else np.nan
        if physm:
            tgt_ce = min(physm, key=lambda m: abs((m["omega_R"] - wR)
                         + 1j * ((-1.0 / m["tau"] if (np.isfinite(m["tau"]) and m["tau"] > 0) else 0.0) - wI)))["omega_R"]
        else:
            tgt_ce = np.nan
        row("cesprit", "multi(6)", naive_ce, tgt_ce)


if __name__ == "__main__":
    main()

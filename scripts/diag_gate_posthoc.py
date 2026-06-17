#!/usr/bin/env python
"""
Post-hoc GATE ceiling test (login-safe, pure numpy/scipy, no training).

Question: the modes_t sweep showed mode count is NOT the lever (tripling modes
leaves QNM ~4x worse than the prior). The cause is the loss INCENTIVE: the
FNO freely reshapes the low-amplitude ringdown tail because that tail is
~1e-7 of the 2D training loss. A principled fix is a time gate that hands the
QNM-bearing tail back to the (already-clean) coarse prior:

    psi_gated(t,x) = psi_prior(t,x) + g(t) * [psi_hybrid(t,x) - psi_prior(t,x)]
    g(t) = 1 / (1 + exp((t - t_ring)/width))      # 1 early (full FNO), ->0 tail

This script measures the CEILING of that idea on the canonical BH we already
have on disk: for a sweep of gate cutoffs t_ring it reports BOTH
  - field rL2 over the FULL 2D field (must stay ~hybrid, i.e. good), AND
  - QNM M4/M2 omega/tau at the observer (must recover toward the prior/fine).

If a cutoff exists where field stays good AND QNM recovers, the gate (or the
equivalent tail-weighted loss) is validated and worth a training run. If field
collapses or QNM never recovers, it is not.

Run (login-safe):
  venv_csd3/bin/python scripts/diag_gate_posthoc.py
"""
from __future__ import annotations

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.qnm import (  # noqa: E402
    qnm_method_2,
    qnm_method_4_window_scan,
)

OMEGA_TRUE = 0.3737
TAU_TRUE = 11.241
XQ = 2.0
CACHE = "outputs/hybrid/fno_sw_richardson/field_cache/canonical.npz"


def banner(s):
    print("\n" + "=" * 72)
    print(s)
    print("=" * 72)


def rl2_2d(pred, ref):
    """Relative L2 over the full 2D field."""
    return float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))


def m4(t, y, te=50.0):
    r = qnm_method_4_window_scan(t, y, 10.0, 18.0, float(te),
                                 n_starts=8, potential="zerilli", ell=2)
    return r["omega"], r["tau"]


def m2(t, y, te=50.0):
    try:
        r = qnm_method_2(t, y, 10.0, float(te))
        return r["omega"], r["tau"]
    except Exception:
        return float("nan"), float("nan")


def pct(v, ref):
    return abs(v - ref) / ref * 100.0


def main():
    d = np.load(CACHE)
    x = d["x"]
    t = d["t"]
    pf = d["psi_fine"]          # (Nt, Nx) truth
    pp = d["psi_coarse_up"]     # (Nt, Nx) coarse prior (upsampled)
    ph = d["psi_hybrid"]        # (Nt, Nx) prior + FNO correction
    ix = int(np.argmin(np.abs(x - XQ)))
    delta = ph - pp             # the FNO correction

    print(f"cache={CACHE}")
    print(f"grid: Nt={t.size} t in [{t[0]:.1f},{t[-1]:.1f}], "
          f"Nx={x.size} x in [{x[0]:.1f},{x[-1]:.1f}]")
    print(f"observer x_q={XQ} -> ix={ix} (x={x[ix]:.4f})")

    # ---- reference rows: prior, hybrid, fine -------------------------------
    banner("BASELINES (canonical BH, M=1)  field rL2 (full 2D) + QNM @x_q=2")
    rl2_p = rl2_2d(pp, pf)
    rl2_h = rl2_2d(ph, pf)
    op_o, op_t = m4(t, pp[:, ix])
    oh_o, oh_t = m4(t, ph[:, ix])
    of_o, of_t = m4(t, pf[:, ix])
    p2o, p2t = m2(t, pp[:, ix])
    h2o, h2t = m2(t, ph[:, ix])
    print(f"{'model':>8} | {'field_rL2':>9} | "
          f"{'M4_om%':>7} {'M4_tau%':>7} | {'M2_om%':>7} {'M2_tau%':>7}")
    print("-" * 64)
    print(f"{'prior':>8} | {rl2_p:>8.3%} | "
          f"{pct(op_o, OMEGA_TRUE):>7.3f} {pct(op_t, TAU_TRUE):>7.3f} | "
          f"{pct(p2o, OMEGA_TRUE):>7.3f} {pct(p2t, TAU_TRUE):>7.3f}")
    print(f"{'hybrid':>8} | {rl2_h:>8.3%} | "
          f"{pct(oh_o, OMEGA_TRUE):>7.3f} {pct(oh_t, TAU_TRUE):>7.3f} | "
          f"{pct(h2o, OMEGA_TRUE):>7.3f} {pct(h2t, TAU_TRUE):>7.3f}")
    print(f"{'fine':>8} | {'   ~0   ':>9} | "
          f"{pct(of_o, OMEGA_TRUE):>7.3f} {pct(of_t, TAU_TRUE):>7.3f} | "
          f"{'  -  ':>7} {'  -  ':>7}")

    # ---- gate sweep --------------------------------------------------------
    banner("GATE SWEEP  psi = prior + sigmoid(t_ring,width) * (hybrid - prior)")
    print("width = 2M.  g->1 for t<<t_ring (full FNO), g->0 for t>>t_ring "
          "(prior tail).")
    print(f"{'t_ring':>7} | {'field_rL2':>9} {'vs_hyb':>7} | "
          f"{'M4_om%':>7} {'M4_tau%':>7} | {'M2_om%':>7} {'M2_tau%':>7}")
    print("-" * 72)
    width = 2.0
    for t_ring in (20, 25, 30, 35, 40, 45):
        g = 1.0 / (1.0 + np.exp((t - float(t_ring)) / width))   # (Nt,)
        pg = pp + g[:, None] * delta
        rl2_g = rl2_2d(pg, pf)
        yo = pg[:, ix]
        go_o, go_t = m4(t, yo)
        g2o, g2t = m2(t, yo)
        print(f"{t_ring:>7} | {rl2_g:>8.3%} {rl2_g/rl2_h:>6.2f}x | "
              f"{pct(go_o, OMEGA_TRUE):>7.3f} {pct(go_t, TAU_TRUE):>7.3f} | "
              f"{pct(g2o, OMEGA_TRUE):>7.3f} {pct(g2t, TAU_TRUE):>7.3f}")

    print("\nread: find a t_ring where field_rL2 stays ~hybrid (vs_hyb ~1x) "
          "AND\nM4_tau% drops from the hybrid row toward the prior/fine row. "
          "That is the\nceiling a tail-weighted loss (or a baked-in gate) "
          "can reach: BOTH metrics.")
    banner("DONE")


if __name__ == "__main__":
    main()

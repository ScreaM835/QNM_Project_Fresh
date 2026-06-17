"""Pressure test for the high-order coarse-FD prior, BEFORE patching production.

Two independent stress axes:

  (1) PARAMETER-SWEEP ROBUSTNESS
      The 100-300x accuracy gain was measured at ONE canonical point
      (M=1, x0=4, sigma=5). The hybrid dataset samples
          M     in [0.8, 1.2]   (potential changes -> peak height/location)
          x0    in [2.0, 6.0]   (off-centre initial pulse)
          sigma in [3.0, 7.0]   (narrow pulses carry more high-k content)
      Narrow + off-centre pulses are the WORST case for dispersion (more
      high-k, longer travel to a boundary). We re-measure S2(bc2)=production
      vs S4(bc4) and DRP7(bc4) at the 8 box corners + centre. If the gain
      collapses for narrow/off-centre pulses, high order is not the answer.

  (2) CFL / STABILITY MARGIN
      Higher-order second-derivative stencils have a LARGER maximum modified
      wavenumber, so RK4 stability caps the timestep at a SMALLER Courant
      number nu = dt/dx. Production runs at nu = 0.2/0.4 = 0.5. We must
      confirm every candidate scheme is still comfortably stable there, and
      report the empirical blow-up nu for each so we know the margin.

Production code untouched. Uses the validated solver/stencils from
coarse_stencil_isolation.py. Login-node safe (tens of seconds, tiny memory).
"""
from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, _THIS)

from coarse_stencil_isolation import (  # noqa: E402
    solve, make_d2,
    DX_COARSE, DT_COARSE, DX_TRUTH, DT_TRUTH,
)

OUTDIR = os.path.abspath(os.path.join(_REPO, "outputs", "exploration", "coarse_stencil"))


def _l2(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err**2)))


# ---------------------------------------------------------------------------
# (1) parameter-sweep robustness
# ---------------------------------------------------------------------------

def param_robustness() -> None:
    print("=== (1) Parameter-sweep robustness ===", flush=True)
    M_lo, M_hi = 0.8, 1.2
    x0_lo, x0_hi = 2.0, 6.0
    s_lo, s_hi = 3.0, 7.0

    pts: List[Tuple[str, float, float, float]] = [("centre", 1.0, 4.0, 5.0)]
    for M_ in (M_lo, M_hi):
        for x0_ in (x0_lo, x0_hi):
            for s_ in (s_lo, s_hi):
                tag = f"M{M_}_x{x0_}_s{s_}"
                pts.append((tag, M_, x0_, s_))
    # the single worst case for dispersion: narrowest pulse, furthest from centre
    pts.append(("WORST_narrow_offcentre", 0.8, 6.0, 3.0))

    sx, st = int(round(DX_COARSE / DX_TRUTH)), int(round(DT_COARSE / DT_TRUTH))

    s2 = make_d2("S2_central", bc_order=2)   # production
    s4 = make_d2("S4_central", bc_order=4)
    drp = make_d2("DRP7", bc_order=4)
    truth_op = make_d2("S6_central", bc_order=6)

    print(f"{'case':>24} {'sigma':>5} {'S2(prod) L2':>12} {'S4bc4 L2':>11} "
          f"{'DRP7bc4 L2':>11} {'S4 gain':>8} {'DRP7 gain':>9}", flush=True)
    worst_gain = np.inf
    for tag, M_, x0_, s_ in pts:
        _, _, phi_t = solve(truth_op, DX_TRUTH, DT_TRUTH, bc_order=6, M_=M_, x0_=x0_, sigma_=s_)
        truth_c = phi_t[::st, ::sx]
        _, _, p2 = solve(s2, DX_COARSE, DT_COARSE, bc_order=2, M_=M_, x0_=x0_, sigma_=s_)
        _, _, p4 = solve(s4, DX_COARSE, DT_COARSE, bc_order=4, M_=M_, x0_=x0_, sigma_=s_)
        _, _, pd = solve(drp, DX_COARSE, DT_COARSE, bc_order=4, M_=M_, x0_=x0_, sigma_=s_)
        l2_2, l2_4, l2_d = _l2(p2 - truth_c), _l2(p4 - truth_c), _l2(pd - truth_c)
        g4, gd = l2_2 / l2_4, l2_2 / l2_d
        worst_gain = min(worst_gain, g4, gd)
        print(f"{tag:>24} {s_:>5.1f} {l2_2:>12.3e} {l2_4:>11.3e} {l2_d:>11.3e} "
              f"{g4:>7.0f}x {gd:>8.0f}x", flush=True)
    print(f"\n  WORST observed gain over all cases (either scheme): {worst_gain:.0f}x", flush=True)
    print("  -> high order holds across the sweep if this stays >> 1.\n", flush=True)


# ---------------------------------------------------------------------------
# (2) CFL / stability margin
# ---------------------------------------------------------------------------

def cfl_margin() -> None:
    print("=== (2) CFL / stability margin (coarse grid dx=0.4) ===", flush=True)
    prod_nu = DT_COARSE / DX_COARSE
    print(f"  production operating point: nu = dt/dx = {DT_COARSE}/{DX_COARSE} = {prod_nu:.3f}", flush=True)

    schemes = {
        "S2_central": make_d2("S2_central", bc_order=2),
        "S4_central": make_d2("S4_central", bc_order=4),
        "S6_central": make_d2("S6_central", bc_order=4),
        "DRP7":       make_d2("DRP7",       bc_order=4),
    }
    # scan Courant number; integrate to t=50 and flag blow-up.
    nus = np.round(np.arange(0.3, 1.61, 0.05), 3)

    def stable_at(op, nu: float) -> bool:
        dt = nu * DX_COARSE
        try:
            _, _, phi = solve(op, DX_COARSE, dt, bc_order=4)
        except Exception:
            return False
        peak = float(np.max(np.abs(phi)))
        return np.isfinite(peak) and peak < 1e3   # IC peak is O(1); 1e3 = blown up

    print(f"{'scheme':>12} {'last stable nu':>15} {'margin over prod':>17}", flush=True)
    limits: Dict[str, float] = {}
    for name, op in schemes.items():
        last_ok = 0.0
        for nu in nus:
            if stable_at(op, float(nu)):
                last_ok = float(nu)
            else:
                break
        limits[name] = last_ok
        print(f"{name:>12} {last_ok:>15.2f} {last_ok / prod_nu:>16.1f}x", flush=True)

    worst = min(limits.values())
    print(f"\n  Tightest stability limit across schemes: nu = {worst:.2f} "
          f"({worst/prod_nu:.1f}x the production nu={prod_nu:.2f}).", flush=True)
    if worst > prod_nu:
        print("  -> production timestep is SAFE for every high-order scheme.", flush=True)
    else:
        print("  -> WARNING: a high-order scheme is NOT stable at the production timestep.", flush=True)


if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)
    t0 = time.time()
    param_robustness()
    cfl_margin()
    print(f"\n[pressure test done in {time.time()-t0:.1f}s]", flush=True)

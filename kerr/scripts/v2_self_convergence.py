"""V.2 self-convergence gate for minimal-gauge hyperboloidal RW.

Goal (kerr/notes/minimal_gauge_derivation.md sec 13, V.2):
    With the FULL Regge-Wheeler potential, evolve identical initial data on
    three NESTED grids and verify the spatial discretisation converges at
    the design order (2nd).

Nested grids: sigma = linspace(eps, 1 - eps, N) with N in {401, 801, 1601}.
Because 400 * 2 = 800 and 800 * 2 = 1600, coarse grid point i coincides
EXACTLY with fine grid point 2 i (and 4 i for the finest). No interpolation
is needed; we compare coincident points.

Self-convergence quantity at fixed final tau:

    e12 = Psi_401            - Psi_801[::2]
    e23 = Psi_801[::2]       - Psi_1601[::4]
    Q   = ||e12|| / ||e23||,        p = log2(Q).

For a 2nd-order-accurate spatial scheme Q -> 4, p -> 2. We also report the
pointwise max-norm version. RK4 time error is O(dt^4) and dt is scaled with
dsigma (fixed CFL ratio), so the spatial error dominates and p should be ~2.

Output: outputs/phase_a/v2_conv_<JOBID>.npz
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_THIS, "..", "..")))

from kerr.src.rwz_minimal_gauge import (
    build_minimal_gauge_op,
    rhs_min,
    state_from_psi,
    cfl_dt,
)
from kerr.src.fd_stencils import d1_central
from kerr.src.mol_rk4 import integrate_state
from kerr.src.dissipation import ko_dissipation


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
M = 1.0
ELL = 2
TAU_FINAL = 50.0
SAFETY = 0.4
SIGMA_KO = 0.02
RESOLUTIONS = [401, 801, 1601]  # nested: 400, 800, 1600 doublings
ID_R0 = 6.0
ID_WIDTH = 1.0
ID_AMP = 1.0
OUTDIR = os.path.abspath(os.path.join(_THIS, "..", "outputs", "phase_a"))


def make_initial_pulse(amp, r0, width):
    def psi0(r):
        return amp * np.exp(-((r - r0) ** 2) / (2.0 * width ** 2))
    return psi0


def evolve(op, dt, n_steps):
    psi0_fn = make_initial_pulse(ID_AMP, ID_R0, ID_WIDTH)
    state0 = state_from_psi(psi0_fn, op, d1_central)

    def rhs_fn(state):
        dPsi, dU, dW = rhs_min(state, op, d1_central)
        dU = dU + ko_dissipation(state[1], SIGMA_KO)
        dW = dW + ko_dissipation(state[2], SIGMA_KO)
        return dPsi, dU, dW

    t0 = time.time()
    state, _, _ = integrate_state(state0, dt, n_steps, rhs_fn, observers=None)
    elapsed = time.time() - t0
    return state[0], elapsed  # final Psi field


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    job = os.environ.get("SLURM_JOB_ID", "local")
    print(f"=== V.2 self-convergence gate (job {job}) ===", flush=True)
    print(f"  M={M}  ell={ELL}  full RW potential", flush=True)
    print(f"  Gaussian(amp={ID_AMP}, r0={ID_R0}M, sigma={ID_WIDTH}M)", flush=True)
    print(f"  tau_final={TAU_FINAL}  safety={SAFETY}  sigma_KO={SIGMA_KO}", flush=True)
    print(f"  nested resolutions = {RESOLUTIONS}", flush=True)

    # Use a SINGLE dt for all grids (set by the coarsest CFL) so that the time
    # error is identical and cancels in the self-convergence differences.
    op_coarse = build_minimal_gauge_op(N=RESOLUTIONS[0], M=M, ell=ELL, include_potential=True)
    dt = cfl_dt(op_coarse, safety=SAFETY)
    n_steps = int(np.ceil(TAU_FINAL / dt))
    dt = TAU_FINAL / n_steps
    print(f"  shared dt = {dt:.4e}   n_steps = {n_steps}", flush=True)
    print("", flush=True)

    psis = {}
    for N in RESOLUTIONS:
        op = build_minimal_gauge_op(N=N, M=M, ell=ELL, include_potential=True)
        psi_final, elapsed = evolve(op, dt, n_steps)
        psis[N] = psi_final
        print(f"  N={N}: evolved in {elapsed:.1f} s, |Psi|_max(tau_f)={np.max(np.abs(psi_final)):.3e}", flush=True)

    # Nested-grid self-convergence at coincident points (the N=401 grid).
    psi1 = psis[401]
    psi2_on1 = psis[801][::2]
    psi3_on1 = psis[1601][::4]
    assert psi1.shape == psi2_on1.shape == psi3_on1.shape, (psi1.shape, psi2_on1.shape, psi3_on1.shape)

    e12 = psi1 - psi2_on1
    e23 = psi2_on1 - psi3_on1
    l2_e12 = float(np.sqrt(np.mean(e12 ** 2)))
    l2_e23 = float(np.sqrt(np.mean(e23 ** 2)))
    max_e12 = float(np.max(np.abs(e12)))
    max_e23 = float(np.max(np.abs(e23)))

    Q_l2 = l2_e12 / l2_e23 if l2_e23 > 0 else float("inf")
    Q_max = max_e12 / max_e23 if max_e23 > 0 else float("inf")
    p_l2 = float(np.log2(Q_l2)) if np.isfinite(Q_l2) and Q_l2 > 0 else float("nan")
    p_max = float(np.log2(Q_max)) if np.isfinite(Q_max) and Q_max > 0 else float("nan")

    print("", flush=True)
    print("=== Self-convergence at tau_final (coincident points, N=401 grid) ===", flush=True)
    print(f"  L2 :  ||e12|| = {l2_e12:.3e}   ||e23|| = {l2_e23:.3e}   Q = {Q_l2:.3f}   p = {p_l2:.3f}", flush=True)
    print(f"  max:  ||e12|| = {max_e12:.3e}   ||e23|| = {max_e23:.3e}   Q = {Q_max:.3f}   p = {p_max:.3f}", flush=True)
    print("", flush=True)
    print("  PASS if p ~ 2 (2nd-order spatial). Values 1.7-2.3 acceptable for", flush=True)
    print("  this coarse 3-grid estimate; < 1.5 or > 2.7 indicates a problem.", flush=True)

    out_path = os.path.join(OUTDIR, f"v2_conv_{job}.npz")
    np.savez(
        out_path,
        resolutions=np.array(RESOLUTIONS),
        dt=np.array(dt), n_steps=np.array(n_steps), tau_final=np.array(TAU_FINAL),
        psi_401=psi1, psi_801_on1=psi2_on1, psi_1601_on1=psi3_on1,
        l2_e12=np.array(l2_e12), l2_e23=np.array(l2_e23),
        max_e12=np.array(max_e12), max_e23=np.array(max_e23),
        Q_l2=np.array(Q_l2), Q_max=np.array(Q_max),
        p_l2=np.array(p_l2), p_max=np.array(p_max),
    )
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()

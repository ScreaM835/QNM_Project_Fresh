"""V.1 flat-propagation gate for minimal-gauge hyperboloidal RW.

Goal (kerr/notes/minimal_gauge_derivation.md sec 13, V.1):
    With V = 0 set the system to the bare wave equation on the
    hyperboloidal sigma-slice. Launch a Gaussian pulse centred at r = 10 M,
    let it propagate outward through scri (sigma = 0) and inward through the
    horizon (sigma = 1). Verify the scheme is reflection-free and stable.

DIAGNOSTIC (important subtlety):
    With V = 0 the field Psi has STATIC ZERO-MODES (Psi = const is an exact
    static solution, and there is a second nonconstant static mode with
    Phi != 0). The Gaussian initial data has a nonzero projection onto these
    modes, so max|Psi| in the late-time window does NOT go to zero and is
    resolution-independent. That is physics of the potential-free problem,
    not a reflection. The correct no-reflection test is on

        Pi := d_tau Psi = (1 - sigma^2) U + sigma^2 W,

    which MUST decay toward zero as the system settles to a static state,
    and whose late-time residual MUST converge to zero with N (2nd order).
    A genuine boundary reflection appears as a non-decaying / slowly
    decaying OSCILLATION in Pi at interior observers.

Pass criteria for V.1:
    (a) finite_final == True at every N (no blow-up; the tanh gauge failed
        exactly here).
    (b) max|Pi| in the quiet window decreases monotonically with N at every
        interior observer (convergence toward a reflection-free static end
        state).
    (c) the convergence ratio approaches ~4 (2nd-order spatial) as N grows;
        we accept >= 2 as evidence of convergence for this coarse gate.

This script does NOT extract QNMs (that is V.3) and does NOT use the RW
potential (that is also V.3). It is a pure boundary / dissipation test.

Output: outputs/phase_a/v1_flat_<JOBID>.npz with the time series and
residuals at every N.
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np

# Allow running from kerr/scripts/ inside the improved repo.
_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_THIS, "..", "..")))

from kerr.src.rwz_minimal_gauge import (
    build_minimal_gauge_op,
    rhs_min,
    recover_pi_phi,
    state_from_psi,
    cfl_dt,
    observer_index,
    scri_index,
)
from kerr.src.fd_stencils import d1_central
from kerr.src.mol_rk4 import integrate_state
from kerr.src.dissipation import ko_dissipation


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
M = 1.0
ELL = 2          # unused (include_potential=False), just for op signature
TAU_FINAL = 100.0
SAFETY = 0.4
SIGMA_KO = 0.02  # very light, applied only to U and W (the "Pi-like" fields)
RESOLUTIONS = [201, 401, 801]
ID_R0 = 10.0
ID_WIDTH = 1.0
ID_AMP = 1.0
OBSERVERS_RM = {"scri_eff": None, "r20M": 20.0, "r10M": 10.0, "r5M": 5.0, "r3M": 3.0}
# Window must start AFTER the launched pulse has fully transited the inner
# region (near the horizon lambda_in is small, so inner observers clear last).
# tau >= 50 is comfortably settled for a pulse launched at r = 10 M; verified
# that all observers reach machine-precision Pi by then (job 30131260).
QUIET_WINDOW = (50.0, 100.0)
OUTDIR = os.path.abspath(os.path.join(_THIS, "..", "outputs", "phase_a"))


def make_initial_pulse(amp, r0, width):
    def psi0(r):
        return amp * np.exp(-((r - r0) ** 2) / (2.0 * width ** 2))
    return psi0


def run_one(N: int):
    op = build_minimal_gauge_op(N=N, M=M, ell=ELL, include_potential=False)
    dt = cfl_dt(op, safety=SAFETY)
    n_steps = int(np.ceil(TAU_FINAL / dt))
    dt = TAU_FINAL / n_steps  # adjust so we land exactly on tau_final

    psi0_fn = make_initial_pulse(ID_AMP, ID_R0, ID_WIDTH)
    state0 = state_from_psi(psi0_fn, op, d1_central)

    # observer indices on this grid
    obs = {}
    for label, rM in OBSERVERS_RM.items():
        obs[label] = scri_index(op) if rM is None else observer_index(op, rM)

    def rhs_fn(state):
        dPsi, dU, dW = rhs_min(state, op, d1_central)
        # KO only on the "Pi-like" fields U and W, interior-only (function
        # already zeroes the outermost 2 cells on each side).
        dU = dU + ko_dissipation(state[1], SIGMA_KO)
        dW = dW + ko_dissipation(state[2], SIGMA_KO)
        return dPsi, dU, dW

    # Record Pi = (1 - sigma^2) U + sigma^2 W (the time derivative of Psi),
    # which must decay to zero in a reflection-free evolution.
    def pi_recorder(state):
        Pi, _ = recover_pi_phi(state[1], state[2], op)
        return Pi

    t0 = time.time()
    state, taus, rec_pi = integrate_state(
        state0, dt, n_steps, rhs_fn,
        record_every=max(1, n_steps // 4000),
        observers=obs, recorder=pi_recorder,
    )
    elapsed = time.time() - t0

    # Keep Psi final snapshot for reference (the gate is on Pi, below).
    Psi_final = state[0]

    # Quiet-window residual of Pi at each observer (THE gate quantity).
    qmask = (taus >= QUIET_WINDOW[0]) & (taus <= QUIET_WINDOW[1])
    pi_residuals = {}
    for k, ts in rec_pi.items():
        pi_residuals[k] = float(np.max(np.abs(ts[qmask])))

    # Bulk sanity: any NaN/Inf at final time?
    finite_final = bool(
        np.all(np.isfinite(state[0])) and np.all(np.isfinite(state[1])) and np.all(np.isfinite(state[2]))
    )
    bulk_max = float(max(np.max(np.abs(state[0])), np.max(np.abs(state[1])), np.max(np.abs(state[2]))))

    return {
        "N": N, "dt": dt, "n_steps": n_steps, "elapsed_s": elapsed,
        "taus": taus, "obs_pi": rec_pi, "obs_indices": obs,
        "pi_residuals": pi_residuals,
        "Psi_final": Psi_final,
        "finite_final": finite_final, "bulk_max_final": bulk_max,
        "sigma_obs": {k: float(op.sigma[obs[k]]) for k in obs},
    }


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    job = os.environ.get("SLURM_JOB_ID", "local")
    print(f"=== V.1 flat-propagation gate (job {job}) ===", flush=True)
    print(f"  M={M}  V=0  Gaussian(amp={ID_AMP}, r0={ID_R0}M, sigma={ID_WIDTH}M)", flush=True)
    print(f"  tau_final={TAU_FINAL}  safety={SAFETY}  sigma_KO={SIGMA_KO}", flush=True)
    print(f"  quiet window = {QUIET_WINDOW}", flush=True)
    print(f"  resolutions = {RESOLUTIONS}", flush=True)
    print("", flush=True)

    all_results = []
    for N in RESOLUTIONS:
        print(f"--- N = {N} ---", flush=True)
        r = run_one(N)
        all_results.append(r)
        print(f"  dt = {r['dt']:.4e}   n_steps = {r['n_steps']}   elapsed = {r['elapsed_s']:.1f} s", flush=True)
        print(f"  finite_final = {r['finite_final']}   bulk_max_final = {r['bulk_max_final']:.3e}", flush=True)
        print(f"  Pi residuals (max |Pi| in quiet window tau in {QUIET_WINDOW}):", flush=True)
        for k in r["obs_pi"]:
            print(f"    {k:10s} (sigma={r['sigma_obs'][k]:.4f}): {r['pi_residuals'][k]:.3e}", flush=True)
        print("", flush=True)

    # Convergence summary: for each observer, print Pi residual vs N and ratios.
    print("=== Convergence of Pi residual (ideal 2nd-order -> 4x reduction per doubling) ===", flush=True)
    obs_keys = list(all_results[0]["obs_pi"].keys())
    for k in obs_keys:
        line = f"  {k:10s}: "
        vals = [r["pi_residuals"][k] for r in all_results]
        line += "  ".join(f"N={r['N']}:{v:.3e}" for r, v in zip(all_results, vals))
        if len(vals) >= 2:
            ratios = [vals[i] / vals[i + 1] if vals[i + 1] > 0 else float("inf") for i in range(len(vals) - 1)]
            line += "   ratios: " + ", ".join(f"{x:.2f}" for x in ratios)
        print(line, flush=True)

    out_path = os.path.join(OUTDIR, f"v1_flat_{job}.npz")
    np.savez(
        out_path,
        resolutions=np.array(RESOLUTIONS),
        **{f"N{r['N']}_taus": r["taus"] for r in all_results},
        **{f"N{r['N']}_pi_{k}": r["obs_pi"][k] for r in all_results for k in r["obs_pi"]},
        **{f"N{r['N']}_sigma_obs_{k}": np.array(r["sigma_obs"][k]) for r in all_results for k in r["sigma_obs"]},
        **{f"N{r['N']}_pi_residual_{k}": np.array(r["pi_residuals"][k]) for r in all_results for k in r["pi_residuals"]},
    )
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()

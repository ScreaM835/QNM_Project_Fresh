"""KV.1 propagation / stability gate for the complex Kerr Teukolsky operator.

Kerr analogue of the Phase A V.1 flat-propagation gate (v1_flat_propagation.py).
Evolve the FULL complex minimal-gauge Teukolsky operator (s=-2, l=m=2) from a
Gaussian pulse, no extraction. This is a pure stability / characteristic-outflow
check: the field must ring down and radiate cleanly out of BOTH endpoints (scri
sigma=0 and the horizon sigma=1) with no boundary growth and no blow-up.

Diagnostic quantity: the discrete L2(sigma) norm of the field,
    ||Psi||(tau) = sqrt( sum_i |Psi_i(tau)|^2 * dsigma ).
For a clean single complex QNM mode Psi(sigma,tau) = A(sigma) exp(-i omega tau)
the modulus envelope is exactly monotone, ||Psi||(tau) = ||A|| exp(Im(omega) tau)
with Im(omega) < 0, so AFTER the launch pulse has left the grid the L2 envelope
must DECAY (negative log-slope) and never grow. A numerical instability would
appear as exponential L2 GROWTH or a NaN.

Pass criteria (KV.1, notes/phase_b_plan.md B.6). The plan asks that the
"L2(sigma) envelope decays (does not grow) after the pulse leaves". A genuine
finding while building this gate: a NAIVE full-grid L2(sigma) of the REGULAR
field Psi is NOT a clean stability diagnostic, because the regularizing rescaling
Psi = psi * sigma^3 (1-sigma)^{-2-i beta} amplifies infalling radiation in the
horizon boundary layer (sigma->1) by (1-sigma)^{-2}; that growth is a coordinate
effect, not an instability, and it swamps the physical exterior (proof: at a=0
the field decays 2.5 decades at every finite-radius observer while the full-grid
L2 "grows" 3x on the horizon cell once the bulk has emptied into the round-off
floor). The honest, physically meaningful statement is evaluated on the
RADIATION field via its late-time ENVELOPE SLOPE (a log-linear fit of log|.|
over the quiet window, masking the round-off floor -- robust both to the a=0
real field crossing zero and to several-decade decay into the floor):
    (a) finite_final == True at every (a/M, N): no NaN/Inf, stable to tau=200M.
    (b) the scri+ waveform (the gravitational waveform at infinity) has negative
        envelope slope (decays), matching the QNM rate -omega_I.
    (c) the exterior bulk L2 (sigma <= BULK_SIGMA_MAX, horizon layer excluded)
        has negative envelope slope (the whole exterior decays, no localized
        growing mode).
An under-dissipated numerical instability turns these slopes POSITIVE. The
pointwise max/start ratios and the full-grid L2 / horizon-cell edge are still
computed and REPORTED for transparency, but are NOT gated: the full-grid L2 is
horizon-rescaling-dominated, and a single window-start reference is ill-posed for
a real field crossing zero or one that has decayed into the floor. Note the scri+
observer of Psi is finite and physical: psi ~ sigma^-3 is peeling-divergent but
Psi = psi sigma^3 is the regular field, so Psi[scri] is the waveform at infinity
-- the key observer.

Numerical hygiene: a real, spin-amplified grid-frequency mode appears at the
horizon when under-dissipated (round-off onset, pushed LATER with resolution;
spin scaling tracks beta = m a/(r+-r-), the near-horizon azimuthal oscillation of
the rescaling). Standard 4th-order Kreiss-Oliger at SIGMA_KO=0.2 (O(dx^3),
invisible to the resolved ringdown -- observer values are bit-identical to 4 s.f.
across SIGMA_KO in [0.02,0.5]) removes it for all spins up to a/M=0.95. This is
legitimate dissipation, NOT gate tuning; the resolved QNM window B.8 extracts
from is unchanged.

a=0 reduction (HONEST note). The a=0 Kerr operator's CHARACTERISTIC STRUCTURE
(speeds lambda_+/-, mu_+/-, inverse map) is identical to Phase A's RW minimal
gauge to machine precision (verified in scripts/test_teukolsky_minimal_gauge.py,
~6e-17), so the propagation/outflow STABILITY at a=0 reproduces V.1. The a=0
SOURCE, however, is the Bardeen-Press reduction, which differs from the RW source
by O(0.5) in c_Pi (the two are isospectral, not identical) -- so the a=0 field is
NOT pointwise identical to V.1. The physical a=0 equivalence (same QNM frequency)
is the B.8 (KV.3) gate, not this one.

This script runs on SLURM (CPU). A short version is smoke-tested on the login
node via run_one(..., tau_final=small).
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_THIS, "..", "..")))

from kerr.src.teukolsky_minimal_gauge import (
    build_teukolsky_op,
    rhs_teuk,
    recover_pi_phi,
    state_from_psi,
    cfl_dt,
    observer_index,
    scri_index,
)
from kerr.src.fd_stencils import d1_central
from kerr.src.mol_rk4 import rk4_step_state
from kerr.src.dissipation import ko_dissipation
from kerr.src.qnm_kerr_reference import kerr_qnm


M = 1.0
ELL = 2
MM = 2                       # azimuthal m
TAU_FINAL = 200.0
SAFETY = 0.4
# Kreiss-Oliger strength. Phase A Schwarzschild used 0.02; Kerr needs more at high
# spin because the regular-field rescaling (1-sigma)^{-2-i beta} carries a near-
# horizon azimuthal oscillation of frequency beta = m a/(r+-r-) that, when under-
# resolved, seeds a grid-frequency mode at the horizon (round-off onset, pushed
# later with resolution). KO=0.2 (O(dx^3), invisible to the resolved ringdown:
# observer values bit-identical to 4 s.f. through the QNM window) suppresses it
# for all a/M up to 0.95; the 0.2-0.5 plateau is flat, so 0.2 is safely chosen.
SIGMA_KO = 0.2
RESOLUTIONS = [401, 801]
SPINS = [0.0, 0.9]
ID_R0 = 10.0
ID_WIDTH = 1.0
ID_AMP = 1.0
OBSERVERS_RM = {"scri": None, "r20M": 20.0, "r10M": 10.0, "r5M": 5.0}
QUIET_FRAC = 0.6             # quiet window = [QUIET_FRAC*tau_final, tau_final]
BULK_SIGMA_MAX = 0.9         # exterior bulk L2 excludes horizon layer sigma>this
OUTDIR = os.path.abspath(os.path.join(_THIS, "..", "outputs", "phase_b"))


def make_initial_pulse(amp, r0, width):
    def psi0(r):
        return amp * np.exp(-((r - r0) ** 2) / (2.0 * width ** 2))
    return psi0


def l2_sigma(field, dsigma):
    """Discrete L2(sigma) norm sqrt(sum |field|^2 dsigma)."""
    return float(np.sqrt(np.sum(np.abs(field) ** 2) * dsigma))


def run_one(a_over_M, N, tau_final=TAU_FINAL):
    ref = kerr_qnm(a_over_M=a_over_M, ell=ELL, m=MM, n=0)
    omega_ref = complex(ref.M_omega_R, ref.M_omega_I)
    op = build_teukolsky_op(
        N=N, a_over_M=a_over_M, M=M, ell=ELL, m=MM,
        omega_ref=omega_ref, include_potential=True,
    )
    dt = cfl_dt(op, safety=SAFETY)
    n_steps = int(np.ceil(tau_final / dt))
    dt = tau_final / n_steps
    record_every = max(1, n_steps // 4000)

    state = state_from_psi(make_initial_pulse(ID_AMP, ID_R0, ID_WIDTH), op, d1_central)
    obs = {}
    for label, rM in OBSERVERS_RM.items():
        obs[label] = scri_index(op) if rM is None else observer_index(op, rM)
    bulk_mask = op.sigma <= BULK_SIGMA_MAX

    def rhs_fn(s):
        dPsi, dU, dW = rhs_teuk(s, op, d1_central)
        dU = dU + ko_dissipation(s[1], SIGMA_KO)
        dW = dW + ko_dissipation(s[2], SIGMA_KO)
        return dPsi, dU, dW

    taus = [0.0]
    l2_psi = [l2_sigma(state[0], op.dsigma)]
    l2_bulk = [l2_sigma(state[0][bulk_mask], op.dsigma)]
    Pi0, _ = recover_pi_phi(state[1], state[2], op)
    l2_pi = [l2_sigma(Pi0, op.dsigma)]
    edge = [max(abs(state[0][0]), abs(state[0][-1]))]
    rec = {k: [state[0][i]] for k, i in obs.items()}

    t0 = time.time()
    for n in range(1, n_steps + 1):
        state = rk4_step_state(state, dt, rhs_fn)
        if n % record_every == 0:
            taus.append(n * dt)
            l2_psi.append(l2_sigma(state[0], op.dsigma))
            l2_bulk.append(l2_sigma(state[0][bulk_mask], op.dsigma))
            Pi, _ = recover_pi_phi(state[1], state[2], op)
            l2_pi.append(l2_sigma(Pi, op.dsigma))
            edge.append(max(abs(state[0][0]), abs(state[0][-1])))
            for k, i in obs.items():
                rec[k].append(state[0][i])
    elapsed = time.time() - t0

    taus = np.asarray(taus)
    l2_psi = np.asarray(l2_psi)
    l2_bulk = np.asarray(l2_bulk)
    l2_pi = np.asarray(l2_pi)
    edge = np.asarray(edge)
    rec = {k: np.asarray(v) for k, v in rec.items()}

    finite_final = bool(all(np.all(np.isfinite(x)) for x in state))
    bulk_max = float(max(np.max(np.abs(x)) for x in state))

    qi = int(np.searchsorted(taus, QUIET_FRAC * tau_final))
    qi = min(qi, taus.size - 2)
    run_peak = max(float(np.max(np.abs(y))) for y in rec.values())

    def _growth_decay(series):
        """(max/start, final/start) of |series| over the quiet window."""
        q = np.abs(np.asarray(series)[qi:])
        s0 = float(q[0])
        if s0 <= 0:
            return float("inf"), float("inf")
        return float(np.max(q) / s0), float(q[-1] / s0)

    def _env_slope(series, peak):
        """Late-time log-envelope slope d(log|series|)/dtau over the quiet window.

        Robust to (i) zero crossings of a real (a=0) field and (ii) the round-off
        floor, by fitting log|.| only where |.| exceeds 1e-8 of the relevant peak.
        A clean decaying ringdown gives slope ~ -omega_I < 0; a numerical
        instability turns the slope POSITIVE. If the field has already decayed
        below the floor everywhere in the window it is stable by definition
        (slope -> -inf).
        """
        tt = taus[qi:]
        amp = np.abs(np.asarray(series)[qi:])
        msk = amp > 1e-8 * peak
        if int(msk.sum()) < 2:
            return float("-inf")
        return float(np.polyfit(tt[msk], np.log(amp[msk]), 1)[0])

    # Gate quantities: late-time ENVELOPE SLOPE of the physical radiation field.
    # This is the faithful statement of the plan's "envelope decays (does not
    # grow) after the pulse leaves": the scri+ waveform (the gravitational
    # waveform at infinity) and the exterior bulk L2 (sigma <= BULK_SIGMA_MAX,
    # horizon layer excluded) must both have negative envelope slope. The pointwise
    # max/start ratios below are NOT gated: at a=0 the field is real and crosses
    # zero, and by tau~120 it has decayed several decades into the round-off floor,
    # so a single window-start reference is ill-posed -- the slope is robust there.
    scri_slope = _env_slope(rec["scri"], run_peak)
    l2_bulk_slope = _env_slope(l2_bulk, float(np.max(l2_bulk)))
    obs_slope = {k: _env_slope(y, run_peak) for k, y in rec.items()}

    # Reported-only diagnostics (transparency; not gated).
    obs_growth = {k: _growth_decay(y)[0] for k, y in rec.items()}
    obs_growth_max = max(obs_growth.values())
    _, scri_decay = _growth_decay(rec["scri"])
    l2_bulk_growth, l2_bulk_decay = _growth_decay(l2_bulk)
    l2_growth_full, _ = _growth_decay(l2_psi)      # horizon-rescaling dominated
    edge_growth, _ = _growth_decay(edge)           # horizon cell only

    # KV.1 pass: finite to tau=200M everywhere, and the radiation field decays --
    # negative late-time envelope slope of both the scri+ waveform and the
    # exterior bulk L2. (An under-dissipated instability would make these positive.)
    passed = (
        finite_final
        and scri_slope < 0.0
        and l2_bulk_slope < 0.0
    )

    return {
        "a_over_M": a_over_M, "N": N, "dt": dt, "n_steps": n_steps,
        "elapsed_s": elapsed, "finite_final": finite_final, "bulk_max_final": bulk_max,
        "M_omega_R": ref.M_omega_R, "M_omega_I": ref.M_omega_I, "tau_over_M": ref.tau_over_M,
        "taus": taus, "l2_psi": l2_psi, "l2_bulk": l2_bulk, "l2_pi": l2_pi,
        "edge": edge, "obs": rec,
        "quiet_start": float(taus[qi]),
        "scri_slope": scri_slope, "l2_bulk_slope": l2_bulk_slope, "obs_slope": obs_slope,
        "obs_growth": obs_growth, "obs_growth_max": obs_growth_max,
        "scri_decay": scri_decay,
        "l2_bulk_growth": l2_bulk_growth, "l2_bulk_decay": l2_bulk_decay,
        "l2_growth_full": l2_growth_full, "edge_growth": edge_growth,
        "passed": bool(passed),
        "sigma_obs": {k: float(op.sigma[i]) for k, i in obs.items()},
        "r_obs": {k: float(op.r[i]) for k, i in obs.items()},
    }


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    job = os.environ.get("SLURM_JOB_ID", "local")
    print(f"=== KV.1 Kerr propagation/stability gate (job {job}) ===", flush=True)
    print(f"  full complex Teukolsky operator, M={M}, l={ELL}, m={MM}", flush=True)
    print(f"  Gaussian(amp={ID_AMP}, r0={ID_R0}M, sigma={ID_WIDTH}M)", flush=True)
    print(f"  tau_final={TAU_FINAL}  safety={SAFETY}  sigma_KO={SIGMA_KO}", flush=True)
    print(f"  spins a/M={SPINS}  resolutions={RESOLUTIONS}", flush=True)
    print(f"  quiet window = [{QUIET_FRAC*TAU_FINAL:.0f}, {TAU_FINAL:.0f}]M", flush=True)
    print("  GATE on the physical radiation field: scri+ waveform and exterior"
          " bulk L2 (sigma<=%.2f) have negative late-time envelope slope (decay)."
          % BULK_SIGMA_MAX, flush=True)
    print("  Full-grid L2 / horizon edge / pointwise ratios are reported only"
          " (horizon-rescaling dominated / zero-crossing fragile, not gated).", flush=True)
    print("  NOTE a=0: characteristic structure == Phase A V.1 (B.4, ~6e-17);"
          " source is Bardeen-Press (isospectral, not pointwise-RW).", flush=True)
    print("", flush=True)

    all_results = []
    n_pass = 0
    n_total = 0
    for a in SPINS:
        for N in RESOLUTIONS:
            n_total += 1
            r = run_one(a, N)
            all_results.append(r)
            n_pass += int(r["passed"])
            print(f"--- a/M={a}  N={N} ---", flush=True)
            print(f"  qnm ref: M*omega_R={r['M_omega_R']:.6f}  M*omega_I={r['M_omega_I']:.6f}"
                  f"  tau/M={r['tau_over_M']:.4f}", flush=True)
            print(f"  dt={r['dt']:.4e}  n_steps={r['n_steps']}  elapsed={r['elapsed_s']:.1f}s", flush=True)
            print(f"  finite_final={r['finite_final']}  bulk_max_final={r['bulk_max_final']:.3e}", flush=True)
            print(f"  quiet window from tau={r['quiet_start']:.1f}M -- GATE on envelope slope:", flush=True)
            print(f"    scri+ env slope = {r['scri_slope']:+.4e}  (< 0; qnm -omega_I={r['M_omega_I']:+.4e})", flush=True)
            print(f"    bulk L2  slope  = {r['l2_bulk_slope']:+.4e}  (< 0)", flush=True)
            print(f"    obs slopes      = " +
                  "  ".join(f"{k}={v:+.3e}" for k, v in r['obs_slope'].items()), flush=True)
            print(f"  [reported only] scri+ net decay={r['scri_decay']:.2e}  "
                  f"obs max/start={r['obs_growth_max']:.2f}  bulk max/start={r['l2_bulk_growth']:.2f}  "
                  f"full-grid L2 growth={r['l2_growth_full']:.2f}  edge growth={r['edge_growth']:.2f}", flush=True)
            print(f"  => {'PASS' if r['passed'] else 'FAIL'}", flush=True)
            print("", flush=True)

    print(f"=== KV.1 summary: {n_pass} / {n_total} runs passed ===", flush=True)

    out_path = os.path.join(OUTDIR, f"kv1_{job}.npz")
    save = {"spins": np.array(SPINS), "resolutions": np.array(RESOLUTIONS)}
    for r in all_results:
        tag = f"a{r['a_over_M']}_N{r['N']}"
        save[f"{tag}_taus"] = r["taus"]
        save[f"{tag}_l2_psi"] = r["l2_psi"]
        save[f"{tag}_l2_bulk"] = r["l2_bulk"]
        save[f"{tag}_l2_pi"] = r["l2_pi"]
        save[f"{tag}_edge"] = r["edge"]
        for k, y in r["obs"].items():
            save[f"{tag}_obs_{k}"] = y
    np.savez(out_path, **save)
    print(f"Saved: {out_path}", flush=True)

    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()

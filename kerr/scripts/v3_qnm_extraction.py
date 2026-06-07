"""V.3 QNM-extraction gate for minimal-gauge hyperboloidal RW.

Goal:
    Full Regge-Wheeler potential, minimal-gauge hyperboloidal evolution,
    fixed-radius observers, and Method-4 two-mode extraction. This is the
    actual Phase A.8 physics gate:

        M*omega -> 0.373672
        tau/M   -> 11.2407

    at the highest resolution, with consistent convergence across grids.

This script runs on SLURM only. Do not execute on a login node.
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
    observer_index,
    scri_index,
)
from kerr.src.fd_stencils import d1_central
from kerr.src.mol_rk4 import integrate_state
from kerr.src.dissipation import ko_dissipation
from kerr.src.extractor_m4 import (
    qnm_method_2,
    qnm_method_5_2d_scan,
    qnm_method_3_esprit,
)
from kerr.src.qnm_kerr_reference import kerr_qnm


M = 1.0
ELL = 2
TAU_FINAL = 180.0
SAFETY = 0.4
SIGMA_KO = 0.02
RESOLUTIONS = [401, 801, 1601]
ID_R0 = 6.0
ID_WIDTH = 1.0
ID_AMP = 1.0
OBSERVERS_RM = {"scri_eff": None, "r50M": 50.0, "r20M": 20.0, "r10M": 10.0}

# Window selection is data-driven, not hand-picked.
#
# Primary estimator: Method 5 (2D plateau scan over BOTH t_start and t_end).
# We do NOT pin t_end. Instead we scan t_end over a range whose UPPER edge is
# kept above the numerical floor: from V.2 the discretization/KO error sits at
# ~1e-5..1e-4, while the ringdown envelope ~ exp(-t/11.24) drops to ~1e-4 of
# peak near t~100M. Letting t_end run past that pulls the late power-law tail
# (RW ~ t^-7) and the floor into the fit. We therefore scan t_end in
# [90, 140] and t_start in [30, 70]; the 2D scan self-diagnoses (tail/floor
# cells fail to plateau and are excluded), and we report the most-stable
# rectangle mean +- std.
SCAN_T0_MIN = 30.0
SCAN_T0_MAX = 70.0
SCAN_TEND_MIN = 90.0
SCAN_TEND_MAX = 140.0
N_STARTS = 10
N_ENDS = 6
MIN_WINDOW = 20.0

# Secondary fixed-window fit (independent reference, NOT the gate). It uses a
# SINGLE-mode damped cosine (Method 2), not the two-mode model: by t~40M the
# first overtone (tau~3.65M) has decayed by ~e^-11, so a two-mode fit on a late
# window is over-parametrised and the bounded NLS fails to converge (returns
# NaN). Single-mode is the physically correct model there and is robust for
# every observer. The window [40, 100] sits inside the QNM regime and below the
# late power-law tail / numerical floor (~t>100M). (The old diagnostic used a
# two-mode fit on [30, 140], which pinned t_end into the tail and produced an
# all-NaN table.)
FIXED_T0 = 40.0
FIXED_TEND = 100.0

# Independent cross-check: ESPRIT (Method 3). This is a SUBSPACE method
# (linear-algebra eigen-decomposition of a Hankel data matrix), mathematically
# unrelated to the nonlinear least-squares fit in M4/M5, and -- unlike a
# Hilbert instantaneous-frequency readout -- it makes NO slow-envelope (high-Q)
# assumption, which matters because the Schwarzschild fundamental is low-Q
# (omega*tau/2 ~ 2, i.e. the envelope e-folds in less than one period). We run
# ESPRIT with model order K=4 (room for the fundamental conjugate pair plus the
# first overtone / a tail mode) over a few QNM-region windows and report the
# median fundamental. Agreement with M5 to ~1e-3 confirms M5 is not
# self-consistently biased.
ESPRIT_K = 4
ESPRIT_WINDOWS = [(45.0, 100.0), (50.0, 95.0), (40.0, 105.0)]

OUTDIR = os.path.abspath(os.path.join(_THIS, "..", "outputs", "phase_a"))


def make_initial_pulse(amp, r0, width):
    def psi0(r):
        return amp * np.exp(-((r - r0) ** 2) / (2.0 * width ** 2))
    return psi0


def run_one(N: int):
    op = build_minimal_gauge_op(N=N, M=M, ell=ELL, include_potential=True)
    dt = cfl_dt(op, safety=SAFETY)
    n_steps = int(np.ceil(TAU_FINAL / dt))
    dt = TAU_FINAL / n_steps
    record_every = max(1, int(round(0.05 / dt)))

    state0 = state_from_psi(make_initial_pulse(ID_AMP, ID_R0, ID_WIDTH), op, d1_central)
    observers = {}
    for label, rM in OBSERVERS_RM.items():
        observers[label] = scri_index(op) if rM is None else observer_index(op, rM)

    def rhs_fn(state):
        dPsi, dU, dW = rhs_min(state, op, d1_central)
        dU = dU + ko_dissipation(state[1], SIGMA_KO)
        dW = dW + ko_dissipation(state[2], SIGMA_KO)
        return dPsi, dU, dW

    t0 = time.time()
    state, taus, series = integrate_state(
        state0, dt, n_steps, rhs_fn,
        observer_field=0, record_every=record_every,
        observers=observers,
    )
    elapsed = time.time() - t0
    finite_final = bool(all(np.all(np.isfinite(x)) for x in state))

    fits = {}
    scans = {}
    esprit = {}
    for label, y in series.items():
        # Secondary: single fixed-window single-mode fit (reference only).
        fits[label] = qnm_method_2(taus, y, FIXED_T0, FIXED_TEND)
        # Primary: Method-5 2D plateau scan over (t_start, t_end).
        scans[label] = qnm_method_5_2d_scan(
            taus, y,
            SCAN_T0_MIN, SCAN_T0_MAX,
            SCAN_TEND_MIN, SCAN_TEND_MAX,
            n_starts=N_STARTS, n_ends=N_ENDS,
            potential="regge_wheeler", ell=ELL,
            min_window=MIN_WINDOW,
        )
        # Independent cross-check: ESPRIT over several windows -> median.
        e_oms, e_taus = [], []
        for (lo, hi) in ESPRIT_WINDOWS:
            r = qnm_method_3_esprit(taus, y, lo, hi, K=ESPRIT_K)
            e_oms.append(r.get("omega", np.nan))
            e_taus.append(r.get("tau", np.nan))
        e_oms = np.array(e_oms, dtype=float)
        e_taus = np.array(e_taus, dtype=float)
        esprit[label] = {
            "omega": float(np.nanmedian(e_oms)) if np.any(np.isfinite(e_oms)) else np.nan,
            "tau": float(np.nanmedian(e_taus)) if np.any(np.isfinite(e_taus)) else np.nan,
            "omega_std": float(np.nanstd(e_oms)) if np.any(np.isfinite(e_oms)) else np.nan,
            "tau_std": float(np.nanstd(e_taus)) if np.any(np.isfinite(e_taus)) else np.nan,
        }

    return {
        "N": N, "dt": dt, "n_steps": n_steps, "record_every": record_every,
        "elapsed_s": elapsed, "finite_final": finite_final,
        "max_abs_final": float(max(np.max(np.abs(x)) for x in state)),
        "taus": taus, "series": series, "fits": fits, "scans": scans,
        "esprit": esprit,
        "sigma_obs": {k: float(op.sigma[i]) for k, i in observers.items()},
        "r_obs": {k: float(op.r[i]) for k, i in observers.items()},
    }


def err_pair(result, ref):
    om = float(result.get("omega", np.nan))
    ta = float(result.get("tau", np.nan))
    eom = abs(om - ref.M_omega_R) / ref.M_omega_R if np.isfinite(om) else np.nan
    eta = abs(ta - ref.tau_over_M) / ref.tau_over_M if np.isfinite(ta) else np.nan
    return om, ta, eom, eta


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    job = os.environ.get("SLURM_JOB_ID", "local")
    ref = kerr_qnm(a_over_M=0.0, ell=2, m=2, n=0)

    print(f"=== V.3 QNM extraction gate (job {job}) ===", flush=True)
    print(f"  reference qnm package: M*omega={ref.M_omega_R:.6f}, tau/M={ref.tau_over_M:.4f}", flush=True)
    print(f"  M={M} ell={ELL} full RW potential, minimal gauge", flush=True)
    print(f"  Gaussian(amp={ID_AMP}, r0={ID_R0}M, sigma={ID_WIDTH}M)", flush=True)
    print(f"  tau_final={TAU_FINAL}", flush=True)
    print(f"  primary = M5 2D scan: t0=[{SCAN_T0_MIN},{SCAN_T0_MAX}], "
          f"t_end=[{SCAN_TEND_MIN},{SCAN_TEND_MAX}], min_window={MIN_WINDOW}", flush=True)
    print(f"  cross-check = ESPRIT (Method 3, K={ESPRIT_K}) over windows {ESPRIT_WINDOWS}", flush=True)
    print(f"  resolutions={RESOLUTIONS}\n", flush=True)

    all_results = []
    for N in RESOLUTIONS:
        print(f"--- N = {N} ---", flush=True)
        out = run_one(N)
        all_results.append(out)
        print(f"  dt={out['dt']:.4e}, n_steps={out['n_steps']}, elapsed={out['elapsed_s']:.1f}s", flush=True)
        print(f"  finite_final={out['finite_final']}  max_abs_final={out['max_abs_final']:.3e}", flush=True)
        for label in out["series"]:
            scan = out["scans"][label]
            om, ta, eom, eta = err_pair(scan, ref)
            print(
                f"  [{label:8s}] r={out['r_obs'][label]:8.3f} sigma={out['sigma_obs'][label]:.5f} "
                f"M5: M*omega={om:.6f} (err {eom:.2e}), tau/M={ta:.4f} (err {eta:.2e}) "
                f"box t0=[{scan.get('t0_plateau_min', np.nan):.1f},"
                f"{scan.get('t0_plateau_max', np.nan):.1f}] "
                f"te=[{scan.get('te_plateau_min', np.nan):.1f},"
                f"{scan.get('te_plateau_max', np.nan):.1f}]",
                flush=True,
            )
            ins = out["esprit"][label]
            iom, ita, ieom, ieta = err_pair(ins, ref)
            print(
                f"  {'':8s}   ESPRIT(K={ESPRIT_K}): M*omega={iom:.6f} (err {ieom:.2e}), "
                f"tau/M={ita:.4f} (err {ieta:.2e}) "
                f"om_spread={float(ins.get('omega_std', np.nan)):.2e}",
                flush=True,
            )
        print("", flush=True)

    print("=== Fixed-window single-mode fit (reference: "
          f"t0={FIXED_T0:.0f}, t1={FIXED_TEND:.0f}) ===", flush=True)
    print(f"{'N':>5} {'obs':>8} {'M*omega':>10} {'err_om':>10} {'tau/M':>10} {'err_tau':>10}", flush=True)
    for out in all_results:
        for label in out["series"]:
            om, ta, eom, eta = err_pair(out["fits"][label], ref)
            print(f"{out['N']:5d} {label:>8s} {om:10.6f} {eom:10.2e} {ta:10.4f} {eta:10.2e}", flush=True)

    print("\n=== M5 2D-scan plateau table (PRIMARY) ===", flush=True)
    print(f"{'N':>5} {'obs':>8} {'M*omega':>10} {'err_om':>10} {'tau/M':>10} {'err_tau':>10} {'om_std':>10} {'tau_std':>10}", flush=True)
    for out in all_results:
        for label in out["series"]:
            scan = out["scans"][label]
            om, ta, eom, eta = err_pair(scan, ref)
            print(
                f"{out['N']:5d} {label:>8s} {om:10.6f} {eom:10.2e} {ta:10.4f} {eta:10.2e} "
                f"{float(scan.get('omega_std', np.nan)):10.2e} {float(scan.get('tau_std', np.nan)):10.2e}",
                flush=True,
            )

    print("\n=== ESPRIT (Method 3) independent cross-check table ===", flush=True)
    print(f"{'N':>5} {'obs':>8} {'M*omega':>10} {'err_om':>10} {'tau/M':>10} {'err_tau':>10} {'om_std':>10}", flush=True)
    for out in all_results:
        for label in out["series"]:
            ins = out["esprit"][label]
            om, ta, eom, eta = err_pair(ins, ref)
            print(
                f"{out['N']:5d} {label:>8s} {om:10.6f} {eom:10.2e} {ta:10.4f} {eta:10.2e} "
                f"{float(ins.get('omega_std', np.nan)):10.2e}",
                flush=True,
            )

    out_path = os.path.join(OUTDIR, f"v3_qnm_{job}.npz")
    save = {"resolutions": np.array(RESOLUTIONS)}
    for out in all_results:
        N = out["N"]
        save[f"N{N}_taus"] = out["taus"]
        for label, y in out["series"].items():
            scan = out["scans"][label]
            ins = out["esprit"][label]
            save[f"N{N}_y_{label}"] = y
            save[f"N{N}_omega_m5_{label}"] = np.array(scan.get("omega", np.nan))
            save[f"N{N}_tau_m5_{label}"] = np.array(scan.get("tau", np.nan))
            save[f"N{N}_omega_m5_std_{label}"] = np.array(scan.get("omega_std", np.nan))
            save[f"N{N}_tau_m5_std_{label}"] = np.array(scan.get("tau_std", np.nan))
            save[f"N{N}_omega_fixed_{label}"] = np.array(out["fits"][label].get("omega", np.nan))
            save[f"N{N}_tau_fixed_{label}"] = np.array(out["fits"][label].get("tau", np.nan))
            save[f"N{N}_omega_esprit_{label}"] = np.array(ins.get("omega", np.nan))
            save[f"N{N}_tau_esprit_{label}"] = np.array(ins.get("tau", np.nan))
    np.savez(out_path, **save)
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()

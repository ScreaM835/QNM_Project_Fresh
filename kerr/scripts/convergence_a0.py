"""A.8b: convergence study + inner-sponge stability test.

Runs the validate_a0 pipeline at three resolutions (n = 501, 1001, 2001)
with a two-sided sponge and the same KO dissipation. For each, fits
M*omega and tau/M with a fixed M4 window [t0, t1] = [20, 120] at both
observers and reports a convergence table.

Acceptance: M*omega converges monotonically toward 0.3737 with rel err
< 1e-3 at the highest resolution; tau/M likewise toward 11.241.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fd_stencils import d1_central, d2_central
from src.initial_data import GaussianID, gaussian
from src.mol_rk4 import integrate, cfl_dt
from src.observers import make_observers, observers_as_indices
from src.rwz_hyperboloidal import build_operator, rhs
from src.dissipation import ko_dissipation, outer_sponge_profile
from src.extractor_m4 import qnm_method_4_two_mode
from src.qnm_kerr_reference import kerr_qnm


M = 1.0
L = 10.0
R_MIN, R_MAX = -20.0, 80.0
T_FINAL = 150.0
SIGMA_KO = 0.05

SPONGE_OUTER = 0.20
SPONGE_GAMMA = 0.5
T0_FIT, T1_FIT = 20.0, 120.0
R_OBS = [("r10M", 10.0), ("r20M", 20.0), ("r50M", 50.0)]


def run_one(n: int):
    r_star = np.linspace(R_MIN, R_MAX, n)
    dx = r_star[1] - r_star[0]
    op = build_operator(r_star, M=M, L=L, ell=2)
    Phi0, Pi0 = gaussian(r_star, GaussianID(A0=1.0, x0=4.0, sigma=5.0))
    observers = make_observers(r_star, R_OBS)
    obs_idx = observers_as_indices(observers)

    gamma = outer_sponge_profile(r_star, SPONGE_OUTER, SPONGE_GAMMA)

    def rhs_fn(Phi, Pi):
        dPhi, dPi = rhs(Phi, Pi, op, d1_central, d2_central)
        dPhi = dPhi + ko_dissipation(Phi, SIGMA_KO)
        dPi = dPi + ko_dissipation(Pi, SIGMA_KO) - gamma * Pi
        return dPhi, dPi

    dt = cfl_dt(dx, float(np.max(np.abs(op.H))), safety=0.4)
    n_steps = int(np.ceil(T_FINAL / dt))
    record_every = max(1, int(round(0.1 / dt)))
    Phi_end, _, taus, series = integrate(Phi0, Pi0, dt, n_steps, rhs_fn,
                                         record_every=record_every, observers=obs_idx)
    if not np.all(np.isfinite(Phi_end)):
        return None
    fits = {}
    for k in series:
        r = qnm_method_4_two_mode(taus, series[k], T0_FIT, T1_FIT, potential="zerilli", ell=2)
        fits[k] = r
    return {"n": n, "dx": dx, "dt": dt, "n_steps": n_steps,
            "max_abs_end": float(np.max(np.abs(Phi_end))),
            "taus": taus, "series": series, "fits": fits,
            "obs": {k: v.r_actual_M for k, v in observers.items()}}


def main():
    ref = kerr_qnm(a_over_M=0.0, ell=2, m=2, n=0)
    print(f"Reference (qnm pkg, s=-2, l=2, m=2, n=0, a=0):  "
          f"M*omega = {ref.M_omega_R:.6f},  tau/M = {ref.tau_over_M:.4f}")
    print(f"Reference (parent paper Zerilli M4):              "
          f"M*omega = 0.3737,   tau/M = 11.241\n")

    grids = [501, 1001, 2001]
    rows = []
    last_out = None
    for n in grids:
        print(f"[run n={n}] ...", flush=True)
        out = run_one(n)
        if out is None:
            print(f"  blew up")
            continue
        last_out = out
        print(f"  dx={out['dx']:.3e}, dt={out['dt']:.3e}, n_steps={out['n_steps']}, "
              f"max|Phi(T)|={out['max_abs_end']:.3e}")
        for k, f in out["fits"].items():
            om = f["omega"]; ta = f["tau"]
            err_om = abs(om - ref.M_omega_R) / ref.M_omega_R if np.isfinite(om) else float("nan")
            err_ta = abs(ta - ref.tau_over_M) / ref.tau_over_M if np.isfinite(ta) else float("nan")
            print(f"  [{k}] M*omega = {om:.6f}  (err {err_om:.2e}),  "
                  f"tau/M = {ta:7.4f}  (err {err_ta:.2e})")
            rows.append((n, k, om, ta, err_om, err_ta))

    print("\n=== Convergence table ===")
    print(f"{'n':>5}  {'obs':>6}  {'M*omega':>10}  {'err_om':>10}  {'tau/M':>10}  {'err_tau':>10}")
    for r in rows:
        n, k, om, ta, eo, et = r
        print(f"{n:>5}  {k:>6}  {om:>10.6f}  {eo:>10.2e}  {ta:>10.4f}  {et:>10.2e}")

    if last_out is not None:
        outpath = Path("outputs/phase_a/convergence_a0.npz")
        outpath.parent.mkdir(parents=True, exist_ok=True)
        np.savez(outpath, taus=last_out["taus"], **{f"y_{k}": v for k, v in last_out["series"].items()})
        print(f"\nSaved highest-resolution waveform -> {outpath}")


if __name__ == "__main__":
    main()

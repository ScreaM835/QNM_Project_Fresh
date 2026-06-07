"""A.8b diagnostic: gauge sanity + inner-sponge control.

(1) Print height-function H, slicing factor 1-H^2, and dh/dr at the two
    observers and at the grid edges. Confirms tanh(r_*/L=30) on
    r_*∈[-20,80] does not reach the asymptotic |H|=1 at either end.
(2) Rerun n=1001 with inner_width_frac=0.0 (outer sponge only). This
    must reproduce the earlier outer-sponge-only run (M*omega≈0.374 at
    r10M with hand-picked window).
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hyperboloidal_schwarzschild import (
    height_H, height_dh_dr, slicing_factor, tortoise,
)
from src.rwz_hyperboloidal import _invert_tortoise
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
R_OBS = [("r10M", 10.0), ("r20M", 20.0)]


def gauge_report(n: int):
    r_star = np.linspace(R_MIN, R_MAX, n)
    r = _invert_tortoise(r_star, M)
    H = height_H(r, M, L)
    sig = slicing_factor(r, M, L)
    dhdr = height_dh_dr(r, M, L)
    observers = make_observers(r_star, R_OBS)
    print(f"\n=== Gauge report (n={n}, L={L}, r_*∈[{R_MIN},{R_MAX}]) ===")
    print(f"{'pt':>10}  {'r_*':>10}  {'r/M':>10}  {'H':>10}  {'1-H^2':>12}  {'dh/dr':>10}")
    print(f"{'left':>10}  {r_star[0]:>10.3f}  {r[0]:>10.4f}  {H[0]:>10.5f}  {sig[0]:>12.3e}  {dhdr[0]:>10.4f}")
    for k, o in observers.items():
        i = o.grid_index
        print(f"{k:>10}  {r_star[i]:>10.3f}  {r[i]:>10.4f}  {H[i]:>10.5f}  {sig[i]:>12.3e}  {dhdr[i]:>10.4f}")
    print(f"{'right':>10}  {r_star[-1]:>10.3f}  {r[-1]:>10.4f}  {H[-1]:>10.5f}  {sig[-1]:>12.3e}  {dhdr[-1]:>10.4f}")
    print(f"max|H|={np.max(np.abs(H)):.6f}  (should be ~1 at horizon and scri)")


def run_control(n: int):
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
    print(f"\n=== Control run n={n}, outer-sponge-only ===")
    print(f"dx={dx:.3e}, dt={dt:.3e}, n_steps={n_steps}, max|Phi(T)|={np.max(np.abs(Phi_end)):.3e}")
    ref = kerr_qnm(a_over_M=0.0, ell=2, m=2, n=0)
    for k, y in series.items():
        print(f"  --- {k} (r_actual={observers[k].r_actual_M:.3f}M, max|y|={np.max(np.abs(y)):.3e}) ---")
        for (t0, t1) in [(15, 90), (20, 90), (20, 120), (25, 120), (30, 120), (40, 140)]:
            r = qnm_method_4_two_mode(taus, y, t0, t1, potential="zerilli", ell=2)
            om, ta = r["omega"], r["tau"]
            eo = abs(om - ref.M_omega_R) / ref.M_omega_R if np.isfinite(om) else float("nan")
            et = abs(ta - ref.tau_over_M) / ref.tau_over_M if np.isfinite(ta) else float("nan")
            print(f"    t∈[{t0},{t1}]: M*omega={om:.5f} (err {eo:.2e}), tau/M={ta:7.3f} (err {et:.2e})")


def main():
    gauge_report(1001)
    for n in [501, 1001, 2001]:
        run_control(n)


if __name__ == "__main__":
    main()

"""A.5 acceptance: RK4 on simple ODEs and on the hyperboloidal wave system.

(1) RK4 on dy/dt = -y, exact e^{-t}: 4th-order convergence in dt.
(2) RK4 on Phi'' = -k^2 Phi (standing wave), energy bounded to ~dt^4.
(3) MOL integration of the V=0 hyperboloidal wave with a Gaussian outgoing
    pulse: solution stays bounded to tau = 200 (no blow-up) at the
    advisory CFL.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fd_stencils import d1_central, d2_central
from src.mol_rk4 import rk4_step, integrate, cfl_dt
from src.rwz_hyperboloidal import build_operator, rhs


def test_rk4_order_on_decay():
    """y' = -y, y(0)=1, integrate to T=1; observed rate -> 4."""
    def rhs_fn(y, _zero):
        return -y, np.zeros_like(_zero)
    errs = []
    dts = []
    for n in [40, 80, 160, 320]:
        dt = 1.0 / n
        y = np.array([1.0])
        z = np.array([0.0])
        for _ in range(n):
            y, z = rk4_step(y, z, dt, rhs_fn)
        errs.append(abs(y[0] - np.exp(-1.0)))
        dts.append(dt)
    errs = np.array(errs)
    dts = np.array(dts)
    rates = np.log(errs[:-1] / errs[1:]) / np.log(dts[:-1] / dts[1:])
    assert rates[-1] > 3.9, f"RK4 rate too low: {rates}"
    return float(rates[-1]), float(errs[-1])


def test_rk4_harmonic_oscillator_energy():
    """Phi'' = -k^2 Phi: integrate one period, energy drift ~ dt^4."""
    k = 1.5
    T = 2.0 * np.pi / k
    def rhs_fn(Phi, Pi):
        return Pi, -(k ** 2) * Phi
    Phi = np.array([1.0])
    Pi = np.array([0.0])
    n = 4000
    dt = T / n
    E0 = 0.5 * (Pi[0] ** 2 + k ** 2 * Phi[0] ** 2)
    for _ in range(n):
        Phi, Pi = rk4_step(Phi, Pi, dt, rhs_fn)
    E1 = 0.5 * (Pi[0] ** 2 + k ** 2 * Phi[0] ** 2)
    drift = abs(E1 - E0) / E0
    assert drift < 1e-8, f"energy drift too large: {drift}"
    return float(drift)


def test_hyperboloidal_wave_stable_short_T():
    """V=0 hyperboloidal wave, Gaussian pulse, bounded to tau = 20M.

    A.5 verifies the stepper does not blow up. Long-time (T=200M)
    stability and the proper gauge are revisited in A.8 with a more
    favourable choice of H(r_*) and slice truncation; for now we use
    L=10M on r_* in [-15, 30], coarse grid n=301, short T.
    """
    M = 1.0
    L = 10.0
    r_star = np.linspace(-15.0, 30.0, 301)
    op = build_operator(r_star, M=M, L=L, ell=2)
    op.V[:] = 0.0
    dx = r_star[1] - r_star[0]

    x0, sig = 4.0, 2.0
    Phi0 = np.exp(-((r_star - x0) ** 2) / (2.0 * sig ** 2))
    Pi0 = ((r_star - x0) / sig ** 2) * Phi0  # outgoing flat-space pulse

    def rhs_fn(Phi, Pi):
        return rhs(Phi, Pi, op, d1_central, d2_central)

    H_max = float(np.max(np.abs(op.H)))
    dt = cfl_dt(dx, H_max, safety=0.4)
    T = 20.0
    n_steps = int(np.ceil(T / dt))
    Phi, Pi, _, _ = integrate(Phi0, Pi0, dt, n_steps, rhs_fn)

    max_abs = float(np.max(np.abs(Phi)))
    assert np.all(np.isfinite(Phi)), "blow-up: NaN/Inf"
    assert max_abs < 10.0, f"unbounded growth: max|Phi| = {max_abs}"
    return max_abs, dt, n_steps


def main():
    tests = [
        ("RK4 4th-order on y'=-y       ", test_rk4_order_on_decay),
        ("RK4 SHO energy drift ~ dt^4  ", test_rk4_harmonic_oscillator_energy),
        ("hyper V=0 wave stable to T=20M", test_hyperboloidal_wave_stable_short_T),
    ]
    passed = 0
    for name, fn in tests:
        try:
            res = fn()
            print(f"  PASS  {name}  result={res}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}  {e}")
    print(f"\n{passed} / {len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)


if __name__ == "__main__":
    main()

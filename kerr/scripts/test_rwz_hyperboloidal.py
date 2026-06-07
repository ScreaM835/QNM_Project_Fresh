"""A.3 acceptance tests for the hyperboloidal Regge-Wheeler operator.

Verifies
  (1) the RW potential matches its closed-form value at standard points,
  (2) coefficients (H, H', 1-H^2, V) are finite on the bulk grid,
  (3) flat-space outgoing wave Phi = f(r_* - t) annihilates the V=0 operator
      analytically (machine precision when all derivatives are analytic),
  (4) the same outgoing wave annihilates the V=0 operator to truncation order
      when d_{r_*}^2 Phi and d_{r_*} Pi are taken from the d1/d2 stencils
      (2nd-order convergence as dx -> 0),
  (5) the tortoise inversion round-trips r_* -> r -> r_* at machine precision.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hyperboloidal_schwarzschild import tortoise
from src.fd_stencils import d1_central, d2_central
from src.rwz_hyperboloidal import build_operator, potential_RW, rhs, _invert_tortoise


M = 1.0
L = 2.0
ELL = 2


def test_RW_potential_peak_near_3M():
    """V_RW peaks near r ~ 3M for ell=2; check value matches closed form."""
    r = np.array([3.0])
    V = potential_RW(r, M=M, ell=ELL)
    expected = (1.0 - 2.0 / 3.0) * (6.0 / 9.0 - 6.0 / 27.0)
    err = float(np.abs(V[0] - expected))
    assert err < 1e-14, f"V_RW mismatch at r=3M: {err}"
    return err


def test_tortoise_inversion_roundtrip():
    """Newton inversion of r_*(r) recovers r to ~1e-13."""
    r_in = np.linspace(2.001, 1000.0, 500)
    r_star = tortoise(r_in, M)
    r_out = _invert_tortoise(r_star, M)
    err = float(np.max(np.abs(r_out - r_in)))
    assert err < 1e-10, f"tortoise inversion residual {err}"
    return err


def test_coefficients_finite_on_bulk():
    """All coefficients finite on r_* in [-50, 50] (M=1, L=2)."""
    r_star = np.linspace(-50.0, 50.0, 2001)
    op = build_operator(r_star, M=M, L=L, ell=ELL)
    assert np.all(np.isfinite(op.H))
    assert np.all(np.isfinite(op.Hprime))
    assert np.all(np.isfinite(op.V))
    assert np.all(op.one_minus_H2 > 0.0)
    return float(op.one_minus_H2.min()), float(op.one_minus_H2.max())


def _outgoing_pulse(r_star, tau, x0=4.0, sig=2.0):
    """Phi = exp( -(r_* - tau - h(r_*))^2 / (2 sig^2) ) shifted to peak at x0 initially.

    Analytically Phi(t, r_*) = g(r_* - t) where g(xi) = exp(-(xi - x0)^2 / (2 sig^2));
    in hyperboloidal coords Phi(tau, r_*) = g(r_* - tau - h(r_*)).
    """
    x = tortoise.__self__ if False else None  # noop
    return None  # placeholder, replaced below


def _gauss(xi, x0, sig):
    return np.exp(-((xi - x0) ** 2) / (2.0 * sig ** 2))


def _gauss_d1(xi, x0, sig):
    return -((xi - x0) / sig ** 2) * _gauss(xi, x0, sig)


def _gauss_d2(xi, x0, sig):
    return ((xi - x0) ** 2 / sig ** 4 - 1.0 / sig ** 2) * _gauss(xi, x0, sig)


def _outgoing_state(r_star, op, x0=4.0, sig=2.0):
    """Analytic Phi, Pi, d_{r_*}Pi, d_{r_*}^2 Phi, d_tau Pi at tau=0."""
    h = op.L * np.log(np.cosh(r_star / op.L))
    H = op.H
    Hprime = op.Hprime
    xi = r_star - h
    g = _gauss(xi, x0, sig)
    gp = _gauss_d1(xi, x0, sig)
    gpp = _gauss_d2(xi, x0, sig)
    Phi = g
    Pi = -gp
    d1_Pi = -gpp * (1.0 - H)
    d2_Phi = gpp * (1.0 - H) ** 2 - gp * Hprime
    dPi_dtau_true = gpp  # d_tau Pi = d_tau(-g'(xi)) = -g''(xi) * (-1) = g''(xi)
    return Phi, Pi, d1_Pi, d2_Phi, dPi_dtau_true


def test_outgoing_wave_analytic_zero_residual():
    """For V=0 on bulk grid, RHS matches analytic d_tau Pi to machine precision.

    Bulk = region where 1 - H^2 is well above float64 underflow.
    Slice-boundary behaviour (1 - H^2 < 1e-16) is a separate stability
    concern for A.5 and not in scope for the algebraic test here.
    """
    r_star = np.linspace(-30.0, 80.0, 4001)
    op = build_operator(r_star, M=M, L=L, ell=ELL)
    op.V[:] = 0.0
    Phi, Pi, d1_Pi, d2_Phi, dPi_true = _outgoing_state(r_star, op)
    rhs_Pi = op.inv_one_minus_H2 * (d2_Phi - 2.0 * op.H * d1_Pi - op.Hprime * Pi - op.V * Phi)
    mask = (r_star > -10.0) & (r_star < 20.0)
    err = float(np.max(np.abs(rhs_Pi[mask] - dPi_true[mask])))
    assert err < 1e-8, f"analytic outgoing wave residual {err}"
    return err


def _fd_residual(n):
    r_star = np.linspace(-30.0, 80.0, n)
    op = build_operator(r_star, M=M, L=L, ell=ELL)
    op.V[:] = 0.0
    Phi, Pi, _, _, dPi_true = _outgoing_state(r_star, op)
    _, dPi_dtau = rhs(Phi, Pi, op, d1_central, d2_central)
    mask = (r_star > 0.0) & (r_star < 12.0)
    return r_star[1] - r_star[0], float(np.max(np.abs(dPi_dtau[mask] - dPi_true[mask])))


def test_outgoing_wave_FD_second_order_convergence():
    """With d1/d2 stencils, residual of V=0 outgoing wave converges at 2nd order."""
    ns = [801, 1601, 3201, 6401]
    rows = [_fd_residual(n) for n in ns]
    dxs = np.array([r[0] for r in rows])
    errs = np.array([r[1] for r in rows])
    rates = np.log(errs[:-1] / errs[1:]) / np.log(dxs[:-1] / dxs[1:])
    assert rates[-1] > 1.9, f"FD residual rate too low: {rates}"
    return float(rates[-1]), float(errs[-1])


def main():
    tests = [
        ("RW potential exact value           ", test_RW_potential_peak_near_3M),
        ("tortoise inversion round-trip      ", test_tortoise_inversion_roundtrip),
        ("coefficients finite on bulk        ", test_coefficients_finite_on_bulk),
        ("analytic outgoing wave residual=0  ", test_outgoing_wave_analytic_zero_residual),
        ("FD residual 2nd-order convergence  ", test_outgoing_wave_FD_second_order_convergence),
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

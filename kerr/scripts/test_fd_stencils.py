"""A.4 acceptance: 2nd-order convergence of d1, d2 on smooth manufactured data."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fd_stencils import d1_central, d2_central


K = 2.0  # wavenumber for manufactured u(x) = sin(K x)


def _grid(n, L=2.0 * np.pi):
    x = np.linspace(0.0, L, n)
    dx = x[1] - x[0]
    return x, dx


def _errors(n):
    x, dx = _grid(n)
    u = np.sin(K * x)
    du_true = K * np.cos(K * x)
    d2u_true = -(K ** 2) * np.sin(K * x)
    e1_int = np.max(np.abs(d1_central(u, dx)[1:-1] - du_true[1:-1]))
    e2_int = np.max(np.abs(d2_central(u, dx)[1:-1] - d2u_true[1:-1]))
    e1_full = np.max(np.abs(d1_central(u, dx) - du_true))
    e2_full = np.max(np.abs(d2_central(u, dx) - d2u_true))
    return dx, e1_int, e2_int, e1_full, e2_full


def _rates(ns):
    rows = [_errors(n) for n in ns]
    dxs = np.array([r[0] for r in rows])
    out = {}
    for k, label in enumerate(["d1 int", "d2 int", "d1 full", "d2 full"], start=1):
        errs = np.array([r[k] for r in rows])
        rates = np.log(errs[:-1] / errs[1:]) / np.log(dxs[:-1] / dxs[1:])
        out[label] = (errs, rates)
    return dxs, out


def test_d1_interior_second_order():
    _, out = _rates([65, 129, 257, 513])
    errs, rates = out["d1 int"]
    assert rates[-1] > 1.95, f"d1 interior rate too low: {rates}"
    return float(rates[-1]), float(errs[-1])


def test_d2_interior_second_order():
    _, out = _rates([65, 129, 257, 513])
    errs, rates = out["d2 int"]
    assert rates[-1] > 1.95, f"d2 interior rate too low: {rates}"
    return float(rates[-1]), float(errs[-1])


def test_d1_full_second_order():
    _, out = _rates([65, 129, 257, 513])
    errs, rates = out["d1 full"]
    assert rates[-1] > 1.9, f"d1 full-grid rate too low: {rates}"
    return float(rates[-1]), float(errs[-1])


def test_d2_full_second_order():
    _, out = _rates([65, 129, 257, 513])
    errs, rates = out["d2 full"]
    assert rates[-1] > 1.9, f"d2 full-grid rate too low: {rates}"
    return float(rates[-1]), float(errs[-1])


def test_exact_on_quadratic():
    """d2 of a quadratic is exact (up to roundoff) interior AND at boundaries."""
    x, dx = _grid(33)
    u = 3.0 * x ** 2 - 2.0 * x + 1.0
    d2 = d2_central(u, dx)
    err = float(np.max(np.abs(d2 - 6.0)))
    assert err < 1e-10, f"d2 not exact on quadratic: {err}"
    return err


def test_exact_on_linear():
    """d1 of a linear function is exact everywhere."""
    x, dx = _grid(33)
    u = 4.0 * x - 7.0
    d1 = d1_central(u, dx)
    err = float(np.max(np.abs(d1 - 4.0)))
    assert err < 1e-12, f"d1 not exact on linear: {err}"
    return err


def main():
    tests = [
        ("d1 interior 2nd-order        ", test_d1_interior_second_order),
        ("d2 interior 2nd-order        ", test_d2_interior_second_order),
        ("d1 full-grid 2nd-order       ", test_d1_full_second_order),
        ("d2 full-grid 2nd-order       ", test_d2_full_second_order),
        ("d1 exact on linear           ", test_exact_on_linear),
        ("d2 exact on quadratic        ", test_exact_on_quadratic),
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

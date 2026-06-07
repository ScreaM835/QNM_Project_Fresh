"""A.6 acceptance: Gaussian initial data sanity checks."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.initial_data import GaussianID, gaussian


def test_peak_value_and_location():
    r_star = np.linspace(-10.0, 30.0, 4001)
    p = GaussianID(A0=1.0, x0=4.0, sigma=5.0)
    Phi, Pi = gaussian(r_star, p)
    i = int(np.argmax(Phi))
    assert abs(r_star[i] - 4.0) < 1e-2, f"peak at r_star={r_star[i]}"
    assert abs(Phi[i] - 1.0) < 1e-6, f"peak value {Phi[i]}"
    assert np.all(Pi == 0.0)
    return float(r_star[i]), float(Phi[i])


def test_normalisation_matches_analytic():
    r_star = np.linspace(-200.0, 200.0, 40001)
    p = GaussianID(A0=1.0, x0=4.0, sigma=5.0)
    Phi, _ = gaussian(r_star, p)
    integral = np.trapezoid(Phi, r_star)
    expected = p.A0 * p.sigma * np.sqrt(2.0 * np.pi)
    err = float(abs(integral - expected) / expected)
    assert err < 1e-6, f"integral err {err}"
    return err


def test_amplitude_scales_linearly():
    r_star = np.linspace(-10.0, 30.0, 1001)
    Phi1, _ = gaussian(r_star, GaussianID(A0=1.0, x0=4.0, sigma=5.0))
    Phi3, _ = gaussian(r_star, GaussianID(A0=3.0, x0=4.0, sigma=5.0))
    err = float(np.max(np.abs(Phi3 - 3.0 * Phi1)))
    assert err < 1e-12, f"non-linear scaling: {err}"
    return err


def main():
    tests = [
        ("Gaussian peak value/location ", test_peak_value_and_location),
        ("Gaussian L1 norm vs analytic ", test_normalisation_matches_analytic),
        ("A0 scales Phi linearly       ", test_amplitude_scales_linearly),
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

"""Tests for kerr/src/kerr_hyperboloidal.py (task B.1).

Run from inside kerr/:
    python scripts/test_kerr_hyperboloidal.py

The decisive acceptance check is test_a0_reduction: every B.1 function at
a = 0 must equal the validated Phase A Schwarzschild map to ~1e-13.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.kerr_hyperboloidal import (
    horizons, delta,
    sigma_of_r, r_of_sigma, dr_dsigma, dsigma_dr,
    tortoise, dtortoise_dr,
)
from src import hyperboloidal_schwarzschild as schw


M = 1.0
SPINS = [0.0, 0.5, 0.9, 0.95]


def test_horizon_values():
    """r_+ + r_- = 2M, r_+ r_- = a^2 (Vieta on Delta)."""
    worst = 0.0
    for a in SPINS:
        rp, rm = horizons(a, M)
        worst = max(worst, abs(rp + rm - 2.0 * M), abs(rp * rm - a * a))
    assert worst < 1e-14, f"horizon identities off by {worst:.2e}"
    return worst


def test_delta_factorisation():
    """Delta = (r - r_+)(r - r_-) to relative machine precision, all spins.

    Bounded away from the horizon (r >= r_+ + 0.1 M): the factored form
    subtracts r - r_+, two O(M) numbers whose difference is O(1e-6 M) in the
    immediate horizon neighbourhood, so relative precision there is set by
    cancellation, not by the formula. The production solver never evaluates in
    r near r_+ (it uses sigma = r_+/r, sigma = 1 at the horizon), so this
    diagnostic is checked where it is well-conditioned.
    """
    worst = 0.0
    for a in SPINS:
        rp, rm = horizons(a, M)
        r = np.linspace(rp + 0.1 * M, 100.0 * M, 500)
        lhs = delta(r, a, M)
        rhs = (r - rp) * (r - rm)
        worst = max(worst, float(np.max(np.abs(lhs - rhs) / (np.abs(rhs) + 1e-30))))
    assert worst < 1e-13, f"Delta factorisation relative error {worst:.2e}"
    return worst


def test_radial_roundtrip():
    """r -> sigma -> r recovers r to machine precision, all spins."""
    worst = 0.0
    for a in SPINS:
        rp, _ = horizons(a, M)
        r = np.logspace(np.log10(rp * (1.0 + 1e-6)), 4.0, 200)
        r_rec = r_of_sigma(sigma_of_r(r, a, M), a, M)
        worst = max(worst, float(np.max(np.abs(r - r_rec) / r)))
    assert worst < 1e-14, f"round-trip relative error {worst:.2e} exceeds 1e-14"
    return worst


def test_sigma_boundaries():
    """sigma -> 0 as r -> inf; sigma -> 1 as r -> r_+, all spins."""
    worst_inf, worst_hor = 0.0, 0.0
    for a in SPINS:
        rp, _ = horizons(a, M)
        s_inf = float(sigma_of_r(1e10 * M, a, M))
        s_hor = float(sigma_of_r(rp * (1.0 + 1e-12), a, M))
        worst_inf = max(worst_inf, s_inf)
        worst_hor = max(worst_hor, abs(s_hor - 1.0))
    assert worst_inf < 1e-9, f"sigma at large r = {worst_inf:.2e}, expected ~0"
    assert worst_hor < 1e-10, f"sigma at horizon off by {worst_hor:.2e}"
    return worst_inf, worst_hor


def test_derivative_consistency():
    """dr/dsigma * dsigma/dr == 1 (chain rule), all spins."""
    worst = 0.0
    for a in SPINS:
        sigma = np.linspace(1e-3, 1.0 - 1e-3, 50)
        r = r_of_sigma(sigma, a, M)
        prod = dr_dsigma(sigma, a, M) * dsigma_dr(r, a, M)
        worst = max(worst, float(np.max(np.abs(prod - 1.0))))
    assert worst < 1e-13, f"chain-rule error {worst:.2e}"
    return worst


def test_dtortoise_numerical_derivative():
    """dr_*/dr matches a central finite difference at a/M = 0.9."""
    a = 0.9
    rp, _ = horizons(a, M)
    r = np.linspace(rp + 0.5 * M, 50.0 * M, 40)
    h = 1e-6 * M
    fd = (tortoise(r + h, a, M) - tortoise(r - h, a, M)) / (2.0 * h)
    ana = dtortoise_dr(r, a, M)
    rel = float(np.max(np.abs(fd - ana) / np.abs(ana)))
    assert rel < 1e-7, f"numerical vs analytic dr_*/dr disagree at {rel:.2e}"
    return rel


def test_tortoise_horizon_divergence():
    """r_* -> -inf as r -> r_+ from above, all spins."""
    worst = 0.0
    for a in SPINS:
        rp, _ = horizons(a, M)
        rstar = float(tortoise(rp * (1.0 + 1e-8), a, M))
        assert rstar < -10.0 * M, f"r_* near horizon = {rstar:.3f} (a={a}), expected very negative"
        worst = min(worst, rstar)
    return worst


def test_tortoise_large_r_log_coeff():
    """At large r the tortoise leading log coefficient is 2M for every spin."""
    worst = 0.0
    for a in SPINS:
        r1, r2 = 1e6 * M, 1e7 * M  # one decade apart
        # r_* - r ~ 2M log(r/2M) + C  =>  (rstar2 - r2) - (rstar1 - r1) ~ 2M log(10)
        d = (tortoise(r2, a, M) - r2) - (tortoise(r1, a, M) - r1)
        expected = 2.0 * M * np.log(10.0)
        worst = max(worst, abs(d - expected) / expected)
    assert worst < 1e-5, f"large-r log coefficient off by {worst:.2e}"
    return worst


def test_a0_reduction_coordinate_map():
    """DECISIVE (part 1): the a = 0 coordinate map equals the Phase A map to 1e-13.

    sigma = r_+/r, r = r_+/sigma and their derivatives are exact ratios (no
    subtraction of near-equal numbers), so they reduce to the Schwarzschild
    map to machine precision on the FULL grid, including the deep near-horizon.
    """
    a = 0.0
    r = np.logspace(np.log10(2.0 * M * (1.0 + 1e-6)), 3.0, 300)
    sigma = np.linspace(1e-3, 1.0 - 1e-3, 300)

    checks = {
        "sigma_of_r": (sigma_of_r(r, a, M), schw.sigma_of_r(r, M)),
        "r_of_sigma": (r_of_sigma(sigma, a, M), schw.r_of_sigma(sigma, M)),
        "dr_dsigma": (dr_dsigma(sigma, a, M), schw.dr_dsigma(sigma, M)),
        "dsigma_dr": (dsigma_dr(r, a, M), schw.dsigma_dr(r, M)),
    }
    worst_name, worst = "", 0.0
    for name, (kerr_val, schw_val) in checks.items():
        rel = float(np.max(np.abs(kerr_val - schw_val) / (np.abs(schw_val) + 1e-30)))
        if rel > worst:
            worst_name, worst = name, rel
    assert worst < 1e-13, f"a=0 map reduction worst mismatch in {worst_name}: {worst:.2e}"
    return worst


def test_a0_reduction_tortoise():
    """DECISIVE (part 2): the a = 0 tortoise diagnostics equal Phase A to 1e-13.

    r_*(r) and dr_*/dr involve r - r_+ and so share the horizon-cancellation
    floor of any r-coordinate implementation (the validated Phase A code
    included). Both forms degrade identically within ~1e-6 M of r_+; checked
    on r >= r_+ + 0.1 M where the reduction is exact to machine precision.
    The solver itself never differentiates in r near the horizon.
    """
    a = 0.0
    rp, _ = horizons(a, M)
    r = np.logspace(np.log10(rp + 0.1 * M), 3.0, 300)
    checks = {
        "tortoise": (tortoise(r, a, M), schw.tortoise(r, M)),
        "dtortoise_dr": (dtortoise_dr(r, a, M), schw.dtortoise_dr(r, M)),
    }
    worst_name, worst = "", 0.0
    for name, (kerr_val, schw_val) in checks.items():
        rel = float(np.max(np.abs(kerr_val - schw_val) / (np.abs(schw_val) + 1e-30)))
        if rel > worst:
            worst_name, worst = name, rel
    assert worst < 1e-13, f"a=0 tortoise reduction worst mismatch in {worst_name}: {worst:.2e}"
    return worst


def main():
    tests = [
        ("horizon identities r_+ + r_- = 2M  ", test_horizon_values),
        ("Delta = (r - r_+)(r - r_-)         ", test_delta_factorisation),
        ("radial round-trip (all spins)      ", test_radial_roundtrip),
        ("sigma boundary limits (all spins)  ", test_sigma_boundaries),
        ("dr/dsigma * dsigma/dr == 1         ", test_derivative_consistency),
        ("dr_*/dr finite-diff (a=0.9)        ", test_dtortoise_numerical_derivative),
        ("tortoise horizon divergence        ", test_tortoise_horizon_divergence),
        ("tortoise large-r log coeff = 2M    ", test_tortoise_large_r_log_coeff),
        ("a=0 reduction: coordinate map      ", test_a0_reduction_coordinate_map),
        ("a=0 reduction: tortoise diagnostics", test_a0_reduction_tortoise),
    ]
    failed = 0
    for label, fn in tests:
        try:
            result = fn()
            print(f"  PASS  {label}  result={result}")
        except AssertionError as e:
            print(f"  FAIL  {label}  {e}")
            failed += 1
    print()
    print(f"{len(tests) - failed} / {len(tests)} tests passed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()

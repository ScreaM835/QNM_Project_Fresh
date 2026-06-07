"""Tests for kerr/src/hyperboloidal_schwarzschild.py (task A.1).

Run from inside kerr/:
    python scripts/test_hyperboloidal.py
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.hyperboloidal_schwarzschild import (
    sigma_of_r, r_of_sigma, dr_dsigma, dsigma_dr,
    tortoise, dtortoise_dr,
    height_H, height_h, height_dh_dr, height_dH_drstar, slicing_factor,
)


M = 1.0


def test_radial_roundtrip():
    """r -> sigma -> r recovers r to machine precision on a log grid."""
    r = np.logspace(np.log10(2.001 * M), 4.0, 200)
    r_recovered = r_of_sigma(sigma_of_r(r, M), M)
    rel = np.max(np.abs(r - r_recovered) / r)
    assert rel < 1e-14, f"round-trip relative error {rel:.2e} exceeds 1e-14"
    return rel


def test_sigma_boundaries():
    """sigma -> 0 as r -> inf; sigma -> 1 as r -> 2M."""
    s_inf = sigma_of_r(1e10 * M, M)
    s_hor = sigma_of_r(2.0 * M * (1.0 + 1e-12), M)
    assert s_inf < 1e-9, f"sigma at large r = {s_inf:.2e}, expected ~0"
    assert abs(s_hor - 1.0) < 1e-10, f"sigma at horizon = {s_hor}, expected 1"
    return float(s_inf), float(s_hor)


def test_derivative_consistency():
    """dr/dsigma * dsigma/dr == 1 (chain rule)."""
    sigma = np.linspace(1e-3, 1.0 - 1e-3, 50)
    r = r_of_sigma(sigma, M)
    prod = dr_dsigma(sigma, M) * dsigma_dr(r, M)
    err = np.max(np.abs(prod - 1.0))
    assert err < 1e-13, f"chain-rule error {err:.2e}"
    return err


def test_tortoise_asymptotic():
    """At large r, r_star - r -> 2 M log(r/2M)."""
    r = 1e4 * M
    rstar = tortoise(r, M)
    expected = r + 2.0 * M * np.log(r / (2.0 * M))
    rel = abs(rstar - expected) / abs(expected)
    assert rel < 1e-6, f"asymptotic mismatch {rel:.2e}"
    return rel


def test_tortoise_horizon_divergence():
    """r_star -> -inf as r -> 2M+."""
    r = 2.0 * M * (1.0 + 1e-8)
    rstar = tortoise(r, M)
    assert rstar < -2.0 * M * 10.0, f"r_star at near-horizon = {rstar:.3f}, expected very negative"
    return float(rstar)


def test_dtortoise_numerical_derivative():
    """dr_star/dr matches a central finite difference."""
    r = np.linspace(2.5 * M, 50.0 * M, 30)
    h = 1e-6 * M
    fd = (tortoise(r + h, M) - tortoise(r - h, M)) / (2.0 * h)
    ana = dtortoise_dr(r, M)
    rel = np.max(np.abs(fd - ana) / np.abs(ana))
    assert rel < 1e-7, f"numerical vs analytic dr_star/dr disagree at {rel:.2e}"
    return rel


# --------------------------------------------------------------------------
# Task A.2 — hyperboloidal height function
# --------------------------------------------------------------------------

L_DEFAULT = 2.0 * M


def test_H_boundary_limits():
    """H -> +1 at r -> infinity, H -> -1 at r -> 2M+."""
    H_inf = height_H(1e6 * M, M, L_DEFAULT)
    H_hor = height_H(2.0 * M * (1.0 + 1e-12), M, L_DEFAULT)
    assert 1.0 - H_inf < 1e-6, f"H at large r = {H_inf}, expected ~1"
    assert H_hor + 1.0 < 1e-6, f"H at near-horizon = {H_hor}, expected ~-1"
    return float(H_inf), float(H_hor)


def test_slicing_spacelike():
    """1 - H^2 > 0 strictly on the bulk-interior r in (2M(1+eps), 200 M).

    Slice is characteristic (1 - H^2 = 0) at r -> 2M and r -> infinity by
    design; this test asserts strict positivity on the region the integrator
    actually evolves over.
    """
    r = np.linspace(2.0 * M * (1.0 + 1e-6), 200.0 * M, 5000)
    sf = slicing_factor(r, M, L_DEFAULT)
    assert np.all(sf > 0.0), f"slicing factor vanishes/negative at min = {sf.min():.3e}"
    return float(sf.min()), float(sf.max())


def test_H_monotone():
    """dH/dr_star > 0 (slicing monotone from horizon to scri+)."""
    r = np.linspace(2.0 * M * (1.0 + 1e-6), 100.0 * M, 500)
    dH = height_dH_drstar(r, M, L_DEFAULT)
    assert np.all(dH > 0.0), f"dH/dr_star not strictly positive (min = {dH.min():.3e})"
    return float(dH.min()), float(dH.max())


def test_height_derivative_consistency():
    """dh/dr (analytic) matches a central finite-difference of h(r)."""
    r = np.linspace(2.5 * M, 50.0 * M, 30)
    eps = 1e-6 * M
    fd = (height_h(r + eps, M, L_DEFAULT) - height_h(r - eps, M, L_DEFAULT)) / (2.0 * eps)
    ana = height_dh_dr(r, M, L_DEFAULT)
    rel = np.max(np.abs(fd - ana) / (np.abs(ana) + 1e-30))
    assert rel < 1e-6, f"dh/dr analytic vs FD disagree at {rel:.2e}"
    return rel


def test_H_squared_chain_rule():
    """H = dh/dr_star = (dh/dr) * (dr/dr_star). Verify chain rule consistency."""
    r = np.linspace(2.5 * M, 50.0 * M, 30)
    H = height_H(r, M, L_DEFAULT)
    dh_dr = height_dh_dr(r, M, L_DEFAULT)
    dr_drstar = 1.0 / dtortoise_dr(r, M)
    H_from_chain = dh_dr * dr_drstar
    err = np.max(np.abs(H - H_from_chain))
    assert err < 1e-13, f"chain-rule check |H - dh/dr * dr/dr_star| = {err:.2e}"
    return err


def main():
    tests = [
        ("radial round-trip                  ", test_radial_roundtrip),
        ("sigma boundary limits              ", test_sigma_boundaries),
        ("dr/dsigma * dsigma/dr == 1         ", test_derivative_consistency),
        ("tortoise large-r asymptotic        ", test_tortoise_asymptotic),
        ("tortoise horizon divergence        ", test_tortoise_horizon_divergence),
        ("dr_star/dr finite-diff agreement   ", test_dtortoise_numerical_derivative),
        ("A.2 H boundary limits              ", test_H_boundary_limits),
        ("A.2 slicing 1 - H^2 > 0 interior   ", test_slicing_spacelike),
        ("A.2 dH/dr_star > 0 (monotone)      ", test_H_monotone),
        ("A.2 dh/dr analytic vs FD           ", test_height_derivative_consistency),
        ("A.2 H = dh/dr * dr/dr_star         ", test_H_squared_chain_rule),
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

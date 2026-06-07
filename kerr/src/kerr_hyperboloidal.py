"""Hyperboloidal radial coordinate map for Kerr (task B.1).

Generalises kerr/src/hyperboloidal_schwarzschild.py (Phase A, a = 0) to
a/M > 0. Only the radial map, horizons, Delta, and the Kerr tortoise are
implemented here; the hyperboloidal height function H(sigma; a) and the
characteristic minimal-gauge construction are the genuine research step of
task B.2 (kerr/notes/kerr_minimal_gauge_derivation.md), not this file.

Conventions (M symbolic; a is the *physical* spin, so a == a_over_M when
M = 1, which is the case throughout this repo):

    r_pm   = M +/- sqrt(M^2 - a^2)            outer / inner horizon
    Delta  = r^2 - 2 M r + a^2 = (r - r_+)(r - r_-)
    sigma  = r_+ / r   in (0, 1]             compactified radius

Boundaries on the closed interval [0, 1]:
    sigma = 0  <->  r = +infinity  <->  future null infinity (scri+).
    sigma = 1  <->  r = r_+        <->  outer (event) horizon.

Kerr tortoise coordinate, dr_*/dr = (r^2 + a^2)/Delta, integrates to

    r_* = r + (2 M r_+ / (r_+ - r_-)) log((r - r_+)/(2M))
            - (2 M r_- / (r_+ - r_-)) log((r - r_-)/(2M)),

singular at r = r_+, asymptotically r_* ~ r + 2 M log(r/2M) at large r (the
leading log coefficient is 2M for every spin, fixed by the ADM mass).

a = 0 reduction (verified numerically in test_kerr_hyperboloidal.py):
    r_+ -> 2M, r_- -> 0, sigma -> 2M/r,
    r_* -> r + 2 M log(r/(2M) - 1)   (the Phase A Schwarzschild map).

References:
    Teukolsky, ApJ 185 635 (1973).
    Panosso Macedo, CQG 37 065019 (2020). Minimal gauge for Kerr.
    Macedo, Jaramillo, Ansorg, PRD 89 064008 (2014). Schwarzschild map.
"""

from __future__ import annotations

import numpy as np


def horizons(a: float, M: float = 1.0):
    """Return (r_+, r_-) = M +/- sqrt(M^2 - a^2). Requires |a| <= M."""
    disc = np.sqrt(M * M - a * a)
    return M + disc, M - disc


def delta(r, a: float, M: float = 1.0):
    """Kerr horizon function Delta = r^2 - 2 M r + a^2 = (r - r_+)(r - r_-)."""
    r = np.asarray(r, dtype=np.float64)
    return r * r - 2.0 * M * r + a * a


def sigma_of_r(r, a: float, M: float = 1.0):
    """Return sigma = r_+ / r. Domain: r > 0."""
    r = np.asarray(r, dtype=np.float64)
    r_plus, _ = horizons(a, M)
    return r_plus / r


def r_of_sigma(sigma, a: float, M: float = 1.0):
    """Return r = r_+ / sigma. Domain: sigma > 0 (avoid sigma == 0)."""
    sigma = np.asarray(sigma, dtype=np.float64)
    r_plus, _ = horizons(a, M)
    return r_plus / sigma


def dr_dsigma(sigma, a: float, M: float = 1.0):
    """Return d r / d sigma = -r_+ / sigma^2. Domain: sigma > 0."""
    sigma = np.asarray(sigma, dtype=np.float64)
    r_plus, _ = horizons(a, M)
    return -r_plus / (sigma * sigma)


def dsigma_dr(r, a: float, M: float = 1.0):
    """Return d sigma / d r = -r_+ / r^2. Domain: r > 0."""
    r = np.asarray(r, dtype=np.float64)
    r_plus, _ = horizons(a, M)
    return -r_plus / (r * r)


def tortoise(r, a: float, M: float = 1.0):
    """Kerr tortoise coordinate r_*.

    r_* = r + (2 M r_+/(r_+ - r_-)) log((r - r_+)/2M)
            - (2 M r_-/(r_+ - r_-)) log((r - r_-)/2M).

    Diverges to -infinity as r -> r_+ from above; asymptotes to
    r + 2 M log(r/2M) for r >> r_+. At a = 0 the second term carries a zero
    coefficient (r_- = 0) and the first reduces to 2 M log(r/2M - 1).
    """
    r = np.asarray(r, dtype=np.float64)
    r_plus, r_minus = horizons(a, M)
    coeff_plus = 2.0 * M * r_plus / (r_plus - r_minus)
    coeff_minus = 2.0 * M * r_minus / (r_plus - r_minus)
    return (
        r
        + coeff_plus * np.log((r - r_plus) / (2.0 * M))
        - coeff_minus * np.log((r - r_minus) / (2.0 * M))
    )


def dtortoise_dr(r, a: float, M: float = 1.0):
    """d r_* / d r = (r^2 + a^2) / Delta. Singular at the outer horizon."""
    r = np.asarray(r, dtype=np.float64)
    return (r * r + a * a) / delta(r, a, M)

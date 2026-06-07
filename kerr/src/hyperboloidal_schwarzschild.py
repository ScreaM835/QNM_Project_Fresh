"""Hyperboloidal radial coordinate map for Schwarzschild.

Task A.1 of kerr/notes/phase_a_plan.md.

Map (Macedo, Jaramillo, Ansorg, PRD 89 064008, 2014):

    sigma = 2 M / r,   r = 2 M / sigma,   sigma in (0, 1].

Boundaries on the closed interval [0, 1]:
    sigma = 0  <->  r = +infinity  <->  future null infinity (scri+).
    sigma = 1  <->  r = 2 M        <->  event horizon.

The standard Schwarzschild tortoise coordinate is

    r_star(r) = r + 2 M * log(r / (2M) - 1),

singular at the horizon, asymptotically r_star ~ r + 2 M log(r / 2M) at large r.

Only the radial map and r_star are implemented here. The hyperboloidal time
slicing (height function h(r)) is task A.2, in this same file once verified.
"""

from __future__ import annotations

import numpy as np


def sigma_of_r(r, M: float = 1.0):
    """Return sigma = 2 M / r. Domain: r > 0."""
    r = np.asarray(r, dtype=np.float64)
    return 2.0 * M / r


def r_of_sigma(sigma, M: float = 1.0):
    """Return r = 2 M / sigma. Domain: sigma > 0 (avoid sigma == 0)."""
    sigma = np.asarray(sigma, dtype=np.float64)
    return 2.0 * M / sigma


def dr_dsigma(sigma, M: float = 1.0):
    """Return d r / d sigma = -2 M / sigma^2. Domain: sigma > 0."""
    sigma = np.asarray(sigma, dtype=np.float64)
    return -2.0 * M / (sigma * sigma)


def dsigma_dr(r, M: float = 1.0):
    """Return d sigma / d r = -2 M / r^2. Domain: r > 0."""
    r = np.asarray(r, dtype=np.float64)
    return -2.0 * M / (r * r)


def tortoise(r, M: float = 1.0):
    """Schwarzschild tortoise coordinate r_star.

    r_star(r) = r + 2 M log(r/(2M) - 1).

    Diverges to -infinity as r -> 2 M from above, asymptotes to
    r + 2 M log(r/(2M)) for r >> 2 M.
    """
    r = np.asarray(r, dtype=np.float64)
    return r + 2.0 * M * np.log(r / (2.0 * M) - 1.0)


def dtortoise_dr(r, M: float = 1.0):
    """d r_star / d r = 1 / (1 - 2 M / r).

    Singular at horizon. Equal to the inverse of the Schwarzschild
    factor f(r) = 1 - 2M/r.
    """
    r = np.asarray(r, dtype=np.float64)
    return 1.0 / (1.0 - 2.0 * M / r)


# ----------------------------------------------------------------------
# Task A.2 — hyperboloidal time slicing
#
# tau = t - h(r), with the height function chosen so that
#     H(r_star) := dh/dr_star = tanh(r_star / L)
#     h(r_star) = L * log(cosh(r_star / L))
# Default L = 2M. Properties:
#   - H -> +1 as r_star -> +inf   (null at scri+)
#   - H -> -1 as r_star -> -inf   (null at horizon, characteristic outflow)
#   - |H| < 1 strictly in the interior, so tau = const surfaces are spacelike
#     for r > 2M (induced metric on the slice is f * (1 - H^2) dr_star^2 +
#     r^2 dOmega^2 with f = 1 - 2M/r > 0 outside the horizon).
#
# Free parameter L sets the transition scale. Default L = 2M places the
# H = 0 crossing at r_star = 0, i.e. r ~ 3.55 M (peak of the Regge-Wheeler
# potential is at r ~ 3 M for ell = 2). This is gauge; if A.3 needs a
# different choice (e.g. PanossoMacedo minimal gauge with polynomial-in-sigma
# coefficients), edit only this section.
# ----------------------------------------------------------------------


def height_H(r, M: float = 1.0, L: float = 2.0):
    """H(r) := dh/dr_star = tanh(r_star(r) / L). Dimensionless, |H| < 1 for r > 2M."""
    return np.tanh(tortoise(r, M) / L)


def height_h(r, M: float = 1.0, L: float = 2.0):
    """h(r) = L * log(cosh(r_star(r) / L)). Time-shift, in units of M."""
    return L * np.log(np.cosh(tortoise(r, M) / L))


def height_dh_dr(r, M: float = 1.0, L: float = 2.0):
    """dh/dr = H(r) * (dr_star/dr) = tanh(r_star/L) / (1 - 2M/r). Singular at horizon."""
    return height_H(r, M, L) * dtortoise_dr(r, M)


def height_dH_drstar(r, M: float = 1.0, L: float = 2.0):
    """dH/dr_star = (1/L) sech^2(r_star/L). Positive, vanishing at both boundaries."""
    rstar_over_L = tortoise(r, M) / L
    return (1.0 / L) / np.cosh(rstar_over_L) ** 2


def slicing_factor(r, M: float = 1.0, L: float = 2.0):
    """1 - H(r)^2; coefficient of dr_star^2 in the induced metric on tau = const.

    Strictly positive for r > 2M and finite (so the slice is spacelike);
    vanishes only in the limits r -> 2M and r -> infinity, where the slice
    becomes characteristic (null) by design.

    Implemented as sech^2(r_star/L) to avoid the float64 saturation
    1 - tanh^2 -> 1 - 1 = 0 that occurs for |r_star|/L > ~18.
    """
    rstar_over_L = tortoise(r, M) / L
    return 1.0 / np.cosh(rstar_over_L) ** 2

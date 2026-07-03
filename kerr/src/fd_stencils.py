"""Central finite-difference operators on a uniform 1D grid.

Second-order accurate interior stencils with one-sided second-order
boundary closures. No physics here — pure numerical-analysis plumbing
called by the hyperboloidal RWZ/Teukolsky operator in A.3+.
"""
from __future__ import annotations

import numpy as np


def d1_central(u: np.ndarray, dx: float) -> np.ndarray:
    """First derivative, 2nd-order central interior, 2nd-order one-sided edges."""
    n = u.shape[-1]
    if n < 3:
        raise ValueError("need at least 3 points for 2nd-order stencil")
    du = np.empty_like(u)
    du[..., 1:-1] = (u[..., 2:] - u[..., :-2]) / (2.0 * dx)
    du[..., 0] = (-3.0 * u[..., 0] + 4.0 * u[..., 1] - u[..., 2]) / (2.0 * dx)
    du[..., -1] = (3.0 * u[..., -1] - 4.0 * u[..., -2] + u[..., -3]) / (2.0 * dx)
    return du


def d2_central(u: np.ndarray, dx: float) -> np.ndarray:
    """Second derivative, 2nd-order central interior, 2nd-order one-sided edges."""
    n = u.shape[-1]
    if n < 4:
        raise ValueError("need at least 4 points for 2nd-order d2 closure")
    d2u = np.empty_like(u)
    d2u[..., 1:-1] = (u[..., 2:] - 2.0 * u[..., 1:-1] + u[..., :-2]) / (dx * dx)
    d2u[..., 0] = (2.0 * u[..., 0] - 5.0 * u[..., 1] + 4.0 * u[..., 2] - u[..., 3]) / (dx * dx)
    d2u[..., -1] = (2.0 * u[..., -1] - 5.0 * u[..., -2] + 4.0 * u[..., -3] - u[..., -4]) / (dx * dx)
    return d2u


def d1_4(u: np.ndarray, dx: float) -> np.ndarray:
    """First derivative, 4th-order central interior + 4th-order one-sided closures.

    Interior uses the 5-point central stencil
    ``(-u[i+2] + 8 u[i+1] - 8 u[i-1] + u[i-2]) / (12 dx)`` (O(dx^4)); the two
    nodes at each end use Fornberg 4th-order one-sided 5-point stencils so the
    whole operator is uniformly 4th-order. Drop-in companion to ``d1_central``
    for the 4th-order solver variant; needs at least 5 points.
    """
    n = u.shape[-1]
    if n < 5:
        raise ValueError("need at least 5 points for 4th-order stencil")
    du = np.empty_like(u)
    # 4th-order central interior
    du[..., 2:-2] = (
        -u[..., 4:] + 8.0 * u[..., 3:-1] - 8.0 * u[..., 1:-3] + u[..., :-4]
    ) / (12.0 * dx)
    # Fornberg 4th-order one-sided (5-point) closures at the two nodes each side
    du[..., 0] = (
        -25.0 * u[..., 0] + 48.0 * u[..., 1] - 36.0 * u[..., 2]
        + 16.0 * u[..., 3] - 3.0 * u[..., 4]
    ) / (12.0 * dx)
    du[..., 1] = (
        -3.0 * u[..., 0] - 10.0 * u[..., 1] + 18.0 * u[..., 2]
        - 6.0 * u[..., 3] + u[..., 4]
    ) / (12.0 * dx)
    du[..., -2] = (
        3.0 * u[..., -1] + 10.0 * u[..., -2] - 18.0 * u[..., -3]
        + 6.0 * u[..., -4] - u[..., -5]
    ) / (12.0 * dx)
    du[..., -1] = (
        25.0 * u[..., -1] - 48.0 * u[..., -2] + 36.0 * u[..., -3]
        - 16.0 * u[..., -4] + 3.0 * u[..., -5]
    ) / (12.0 * dx)
    return du

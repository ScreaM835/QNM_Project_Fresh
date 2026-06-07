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

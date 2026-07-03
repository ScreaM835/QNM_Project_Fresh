"""Numerical dissipation and outer sponge for the hyperboloidal MOL system.

Two stabilisation knobs, both standard in production NR codes.

(1) Kreiss-Oliger (KO) dissipation. For a 2nd-order spatial scheme the
    4th-difference operator
        Q u_i = -(sigma / 16) * (u_{i+2} - 4 u_{i+1} + 6 u_i - 4 u_{i-1} + u_{i-2})
    is added directly to the RHS. It damps grid-scale modes without
    affecting 2nd-order accuracy on smooth solutions. sigma in [0, 1];
    typical values 0.02 - 0.2.

(2) Outer sponge. A smooth damping coefficient gamma(r_*) that ramps from
    0 in the interior to gamma_max in the outermost fraction of the grid,
    added to dPi/dtau as -gamma * Pi. This absorbs would-be reflections
    from the truncated outer boundary.
"""
from __future__ import annotations

import numpy as np


def ko_dissipation(u: np.ndarray, sigma_ko: float) -> np.ndarray:
    """Kreiss-Oliger fourth-difference dissipation; zero in the outermost 2 cells."""
    q = np.zeros_like(u)
    q[2:-2] = -(sigma_ko / 16.0) * (
        u[4:] - 4.0 * u[3:-1] + 6.0 * u[2:-2] - 4.0 * u[1:-3] + u[:-4]
    )
    return q


def ko_dissipation_6(u: np.ndarray, sigma_ko: float) -> np.ndarray:
    """Kreiss-Oliger sixth-difference dissipation; zero in the outermost 3 cells.

    The order-6 KO operator matched to a 4th-order spatial scheme (``d1_4``):
        Q u_i = +(sigma / 64) (u_{i+3} - 6 u_{i+2} + 15 u_{i+1} - 20 u_i
                               + 15 u_{i-1} - 6 u_{i-2} + u_{i-3})
    It damps grid-scale modes without affecting 4th-order accuracy on smooth
    solutions (one order higher than ``ko_dissipation``). The ``+sigma/64`` sign
    is the dissipative branch for r=3. Needs at least 7 points.
    """
    q = np.zeros_like(u)
    q[3:-3] = (sigma_ko / 64.0) * (
        u[6:] - 6.0 * u[5:-1] + 15.0 * u[4:-2] - 20.0 * u[3:-3]
        + 15.0 * u[2:-4] - 6.0 * u[1:-5] + u[:-6]
    )
    return q


def outer_sponge_profile(r_star: np.ndarray, width_frac: float = 0.2, gamma_max: float = 1.0) -> np.ndarray:
    """gamma(r_*): 0 on inner (1 - width_frac) of grid, smoothly to gamma_max at outer edge."""
    n = r_star.shape[0]
    n_sponge = max(1, int(np.floor(width_frac * n)))
    gamma = np.zeros_like(r_star)
    idx = np.arange(n_sponge)
    s = idx / max(1, n_sponge - 1)  # 0 ... 1
    gamma[-n_sponge:] = gamma_max * s ** 4
    return gamma


def two_sided_sponge_profile(
    r_star: np.ndarray,
    inner_width_frac: float = 0.1,
    outer_width_frac: float = 0.2,
    gamma_max: float = 1.0,
) -> np.ndarray:
    """Symmetric sponge: ramps up at both ends, zero in the bulk.

    Required when both r_*_min and r_*_max are artificial truncations
    (the case here: r_* = -20 M is not the horizon, just a wall).
    """
    n = r_star.shape[0]
    gamma = np.zeros_like(r_star)

    n_in = max(1, int(np.floor(inner_width_frac * n)))
    s_in = np.arange(n_in) / max(1, n_in - 1)  # 0 at deepest, 1 at bulk edge
    gamma[:n_in] = gamma_max * (1.0 - s_in) ** 4

    n_out = max(1, int(np.floor(outer_width_frac * n)))
    s_out = np.arange(n_out) / max(1, n_out - 1)  # 0 at bulk edge, 1 at outer
    gamma[-n_out:] = np.maximum(gamma[-n_out:], gamma_max * s_out ** 4)
    return gamma

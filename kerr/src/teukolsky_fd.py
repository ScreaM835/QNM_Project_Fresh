"""Time-domain Teukolsky integrator for (s=-2, l=m=2) on Kerr.

Phase A scope: hyperboloidal-slicing 1+1-D solver. NOT YET IMPLEMENTED.

Equation (Teukolsky 1973, s=-2 master variable psi, single (l,m)):

    A(r,a) psi_tt + B(r,a) psi_tr + C(r,a) psi_rr
       + D(r,a) psi_t + E(r,a) psi_r + V_{lm}(r,a) psi = 0

Recast on hyperboloidal slices (Zenginoglu 2011) with compactified radial
coordinate sigma in [0, 1] mapping horizon (sigma=0) to scri+ (sigma=1).
The hyperboloidal form removes the outer-boundary problem and damps
in-going modes at the horizon, giving long-time stability without sponge
layers.

a -> 0 limit must reduce to the existing Schwarzschild Zerilli pipeline at
the level of M4 plateau-extracted (omega_220, tau_220). This is the
Phase A acceptance test in scripts/validate_a0.py.

References:
    Teukolsky 1973 ApJ 185 635.
    Zenginoglu 2011 PRD 83 127502 (hyperboloidal compactification).
    Harms, Bernuzzi, Bruegmann 2014 CQG 31 245004 (hyperboloidal Teukolsky FD).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class TeukolskyConfig:
    a_over_M: float            # spin parameter a/M, range [0, 0.99]
    M: float = 1.0             # mass
    ell: int = 2
    m: int = 2
    s: int = -2                # spin weight
    sigma_n: int = 401         # hyperboloidal radial grid points
    t_max: float = 200.0       # final time / M
    dt: float = 0.05           # time step / M  (CFL-respecting)
    cfl_safety: float = 0.5
    # Gaussian initial-data pulse, parameters in tortoise-equivalent coords.
    x0: float = 4.0
    sigma: float = 5.0
    A0: float = 1.0


def solve_teukolsky(cfg: TeukolskyConfig) -> dict:
    """Integrate the (s=-2, l, m) Teukolsky master equation on Kerr.

    Returns:
        dict with keys
            t:   shape (Nt,) time grid, in units of M.
            sigma: shape (Nx,) hyperboloidal radial grid.
            psi: shape (Nx, Nt) complex master variable.
            x_obs_indices, x_obs_values: observer indices (e.g. xq=2M, 10M
                in tortoise units mapped back to sigma).
    """
    raise NotImplementedError(
        "Phase A: time-domain Teukolsky integrator. "
        "Choose: (a) hand-code hyperboloidal RK4-MOL, (b) wrap an existing "
        "time-domain library (Markakis TGRF / Sundararajan-Khanna). "
        "Decision is open."
    )


def map_tortoise_to_sigma(x_star: np.ndarray, a_over_M: float, M: float) -> np.ndarray:
    """Map tortoise coordinate x* in [x_min, x_max] M to hyperboloidal sigma in [0,1]."""
    raise NotImplementedError("Phase A.")


def map_sigma_to_tortoise(sigma: np.ndarray, a_over_M: float, M: float) -> np.ndarray:
    """Inverse of map_tortoise_to_sigma."""
    raise NotImplementedError("Phase A.")

"""Minimal-gauge hyperboloidal Regge-Wheeler operator on Schwarzschild.

Derived analytically in kerr/notes/minimal_gauge_derivation.md. Compactified
radial coordinate sigma = 2 M / r in [0, 1] (sigma = 0 is scri, sigma = 1 is
horizon). Height function H(sigma) = 1 - 2 sigma^2. Three-field characteristic
form (Psi, U, W) with bounded coefficients on the closed interval and
characteristic outflow at both endpoints (no boundary data injected).

This module supersedes rwz_hyperboloidal.py for the Schwarzschild axial RW
problem; the tortoise + tanh-gauge implementation in that file is left in
place for diagnostic purposes only and is NOT to be used for production
extractions (it is non-hyperboloidal in practice and CFL-unstable near scri).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Tabulated coefficients on a uniform sigma grid.
# All formulas are closed-form analytic; nothing is fit, nothing is solved.
# Section references point to kerr/notes/minimal_gauge_derivation.md.
# ----------------------------------------------------------------------


@dataclass
class MinGaugeOp:
    sigma: np.ndarray            # uniform grid on [eps, 1 - eps]
    dsigma: float
    M: float
    ell: int
    # Geometry
    r: np.ndarray                # r = 2 M / sigma  (areal radius at each grid point)
    H: np.ndarray                # = 1 - 2 sigma^2
    # Characteristic speeds in (sigma, tau)
    lambda_out: np.ndarray       # = -(1 - sigma) / (4 M),    eq (13a)
    lambda_in:  np.ndarray       # =  sigma^2 / [4 M (1 + sigma)], eq (13b)
    # Characteristic-variable coefficients
    mu_plus:    np.ndarray       # =  sigma^2 / [4 M (1 + sigma)],     eq (22a) (= lambda_in)
    mu_minus:   np.ndarray       # = -(1 - sigma) / (4 M),             eq (22b) (= lambda_out)
    mu_plus_d:  np.ndarray       # d mu_+ / d sigma = sigma (2 + sigma) / [4 M (1 + sigma)^2], eq (31a)
    mu_minus_d: np.ndarray       # d mu_- / d sigma = 1 / (4 M),       eq (31b)
    inv_dmu:    np.ndarray       # = 4 M (1 + sigma)  (= 1 / (mu_+ - mu_-)), used to recover Phi
    one_minus_sigma2: np.ndarray # weight on U in Pi recovery, eq (23b)
    sigma2:     np.ndarray       # weight on W in Pi recovery, eq (23b)
    # Bounded source coefficients (eqs 26a-c, regularised: numerator/(1-H^2))
    c_Phi: np.ndarray            # = -(3 sigma - 2) sigma / [16 M^2 (1 + sigma)],   eq (26a)
    c_Pi:  np.ndarray            # = -  sigma / [2 M (1 + sigma)],                  eq (26b)
    c_Psi: np.ndarray            # = -[l(l+1) - 3 sigma] / [16 M^2 (1 + sigma)],    eq (26c)


def build_minimal_gauge_op(
    N: int,
    M: float = 1.0,
    ell: int = 2,
    eps_scri: float = 1e-8,
    eps_horizon: float = 1e-8,
    include_potential: bool = True,
) -> MinGaugeOp:
    """Construct the minimal-gauge operator on a uniform sigma grid.

    The two endpoints sigma = 0 and sigma = 1 are not sampled exactly to avoid
    division by zero in r = 2 M / sigma; tiny insets eps_* are used. These
    insets do NOT change the boundary-condition story: the characteristic
    speeds (13a, 13b) are already polynomial in sigma at the true endpoints,
    and the insets are O(eps) corrections to coefficients otherwise evaluated
    at the endpoints. Use the same eps_scri across resolutions for honest
    convergence.

    include_potential = False sets c_Psi -> 0, i.e. V = 0. Used for the
    Phase A.8-fix flat-propagation gate (V.1 in section 13 of the derivation).
    """
    if N < 7:
        raise ValueError("need at least 7 points for 2nd-order central + KO")
    sigma = np.linspace(eps_scri, 1.0 - eps_horizon, N)
    dsigma = sigma[1] - sigma[0]
    r = 2.0 * M / sigma
    H = 1.0 - 2.0 * sigma ** 2
    one_plus_sigma = 1.0 + sigma
    one_minus_sigma = 1.0 - sigma
    sigma2 = sigma * sigma

    lambda_out = -one_minus_sigma / (4.0 * M)
    lambda_in = sigma2 / (4.0 * M * one_plus_sigma)
    mu_plus = lambda_in.copy()
    mu_minus = lambda_out.copy()
    mu_plus_d = sigma * (2.0 + sigma) / (4.0 * M * one_plus_sigma ** 2)
    mu_minus_d = np.full_like(sigma, 1.0 / (4.0 * M))
    inv_dmu = 4.0 * M * one_plus_sigma
    one_minus_sigma2 = 1.0 - sigma2

    # Regularised source coefficients (eqs 26a-c with the explicit sign from
    # the d_tau Pi equation; the analytic combination (1 - H^2)^{-1} * S_Pi^raw
    # gives -1 times each ratio).
    c_Phi = -(3.0 * sigma - 2.0) * sigma / (16.0 * M ** 2 * one_plus_sigma)
    c_Pi = -sigma / (2.0 * M * one_plus_sigma)
    if include_potential:
        c_Psi = -(ell * (ell + 1) - 3.0 * sigma) / (16.0 * M ** 2 * one_plus_sigma)
    else:
        c_Psi = np.zeros_like(sigma)

    return MinGaugeOp(
        sigma=sigma, dsigma=dsigma, M=M, ell=ell,
        r=r, H=H,
        lambda_out=lambda_out, lambda_in=lambda_in,
        mu_plus=mu_plus, mu_minus=mu_minus,
        mu_plus_d=mu_plus_d, mu_minus_d=mu_minus_d,
        inv_dmu=inv_dmu, one_minus_sigma2=one_minus_sigma2, sigma2=sigma2,
        c_Phi=c_Phi, c_Pi=c_Pi, c_Psi=c_Psi,
    )


# ----------------------------------------------------------------------
# Inverse map and RHS.
# State variables are (Psi, U, W) where U = Pi + mu_+ Phi, W = Pi + mu_- Phi.
# Inverse (eq 23):
#     Phi = inv_dmu * (U - W)
#     Pi  = (1 - sigma^2) U + sigma^2 W
# ----------------------------------------------------------------------


def recover_pi_phi(U: np.ndarray, W: np.ndarray, op: MinGaugeOp) -> Tuple[np.ndarray, np.ndarray]:
    Phi = op.inv_dmu * (U - W)
    Pi = op.one_minus_sigma2 * U + op.sigma2 * W
    return Pi, Phi


def rhs_min(
    state: Tuple[np.ndarray, np.ndarray, np.ndarray],
    op: MinGaugeOp,
    d1: Callable[[np.ndarray, float], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right-hand side of the (Psi, U, W) system.

    Following section 9 of the derivation:
        d_tau Psi = Pi
        d_tau U   = -lambda_out * d_sigma U + S_Pi + mu_+'  * Phi * lambda_out
        d_tau W   = -lambda_in  * d_sigma W + S_Pi + mu_-'  * Phi * lambda_in

    Note we form S_Pi (= -c_Phi*Phi - c_Pi*Pi - c_Psi*Psi taken with the SIGN
    already absorbed into c_Phi/c_Pi/c_Psi above) using the recovered Pi, Phi.
    """
    Psi, U, W = state
    Pi, Phi = recover_pi_phi(U, W, op)
    dU_dsigma = d1(U, op.dsigma)
    dW_dsigma = d1(W, op.dsigma)
    S_Pi = op.c_Phi * Phi + op.c_Pi * Pi + op.c_Psi * Psi
    dPsi = Pi
    dU = -op.lambda_out * dU_dsigma + S_Pi + op.mu_plus_d * Phi * op.lambda_out
    dW = -op.lambda_in * dW_dsigma + S_Pi + op.mu_minus_d * Phi * op.lambda_in
    return dPsi, dU, dW


def cfl_dt(op: MinGaugeOp, safety: float = 0.4) -> float:
    """CFL bound for explicit RK4: dt <= 4 M * safety * dsigma (eq 32)."""
    lam_max = float(max(np.max(np.abs(op.lambda_out)), np.max(np.abs(op.lambda_in))))
    if lam_max <= 0.0:
        raise RuntimeError("degenerate characteristic speeds")
    return safety * op.dsigma / lam_max


# ----------------------------------------------------------------------
# Initial data prescription (section 10b).
# Given Psi_0(r), set Pi(0) = 0, Phi(0) = d_sigma Psi(0) via the SAME FD
# stencil as the evolution operator (NOT analytic) so the discrete
# compatibility relation d_tau Phi = d_sigma Pi holds to machine precision
# at t = 0.
# ----------------------------------------------------------------------


def state_from_psi(
    psi0_of_r: Callable[[np.ndarray], np.ndarray],
    op: MinGaugeOp,
    d1: Callable[[np.ndarray, float], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map a callable psi0(r) -> initial (Psi, U, W)."""
    Psi0 = psi0_of_r(op.r)
    Phi0 = d1(Psi0, op.dsigma)
    Pi0 = np.zeros_like(Psi0)
    U0 = Pi0 + op.mu_plus * Phi0
    W0 = Pi0 + op.mu_minus * Phi0
    return Psi0, U0, W0


# ----------------------------------------------------------------------
# Observer index helper.
# Observer location is specified by r/M; sigma_obs = 2 / (r/M).
# ----------------------------------------------------------------------


def observer_index(op: MinGaugeOp, r_over_M: float) -> int:
    """Nearest-grid-point index for an observer at r = r_over_M * M."""
    sigma_obs = 2.0 / r_over_M
    return int(np.argmin(np.abs(op.sigma - sigma_obs)))


def scri_index(op: MinGaugeOp) -> int:
    """Index of the grid point closest to scri (smallest sigma)."""
    return int(np.argmin(op.sigma))

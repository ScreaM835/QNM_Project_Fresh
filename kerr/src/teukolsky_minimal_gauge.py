"""Minimal-gauge hyperboloidal Teukolsky operator on Kerr (s = -2, l = m = 2).

Complex generalisation of `rwz_minimal_gauge.py` to the spinning Teukolsky
master equation. Derived in `notes/kerr_minimal_gauge_derivation.md` and the
closed forms are emitted + validated by
`scripts/derive_first_order_system.py` (exact Gaussian-rational arithmetic).

Compactified coordinate sigma = r_plus / r in [0, 1] (sigma = 0 scri, sigma = 1
horizon). Height H(sigma) = 1 - 2 sigma^2 (spin-INDEPENDENT, same as Phase A).
Three-field characteristic form (Psi, U, W) with bounded coefficients and
characteristic outflow at both endpoints (no boundary data). The state is
COMPLEX: frame dragging enters through the complex source coefficients
(c_Pi, c_Phi, c_Psi); the characteristic structure (speeds, mu_pm, inverse
map) is real and reduces EXACTLY to the validated Phase A minimal gauge at
a = 0 (the a=0 source reduces to Bardeen-Press, not Regge-Wheeler -- the two
are isospectral, so the physical a=0 check is the QNM frequency, B.8).

This module supersedes the `teukolsky_fd.py` stub (left for reference only).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

try:
    from .spheroidal import teukolsky_lambda
except ImportError:  # allow running as a stand-alone script
    from spheroidal import teukolsky_lambda


# ----------------------------------------------------------------------
# Tabulated complex coefficients on a uniform sigma grid.
# All formulas are closed-form analytic (no fit, no solve). They are emitted
# verbatim by scripts/derive_first_order_system.py (--pycode) and re-validated
# against lambdify in scripts/test_teukolsky_minimal_gauge.py.
# Symbol map in the emitted forms: sg = sigma, rp = r_plus, rm = r_minus,
# bt = beta = m a / (r+ - r-), lm = lambda (frozen Teukolsky separation const).
# ----------------------------------------------------------------------


@dataclass
class KerrMinGaugeOp:
    sigma: np.ndarray            # uniform grid on [eps, 1 - eps]
    dsigma: float
    M: float
    ell: int
    m: int
    a_over_M: float
    # Geometry
    r_plus: float
    r_minus: float
    beta: float                  # = m a / (r+ - r-)  (real, corotating phase rate)
    r: np.ndarray                # r = r_plus / sigma
    Delta: np.ndarray            # (r - r+)(r - r-)
    H: np.ndarray                # 1 - 2 sigma^2
    # Characteristic speeds in (sigma, tau)  (REAL)
    lambda_out: np.ndarray       # eq (7a)
    lambda_in: np.ndarray        # eq (7b)
    # Characteristic-variable coefficients  (REAL; mu_+ = lam_in, mu_- = lam_out)
    mu_plus: np.ndarray
    mu_minus: np.ndarray
    mu_plus_d: np.ndarray
    mu_minus_d: np.ndarray
    inv_dmu: np.ndarray          # 1 / (mu_+ - mu_-)
    one_minus_sigma2: np.ndarray # weight on U in Pi recovery (spin-independent)
    sigma2: np.ndarray           # weight on W in Pi recovery (spin-independent)
    # Frozen separation data
    omega_ref: complex
    lam: complex
    # Bounded COMPLEX source coefficients (derivation sec 5; a=0 -> Bardeen-Press)
    c_Phi: np.ndarray
    c_Pi: np.ndarray
    c_Psi: np.ndarray


def build_teukolsky_op(
    N: int,
    a_over_M: float,
    M: float = 1.0,
    ell: int = 2,
    m: int = 2,
    omega_ref: complex | None = None,
    n: int = 0,
    lam: complex | None = None,
    eps_scri: float = 1e-8,
    eps_horizon: float = 1e-8,
    include_potential: bool = True,
) -> KerrMinGaugeOp:
    """Construct the complex minimal-gauge Teukolsky operator on a uniform grid.

    Parameters
    ----------
    a_over_M : dimensionless Kerr spin (0 <= a/M <= 0.95 in scope).
    omega_ref : reference QNM frequency M*omega (complex, M=1 convention) used to
        freeze the spheroidal separation constant. Required unless `lam` is given
        or `include_potential` is False.
    lam : optional explicit Teukolsky separation constant (overrides omega_ref).
    include_potential : if False, set c_Psi -> 0 (V = 0 flat-propagation gate).

    The two endpoints are inset by eps_* to avoid r = r_plus/sigma blowups; use
    the same eps across resolutions for honest convergence. All source
    coefficients are analytically cancelled (horizon-stable), so the insets are
    O(eps) corrections, never 0/0.
    """
    if N < 7:
        raise ValueError("need at least 7 points for 2nd-order central + KO")
    if not (0.0 <= a_over_M < 1.0):
        raise ValueError("a_over_M must be in [0, 1)")

    chi = float(a_over_M)
    root = np.sqrt(1.0 - chi * chi)            # sqrt(1 - (a/M)^2)
    rp = M * (1.0 + root)                       # r_plus
    rm = M * (1.0 - root)                       # r_minus
    bt = m * chi / (2.0 * root) if root > 0.0 else np.inf   # beta = m a/(r+-r-)

    if lam is None:
        if include_potential:
            if omega_ref is None:
                raise ValueError("omega_ref or lam required when include_potential")
            lam = teukolsky_lambda(chi, ell, m, -2, omega_ref)
        else:
            lam = 0.0 + 0.0j
    lm = complex(lam)

    sg = np.linspace(eps_scri, 1.0 - eps_horizon, N)
    dsigma = sg[1] - sg[0]
    r = rp / sg
    Delta = (r - rp) * (r - rm)
    H = 1.0 - 2.0 * sg ** 2
    one_minus_sigma2 = 1.0 - sg ** 2
    sigma2 = sg ** 2

    # ---- characteristic speeds (emitted forms) ----
    lam_out = (-rm * sg ** 2 + rm * sg + rp * sg - rp) / (2 * rm * rp * sg ** 2 + 2 * rp ** 2)
    lam_in = (-rm * sg ** 3 + rp * sg ** 2) / (
        2 * rm * rp * sg ** 3 + 2 * rm * rp * sg ** 2 + 2 * rp ** 2 * sg + 2 * rp ** 2)

    # mu_+ = lam_in, mu_- = lam_out (verified exact for all spin); derivatives emitted
    mu_plus = lam_in.copy()
    mu_minus = lam_out.copy()
    mu_plus_d = (-rm ** 2 * sg ** 4 - rm * rp * sg ** 4 - 2 * rm * rp * sg ** 3
                 - 3 * rm * rp * sg ** 2 + rp ** 2 * sg ** 2 + 2 * rp ** 2 * sg) / (
        2 * rm ** 2 * rp * sg ** 6 + 4 * rm ** 2 * rp * sg ** 5 + 2 * rm ** 2 * rp * sg ** 4
        + 4 * rm * rp ** 2 * sg ** 4 + 8 * rm * rp ** 2 * sg ** 3 + 4 * rm * rp ** 2 * sg ** 2
        + 2 * rp ** 3 * sg ** 2 + 4 * rp ** 3 * sg + 2 * rp ** 3)
    mu_minus_d = (-rm ** 2 * sg ** 2 - rm * rp * sg ** 2 + rm * rp + rp ** 2) / (
        2 * rm ** 2 * rp * sg ** 4 + 4 * rm * rp ** 2 * sg ** 2 + 2 * rp ** 3)
    inv_dmu = (-2 * rm * rp * sg ** 3 - 2 * rm * rp * sg ** 2 - 2 * rp ** 2 * sg - 2 * rp ** 2) / (
        rm * sg - rp)

    # ---- complex source coefficients (emitted forms) ----
    c_Pi = (2j * bt * rm ** 2 * sg ** 4 + 2j * bt * rm ** 2 * sg ** 3 + 1j * bt * rm ** 2 * sg ** 2
            - 2j * bt * rm * rp * sg ** 3 + 2j * bt * rm * rp * sg + 1j * bt * rm * rp
            - 2j * bt * rp ** 2 * sg - 2j * bt * rp ** 2
            + 2 * rm ** 2 * sg ** 3 + 3 * rm ** 2 * sg ** 2 - 2 * rm * rp * sg ** 3
            - 4 * rm * rp * sg ** 2 - rm * rp * sg - 2 * rp ** 2) / (
        2 * rm ** 2 * rp * sg ** 5 + 2 * rm ** 2 * rp * sg ** 4 + 4 * rm * rp ** 2 * sg ** 3
        + 4 * rm * rp ** 2 * sg ** 2 + 2 * rp ** 3 * sg + 2 * rp ** 3)
    c_Phi = (-2j * bt * rm ** 2 * sg ** 4 + 4j * bt * rm * rp * sg ** 3 - 2j * bt * rp ** 2 * sg ** 2
             - 3 * rm ** 2 * sg ** 3 + rm * rp * sg ** 3 + 5 * rm * rp * sg ** 2
             - rp ** 2 * sg ** 2 - 2 * rp ** 2 * sg) / (
        4 * rm ** 2 * rp ** 2 * sg ** 5 + 4 * rm ** 2 * rp ** 2 * sg ** 4 + 8 * rm * rp ** 3 * sg ** 3
        + 8 * rm * rp ** 3 * sg ** 2 + 4 * rp ** 4 * sg + 4 * rp ** 4)
    if include_potential:
        c_Psi = (1j * bt * rm ** 4 * sg ** 3 - 2j * bt * rm ** 4 * sg ** 2 - 2j * bt * rm ** 3 * rp * sg ** 3
                 + 3j * bt * rm ** 3 * rp * sg ** 2 + 4j * bt * rm ** 3 * rp * sg + 1j * bt * rm ** 2 * rp ** 2 * sg ** 3
                 - 10j * bt * rm ** 2 * rp ** 2 * sg - 1j * bt * rm * rp ** 3 * sg ** 2 + 8j * bt * rm * rp ** 3 * sg
                 - 2j * bt * rp ** 4 * sg
                 + lm * rm ** 3 * rp * sg - 2 * lm * rm ** 2 * rp ** 2 * sg - lm * rm ** 2 * rp ** 2
                 + lm * rm * rp ** 3 * sg + 2 * lm * rm * rp ** 3 - lm * rp ** 4
                 + 3 * rm ** 4 * sg ** 2 + 4 * rm ** 3 * rp * sg ** 3 - 3 * rm ** 3 * rp * sg ** 2
                 - 3 * rm ** 3 * rp * sg - 3 * rm ** 2 * rp ** 2 * sg ** 2 + 7 * rm ** 2 * rp ** 2 * sg
                 - rm * rp ** 3 * sg ** 2 - 5 * rm * rp ** 3 * sg + rp ** 4 * sg) / (
            4 * rm ** 4 * rp ** 2 * sg ** 5 + 4 * rm ** 4 * rp ** 2 * sg ** 4 - 8 * rm ** 3 * rp ** 3 * sg ** 5
            - 8 * rm ** 3 * rp ** 3 * sg ** 4 + 8 * rm ** 3 * rp ** 3 * sg ** 3 + 8 * rm ** 3 * rp ** 3 * sg ** 2
            + 4 * rm ** 2 * rp ** 4 * sg ** 5 + 4 * rm ** 2 * rp ** 4 * sg ** 4 - 16 * rm ** 2 * rp ** 4 * sg ** 3
            - 16 * rm ** 2 * rp ** 4 * sg ** 2 + 4 * rm ** 2 * rp ** 4 * sg + 4 * rm ** 2 * rp ** 4
            + 8 * rm * rp ** 5 * sg ** 3 + 8 * rm * rp ** 5 * sg ** 2 - 8 * rm * rp ** 5 * sg - 8 * rm * rp ** 5
            + 4 * rp ** 6 * sg + 4 * rp ** 6)
    else:
        c_Psi = np.zeros_like(sg, dtype=np.complex128)

    return KerrMinGaugeOp(
        sigma=sg, dsigma=dsigma, M=M, ell=ell, m=m, a_over_M=chi,
        r_plus=rp, r_minus=rm, beta=float(bt),
        r=r, Delta=Delta, H=H,
        lambda_out=lam_out, lambda_in=lam_in,
        mu_plus=mu_plus, mu_minus=mu_minus,
        mu_plus_d=mu_plus_d, mu_minus_d=mu_minus_d,
        inv_dmu=inv_dmu, one_minus_sigma2=one_minus_sigma2, sigma2=sigma2,
        omega_ref=(complex(omega_ref) if omega_ref is not None else complex("nan")),
        lam=lm,
        c_Phi=np.asarray(c_Phi, dtype=np.complex128),
        c_Pi=np.asarray(c_Pi, dtype=np.complex128),
        c_Psi=np.asarray(c_Psi, dtype=np.complex128),
    )


# ----------------------------------------------------------------------
# Inverse map and RHS (complex), mirroring rwz_minimal_gauge.py.
# State (Psi, U, W) with U = Pi + mu_+ Phi, W = Pi + mu_- Phi.
#     Phi = inv_dmu * (U - W)
#     Pi  = (1 - sigma^2) U + sigma^2 W
# ----------------------------------------------------------------------


def recover_pi_phi(U: np.ndarray, W: np.ndarray, op: KerrMinGaugeOp
                   ) -> Tuple[np.ndarray, np.ndarray]:
    Phi = op.inv_dmu * (U - W)
    Pi = op.one_minus_sigma2 * U + op.sigma2 * W
    return Pi, Phi


def rhs_teuk(
    state: Tuple[np.ndarray, np.ndarray, np.ndarray],
    op: KerrMinGaugeOp,
    d1: Callable[[np.ndarray, float], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right-hand side of the complex (Psi, U, W) system (derivation sec 8).

        d_tau Psi = Pi
        d_tau U   = -lambda_out d_sigma U + S_Pi + mu_+' Phi lambda_out
        d_tau W   = -lambda_in  d_sigma W + S_Pi + mu_-' Phi lambda_in

    with S_Pi = c_Phi Phi + c_Pi Pi + c_Psi Psi (signs absorbed into the
    coefficients). Structurally identical to rwz_minimal_gauge.rhs_min; the only
    difference is complex coefficients/state.
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


def cfl_dt(op: KerrMinGaugeOp, safety: float = 0.4) -> float:
    """CFL bound for explicit RK4: dt <= safety * dsigma / max|lambda|."""
    lam_max = float(max(np.max(np.abs(op.lambda_out)), np.max(np.abs(op.lambda_in))))
    if lam_max <= 0.0:
        raise RuntimeError("degenerate characteristic speeds")
    return safety * op.dsigma / lam_max


def state_from_psi(
    psi0_of_r: Callable[[np.ndarray], np.ndarray],
    op: KerrMinGaugeOp,
    d1: Callable[[np.ndarray, float], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map a callable psi0(r) -> initial complex (Psi, U, W).

    Pi(0) = 0, Phi(0) = d_sigma Psi(0) via the SAME FD stencil as the evolution
    (not analytic), so the discrete compatibility relation holds at t = 0.
    """
    Psi0 = np.asarray(psi0_of_r(op.r), dtype=np.complex128)
    Phi0 = d1(Psi0, op.dsigma)
    Pi0 = np.zeros_like(Psi0)
    U0 = Pi0 + op.mu_plus * Phi0
    W0 = Pi0 + op.mu_minus * Phi0
    return Psi0, U0, W0


def observer_index(op: KerrMinGaugeOp, r_over_M: float) -> int:
    """Nearest-grid-point index for an observer at r = r_over_M * M.

    sigma = r_plus / r, so sigma_obs = (r_plus / M) / r_over_M.
    """
    sigma_obs = (op.r_plus / op.M) / r_over_M
    return int(np.argmin(np.abs(op.sigma - sigma_obs)))


def scri_index(op: KerrMinGaugeOp) -> int:
    """Index of the grid point closest to scri (smallest sigma)."""
    return int(np.argmin(op.sigma))

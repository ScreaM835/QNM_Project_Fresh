"""Hyperboloidal Chebyshev spectral QNM solver.

Implements the Jaramillo-Macedo-Sheikh 2020 (arXiv:2004.06434) hyperboloidal
formulation: a compactified slicing on which the wave operator becomes a
first-order-in-time generalised eigenvalue problem on a finite interval, with
characteristic boundaries (no PML needed). The QNMs are eigenvalues of a
non-self-adjoint operator and the spectrum includes all overtones, in contrast
to the PML formulation in spectral_qnm.py which is spectrally unstable for
n >= 1.

Equation form (JMS 2020 §II):

    L * u = i * omega * u,    L = i * (1/L1) * [L0  +  L2 * d/dtau]

We linearise to first order in time by introducing u = (phi, psi=d_tau phi),
giving the generalised eigenvalue problem (JMS eq. 2.12):

    A u = i omega B u,   A = [[0, 1], [L0, L2]],   B = diag(1, L1)

where L0, L1, L2 are second-order spatial operators on x in [0, 1].

For Poschl-Teller (PT), the JMS choice gives (in units where V0=1):

    L0 = -d/dx [ (1 - x^2) d/dx ] + V_PT(x)
    L1 = 1 + x^2
    L2 = 2 x d/dx + 1

with V_PT(x) = 1.   The analytic QNM spectrum is

    omega_n = +/- sqrt(V0 - 1/4) - i (n + 1/2),   n = 0, 1, 2, ...

so for V0=1 we expect omega_n = +/- sqrt(3)/2 - i(n+1/2)
                              = +/- 0.8660254037844386 - i(0.5, 1.5, 2.5, ...).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.linalg import eig


# ---------------------------------------------------------------------------
# Chebyshev-Lobatto differentiation matrix on [-1, 1] (Trefethen, SMM, p. 53)
# ---------------------------------------------------------------------------
def cheb_diff(N: int):
    """Return Chebyshev-Lobatto nodes x[0..N] on [-1,1] and (N+1)x(N+1) D matrix."""
    if N == 0:
        return np.array([1.0]), np.array([[0.0]])
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.ones(N + 1)
    c[0] = 2.0
    c[N] = 2.0
    c = c * (-1.0) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))
    D = D - np.diag(D.sum(axis=1))
    return x, D


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class HyperboloidalResult:
    omegas: np.ndarray         # complex QNM frequencies (sorted by Im, descending)
    eigvecs: np.ndarray        # corresponding eigenvectors of (phi, psi)
    x: np.ndarray              # Chebyshev grid on [0, 1]
    N: int


# ---------------------------------------------------------------------------
# Poschl-Teller hyperboloidal solver
# ---------------------------------------------------------------------------
def solve_poschl_teller(N: int = 100,
                        n_return: int = 10) -> HyperboloidalResult:
    """Hyperboloidal Chebyshev solver for the Poschl-Teller potential.

    Following Jaramillo, Macedo & Sheikh 2020 (arXiv:2004.06434), the
    compactified hyperboloidal Bizon-Mach coordinates put the PT wave
    equation in the form (their Eqs. 4.43-4.44, with rescaling V0 -> 1):

        d_tau^2 phi = L1 phi + L2 d_tau phi,   x in [-1, 1]

    with
        L1 = d_x[(1 - x^2) d_x] - 1
        L2 = -(2 x d_x + 1)

    Linearising with psi = d_tau phi gives
        d_tau u = M u,   M = [[0, I], [L1, L2]],   u = (phi, psi),
    so eigenvalue lambda = -i omega (ansatz e^{-i omega tau}) and
    omega = i * lambda.

    Analytic spectrum (V0=1):  omega_n = +/- sqrt(3)/2 - i(n+1/2).
    """
    x, D = cheb_diff(N)
    I = np.eye(N + 1)
    X = np.diag(x)

    # L1 = d_x[(1 - x^2) d_x] - 1
    L1op = D @ np.diag(1.0 - x * x) @ D - I
    # L2 = -(2 x d_x + 1)
    L2op = -(2.0 * X @ D + I)

    n = N + 1
    M = np.zeros((2 * n, 2 * n), dtype=complex)
    M[:n, n:] = I
    M[n:, :n] = L1op
    M[n:, n:] = L2op

    lam, vec = np.linalg.eig(M)
    omega = 1j * lam

    # Discard infinities / numerical garbage
    good = np.isfinite(omega) & (np.abs(omega) < 1e6)
    omega = omega[good]
    vec = vec[:, good]

    # Physical QNMs have Im(omega) < 0 (decaying) AND nonzero real part
    # (filters out a single spurious near-zero mode from the discretisation).
    phys = (omega.imag < -0.1) & (np.abs(omega.real) > 1e-3)
    omega = omega[phys]
    vec = vec[:, phys]
    order = np.lexsort((-np.abs(omega.real), -omega.imag))  # least-damped first
    omega = omega[order]
    vec = vec[:, order]

    return HyperboloidalResult(
        omegas=omega[:n_return],
        eigvecs=vec[:, :n_return],
        x=x,
        N=N,
    )


# ---------------------------------------------------------------------------
# Analytic Poschl-Teller spectrum for validation
# ---------------------------------------------------------------------------
def poschl_teller_analytic(V0: float = 1.0, n_max: int = 5) -> list[complex]:
    """Analytic QNM frequencies of the PT potential.

    omega_n = sqrt(V0 - 1/4) - i (n + 1/2),  n = 0, 1, 2, ...
    (positive-real branch; the negative branch is the complex conjugate of
    -omega_n.)
    """
    re = np.sqrt(V0 - 0.25)
    return [complex(re, -(n + 0.5)) for n in range(n_max + 1)]


# ---------------------------------------------------------------------------
# Schwarzschild Regge-Wheeler / Zerilli hyperboloidal solver
# ---------------------------------------------------------------------------
def solve_schwarzschild(ell: int, N: int = 200, M: float = 1.0,
                        parity: str = 'RW', s: int = 2,
                        n_return: int = 20) -> HyperboloidalResult:
    """Hyperboloidal Chebyshev solver for Schwarzschild RW/Zerilli QNMs.

    Implements JMS 2020 Eqs. (5.7)-(5.11) (arXiv:2004.06434). The
    minimal-gauge compactification uses sigma = 2M/r on sigma in [0, 1],
    with sigma=0 at future null infinity and sigma=1 at the BH horizon.
    Coordinates are rescaled by lambda = 4M, so eigenvalues are 4M*omega.

    Operators (JMS Eq. 5.10-5.11):
        w(sigma) = 2(1+sigma)
        p(sigma) = 2 sigma^2 (1 - sigma)
        gamma(sigma) = 1 - 2 sigma^2,    gamma'(sigma) = -4 sigma
        L1 = (1/w) [ d_s(p d_s) - V_tilde_ell ]
        L2 = (1/w) [ 2 gamma d_s - 4 sigma ]

    Rescaled potentials (JMS Eq. 5.12):
        V_tilde^RW = 2 [ ell(ell+1) + (1 - s^2) sigma ]   (axial, s=2 gravity)
        V_tilde^Z  = 2 [ sigma + (2n/3)(1 + 4n(3+2n)/(2n+3 sigma)^2) ]
                     with n = (ell-1)(ell+2)/2.

    Returns omegas in physical units (i.e. M*omega / M = omega; multiply by M
    externally if you want dimensionless M*omega).
    """
    # Chebyshev-Lobatto on [-1,1] then map to sigma in [0,1] via sigma = (1-y)/2.
    # JMS use sigma=0 at scri+ (radial infinity) and sigma=1 at horizon.
    y, Dy = cheb_diff(N)
    sigma = 0.5 * (1.0 - y)         # y=+1 -> sigma=0,  y=-1 -> sigma=1
    D = -2.0 * Dy                   # d/d sigma = (d sigma / dy)^{-1} * d/dy = -2 d/dy
    Iop = np.eye(N + 1)

    w = 2.0 * (1.0 + sigma)
    p = 2.0 * sigma * sigma * (1.0 - sigma)
    gamma = 1.0 - 2.0 * sigma * sigma

    if parity.upper() == 'RW':
        V_tilde = 2.0 * (ell * (ell + 1) + (1 - s * s) * sigma)
    elif parity.upper() in ('Z', 'ZERILLI'):
        nZ = 0.5 * (ell - 1) * (ell + 2)
        V_tilde = 2.0 * (sigma + (2.0 * nZ / 3.0) *
                         (1.0 + 4.0 * nZ * (3.0 + 2.0 * nZ) /
                          (2.0 * nZ + 3.0 * sigma) ** 2))
    else:
        raise ValueError(f"parity must be 'RW' or 'Z', got {parity!r}")

    invw = np.diag(1.0 / w)
    Pmat = np.diag(p)
    Gmat = np.diag(gamma)
    Vmat = np.diag(V_tilde)

    # L1 = (1/w) [ d_s(p d_s) - V_tilde ]
    L1op = invw @ (D @ Pmat @ D - Vmat)
    # L2 = (1/w) [ 2 gamma d_s + (d_s gamma) ] = (1/w) [ 2 gamma d_s - 4 sigma ]
    L2op = invw @ (2.0 * Gmat @ D + np.diag(-4.0 * sigma))

    n = N + 1
    Mmat = np.zeros((2 * n, 2 * n), dtype=complex)
    Mmat[:n, n:] = Iop
    Mmat[n:, :n] = L1op
    Mmat[n:, n:] = L2op

    lam, vec = np.linalg.eig(Mmat)
    # In JMS rescaled units, lam = -i * (4M) * omega, so omega = i * lam / (4M).
    omega = 1j * lam / (4.0 * M)

    good = np.isfinite(omega) & (np.abs(omega) < 1e6)
    omega = omega[good]
    vec = vec[:, good]

    # Physical QNMs: decaying (Im < 0). JMS document a spurious branch-cut
    # signature concentrated on / near the imaginary axis (small |Re|), which we
    # filter out by requiring |Re(omega)| > re_floor.
    re_floor = 0.05 / (4.0 * M)
    phys = (omega.imag < -1e-3) & (np.abs(omega.real) > re_floor)
    omega = omega[phys]
    vec = vec[:, phys]
    order = np.lexsort((-np.abs(omega.real), -omega.imag))  # least-damped first
    omega = omega[order]
    vec = vec[:, order]

    return HyperboloidalResult(
        omegas=omega[:n_return],
        eigvecs=vec[:, :n_return],
        x=sigma,
        N=N,
    )


def solve_schwarzschild_convergent(ell: int, N: int = 120, dN: int = 40,
                                   M: float = 1.0, parity: str = 'RW',
                                   s: int = 2, tol: float = 1e-4,
                                   n_return: int = 8) -> HyperboloidalResult:
    """Schwarzschild QNMs filtered by convergence across two resolutions.

    Solves at N and N+dN; keeps only eigenvalues whose nearest match in the
    other grid is within `tol`. This removes the dense numerical branch-cut
    cloud (JMS Fig. 13), keeping only the genuine QNMs.
    """
    a = solve_schwarzschild(ell=ell, N=N, M=M, parity=parity, s=s,
                            n_return=10 * n_return)
    b = solve_schwarzschild(ell=ell, N=N + dN, M=M, parity=parity, s=s,
                            n_return=10 * n_return)
    keep_omegas = []
    keep_vecs = []
    for j, wa in enumerate(a.omegas):
        d = np.abs(b.omegas - wa)
        if d.min() < tol:
            keep_omegas.append(wa)
            keep_vecs.append(a.eigvecs[:, j])
    if not keep_omegas:
        return HyperboloidalResult(
            omegas=np.array([], dtype=complex),
            eigvecs=np.zeros((2 * (N + 1), 0), dtype=complex),
            x=a.x, N=N,
        )
    keep_omegas = np.array(keep_omegas)
    keep_vecs = np.column_stack(keep_vecs)
    order = np.lexsort((-np.abs(keep_omegas.real), -keep_omegas.imag))
    keep_omegas = keep_omegas[order]
    keep_vecs = keep_vecs[:, order]
    return HyperboloidalResult(
        omegas=keep_omegas[:n_return],
        eigvecs=keep_vecs[:, :n_return],
        x=a.x,
        N=N,
    )

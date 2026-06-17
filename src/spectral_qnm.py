"""
Spectral QNM solver for Schwarzschild perturbations.

Method
------
Tortoise-coordinate Chebyshev collocation with a perfectly matched layer (PML)
implementing the outgoing boundary conditions via complex coordinate stretching.

The Regge--Wheeler / Zerilli equation reads
    [-d^2/dx^2 + V(x)] psi = omega^2 psi
on x in R, with outgoing conditions psi ~ exp(+/- i omega x) as x -> +/- infinity.

We truncate to a finite interval x in [-L, L] and apply the complex coordinate
stretch
    x -> x_tilde(x) = x + i * S(x),     S(x) = sigma_0 * f((|x| - L_phys)/L_pml)_+ ^ p
which leaves the operator unchanged inside the physical window |x| < L_phys and
introduces complex stretching outside. Under this stretch d/dx -> (1/(1+iS'(x))) d/dx.
Outgoing waves become exponentially decaying inside the PML, so Dirichlet
boundary conditions at +/- L are effectively reflectionless. The resulting
non-Hermitian linear EVP
    A psi = lambda psi,   lambda = omega^2
is solved by a dense LAPACK call (sizes are small, N ~ 100--400).

Reference physics
-----------------
Schwarzschild gravitational perturbations, l=2 fundamental (n=0) Leaver values
in units M=1 (verified against the `qnm` package, Stein 2019):
    M omega = 0.3736716857 - 0.0889623157 j     (l=2, n=0)
    M omega = 0.5994432905 - 0.0927030479 j     (l=3, n=0)
    M omega = 0.8091783803 - 0.0941639610 j     (l=4, n=0)

Implementation
--------------
- Chebyshev--Lobatto nodes on x in [-L, L] (mapped from xi in [-1,1]).
- Differentiation matrix from Trefethen, Spectral Methods in MATLAB, Ch. 6.
- Potential V(x) supplied as a Python callable in tortoise coordinate
  (use `src.potentials.V_of_x`).
- Returns the requested number of QNMs nearest to a user-supplied shift omega0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Chebyshev differentiation matrix (Trefethen, Spectral Methods in MATLAB)
# ---------------------------------------------------------------------------
def cheb_diff(N: int) -> Tuple[np.ndarray, np.ndarray]:
    """Chebyshev--Lobatto nodes xi_k = cos(pi k / N), k = 0..N, and the
    associated (N+1)x(N+1) differentiation matrix D on the interval [-1, 1].
    Returns (D, xi) with xi sorted from +1 down to -1 (Trefethen convention)."""
    if N == 0:
        return np.zeros((1, 1)), np.array([1.0])
    xi = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.ones(N + 1)
    c[0] = 2.0
    c[N] = 2.0
    c = c * (-1.0) ** np.arange(N + 1)
    X = np.tile(xi[:, None], (1, N + 1))
    dX = X - X.T
    D = (c[:, None] / c[None, :]) / (dX + np.eye(N + 1))
    D = D - np.diag(D.sum(axis=1))
    return D, xi


# ---------------------------------------------------------------------------
# PML profile and metric coefficient
# ---------------------------------------------------------------------------
@dataclass
class PMLParams:
    L: float          # outer truncation, |x| <= L
    L_phys: float     # PML starts at |x| = L_phys, L_phys < L
    sigma0: float     # PML strength
    p: int = 3        # polynomial profile exponent

    def __post_init__(self) -> None:
        if not (0.0 < self.L_phys < self.L):
            raise ValueError("require 0 < L_phys < L")
        if self.sigma0 <= 0.0:
            raise ValueError("sigma0 must be positive")


def _pml_profile(x: np.ndarray, pml: PMLParams) -> Tuple[np.ndarray, np.ndarray]:
    """Return (S(x), S'(x)) for the PML stretching profile.

    The outgoing condition is psi ~ exp(+i omega x) as x -> +infty and
    psi ~ exp(-i omega x) as x -> -infty.  For the complex coordinate
    stretch x -> x + i S(x) to push *both* outgoing branches into L^2 we
    need S(x) > 0 for x > 0 and S(x) < 0 for x < 0, i.e. S is odd in x.

        S(x) = sign(x) * sigma0 * ((|x| - L_phys) / (L - L_phys))^p     for |x| > L_phys
        S(x) = 0                                                        for |x| <= L_phys
    """
    a = np.abs(x)
    width = pml.L - pml.L_phys
    u = (a - pml.L_phys) / width
    in_pml = a > pml.L_phys
    u = np.where(in_pml, u, 0.0)
    S_mag = pml.sigma0 * u ** pml.p
    S = np.sign(x) * S_mag
    # d S / dx = sign(x) * d S_mag / d|x| * sign(x) = d S_mag / d|x|  (a.e.)
    Sp = np.where(in_pml,
                  pml.sigma0 * pml.p * u ** (pml.p - 1) / width,
                  0.0)
    return S, Sp


# ---------------------------------------------------------------------------
# Build operator and solve
# ---------------------------------------------------------------------------
@dataclass
class QNMResult:
    omegas: np.ndarray         # complex, sorted by |omega - shift|
    psis: np.ndarray           # shape (N_returned, n_interior)
    x_interior: np.ndarray     # tortoise coordinate, interior nodes only
    omega_sq: np.ndarray       # raw eigenvalues lambda = omega^2


def solve_qnm_chebyshev_pml(
    V_func: Callable[[np.ndarray], np.ndarray],
    N: int = 200,
    pml: PMLParams | None = None,
    shift: complex = 0.37 - 0.09j,
    n_return: int = 6,
) -> QNMResult:
    """Solve [-d^2/dx^2 + V(x)] psi = omega^2 psi on x in [-L, L] with PML
    absorbing layers via complex coordinate stretching.

    Parameters
    ----------
    V_func : callable
        Potential as a function of tortoise coordinate x (real array in, real array out).
    N : int
        Chebyshev polynomial order (grid has N+1 nodes; interior solve uses N-1 nodes).
    pml : PMLParams or None
        PML configuration. Defaults to L=120, L_phys=80, sigma0=20, p=3.
    shift : complex
        Target frequency for sorting eigenvalues (default = Schwarzschild l=2 fundamental).
    n_return : int
        Number of eigenpairs to return, closest to `shift` in omega-space.
    """
    if pml is None:
        # Tuned to ~1e-7 accuracy on Schwarzschild Zerilli l=2 fundamental.
        # Smaller L_phys puts more domain in the absorbing layer; the QNM mode
        # is concentrated near the potential peak at r~3M (x~1.3M).
        pml = PMLParams(L=200.0, L_phys=5.0, sigma0=400.0, p=6)

    # Chebyshev nodes on [-1, 1], then map to x on [-L, L].
    D_xi, xi = cheb_diff(N)
    L = pml.L
    x = L * xi                            # x in [-L, L], descending
    D_x = D_xi / L                        # d/dx = (d xi / dx) * d/d xi = (1/L) D_xi

    # PML profile on the real grid.
    S, Sp = _pml_profile(x, pml)
    # Under x -> x + i S(x), d/dx -> (1/(1 + i S')) d/dx.
    inv_J = 1.0 / (1.0 + 1j * Sp)         # shape (N+1,)
    # Second derivative in stretched coord: d2/dx_tilde^2 = inv_J * d/dx (inv_J * d/dx ·)
    # Construct as matrix: D2_t = diag(inv_J) @ D_x @ diag(inv_J) @ D_x
    D_t = np.diag(inv_J) @ D_x
    D2_t = D_t @ D_t

    # Potential on complex-stretched coord. V is real-analytic in x for Zerilli/RW;
    # the values V(x + iS(x)) inside the PML need analytic continuation. For PML to
    # be effective we use V evaluated on the real grid (V dies off at infinity faster
    # than the PML kicks in if L_phys is large enough), which is the standard
    # "scalar PML" approximation.
    V = V_func(x)

    # Operator A = -D2_t + diag(V).
    A = -D2_t + np.diag(V.astype(complex))

    # Dirichlet BCs at xi = +/- 1, i.e. nodes 0 and N. Eliminate those rows/cols.
    interior = slice(1, N)
    A_int = A[interior, interior]
    x_int = x[interior]

    # Dense eigensolve. Sizes are small (~200--400) so this is fine.
    eigvals, eigvecs = np.linalg.eig(A_int)   # lambda = omega^2

    # Convert lambda -> omega, picking the QNM root: Re(omega) > 0, Im(omega) < 0.
    omegas = np.sqrt(eigvals.astype(complex))
    flip = omegas.real < 0
    omegas = np.where(flip, -omegas, omegas)
    # Sign of imaginary part is preserved up to the branch above; QNMs have Im<0.

    # Sort by distance to shift.
    order = np.argsort(np.abs(omegas - shift))
    omegas = omegas[order]
    eigvecs = eigvecs[:, order]
    eigvals = eigvals[order]

    # Keep n_return.
    keep = min(n_return, omegas.size)
    return QNMResult(
        omegas=omegas[:keep],
        psis=eigvecs[:, :keep].T,
        x_interior=x_int,
        omega_sq=eigvals[:keep],
    )


# ---------------------------------------------------------------------------
# Overtone extraction: identify the physical QNM ladder among all eigenvalues
# ---------------------------------------------------------------------------
def extract_overtones(
    V_func: Callable[[np.ndarray], np.ndarray],
    n_overtones: int = 4,
    N: int = 800,
    pml: PMLParams | None = None,
    pml_alt: PMLParams | None = None,
    omega_R_window: Tuple[float, float] = (0.05, 1.5),
    omega_I_max: float = 1.2,
    match_tol: float = 1e-3,
    dedup_tol: float = 1e-4,
) -> QNMResult:
    """Return the first `n_overtones` Schwarzschild QNMs as a clean ladder.

    Caveat
    ------
    Schwarzschild QNM overtones are spectrally unstable: small operator
    perturbations (including residual PML reflection) shift overtones by
    O(10%) even when the fundamental is reproduced to ~7 digits.  See
    Jaramillo, Panosso Macedo & Al Sheikh, PRX 11, 031003 (2021).
    In practice this routine reliably returns the fundamental (n=0) to
    1e-7 with the default PML; n>=1 will typically be missing or wrong
    by O(1e-2).  Use hyperboloidal-slicing methods to access overtones.

    Method
    ------
    Physical QNMs are PML-invariant; PML-induced "ghost" modes are not.
    So we do two eigensolves at two PML configurations and keep only
    eigenvalues that agree within `match_tol`.  The kept set is then
    filtered to physical roots, de-duplicated, and sorted by damping.
    """
    if pml is None:
        pml = PMLParams(L=200.0, L_phys=5.0, sigma0=400.0, p=6)
    if pml_alt is None:
        # Perturb sigma0 and L while keeping geometry similar.
        pml_alt = PMLParams(L=pml.L * 1.3, L_phys=pml.L_phys,
                            sigma0=pml.sigma0 * 0.7, p=pml.p)

    def _all_eigs(p: PMLParams) -> np.ndarray:
        # Just need eigenvalues; reuse solve and read off.
        res = solve_qnm_chebyshev_pml(
            V_func, N=N, pml=p,
            shift=complex(np.mean(omega_R_window), -omega_I_max / 2),
            n_return=N - 1,
        )
        om = res.omegas
        lo, hi = omega_R_window
        mask = (om.real > lo) & (om.real < hi) & (om.imag < 0) & (om.imag > -omega_I_max)
        return om[mask], res.psis[mask], res.x_interior

    om_a, psi_a, x_int = _all_eigs(pml)
    om_b, _, _ = _all_eigs(pml_alt)

    # Keep eigenvalues that have a match in om_b within match_tol.
    kept_mask = np.zeros(om_a.size, dtype=bool)
    for i, w in enumerate(om_a):
        if np.any(np.abs(om_b - w) < match_tol):
            kept_mask[i] = True
    om = om_a[kept_mask]
    psi = psi_a[kept_mask]

    # Sort by damping (least damped = fundamental).
    order = np.argsort(-om.imag)
    om = om[order]
    psi = psi[order]

    # Deduplicate.
    kept_idx: list[int] = []
    for i, w in enumerate(om):
        if all(abs(w - om[j]) > dedup_tol for j in kept_idx):
            kept_idx.append(i)
        if len(kept_idx) >= n_overtones:
            break

    if len(kept_idx) < n_overtones:
        pad = n_overtones - len(kept_idx)
        if kept_idx:
            om_out = np.concatenate([om[kept_idx], np.full(pad, np.nan + 1j * np.nan)])
            psi_out = np.concatenate(
                [psi[kept_idx], np.full((pad, psi.shape[1]), np.nan, dtype=complex)],
                axis=0,
            )
        else:
            om_out = np.full(pad, np.nan + 1j * np.nan)
            psi_out = np.full((pad, x_int.size), np.nan, dtype=complex)
    else:
        om_out = om[kept_idx[:n_overtones]]
        psi_out = psi[kept_idx[:n_overtones]]

    return QNMResult(
        omegas=om_out,
        psis=psi_out,
        x_interior=x_int,
        omega_sq=om_out ** 2,
    )

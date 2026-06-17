from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, Tuple

import numpy as np

from .potentials import V_of_x
from .initial_data import gaussian_phi, gaussian_phi_t


def _one_sided_dx_left(u: np.ndarray, dx: float) -> float:
    # 2nd-order one-sided derivative at i=0
    return (-3.0 * u[0] + 4.0 * u[1] - 1.0 * u[2]) / (2.0 * dx)


def _one_sided_dx_right(u: np.ndarray, dx: float) -> float:
    # 2nd-order one-sided derivative at i=-1
    return (3.0 * u[-1] - 4.0 * u[-2] + 1.0 * u[-3]) / (2.0 * dx)


def _second_derivative(u: np.ndarray, dx: float) -> np.ndarray:
    """
    2nd-order finite-difference approximation to u_xx on a uniform grid.
    Uses one-sided 2nd-order formulas on the boundaries.
    """
    N = u.size
    uxx = np.empty_like(u)
    # interior
    uxx[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (dx**2)
    # boundaries (2nd-order one-sided second derivative)
    uxx[0] = (2.0 * u[0] - 5.0 * u[1] + 4.0 * u[2] - 1.0 * u[3]) / (dx**2)
    uxx[-1] = (2.0 * u[-1] - 5.0 * u[-2] + 4.0 * u[-3] - 1.0 * u[-4]) / (dx**2)
    return uxx


def _apply_radiative_bc(u: np.ndarray, v: np.ndarray, dx: float) -> None:
    """
    Radiative (Sommerfeld) boundary conditions, consistent with the target paper:

      left (x -> -∞):  (∂t - ∂x)u = 0  =>  v = u_x
      right (x -> +∞): (∂t + ∂x)u = 0  =>  v = -u_x
    """
    ux_l = _one_sided_dx_left(u, dx)
    ux_r = _one_sided_dx_right(u, dx)
    v[0] = ux_l
    v[-1] = -ux_r


# ---------------------------------------------------------------------------
# Optional high-order / dispersion-relation-preserving spatial schemes.
#
# These were validated in isolation (scripts/exploration/coarse_stencil_*.py):
# on the canonical coarse grid (dx=0.4) they cut the coarse-prior L2 error by
# ~100x (4th-order central) to ~300x (DRP7) versus the default 2nd-order scheme,
# with the gain holding across the full hybrid parameter sweep and a >=2.3x CFL
# margin at the production timestep. They exist to make the CHEAP coarse FD prior
# more accurate per grid point (high-d scaling) without adding points.
#
# The default scheme ("central2", bc_order=2) reproduces the original solver
# exactly; the helpers below are only used when a different scheme is requested.
# ---------------------------------------------------------------------------


def _fornberg_weights(z: float, nodes: Tuple[float, ...], m: int) -> np.ndarray:
    """Fornberg (1988) finite-difference weights for the m-th derivative at z on
    arbitrary nodes. Returns the weight vector for the m-th derivative."""
    x = np.asarray(nodes, dtype=float)
    n = x.size
    c = np.zeros((n, m + 1))
    c1 = 1.0
    c4 = x[0] - z
    c[0, 0] = 1.0
    for i in range(1, n):
        mn = min(i, m)
        c2 = 1.0
        c5 = c4
        c4 = x[i] - z
        for j in range(i):
            c3 = x[i] - x[j]
            c2 *= c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    c[i, k] = c1 * (k * c[i - 1, k - 1] - c5 * c[i - 1, k]) / c2
                c[i, 0] = -c1 * c5 * c[i - 1, 0] / c2
            for k in range(mn, 0, -1):
                c[j, k] = (c4 * c[j, k] - k * c[j, k - 1]) / c3
            c[j, 0] = c4 * c[j, 0] / c3
        c1 = c2
    return c[:, m]


@lru_cache(maxsize=8)
def _drp7_half_coeffs(eta: float = 1.5) -> Tuple[float, float, float, float]:
    """Symmetric 7-point dispersion-relation-preserving (Tam & Webb) second
    derivative. The two 4th-order constraints leave one free parameter, fixed by
    minimising the band-integrated modified-wavenumber error over w in [0, eta].
    Returns half-coefficients [a0(centre), a1, a2, a3]; a_{-j}=a_j."""
    from scipy.optimize import minimize_scalar

    w = np.linspace(0.0, eta, 2000)
    w2 = w**2

    def coeffs(s: float) -> Tuple[float, float, float, float]:
        a3 = s
        a2 = -1.0 / 12.0 - 6.0 * s
        a1 = 4.0 / 3.0 + 15.0 * s
        a0 = -2.0 * (a1 + a2 + a3)
        return a0, a1, a2, a3

    def band_err(s: float) -> float:
        _, a1, a2, a3 = coeffs(s)
        f = 2.0 * (a1 * (1 - np.cos(w)) + a2 * (1 - np.cos(2 * w))
                   + a3 * (1 - np.cos(3 * w)))
        return float(np.trapezoid((f - w2) ** 2, w))

    res = minimize_scalar(band_err, bounds=(-0.05, 0.05), method="bounded")
    return coeffs(res.x)


def _interior_half_coeffs(scheme: str) -> np.ndarray:
    """Symmetric interior half-coefficients [c0(centre), c1, ...] for u'' * dx^2."""
    if scheme == "central2":
        return np.array([-2.0, 1.0])
    if scheme == "central4":
        return np.array([-30.0, 16.0, -1.0]) / 12.0
    if scheme == "central6":
        return np.array([-490.0, 270.0, -27.0, 2.0]) / 180.0
    if scheme == "drp7":
        return np.array(_drp7_half_coeffs())
    raise ValueError(f"unknown fd scheme {scheme!r} "
                     "(expected central2/central4/central6/drp7)")


def _build_second_derivative(scheme: str, bc_order: int) -> Callable[[np.ndarray, float], np.ndarray]:
    """Return a u_xx operator: central interior stencil of `scheme`, Fornberg
    one-sided closure of accuracy `bc_order` on the hw boundary rows at each end."""
    half = _interior_half_coeffs(scheme)
    hw = half.size - 1
    # Boundary weights depend only on dx; cache per dx (constant within a solve).
    _wcache: Dict[float, Tuple[int, np.ndarray, np.ndarray]] = {}

    def _bnd_weights(dx: float, N: int) -> Tuple[int, np.ndarray, np.ndarray]:
        cached = _wcache.get(dx)
        if cached is not None:
            return cached
        nw = min(max(bc_order + 2, hw), N)
        nodes = tuple((dx * np.arange(nw)).tolist())
        # rows: left boundary point i uses nodes 0..nw-1; right is the mirror.
        wl = np.stack([_fornberg_weights(dx * i, nodes, 2) for i in range(hw)])
        wr = np.stack([_fornberg_weights(dx * (nw - 1 - i), nodes, 2) for i in range(hw)])
        out = (nw, wl, wr)
        _wcache[dx] = out
        return out

    def op(u: np.ndarray, dx: float) -> np.ndarray:
        N = u.size
        uxx = np.empty_like(u)
        acc = half[0] * u[hw:N - hw]
        for k in range(1, hw + 1):
            acc = acc + half[k] * (u[hw + k:N - hw + k] + u[hw - k:N - hw - k])
        uxx[hw:N - hw] = acc / dx**2
        nw, wl, wr = _bnd_weights(dx, N)
        for i in range(hw):
            uxx[i] = wl[i] @ u[:nw]
            uxx[N - 1 - i] = wr[i] @ u[N - nw:N]
        return uxx

    return op


def _build_radiative_bc(bc_order: int) -> Callable[[np.ndarray, np.ndarray, float], None]:
    """Radiative (Sommerfeld) BC using a one-sided first derivative of accuracy
    `bc_order`. bc_order=2 reproduces the default _apply_radiative_bc exactly."""
    if bc_order == 2:
        return _apply_radiative_bc

    _wcache: Dict[float, Tuple[int, np.ndarray, np.ndarray]] = {}

    def _bc_weights(dx: float) -> Tuple[int, np.ndarray, np.ndarray]:
        cached = _wcache.get(dx)
        if cached is not None:
            return cached
        n = bc_order + 1
        nodes = tuple((dx * np.arange(n)).tolist())
        out = (n, _fornberg_weights(0.0, nodes, 1), _fornberg_weights(dx * (n - 1), nodes, 1))
        _wcache[dx] = out
        return out

    def apply(u: np.ndarray, v: np.ndarray, dx: float) -> None:
        n, wl, wr = _bc_weights(dx)
        v[0] = wl @ u[:n]
        v[-1] = -(wr @ u[-n:])

    return apply


def solve_fd(config: Dict) -> Dict[str, np.ndarray]:
    """
    Solve the 1+1 master equation with a method-of-lines FD scheme and RK4 time integration.

    PDE convention used here is the *standard* stable form:
        u_tt - u_xx + V(x) u = 0  =>  u_tt = u_xx - V u.

    Returns a dict containing x, t, phi[t_index, x_index], and V(x).
    """
    M = float(config["physics"]["M"])
    l = int(config["physics"]["l"])
    potential = config["physics"]["potential"]

    xmin = float(config["domain"]["xmin"])
    xmax = float(config["domain"]["xmax"])
    tmin = float(config["domain"]["tmin"])
    tmax = float(config["domain"]["tmax"])

    dx = float(config["fd"]["dx"])
    dt = float(config["fd"]["dt"])

    # Spatial scheme. Default reproduces the original 2nd-order solver exactly;
    # "central4"/"central6"/"drp7" give a higher-resolution coarse prior (see
    # the module docstring above and scripts/exploration/coarse_stencil_*.py).
    # NB: fd.scheme (e.g. "rk4_mol") is a pre-existing time-integrator label and
    # is intentionally NOT used here; the spatial scheme is fd.space_scheme.
    space_scheme = str(config["fd"].get("space_scheme", "central2")).lower()
    bc_order = int(config["fd"].get("bc_order", 2))

    if space_scheme == "central2" and bc_order == 2:
        d2 = _second_derivative
        apply_bc = _apply_radiative_bc
    else:
        d2 = _build_second_derivative(space_scheme, bc_order)
        apply_bc = _build_radiative_bc(bc_order)

    # derive grid sizes so dx and dt are honored exactly over the closed intervals
    Nx = int(round((xmax - xmin) / dx)) + 1
    Nt = int(round((tmax - tmin) / dt))

    x = xmin + dx * np.arange(Nx)
    t = tmin + dt * np.arange(Nt + 1)

    Vx = V_of_x(x, M=M, l=l, potential=potential)

    A = float(config["initial_data"]["A"])
    x0 = float(config["initial_data"]["x0"])
    sigma = float(config["initial_data"]["sigma"])
    profile = config["initial_data"]["velocity_profile"]

    u = gaussian_phi(x, A=A, x0=x0, sigma=sigma)
    v = gaussian_phi_t(x, A=A, x0=x0, sigma=sigma, profile=profile)

    # enforce BC at t=0
    apply_bc(u, v, dx)

    phi = np.zeros((Nt + 1, Nx), dtype=float)
    phi[0] = u.copy()

    def rhs(u_: np.ndarray, v_: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # u_t = v
        du = v_.copy()
        # v_t = u_xx - V u
        uxx = d2(u_, dx)
        dv = uxx - Vx * u_
        return du, dv

    for n in range(Nt):
        # RK4
        k1u, k1v = rhs(u, v)

        u2 = u + 0.5 * dt * k1u
        v2 = v + 0.5 * dt * k1v
        apply_bc(u2, v2, dx)
        k2u, k2v = rhs(u2, v2)

        u3 = u + 0.5 * dt * k2u
        v3 = v + 0.5 * dt * k2v
        apply_bc(u3, v3, dx)
        k3u, k3v = rhs(u3, v3)

        u4 = u + dt * k3u
        v4 = v + dt * k3v
        apply_bc(u4, v4, dx)
        k4u, k4v = rhs(u4, v4)

        u = u + (dt / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
        v = v + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)

        apply_bc(u, v, dx)
        phi[n + 1] = u.copy()

    return {"x": x, "t": t, "phi": phi, "V": Vx, "dx": dx, "dt": dt}

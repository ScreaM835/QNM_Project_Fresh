"""Isolation test: does a higher-order / compact spatial stencil reduce the
coarse-FD dispersion error that shows up as the characteristic stripes in the
hybrid baseline heatmap (paper Fig. 42)?

Context
-------
The hybrid surrogate uses a CHEAP coarse FD solve as a prior that an FNO then
corrects. The coarse solve is method-of-lines: 2nd-order central second
derivative (u_xx = [1,-2,1]/dx^2), RK4 in time, Sommerfeld BCs, for
    u_tt = u_xx - V(x) u .
Its error is dominated by DISPERSION: the 2nd-order Laplacian propagates short
wavelengths at the wrong phase speed, with per-wavelength error ~ (k dx)^2 that
accumulates with propagation distance. That is exactly the coherent
along-characteristic striping seen in Fig. 42.

This script tests, IN ISOLATION (production code untouched), whether spending a
few more flops per point on the SAME coarse grid -- i.e. raising resolving
power per degree of freedom -- shrinks that error. We compare on the canonical
Schwarzschild/Zerilli setup, same coarse grid (k=2, dx=0.4, dt=0.2):

    S2   2nd-order central        (= production baseline)
    S4   4th-order central
    S6   6th-order central
    SC4  compact 4th-order Pade   (Lele alpha=1/10, near-spectral resolution)

against a converged reference (6th-order on a 4x-finer grid). Only the spatial
second-derivative operator changes; RK4, BCs, initial data and potential are
identical to src/fd_solver.py.

Login-node safe: single short test (<10 s, tiny memory). Pure analysis, no
training, no production writes.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Callable, Dict, Tuple

import numpy as np
from scipy.linalg import solve_banded

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", ".."))
sys.path.insert(0, _REPO)

from src.potentials import V_of_x
from src.initial_data import gaussian_phi, gaussian_phi_t

# ---------------------------------------------------------------------------
# Canonical config (matches configs/hybrid_sw_dataset.yaml)
# ---------------------------------------------------------------------------
M = 1.0
ELL = 2
POTENTIAL = "zerilli"
XMIN, XMAX = -50.0, 150.0
TMIN, TMAX = 0.0, 50.0
ID_A, ID_X0, ID_SIGMA = 1.0, 4.0, 5.0
ID_VPROFILE = "outgoing"

# Grids. Truth is 4x finer than the coarse grid in BOTH x and t, so the coarse
# grid points are an exact subset of the truth grid points (no interpolation
# needed to compare).
DX_COARSE, DT_COARSE = 0.4, 0.2     # hybrid k=2 coarse grid
DX_FINE, DT_FINE = 0.2, 0.1         # hybrid "fine" reference grid (2nd order)
DX_TRUTH, DT_TRUTH = 0.1, 0.05      # converged reference (6th order)

OUTDIR = os.path.abspath(os.path.join(_REPO, "outputs", "exploration", "coarse_stencil"))


# ---------------------------------------------------------------------------
# Spatial second-derivative operators on a uniform grid.
# Interior uses the central stencil; near-boundary points drop to the highest
# central order that fits; the two end points use a 2nd-order one-sided closure
# (identical convention to src/fd_solver.py at the very edges).
# ---------------------------------------------------------------------------

def _edge_one_sided(u: np.ndarray, dx: float, uxx: np.ndarray) -> None:
    uxx[0] = (2.0 * u[0] - 5.0 * u[1] + 4.0 * u[2] - 1.0 * u[3]) / dx**2
    uxx[-1] = (2.0 * u[-1] - 5.0 * u[-2] + 4.0 * u[-3] - 1.0 * u[-4]) / dx**2


def fornberg_weights(z: float, x: np.ndarray, m: int) -> np.ndarray:
    """Fornberg (1988) finite-difference weights for the m-th derivative at point
    z, using arbitrary nodes x. Returns the length-len(x) weight vector for the
    m-th derivative (exact, no hand-transcribed coefficient tables)."""
    x = np.asarray(x, dtype=float)
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


def drp7_coeffs(eta: float) -> Tuple[float, float, float, float]:
    """Derive the symmetric 7-point DRP second-derivative half-coefficients
    [a0, a1, a2, a3] (a_{-j}=a_j) by spending the spare freedom of the 4th-order
    family on minimising the band-integrated modified-wavenumber error."""
    from scipy.optimize import minimize_scalar

    w = np.linspace(0.0, eta, 2000)
    w2 = w**2

    def coeffs(s: float):
        a3 = s
        a2 = -1.0 / 12.0 - 6.0 * s
        a1 = 4.0 / 3.0 + 15.0 * s
        a0 = -2.0 * (a1 + a2 + a3)
        return a0, a1, a2, a3

    def band_err(s: float) -> float:
        _, a1, a2, a3 = coeffs(s)
        f = 2.0 * (a1 * (1 - np.cos(w)) + a2 * (1 - np.cos(2 * w)) + a3 * (1 - np.cos(3 * w)))
        return float(np.trapezoid((f - w2) ** 2, w))

    res = minimize_scalar(band_err, bounds=(-0.05, 0.05), method="bounded")
    return coeffs(res.x)


# Symmetric interior half-coefficients [c0(center), c1, ..., c_hw] for u'' dx^2.
def _half_coeffs(name: str, eta: float = 1.5) -> np.ndarray:
    if name == "S2_central":
        return np.array([-2.0, 1.0])
    if name == "S4_central":
        return np.array([-30.0, 16.0, -1.0]) / 12.0
    if name == "S6_central":
        return np.array([-490.0, 270.0, -27.0, 2.0]) / 180.0
    if name == "DRP7":
        return np.array(drp7_coeffs(eta))
    raise ValueError(f"no half-coeffs for {name}")


def _fill_d2_boundaries(u: np.ndarray, dx: float, uxx: np.ndarray,
                        hw: int, order: int) -> None:
    """Fill the hw boundary rows at each end with one-sided Fornberg
    second-derivative weights of formal accuracy `order` (a one-sided 2nd
    derivative needs order+2 nodes). Translation invariance lets the same
    node weights serve the mirrored right end."""
    N = u.size
    nw = max(order + 2, hw)   # order+2 nodes => one-sided d2 of accuracy `order`
    nw = min(nw, N)
    xn = dx * np.arange(nw)
    for i in range(hw):
        uxx[i] = fornberg_weights(dx * i, xn, 2) @ u[:nw]
        uxx[N - 1 - i] = fornberg_weights(dx * (nw - 1 - i), xn, 2) @ u[N - nw:N]


def make_d2(name: str, bc_order: int, eta: float = 1.5):
    """Build a second-derivative operator with the central interior stencil of
    `name` and a Fornberg one-sided boundary closure of accuracy `bc_order`."""
    half = _half_coeffs(name, eta)
    hw = half.size - 1

    def op(u: np.ndarray, dx: float) -> np.ndarray:
        N = u.size
        uxx = np.empty_like(u)
        acc = half[0] * u[hw:N - hw]
        for k in range(1, hw + 1):
            acc = acc + half[k] * (u[hw + k:N - hw + k] + u[hw - k:N - hw - k])
        uxx[hw:N - hw] = acc / dx**2
        _fill_d2_boundaries(u, dx, uxx, hw, bc_order)
        return uxx

    return op


def d2_order2(u: np.ndarray, dx: float) -> np.ndarray:
    uxx = np.empty_like(u)
    uxx[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / dx**2
    _edge_one_sided(u, dx, uxx)
    return uxx


def d2_order4(u: np.ndarray, dx: float) -> np.ndarray:
    uxx = np.empty_like(u)
    uxx[2:-2] = (-u[:-4] + 16.0 * u[1:-3] - 30.0 * u[2:-2]
                 + 16.0 * u[3:-1] - u[4:]) / (12.0 * dx**2)
    # near-boundary: 2nd-order central
    uxx[1] = (u[2] - 2.0 * u[1] + u[0]) / dx**2
    uxx[-2] = (u[-1] - 2.0 * u[-2] + u[-3]) / dx**2
    _edge_one_sided(u, dx, uxx)
    return uxx


def d2_order6(u: np.ndarray, dx: float) -> np.ndarray:
    uxx = np.empty_like(u)
    uxx[3:-3] = (2.0 * u[:-6] - 27.0 * u[1:-5] + 270.0 * u[2:-4]
                 - 490.0 * u[3:-3] + 270.0 * u[4:-2] - 27.0 * u[5:-1]
                 + 2.0 * u[6:]) / (180.0 * dx**2)
    # near-boundary: drop order gracefully (4th at the second ring, 2nd at first)
    uxx[2] = (-u[0] + 16.0 * u[1] - 30.0 * u[2] + 16.0 * u[3] - u[4]) / (12.0 * dx**2)
    uxx[-3] = (-u[-5] + 16.0 * u[-4] - 30.0 * u[-3] + 16.0 * u[-2] - u[-1]) / (12.0 * dx**2)
    uxx[1] = (u[2] - 2.0 * u[1] + u[0]) / dx**2
    uxx[-2] = (u[-1] - 2.0 * u[-2] + u[-3]) / dx**2
    _edge_one_sided(u, dx, uxx)
    return uxx


def make_drp7(eta: float = 1.5):
    """Tam & Webb-style DISPERSION-RELATION-PRESERVING 7-point second derivative.

    Symmetric stencil  u'' ~ (1/dx^2) sum_{j=-3}^{3} a_j u_{i+j}, with a_{-j}=a_j.
    Two constraints fix consistency and 4th-order accuracy and leave ONE free
    parameter (a_3); the rest of the 6th-order condition is spent instead on
    minimising the modified-wavenumber error over a band w in [0, eta].

        a_2 = -1/12 - 6 a_3 ,   a_1 = 4/3 + 15 a_3 ,   a_0 = -2(a_1+a_2+a_3).

    The second-derivative modified wavenumber is
        f(w) = 2[a_1(1-cos w) + a_2(1-cos 2w) + a_3(1-cos 3w)]  ->  target w^2.
    a_3 is chosen to minimise int_0^eta (f(w)-w^2)^2 dw (a_3=1/90 -> pure 6th).
    Coefficients are DERIVED here, not hand-copied.
    """
    a0, a1, a2, a3 = drp7_coeffs(eta)
    print(f"  [DRP7] band eta={eta:.2f}  a3={a3:.6e} (6th-order would be {1/90:.6e})", flush=True)
    print(f"         a0={a0:.6e} a1={a1:.6e} a2={a2:.6e} a3={a3:.6e}", flush=True)

    def d2_drp7(u: np.ndarray, dx: float) -> np.ndarray:
        uxx = np.empty_like(u)
        uxx[3:-3] = (a3 * (u[:-6] + u[6:]) + a2 * (u[1:-5] + u[5:-1])
                     + a1 * (u[2:-4] + u[4:-2]) + a0 * u[3:-3]) / dx**2
        # graceful boundary drop (same closure as the 6th-order scheme)
        uxx[2] = (-u[0] + 16.0 * u[1] - 30.0 * u[2] + 16.0 * u[3] - u[4]) / (12.0 * dx**2)
        uxx[-3] = (-u[-5] + 16.0 * u[-4] - 30.0 * u[-3] + 16.0 * u[-2] - u[-1]) / (12.0 * dx**2)
        uxx[1] = (u[2] - 2.0 * u[1] + u[0]) / dx**2
        uxx[-2] = (u[-1] - 2.0 * u[-2] + u[-3]) / dx**2
        _edge_one_sided(u, dx, uxx)
        return uxx

    return d2_drp7


def _compact4_banded(n: int, alpha: float) -> np.ndarray:
    """Banded LHS (3, n) for the tridiagonal compact system.
    Interior rows: alpha f''_{i-1} + f''_i + alpha f''_{i+1} = rhs.
    Boundary rows 0 and n-1 are identity (explicit closure in the rhs)."""
    ab = np.zeros((3, n))
    ab[1, :] = 1.0  # main diagonal
    # superdiagonal ab[0, j] = A[j-1, j]; subdiagonal ab[2, j] = A[j+1, j]
    for i in range(1, n - 1):
        ab[0, i + 1] = alpha
        ab[2, i - 1] = alpha
    return ab


def make_compact4(n: int, alpha: float = 0.1, a: float = 1.2):
    """Lele (1992) 4th-order compact second derivative, near-spectral resolution.
        (1/10) f''_{i-1} + f''_i + (1/10) f''_{i+1}
            = (6/5) (f_{i+1} - 2 f_i + f_{i-1}) / dx^2
    Boundary points use a 2nd-order one-sided explicit closure."""
    ab = _compact4_banded(n, alpha)

    def d2_compact4(u: np.ndarray, dx: float) -> np.ndarray:
        rhs = np.empty_like(u)
        rhs[1:-1] = a * (u[2:] - 2.0 * u[1:-1] + u[:-2]) / dx**2
        rhs[0] = (2.0 * u[0] - 5.0 * u[1] + 4.0 * u[2] - 1.0 * u[3]) / dx**2
        rhs[-1] = (2.0 * u[-1] - 5.0 * u[-2] + 4.0 * u[-3] - 1.0 * u[-4]) / dx**2
        return solve_banded((1, 1), ab, rhs)

    return d2_compact4


# ---------------------------------------------------------------------------
# Method-of-lines solver (mirrors src/fd_solver.py exactly, pluggable d2 op).
# ---------------------------------------------------------------------------

def _one_sided_dx_left(u: np.ndarray, dx: float) -> float:
    return (-3.0 * u[0] + 4.0 * u[1] - 1.0 * u[2]) / (2.0 * dx)


def _one_sided_dx_right(u: np.ndarray, dx: float) -> float:
    return (3.0 * u[-1] - 4.0 * u[-2] + 1.0 * u[-3]) / (2.0 * dx)


def _dx_left(u: np.ndarray, dx: float, order: int) -> float:
    if order == 2:
        return _one_sided_dx_left(u, dx)
    n = order + 1
    return float(fornberg_weights(0.0, dx * np.arange(n), 1) @ u[:n])


def _dx_right(u: np.ndarray, dx: float, order: int) -> float:
    if order == 2:
        return _one_sided_dx_right(u, dx)
    n = order + 1
    return float(fornberg_weights(dx * (n - 1), dx * np.arange(n), 1) @ u[-n:])


def _apply_radiative_bc(u: np.ndarray, v: np.ndarray, dx: float, order: int = 2) -> None:
    v[0] = _dx_left(u, dx, order)
    v[-1] = -_dx_right(u, dx, order)


def solve(d2_op: Callable[[np.ndarray, float], np.ndarray],
          dx: float, dt: float, bc_order: int = 2,
          M_: float = M, x0_: float = ID_X0, sigma_: float = ID_SIGMA
          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Nx = int(round((XMAX - XMIN) / dx)) + 1
    Nt = int(round((TMAX - TMIN) / dt))
    x = XMIN + dx * np.arange(Nx)
    t = TMIN + dt * np.arange(Nt + 1)
    Vx = V_of_x(x, M=M_, l=ELL, potential=POTENTIAL)

    u = gaussian_phi(x, A=ID_A, x0=x0_, sigma=sigma_)
    v = gaussian_phi_t(x, A=ID_A, x0=x0_, sigma=sigma_, profile=ID_VPROFILE)
    _apply_radiative_bc(u, v, dx, bc_order)

    phi = np.empty((Nt + 1, Nx), dtype=float)
    phi[0] = u

    def rhs(u_, v_):
        return v_, d2_op(u_, dx) - Vx * u_

    for n in range(Nt):
        k1u, k1v = rhs(u, v)
        u2, v2 = u + 0.5 * dt * k1u, v + 0.5 * dt * k1v
        _apply_radiative_bc(u2, v2, dx, bc_order); k2u, k2v = rhs(u2, v2)
        u3, v3 = u + 0.5 * dt * k2u, v + 0.5 * dt * k2v
        _apply_radiative_bc(u3, v3, dx, bc_order); k3u, k3v = rhs(u3, v3)
        u4, v4 = u + dt * k3u, v + dt * k3v
        _apply_radiative_bc(u4, v4, dx, bc_order); k4u, k4v = rhs(u4, v4)
        u = u + (dt / 6.0) * (k1u + 2 * k2u + 2 * k3u + k4u)
        v = v + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        _apply_radiative_bc(u, v, dx, bc_order)
        phi[n + 1] = u
    return x, t, phi


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    print("=== Coarse-stencil isolation test (canonical Schwarzschild/Zerilli) ===", flush=True)
    print(f"  domain x in [{XMIN},{XMAX}], t in [{TMIN},{TMAX}]; V={POTENTIAL} M={M} l={ELL}", flush=True)
    print(f"  coarse grid dx={DX_COARSE} dt={DT_COARSE}; truth dx={DX_TRUTH} dt={DT_TRUTH} (6th order)\n", flush=True)

    # --- converged reference ---
    t0 = time.time()
    xt, tt, phi_truth = solve(d2_order6, DX_TRUTH, DT_TRUTH)
    print(f"  truth solve (6th, dx={DX_TRUTH}): {time.time()-t0:.2f}s, grid {phi_truth.shape}", flush=True)

    sx = int(round(DX_COARSE / DX_TRUTH))   # spatial stride (=4)
    st = int(round(DT_COARSE / DT_TRUTH))   # temporal stride (=4)
    truth_on_coarse = phi_truth[::st, ::sx]

    compact = make_compact4(int(round((XMAX - XMIN) / DX_COARSE)) + 1)
    drp7 = make_drp7(eta=1.5)
    schemes = {
        "S2_central":  d2_order2,
        "S4_central":  d2_order4,
        "S6_central":  d2_order6,
        "SC4_compact": compact,
        "DRP7":        drp7,
    }
    interior_order = {"S2_central": 2, "S4_central": 4, "S6_central": 6,
                      "SC4_compact": 4, "DRP7": 4}

    # wave zone mask (where the dispersion stripes live): |x*| beyond the
    # potential peak, excluding the near-peak scattering region.
    xc, tc, _ = solve(d2_order2, DX_COARSE, DT_COARSE)
    wave_mask = np.abs(xc) > 10.0

    results: Dict[str, Dict] = {}
    err_fields: Dict[str, np.ndarray] = {}
    for name, op in schemes.items():
        t0 = time.time()
        x_c, t_c, phi_c = solve(op, DX_COARSE, DT_COARSE)
        wall = time.time() - t0
        err = phi_c - truth_on_coarse
        err_fields[name] = err
        l2 = float(np.sqrt(np.mean(err**2)))
        linf = float(np.max(np.abs(err)))
        l2_wave = float(np.sqrt(np.mean(err[:, wave_mask]**2)))
        results[name] = {"l2": l2, "linf": linf, "l2_wave": l2_wave,
                         "wall_s": wall, "order": interior_order[name]}

    # reference: production "fine" 2nd-order grid (dx=0.2) error too, for scale
    x_f, t_f, phi_f = solve(d2_order2, DX_FINE, DT_FINE)
    sxf, stf = int(round(DX_FINE / DX_TRUTH)), int(round(DT_FINE / DT_TRUTH))
    err_fine = phi_f - phi_truth[::stf, ::sxf]
    fine_l2 = float(np.sqrt(np.mean(err_fine**2)))

    print("\n=== Coarse-grid error vs converged truth (same dx=0.4, dt=0.2) ===", flush=True)
    print(f"{'scheme':>14} {'order':>5} {'L2':>11} {'Linf':>11} {'L2(wave)':>11} {'wall_s':>8} {'L2 vs S2':>9}", flush=True)
    base_l2 = results["S2_central"]["l2"]
    for name, r in results.items():
        print(f"{name:>14} {r['order']:>5d} {r['l2']:>11.3e} {r['linf']:>11.3e} "
              f"{r['l2_wave']:>11.3e} {r['wall_s']:>8.3f} {base_l2/r['l2']:>8.1f}x", flush=True)
    print(f"\n  (for scale) production FINE 2nd-order dx=0.2 global L2 = {fine_l2:.3e}", flush=True)
    print(f"  S2 coarse / fine L2 ratio = {base_l2/fine_l2:.1f}x worse than the fine reference", flush=True)

    # --- heatmaps ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        allerr = np.concatenate([np.abs(e).ravel() for e in err_fields.values()])
        vmax = float(np.percentile(allerr, 99.9))
        vmin = max(vmax * 1e-4, 1e-7)
        fig, axes = plt.subplots(1, len(err_fields), figsize=(5 * len(err_fields), 5),
                                 constrained_layout=True)
        for ax, (name, err) in zip(axes, err_fields.items()):
            im = ax.pcolormesh(t_c, x_c, np.abs(err).T,
                               norm=LogNorm(vmin=vmin, vmax=vmax),
                               cmap="magma_r", shading="auto")
            ax.set_title(f"{name}  (L2={results[name]['l2']:.2e})")
            ax.set_xlabel("t / M")
            ax.set_ylabel("x* / M")
        fig.colorbar(im, ax=axes, location="right", shrink=0.8,
                     label="|Phi_coarse - Phi_truth|")
        fig.suptitle("Coarse-FD pointwise error vs stencil order (same coarse grid dx=0.4)",
                     fontsize=14)
        figpath = os.path.join(OUTDIR, "coarse_stencil_error_heatmaps.png")
        fig.savefig(figpath, dpi=120)
        print(f"\nSaved heatmaps: {figpath}", flush=True)
    except Exception as e:  # pragma: no cover
        print(f"\n[plot skipped: {e}]", flush=True)

    np.savez(os.path.join(OUTDIR, "coarse_stencil_errors.npz"),
             x_coarse=x_c, t_coarse=t_c,
             **{f"err_{k}": v for k, v in err_fields.items()},
             **{f"l2_{k}": results[k]["l2"] for k in results})
    print(f"Saved arrays: {os.path.join(OUTDIR, 'coarse_stencil_errors.npz')}", flush=True)


def boundary_floor_sweep() -> None:
    """Test (c): does upgrading the boundary closure (one-sided d2 rows AND the
    radiative-BC first derivative) lower the Linf floor that all high-order
    interior schemes plateau at? Compares bc_order in {2,4,6} for S6 and DRP7
    against a high-order-boundary truth so the truth's own boundary error does
    not contaminate the measurement."""
    print("\n\n=== (c) Boundary-closure floor sweep ===", flush=True)

    # high-order-boundary truth (interior 6th, boundary 6th) at the fine truth grid
    xt, tt, phi_truth = solve(make_d2("S6_central", bc_order=6), DX_TRUTH, DT_TRUTH, bc_order=6)
    sx, st = int(round(DX_COARSE / DX_TRUTH)), int(round(DT_COARSE / DT_TRUTH))
    truth_on_coarse = phi_truth[::st, ::sx]

    # column masks on the coarse grid: "boundary band" = near either x edge.
    x_c = XMIN + DX_COARSE * np.arange(truth_on_coarse.shape[1])
    BAND = 6  # coarse points (= 2.4 in x*) from each end
    bnd_cols = np.zeros(x_c.size, dtype=bool)
    bnd_cols[:BAND] = True
    bnd_cols[-BAND:] = True
    int_cols = ~bnd_cols

    def metrics(err: np.ndarray) -> Dict[str, float]:
        return {
            "l2": float(np.sqrt(np.mean(err**2))),
            "linf": float(np.max(np.abs(err))),
            "linf_bnd": float(np.max(np.abs(err[:, bnd_cols]))),
            "linf_int": float(np.max(np.abs(err[:, int_cols]))),
        }

    schemes = ["S6_central", "DRP7"]
    bc_orders = [2, 4, 6]
    print(f"{'scheme':>12} {'bc_order':>8} {'L2':>11} {'Linf':>11} "
          f"{'Linf(bnd)':>11} {'Linf(int)':>11}", flush=True)
    fields: Dict[str, np.ndarray] = {}
    for name in schemes:
        for bc in bc_orders:
            _, _, phi_c = solve(make_d2(name, bc_order=bc), DX_COARSE, DT_COARSE, bc_order=bc)
            err = phi_c - truth_on_coarse
            m = metrics(err)
            fields[f"{name}_bc{bc}"] = err
            print(f"{name:>12} {bc:>8d} {m['l2']:>11.3e} {m['linf']:>11.3e} "
                  f"{m['linf_bnd']:>11.3e} {m['linf_int']:>11.3e}", flush=True)

    # figure: DRP7 error at bc_order 2 / 4 / 6
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        t_c = TMIN + DT_COARSE * np.arange(truth_on_coarse.shape[0])
        panels = [f"DRP7_bc{bc}" for bc in bc_orders]
        allerr = np.concatenate([np.abs(fields[k]).ravel() for k in panels])
        vmax = float(np.percentile(allerr, 99.9))
        vmin = max(vmax * 1e-4, 1e-8)
        fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5),
                                 constrained_layout=True)
        for ax, key in zip(axes, panels):
            err = fields[key]
            im = ax.pcolormesh(t_c, x_c, np.abs(err).T,
                               norm=LogNorm(vmin=vmin, vmax=vmax),
                               cmap="magma_r", shading="auto")
            ax.set_title(f"{key}  (Linf={np.max(np.abs(err)):.2e})")
            ax.set_xlabel("t / M")
            ax.set_ylabel("x* / M")
        fig.colorbar(im, ax=axes, location="right", shrink=0.8,
                     label="|Phi_coarse - Phi_truth|")
        fig.suptitle("DRP7 pointwise error vs boundary-closure order (interior fixed)",
                     fontsize=14)
        figpath = os.path.join(OUTDIR, "boundary_floor_sweep.png")
        fig.savefig(figpath, dpi=120)
        print(f"\nSaved boundary sweep figure: {figpath}", flush=True)
    except Exception as e:  # pragma: no cover
        print(f"\n[plot skipped: {e}]", flush=True)


if __name__ == "__main__":
    main()
    boundary_floor_sweep()

"""Hyperboloidal Regge-Wheeler operator on Schwarzschild (a=0).

Derivation. The flat Regge-Wheeler equation on Schwarzschild reads

    -d_t^2 Phi + d_{r_*}^2 Phi - V(r) Phi = 0,

with V(r) = (1 - 2M/r) [ ell(ell+1)/r^2 - 6 M / r^3 ] for axial parity.
Hyperboloidal time tau = t - h(r_*) gives

    d_t|_{r_*}      = d_tau|_{r_*}
    d_{r_*}|_t      = d_{r_*}|_tau - H(r_*) d_tau,        H := dh/dr_*

and the operator becomes

    (H^2 - 1) d_tau^2 Phi - 2 H d_{r_*} d_tau Phi - H' d_tau Phi
        + d_{r_*}^2 Phi - V(r) Phi = 0.

Introducing Pi := d_tau Phi the first-order MOL system is

    d_tau Phi = Pi
    d_tau Pi  = (1 / (1 - H^2)) [ d_{r_*}^2 Phi - 2 H d_{r_*} Pi - H' Pi - V Phi ].

Coefficients are regular on the bulk interior (1 - H^2 > 0) and become
characteristic in the limits r_* -> +inf (scri+) and r_* -> -inf (horizon),
which is the intended outflow behaviour of the hyperboloidal slice.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def potential_RW(r, M: float = 1.0, ell: int = 2):
    """Axial Regge-Wheeler potential V(r)."""
    f = 1.0 - 2.0 * M / r
    return f * (ell * (ell + 1) / r ** 2 - 6.0 * M / r ** 3)


@dataclass
class HyperOp:
    """Pre-tabulated coefficients on a fixed r_* grid."""

    r_star: np.ndarray
    r: np.ndarray
    H: np.ndarray
    Hprime: np.ndarray  # dH/dr_*
    V: np.ndarray
    one_minus_H2: np.ndarray  # = sech^2(r_*/L), strictly > 0 on bulk
    inv_one_minus_H2: np.ndarray
    M: float
    L: float
    ell: int


def _invert_tortoise(r_star: np.ndarray, M: float, tol: float = 1e-13, itmax: int = 200) -> np.ndarray:
    """Numerically invert r_*(r) = r + 2M ln(r/(2M) - 1) for r > 2M."""
    r = np.where(r_star > 0.0, r_star, 2.0 * M * (1.0 + np.exp(r_star / (2.0 * M) - 1.0)))
    r = np.maximum(r, 2.0 * M * (1.0 + 1e-14))
    for _ in range(itmax):
        f = r + 2.0 * M * np.log(r / (2.0 * M) - 1.0) - r_star
        fprime = 1.0 / (1.0 - 2.0 * M / r)
        dr = -f / fprime
        r = r + dr
        r = np.maximum(r, 2.0 * M * (1.0 + 1e-15))
        if np.max(np.abs(dr)) < tol:
            break
    return r


def build_operator(r_star: np.ndarray, M: float = 1.0, L: float = 2.0, ell: int = 2) -> HyperOp:
    """Tabulate (H, H', V, 1-H^2) on the given r_* grid for the tanh(r_*/L) gauge."""
    r = _invert_tortoise(r_star, M)
    x = r_star / L
    H = np.tanh(x)
    sech2 = 1.0 / np.cosh(x) ** 2
    Hprime = sech2 / L
    V = potential_RW(r, M, ell)
    return HyperOp(
        r_star=r_star, r=r, H=H, Hprime=Hprime, V=V,
        one_minus_H2=sech2, inv_one_minus_H2=1.0 / sech2,
        M=M, L=L, ell=ell,
    )


def rhs(Phi: np.ndarray, Pi: np.ndarray, op: HyperOp, d1, d2):
    """Right-hand side of the MOL system.

    d_tau Phi = Pi
    d_tau Pi  = inv_(1-H^2) * [ d_{r_*}^2 Phi - 2 H d_{r_*} Pi - H' Pi - V Phi ]
    """
    dx = op.r_star[1] - op.r_star[0]
    d2_Phi = d2(Phi, dx)
    d1_Pi = d1(Pi, dx)
    dPhi_dtau = Pi
    dPi_dtau = op.inv_one_minus_H2 * (d2_Phi - 2.0 * op.H * d1_Pi - op.Hprime * Pi - op.V * Phi)
    return dPhi_dtau, dPi_dtau

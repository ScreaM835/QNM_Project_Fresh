"""Spin-weighted spheroidal separation constant for the Kerr Teukolsky equation.

In the 1+1D (single azimuthal mode) hyperboloidal reduction the angular sector
is frozen at the spheroidal eigenvalue ``_sA_lm(c)``, ``c = a*omega``, evaluated
at the reference QNM frequency.  This module sources ``_sA_lm`` from the angular
solver of the ``qnm`` package (Stein 2019) -- the same package wrapped in
``qnm_kerr_reference.py`` -- and assembles the Teukolsky radial separation
constant

    lambda = _sA_lm + a^2 omega^2 - 2 a m omega           (M = 1 units)

used by the minimal-gauge radial operator (notes/kerr_minimal_gauge_derivation
section 1, where the zeroth-order coefficient carries ``-lambda``).

Conventions
-----------
* Geometric units, ``M = 1``, so ``a_over_M == a`` and ``c == a*omega``.
* ``qnm``'s angular eigenvalue ``A`` uses the convention
  ``A(a=0) = l(l+1) - s(s+1)`` (= 4 for ``s=-2, l=2``), the same convention in
  which the term-by-term operator check gives ``C_0(a=0) = -lambda = -4``.
"""

from __future__ import annotations

from dataclasses import dataclass

from qnm.angular import sep_const_closest, swsphericalh_A


@dataclass
class SeparationConstant:
    """Frozen angular/radial separation data at a single ``(a, omega)``."""

    a_over_M: float
    ell: int
    m: int
    s: int
    omega: complex   # M * omega (reference QNM frequency)
    c: complex       # oblateness a*omega
    sA: complex      # spheroidal eigenvalue  _sA_lm(c)
    lam: complex     # Teukolsky lambda = sA + a^2 omega^2 - 2 a m omega


def sep_constant(a_over_M: float, ell: int, m: int, s: int, omega: complex,
                 l_max: int | None = None) -> complex:
    """Spin-weighted spheroidal eigenvalue ``_sA_lm(c)``, ``c = a_over_M*omega``.

    At ``a = 0`` (``c = 0``) the spheroidal harmonics reduce to spin-weighted
    *spherical* harmonics and the eigenvalue is exactly ``l(l+1) - s(s+1)``.
    For ``a > 0`` the eigenvalue closest to that spherical value is selected from
    the spherical-spheroidal decomposition matrix of the ``qnm`` package.
    """
    A0 = swsphericalh_A(s, ell, m)             # l(l+1) - s(s+1), exact integer
    if a_over_M == 0.0:
        return complex(A0)
    c = a_over_M * omega
    if l_max is None:
        l_max = ell + 12                       # >> enough for |c| <~ 1
    return complex(sep_const_closest(A0, s, c, m, l_max))


def teukolsky_lambda(a_over_M: float, ell: int, m: int, s: int, omega: complex,
                     l_max: int | None = None) -> complex:
    """Teukolsky radial separation constant ``lambda = _sA + a^2 w^2 - 2 a m w``."""
    sA = sep_constant(a_over_M, ell, m, s, omega, l_max=l_max)
    return sA + (a_over_M ** 2) * omega ** 2 - 2.0 * a_over_M * m * omega


def separation_constant(a_over_M: float, ell: int, m: int, s: int,
                        omega: complex, l_max: int | None = None
                        ) -> SeparationConstant:
    """Bundle the spheroidal eigenvalue and Teukolsky ``lambda`` at ``(a, omega)``."""
    sA = sep_constant(a_over_M, ell, m, s, omega, l_max=l_max)
    c = a_over_M * omega
    lam = sA + (a_over_M ** 2) * omega ** 2 - 2.0 * a_over_M * m * omega
    return SeparationConstant(
        a_over_M=a_over_M, ell=ell, m=m, s=s, omega=omega,
        c=c, sA=sA, lam=lam,
    )


def frozen_separation_constant(a_over_M: float, ell: int = 2, m: int = 2,
                               n: int = 0, s: int = -2,
                               l_max: int | None = None
                               ) -> SeparationConstant:
    """Frozen separation data at the reference Kerr QNM frequency ``omega_ref(a)``.

    Pulls ``omega_ref(a)`` from ``qnm`` via ``qnm_kerr_reference.kerr_qnm`` and
    freezes ``_sA`` and ``lambda`` there -- the values the single-mode radial
    solver uses.
    """
    try:
        from .qnm_kerr_reference import kerr_qnm
    except ImportError:  # allow running as a stand-alone script (python src/spheroidal.py)
        from qnm_kerr_reference import kerr_qnm
    q = kerr_qnm(a_over_M, ell=ell, m=m, n=n)
    omega = complex(q.M_omega_R, q.M_omega_I)
    return separation_constant(a_over_M, ell, m, s, omega, l_max=l_max)


if __name__ == "__main__":
    for a in (0.0, 0.5, 0.9):
        sc = frozen_separation_constant(a, 2, 2, 0, -2)
        print(f"a/M={a:>4}  c={sc.c:+.5f}  sA={sc.sA:+.6f}  lambda={sc.lam:+.6f}")

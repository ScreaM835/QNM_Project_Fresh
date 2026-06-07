"""Wrapper around the `qnm` PyPI package (Stein 2019) for Kerr QNM reference values.

Provides M*omega and tau/M for the dominant (l, m, n) modes used as the
extraction target throughout this repo.
"""

from __future__ import annotations

from dataclasses import dataclass
import qnm


@dataclass
class KerrQNM:
    a_over_M: float
    ell: int
    m: int
    n: int
    M_omega_R: float   # M * Re(omega)
    M_omega_I: float   # M * Im(omega)   (negative for decaying modes)

    @property
    def tau_over_M(self) -> float:
        return -1.0 / self.M_omega_I


def kerr_qnm(a_over_M: float, ell: int = 2, m: int = 2, n: int = 0) -> KerrQNM:
    """Return Leaver-equivalent Kerr (l, m, n) QNM, normalised by M=1."""
    seq = qnm.modes_cache(s=-2, l=ell, m=m, n=n)
    omega, _, _ = seq(a=a_over_M)
    return KerrQNM(
        a_over_M=a_over_M,
        ell=ell,
        m=m,
        n=n,
        M_omega_R=float(omega.real),
        M_omega_I=float(omega.imag),
    )


if __name__ == "__main__":
    for a in (0.0, 0.5, 0.9):
        q0 = kerr_qnm(a, 2, 2, 0)
        q1 = kerr_qnm(a, 2, 2, 1)
        print(f"a/M={a:>4}  (2,2,0) Mw={q0.M_omega_R:.6f}  tau/M={q0.tau_over_M:.4f}  "
              f"(2,2,1) Mw={q1.M_omega_R:.6f}  tau/M={q1.tau_over_M:.4f}")

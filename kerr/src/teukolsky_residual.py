"""Torch port of the minimal-gauge Teukolsky coefficients + the second-order
PINN residual (Phase C.3).

The Phase B FD solver (`teukolsky_minimal_gauge.build_teukolsky_op`) evolves the
first-order characteristic system ``(Psi, U, W)`` with ``U = Pi + lam_in Phi``,
``W = Pi + lam_out Phi``, ``Pi = d_tau Psi``, ``Phi = d_sigma Psi``. Eliminating
``U, W`` (the ``lam'`` terms cancel identically -- verified from both the U and W
equations) collapses that system to a single **second-order complex scalar** PDE
in ``(sigma, tau)``:

    R[Psi] = d_tau^2 Psi
           + (lam_in + lam_out) d_tau d_sigma Psi
           + lam_in lam_out      d_sigma^2 Psi
           - c_Pi  d_tau Psi
           - c_Phi d_sigma Psi
           - c_Psi Psi
           = 0 .

This is the residual the PINN minimises (no FD field, no boundary terms -- the
hyperboloidal characteristics are outflow at both ends -- and no Kreiss-Oliger
dissipation, which is an FD-only regulariser).

The coefficient closed forms below are **copied verbatim** from
`teukolsky_minimal_gauge.build_teukolsky_op` (only ``np`` -> ``torch`` and the
complex promotion differ), so the torch coefficients reproduce the validated FD
operator to machine precision. ``scripts``/tests gate that at <= 1e-10.

Everything is FP64 (real -> ``torch.float64``, complex -> ``torch.complex128``).
FP32 is out of spec for this module (Xu et al. 2025, arXiv:2505.10949).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

try:
    from kerr.src.spheroidal import teukolsky_lambda
except ImportError:  # allow running as a stand-alone script / from kerr/src
    try:
        from .spheroidal import teukolsky_lambda
    except ImportError:
        from spheroidal import teukolsky_lambda


REAL_DTYPE = torch.float64
COMPLEX_DTYPE = torch.complex128


@dataclass
class KerrCoeffs:
    """Closed-form minimal-gauge Teukolsky coefficients for a single spin.

    The geometry scalars ``(r_plus, r_minus, beta, lam)`` are frozen at
    construction (``beta`` is the corotating phase rate ``m a / (r+ - r-)`` and
    ``lam`` is the Teukolsky separation constant at the reference QNM frequency).
    The five coefficients are then closed-form rational functions of ``sigma``
    alone, evaluated elementwise in torch on the PINN collocation points.

    ``lambda_in``/``lambda_out`` are real; ``c_pi``/``c_phi``/``c_psi`` are
    complex (they carry the frame-dragging ``i beta`` and the complex ``lam``).
    """

    r_plus: float
    r_minus: float
    beta: float
    lam: complex
    a_over_M: float
    ell: int = 2
    m: int = 2
    M: float = 1.0

    # ---- real characteristic speeds --------------------------------------
    def lambda_out(self, sg: torch.Tensor) -> torch.Tensor:
        rp, rm = self.r_plus, self.r_minus
        return (-rm * sg ** 2 + rm * sg + rp * sg - rp) / (
            2 * rm * rp * sg ** 2 + 2 * rp ** 2)

    def lambda_in(self, sg: torch.Tensor) -> torch.Tensor:
        rp, rm = self.r_plus, self.r_minus
        return (-rm * sg ** 3 + rp * sg ** 2) / (
            2 * rm * rp * sg ** 3 + 2 * rm * rp * sg ** 2
            + 2 * rp ** 2 * sg + 2 * rp ** 2)

    # ---- complex source coefficients -------------------------------------
    def c_pi(self, sg: torch.Tensor) -> torch.Tensor:
        rp, rm, bt = self.r_plus, self.r_minus, self.beta
        z = sg.to(COMPLEX_DTYPE)
        num = (2j * bt * rm ** 2 * z ** 4 + 2j * bt * rm ** 2 * z ** 3
               + 1j * bt * rm ** 2 * z ** 2
               - 2j * bt * rm * rp * z ** 3 + 2j * bt * rm * rp * z
               + 1j * bt * rm * rp
               - 2j * bt * rp ** 2 * z - 2j * bt * rp ** 2
               + 2 * rm ** 2 * z ** 3 + 3 * rm ** 2 * z ** 2
               - 2 * rm * rp * z ** 3 - 4 * rm * rp * z ** 2
               - rm * rp * z - 2 * rp ** 2)
        den = (2 * rm ** 2 * rp * z ** 5 + 2 * rm ** 2 * rp * z ** 4
               + 4 * rm * rp ** 2 * z ** 3 + 4 * rm * rp ** 2 * z ** 2
               + 2 * rp ** 3 * z + 2 * rp ** 3)
        return num / den

    def c_phi(self, sg: torch.Tensor) -> torch.Tensor:
        rp, rm, bt = self.r_plus, self.r_minus, self.beta
        z = sg.to(COMPLEX_DTYPE)
        num = (-2j * bt * rm ** 2 * z ** 4 + 4j * bt * rm * rp * z ** 3
               - 2j * bt * rp ** 2 * z ** 2
               - 3 * rm ** 2 * z ** 3 + rm * rp * z ** 3 + 5 * rm * rp * z ** 2
               - rp ** 2 * z ** 2 - 2 * rp ** 2 * z)
        den = (4 * rm ** 2 * rp ** 2 * z ** 5 + 4 * rm ** 2 * rp ** 2 * z ** 4
               + 8 * rm * rp ** 3 * z ** 3 + 8 * rm * rp ** 3 * z ** 2
               + 4 * rp ** 4 * z + 4 * rp ** 4)
        return num / den

    def c_psi(self, sg: torch.Tensor) -> torch.Tensor:
        rp, rm, bt = self.r_plus, self.r_minus, self.beta
        lm = self.lam
        z = sg.to(COMPLEX_DTYPE)
        num = (1j * bt * rm ** 4 * z ** 3 - 2j * bt * rm ** 4 * z ** 2
               - 2j * bt * rm ** 3 * rp * z ** 3 + 3j * bt * rm ** 3 * rp * z ** 2
               + 4j * bt * rm ** 3 * rp * z + 1j * bt * rm ** 2 * rp ** 2 * z ** 3
               - 10j * bt * rm ** 2 * rp ** 2 * z - 1j * bt * rm * rp ** 3 * z ** 2
               + 8j * bt * rm * rp ** 3 * z - 2j * bt * rp ** 4 * z
               + lm * rm ** 3 * rp * z - 2 * lm * rm ** 2 * rp ** 2 * z
               - lm * rm ** 2 * rp ** 2 + lm * rm * rp ** 3 * z
               + 2 * lm * rm * rp ** 3 - lm * rp ** 4
               + 3 * rm ** 4 * z ** 2 + 4 * rm ** 3 * rp * z ** 3
               - 3 * rm ** 3 * rp * z ** 2 - 3 * rm ** 3 * rp * z
               - 3 * rm ** 2 * rp ** 2 * z ** 2 + 7 * rm ** 2 * rp ** 2 * z
               - rm * rp ** 3 * z ** 2 - 5 * rm * rp ** 3 * z + rp ** 4 * z)
        den = (4 * rm ** 4 * rp ** 2 * z ** 5 + 4 * rm ** 4 * rp ** 2 * z ** 4
               - 8 * rm ** 3 * rp ** 3 * z ** 5 - 8 * rm ** 3 * rp ** 3 * z ** 4
               + 8 * rm ** 3 * rp ** 3 * z ** 3 + 8 * rm ** 3 * rp ** 3 * z ** 2
               + 4 * rm ** 2 * rp ** 4 * z ** 5 + 4 * rm ** 2 * rp ** 4 * z ** 4
               - 16 * rm ** 2 * rp ** 4 * z ** 3 - 16 * rm ** 2 * rp ** 4 * z ** 2
               + 4 * rm ** 2 * rp ** 4 * z + 4 * rm ** 2 * rp ** 4
               + 8 * rm * rp ** 5 * z ** 3 + 8 * rm * rp ** 5 * z ** 2
               - 8 * rm * rp ** 5 * z - 8 * rm * rp ** 5
               + 4 * rp ** 6 * z + 4 * rp ** 6)
        return num / den

    def all_coeffs(self, sg: torch.Tensor
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                              torch.Tensor, torch.Tensor]:
        """``(lambda_in, lambda_out, c_pi, c_phi, c_psi)`` on grid ``sg``."""
        return (self.lambda_in(sg), self.lambda_out(sg),
                self.c_pi(sg), self.c_phi(sg), self.c_psi(sg))


def coeffs_from_spin(a_over_M: float, omega_ref: complex | None = None,
                     ell: int = 2, m: int = 2, M: float = 1.0,
                     lam: complex | None = None) -> KerrCoeffs:
    """Build :class:`KerrCoeffs` from spin + reference frequency.

    Mirrors the geometry/separation-constant setup of ``build_teukolsky_op``
    exactly: ``r_pm = M(1 +- sqrt(1 - chi^2))``, ``beta = m chi / (2 sqrt(...))``,
    and ``lam = teukolsky_lambda(chi, ell, m, -2, omega_ref)`` frozen at the
    reference QNM (unless ``lam`` is supplied directly).
    """
    chi = float(a_over_M)
    if not (0.0 <= chi < 1.0):
        raise ValueError("a_over_M must be in [0, 1)")
    root = float(np.sqrt(1.0 - chi * chi))
    rp = M * (1.0 + root)
    rm = M * (1.0 - root)
    if root <= 0.0:
        raise ValueError("a_over_M = 1 (extremal) is out of scope")
    bt = m * chi / (2.0 * root)
    if lam is None:
        if omega_ref is None:
            raise ValueError("omega_ref or lam is required")
        lam = teukolsky_lambda(chi, ell, m, -2, complex(omega_ref))
    return KerrCoeffs(r_plus=rp, r_minus=rm, beta=float(bt), lam=complex(lam),
                      a_over_M=chi, ell=ell, m=m, M=M)


def second_order_residual(
    p_t: torch.Tensor, q_t: torch.Tensor,
    p_s: torch.Tensor, q_s: torch.Tensor,
    p_tt: torch.Tensor, q_tt: torch.Tensor,
    p_ss: torch.Tensor, q_ss: torch.Tensor,
    p_ts: torch.Tensor, q_ts: torch.Tensor,
    p: torch.Tensor, q: torch.Tensor,
    lam_in: torch.Tensor, lam_out: torch.Tensor,
    c_pi: torch.Tensor, c_phi: torch.Tensor, c_psi: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Real/imag parts of ``R[Psi]`` from precomputed derivatives + coefficients.

    ``Psi = p + i q``; subscripts ``t``/``s`` denote ``d_tau``/``d_sigma``.
    Splitting the complex residual into two real channels (so the network's two
    real outputs train against two real residuals):

        R_re = p_tt + A p_ts + B p_ss
               - (cr_pi  p_t - ci_pi  q_t)
               - (cr_phi p_s - ci_phi q_s)
               - (cr_psi p   - ci_psi q)
        R_im = q_tt + A q_ts + B q_ss
               - (cr_pi  q_t + ci_pi  p_t)
               - (cr_phi q_s + ci_phi p_s)
               - (cr_psi q   + ci_psi p)

    with ``A = lam_in + lam_out`` (real), ``B = lam_in lam_out`` (real).
    """
    A = lam_in + lam_out
    B = lam_in * lam_out
    cpi_r, cpi_i = c_pi.real, c_pi.imag
    cph_r, cph_i = c_phi.real, c_phi.imag
    cps_r, cps_i = c_psi.real, c_psi.imag

    r_re = (p_tt + A * p_ts + B * p_ss
            - (cpi_r * p_t - cpi_i * q_t)
            - (cph_r * p_s - cph_i * q_s)
            - (cps_r * p - cps_i * q))
    r_im = (q_tt + A * q_ts + B * q_ss
            - (cpi_r * q_t + cpi_i * p_t)
            - (cph_r * q_s + cph_i * p_s)
            - (cps_r * q + cps_i * p))
    return r_re, r_im

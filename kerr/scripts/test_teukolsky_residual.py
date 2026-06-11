"""C.3 acceptance: torch Teukolsky residual module vs the validated FD operator.

Two exact checks + one finiteness guard:

  1. COEFFICIENT PORT (the C.3 gate).  The torch ``KerrCoeffs`` arrays
     (lambda_in, lambda_out, c_pi, c_phi, c_psi) reproduce
     ``teukolsky_minimal_gauge.build_teukolsky_op`` -- the operator validated to
     1e-12 against Phase A and against the `qnm` package in B.4/B.8 -- to
     <= 1e-10 on the production sigma grid, at spins {0, 0.5, 0.7, 0.9}.
     This is what lets the PINN trust the ported coefficients.

  2. RESIDUAL ALGEBRA.  ``second_order_residual`` splits the complex residual
     R[Psi] = p+iq into two real channels.  Fed independent random derivative
     slots, its (r_re, r_im) must equal the complex128 reference
       R = Psi_tt + (lin+lout) Psi_ts + lin*lout Psi_ss
           - c_pi Psi_t - c_phi Psi_s - c_psi Psi
     to <= 1e-12 (catches any sign error in the Re/Im bookkeeping).

  3. FINITENESS.  All five coefficients finite at both endpoint insets for every
     spin in scope (mirrors the FD operator's B.4 guard).
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.teukolsky_minimal_gauge import build_teukolsky_op
from src.teukolsky_residual import (
    coeffs_from_spin,
    second_order_residual,
    COMPLEX_DTYPE,
)
from src.qnm_kerr_reference import kerr_qnm

ELL, MM, M = 2, 2, 1.0
SPINS = (0.0, 0.5, 0.7, 0.9)
N = 801                      # the B.9 / corpus production grid


def _omega(a):
    q = kerr_qnm(a, ELL, MM, 0)
    return complex(q.M_omega_R, q.M_omega_I)


OMEGA_REF = {a: _omega(a) for a in SPINS}


def test_coeffs_match_fd_operator():
    """(1) torch coeffs == build_teukolsky_op to <= 1e-10 on the corpus grid."""
    worst = {}
    for a in SPINS:
        op = build_teukolsky_op(N, a, M, ELL, MM, omega_ref=OMEGA_REF[a])
        sg = torch.as_tensor(op.sigma, dtype=torch.float64)
        # Use the operator's own frozen lam so this isolates the closed-form
        # transcription (coeffs_from_spin re-derives the identical lam below).
        c = coeffs_from_spin(a, omega_ref=OMEGA_REF[a], ell=ELL, m=MM, M=M)
        lin, lout, cpi, cph, cps = c.all_coeffs(sg)
        pairs = {
            "lambda_in": (lin.numpy(), op.lambda_in),
            "lambda_out": (lout.numpy(), op.lambda_out),
            "c_Pi": (cpi.numpy(), op.c_Pi),
            "c_Phi": (cph.numpy(), op.c_Phi),
            "c_Psi": (cps.numpy(), op.c_Psi),
        }
        for name, (got, ref) in pairs.items():
            d = float(np.max(np.abs(got - ref)))
            worst[(a, name)] = d
            assert d <= 1e-10, f"a={a} {name}: max|torch-FD| = {d:.3e} > 1e-10"
    mx = max(worst.values())
    return f"max|torch-FD| over all spins/coeffs = {mx:.2e}"


def test_lam_matches_operator():
    """coeffs_from_spin freezes the SAME separation constant as the FD op."""
    worst = 0.0
    for a in SPINS:
        op = build_teukolsky_op(N, a, M, ELL, MM, omega_ref=OMEGA_REF[a])
        c = coeffs_from_spin(a, omega_ref=OMEGA_REF[a], ell=ELL, m=MM, M=M)
        d = abs(complex(c.lam) - complex(op.lam))
        worst = max(worst, d)
        assert d <= 1e-12, f"a={a}: |lam_torch - lam_FD| = {d:.3e}"
    return f"max|dlam| = {worst:.2e}"


def test_residual_complex_split():
    """(2) Re/Im split == complex128 reference to <= 1e-12."""
    torch.manual_seed(0)
    a = 0.7
    op = build_teukolsky_op(N, a, M, ELL, MM, omega_ref=OMEGA_REF[a])
    sg = torch.as_tensor(op.sigma, dtype=torch.float64)
    c = coeffs_from_spin(a, omega_ref=OMEGA_REF[a], ell=ELL, m=MM, M=M)
    lin, lout, cpi, cph, cps = c.all_coeffs(sg)

    n = sg.shape[0]
    def rnd():
        return torch.randn(n, dtype=torch.float64)
    p, q = rnd(), rnd()
    p_t, q_t = rnd(), rnd()
    p_s, q_s = rnd(), rnd()
    p_tt, q_tt = rnd(), rnd()
    p_ss, q_ss = rnd(), rnd()
    p_ts, q_ts = rnd(), rnd()

    r_re, r_im = second_order_residual(
        p_t, q_t, p_s, q_s, p_tt, q_tt, p_ss, q_ss, p_ts, q_ts, p, q,
        lin, lout, cpi, cph, cps)

    # complex128 reference (independent of the real-split implementation)
    j = 1j
    Psi = (p + j * q).numpy()
    Psi_t = (p_t + j * q_t).numpy()
    Psi_s = (p_s + j * q_s).numpy()
    Psi_tt = (p_tt + j * q_tt).numpy()
    Psi_ss = (p_ss + j * q_ss).numpy()
    Psi_ts = (p_ts + j * q_ts).numpy()
    A = (lin + lout).numpy()
    B = (lin * lout).numpy()
    R = (Psi_tt + A * Psi_ts + B * Psi_ss
         - cpi.numpy() * Psi_t - cph.numpy() * Psi_s - cps.numpy() * Psi)
    d_re = float(np.max(np.abs(r_re.numpy() - R.real)))
    d_im = float(np.max(np.abs(r_im.numpy() - R.imag)))
    assert d_re <= 1e-12, f"Re mismatch {d_re:.3e}"
    assert d_im <= 1e-12, f"Im mismatch {d_im:.3e}"
    return f"max|dRe|={d_re:.2e}, max|dIm|={d_im:.2e}"


def test_all_coeffs_finite():
    """(3) coeffs finite at both endpoint insets, all spins in scope."""
    bad = []
    for a in (0.0, 0.5, 0.7, 0.9, 0.95):
        c = coeffs_from_spin(a, omega_ref=_omega(a), ell=ELL, m=MM, M=M)
        sg = torch.as_tensor(
            np.linspace(1e-8, 1.0 - 1e-8, N), dtype=torch.float64)
        for name, arr in zip(
            ("lin", "lout", "cpi", "cph", "cps"), c.all_coeffs(sg)):
            if not bool(torch.all(torch.isfinite(arr.to(COMPLEX_DTYPE)))):
                bad.append((a, name))
    assert not bad, f"non-finite: {bad}"
    return "all finite"


def main():
    tests = [
        ("coeffs == FD operator (<=1e-10)    ", test_coeffs_match_fd_operator),
        ("lam == FD operator (<=1e-12)       ", test_lam_matches_operator),
        ("residual Re/Im split (<=1e-12)     ", test_residual_complex_split),
        ("all coefficients finite            ", test_all_coeffs_finite),
    ]
    passed = 0
    for name, fn in tests:
        try:
            res = fn()
            print(f"  PASS  {name}  result={res}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}  {e}")
    print(f"\n{passed} / {len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)


if __name__ == "__main__":
    main()

"""B.3 acceptance: spin-weighted spheroidal separation constant.

Acceptance (notes/phase_b_plan.md, B.3):
  * a = 0  =>  _sA = l(l+1) - s(s+1) = 4  exactly;
  * s=-2, l=m=2, a/M=0.9  matches qnm's own angular eigenvalue (the
    Berti-Cardoso-Will-equivalent reference) to 1e-6.

Supporting rigour (literature-independent anchors so the qnm match is not
self-referential):
  * lambda(a=0) = 4, tying B.3 to the operator zeroth-order coeff C_0(a=0)=-4;
  * the eigenvalue is converged in the matrix truncation l_max;
  * real oblateness c gives a real eigenvalue (spin-weighted spheroidal reality);
  * the small-c slope reproduces the analytic leading coefficient
    -2 m s^2 / [l(l+1)] = -8/3.
"""
import sys
from pathlib import Path

import qnm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.spheroidal import sep_constant, teukolsky_lambda


S, L, M, N = -2, 2, 2, 0
A0_EXACT = L * (L + 1) - S * (S + 1)            # = 4


def _qnm_seq_at(a):
    """qnm's omega and its own angular eigenvalue A at spin a (independent path)."""
    seq = qnm.modes_cache(s=S, l=L, m=M, n=N)
    omega, A, _ = seq(a=a)
    return complex(omega), complex(A)


def test_a0_separation_constant_exact():
    """a = 0  =>  _sA = l(l+1) - s(s+1) = 4 exactly (real)."""
    sA = sep_constant(0.0, L, M, S, 0.3 - 0.09j)   # omega is irrelevant at a=0
    assert sA == complex(A0_EXACT), f"_sA(a=0) = {sA}, expected {A0_EXACT}"
    assert sA.imag == 0.0, f"_sA(a=0) not real: {sA}"
    return sA


def test_lambda_a0_ties_to_operator():
    """lambda(a=0) = _sA + a^2 w^2 - 2 a m w = 4, matching C_0(a=0) = -lambda = -4."""
    lam = teukolsky_lambda(0.0, L, M, S, 0.3 - 0.09j)
    assert lam == complex(A0_EXACT), f"lambda(a=0) = {lam}, expected {A0_EXACT}"
    return lam


def test_a09_matches_qnm_angular_eigenvalue():
    """a/M=0.9: _sA matches qnm's own angular eigenvalue to < 1e-6."""
    omega, A_qnm = _qnm_seq_at(0.9)
    sA = sep_constant(0.9, L, M, S, omega)
    resid = abs(sA - A_qnm)
    assert resid < 1e-6, f"_sA(0.9) = {sA} vs qnm {A_qnm}, |resid| = {resid:.2e}"
    return resid


def test_truncation_converged():
    """The eigenvalue is converged in the matrix truncation l_max (not an artifact)."""
    omega, _ = _qnm_seq_at(0.9)
    v_lo = sep_constant(0.9, L, M, S, omega, l_max=L + 6)
    v_hi = sep_constant(0.9, L, M, S, omega, l_max=L + 24)
    diff = abs(v_lo - v_hi)
    assert diff < 1e-10, f"l_max not converged: |v(l+6) - v(l+24)| = {diff:.2e}"
    return diff


def test_real_c_gives_real_eigenvalue():
    """Real oblateness c = a*omega yields a real spheroidal eigenvalue."""
    sA = sep_constant(0.9, L, M, S, 0.5 + 0.0j)    # real omega => real c
    assert abs(sA.imag) < 1e-12, f"real c gave complex eigenvalue: {sA}"
    return sA.real


def test_small_c_analytic_slope():
    """Small-c slope -> analytic leading coefficient -2 m s^2 / [l(l+1)] = -8/3."""
    analytic = -2.0 * M * S ** 2 / (L * (L + 1))   # = -8/3
    c = 1e-5
    sA = sep_constant(1.0, L, M, S, c + 0.0j)      # a=1 so c is real = 1e-5
    slope = (sA - A0_EXACT) / c
    assert abs(slope.imag) < 1e-8, f"slope not real: {slope}"
    err = abs(slope.real - analytic)
    assert err < 1e-4, f"small-c slope {slope.real:.6f} vs analytic {analytic:.6f}"
    return slope.real, analytic


def main():
    tests = [
        ("a=0 separation constant = 4 exact ", test_a0_separation_constant_exact),
        ("lambda(a=0) = 4 (ties to C_0=-4)  ", test_lambda_a0_ties_to_operator),
        ("a=0.9 matches qnm to 1e-6         ", test_a09_matches_qnm_angular_eigenvalue),
        ("l_max truncation converged        ", test_truncation_converged),
        ("real c -> real eigenvalue         ", test_real_c_gives_real_eigenvalue),
        ("small-c analytic slope = -8/3     ", test_small_c_analytic_slope),
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

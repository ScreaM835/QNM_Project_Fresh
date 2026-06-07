"""B.4 acceptance: complex minimal-gauge Teukolsky operator.

Acceptance (notes/phase_b_plan.md, B.4):
  1. All coefficients finite (no NaN/Inf) at sigma in {1e-8, 1-1e-8} for
     a/M in {0, 0.5, 0.9, 0.95}.
  2. At a=0 the characteristic-structure arrays (lambda_pm, mu_pm, mu_pm',
     inverse map) equal those of build_minimal_gauge_op to 1e-12.
  3. The a=0 source/potential arrays equal the Bardeen-Press reduction
     (derivation eq 11), NOT the Regge-Wheeler source.

Supporting rigour:
  4. Transcription check: every numpy coefficient array matches the exact
     symbolic closed form (lambdified from scripts/derive_first_order_system)
     to ~1e-12 at all four spins -- catches paste errors in the spin-dependent
     (r_minus, beta) terms that a=0 alone cannot see.
  5. RHS wiring: at a=0, rhs_teuk on identical real (Psi,U,W) reproduces the
     validated Phase A rhs_min to ~1e-11 (the whole right-hand side, not just
     the coefficient arrays).
"""
import sys
from pathlib import Path

import numpy as np
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.fd_stencils import d1_central
from src.qnm_kerr_reference import kerr_qnm
from src.rwz_minimal_gauge import build_minimal_gauge_op, rhs_min
from src.teukolsky_minimal_gauge import build_teukolsky_op, rhs_teuk
from derive_first_order_system import closed_forms

ELL, MM = 2, 1.0
SPINS = (0.0, 0.5, 0.9, 0.95)


def _omega(a):
    """Reference M*omega for the (2,2,0) mode from qnm, used to freeze lam."""
    q = kerr_qnm(a, ELL, 2, 0)
    return complex(q.M_omega_R, q.M_omega_I)


OMEGA_REF = {a: _omega(a) for a in SPINS}


def test_all_coefficients_finite():
    """(1) No NaN/Inf at either endpoint inset for every spin."""
    bad = []
    for a in SPINS:
        op = build_teukolsky_op(401, a, MM, ELL, 2, omega_ref=OMEGA_REF[a])
        arrays = {
            "lambda_out": op.lambda_out, "lambda_in": op.lambda_in,
            "mu_plus": op.mu_plus, "mu_minus": op.mu_minus,
            "mu_plus_d": op.mu_plus_d, "mu_minus_d": op.mu_minus_d,
            "inv_dmu": op.inv_dmu, "c_Pi": op.c_Pi, "c_Phi": op.c_Phi,
            "c_Psi": op.c_Psi,
        }
        for name, arr in arrays.items():
            if not np.all(np.isfinite(arr)):
                bad.append((a, name))
    assert not bad, f"non-finite coefficients: {bad}"
    return "all finite"


def test_a0_characteristic_matches_phase_a():
    """(2) a=0 characteristic structure == build_minimal_gauge_op to 1e-12."""
    N = 401
    k = build_teukolsky_op(N, 0.0, MM, ELL, 2, omega_ref=OMEGA_REF[0.0])
    a = build_minimal_gauge_op(N, MM, ELL)
    worst = 0.0
    for name in ("lambda_out", "lambda_in", "mu_plus", "mu_minus",
                 "mu_plus_d", "mu_minus_d", "inv_dmu",
                 "one_minus_sigma2", "sigma2"):
        d = float(np.max(np.abs(getattr(k, name) - getattr(a, name))))
        worst = max(worst, d)
        assert d < 1e-12, f"{name}: max|kerr-phaseA| = {d:.2e}"
    return worst


def test_a0_source_is_bardeen_press_not_rw():
    """(3) a=0 source == Bardeen-Press (eq 11), and differs from Regge-Wheeler."""
    N = 401
    k = build_teukolsky_op(N, 0.0, MM, ELL, 2, omega_ref=OMEGA_REF[0.0])
    sg = k.sigma
    one_plus = 1.0 + sg
    bp_c_Pi = -1.0 / (2.0 * one_plus)
    bp_c_Phi = -sg * (sg + 2.0) / (16.0 * one_plus)
    bp_c_Psi = (sg - 4.0) / (16.0 * one_plus)
    # imaginary parts must be exactly zero at a=0 (beta=0)
    assert np.max(np.abs(k.c_Pi.imag)) < 1e-14, "c_Pi not real at a=0"
    assert np.max(np.abs(k.c_Phi.imag)) < 1e-14, "c_Phi not real at a=0"
    assert np.max(np.abs(k.c_Psi.imag)) < 1e-14, "c_Psi not real at a=0"
    ePi = float(np.max(np.abs(k.c_Pi.real - bp_c_Pi)))
    ePhi = float(np.max(np.abs(k.c_Phi.real - bp_c_Phi)))
    ePsi = float(np.max(np.abs(k.c_Psi.real - bp_c_Psi)))
    assert ePi < 1e-12 and ePhi < 1e-12 and ePsi < 1e-12, \
        f"a=0 source != Bardeen-Press: {ePi:.1e},{ePhi:.1e},{ePsi:.1e}"
    # confirm it is NOT the RW source (must differ by O(1))
    rw_c_Pi = -sg / (2.0 * one_plus)
    assert float(np.max(np.abs(k.c_Pi.real - rw_c_Pi))) > 0.1, \
        "a=0 c_Pi coincides with RW (should be Bardeen-Press)"
    return ePi, ePhi, ePsi


def test_transcription_matches_symbolic():
    """(4) numpy arrays == exact symbolic closed forms (lambdify) at all spins."""
    cf = closed_forms()
    syms = sp.symbols("sigma r_plus r_minus beta lambda_sep")
    funcs = {name: sp.lambdify(syms, expr, "numpy") for name, expr in cf.items()}
    worst = 0.0
    for a in SPINS:
        op = build_teukolsky_op(401, a, MM, ELL, 2, omega_ref=OMEGA_REF[a])
        rp, rm = op.r_plus, op.r_minus
        bt, lm = op.beta, op.lam
        sg = op.sigma
        targets = {
            "lambda_out": op.lambda_out, "lambda_in": op.lambda_in,
            "mu_plus": op.mu_plus, "mu_minus": op.mu_minus,
            "mu_plus_d": op.mu_plus_d, "mu_minus_d": op.mu_minus_d,
            "inv_dmu": op.inv_dmu,
            "c_Pi": op.c_Pi, "c_Phi": op.c_Phi, "c_Psi": op.c_Psi,
        }
        for name, arr in targets.items():
            ref = funcs[name](sg, rp, rm, bt, lm)
            ref = np.asarray(ref) * np.ones_like(sg)   # broadcast constants
            d = float(np.max(np.abs(np.asarray(arr, dtype=np.complex128) - ref)))
            worst = max(worst, d)
            assert d < 1e-11, f"a={a} {name}: max|numpy-symbolic| = {d:.2e}"
    return worst


def test_rhs_wiring_matches_phase_a_algebra():
    """(5) rhs_teuk wires the characteristic arrays the SAME way as the
    validated Phase A rhs_min. We force the source coefficients equal (the only
    physical difference at a=0 is Bardeen-Press vs Regge-Wheeler source), so the
    comparison isolates the principal-part wiring (lambda_pm, mu_pm', inverse
    map). A swapped lambda_in/lambda_out or mu' would fail here."""
    N = 257
    k = build_teukolsky_op(N, 0.0, MM, ELL, 2, omega_ref=OMEGA_REF[0.0])
    a = build_minimal_gauge_op(N, MM, ELL)
    # overwrite Phase A (RW) source with the Kerr (Bardeen-Press) source so only
    # the wiring of the identical characteristic arrays is under test
    a.c_Pi[:] = k.c_Pi.real
    a.c_Phi[:] = k.c_Phi.real
    a.c_Psi[:] = k.c_Psi.real
    rng = np.random.default_rng(0)
    Psi = rng.standard_normal(N)
    U = rng.standard_normal(N)
    W = rng.standard_normal(N)
    dk = rhs_teuk((Psi.astype(np.complex128), U.astype(np.complex128),
                   W.astype(np.complex128)), k, d1_central)
    da = rhs_min((Psi.copy(), U.copy(), W.copy()), a, d1_central)
    worst = 0.0
    for xk, xa, nm in zip(dk, da, ("dPsi", "dU", "dW")):
        assert np.max(np.abs(np.asarray(xk).imag)) < 1e-12, f"{nm} gained imag part"
        d = float(np.max(np.abs(np.asarray(xk).real - xa)))
        worst = max(worst, d)
        assert d < 1e-11, f"{nm}: max|kerr-phaseA| = {d:.2e}"
    return worst


def main():
    tests = [
        ("all coefficients finite           ", test_all_coefficients_finite),
        ("a=0 chars == Phase A (1e-12)       ", test_a0_characteristic_matches_phase_a),
        ("a=0 source == Bardeen-Press        ", test_a0_source_is_bardeen_press_not_rw),
        ("transcription == symbolic (4 spins)", test_transcription_matches_symbolic),
        ("rhs wiring == Phase A algebra      ", test_rhs_wiring_matches_phase_a_algebra),
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

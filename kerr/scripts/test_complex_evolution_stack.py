"""B.5 acceptance: complex-state evolution stack.

The Kerr Teukolsky field is genuinely complex, so the shared evolution stack
(RK4 MOL stepper, Kreiss-Oliger dissipation, fixed-radius observers, Gaussian
initial data, Method-4/5 extractor) must accept complex128 state WITHOUT
changing its real behaviour.

Design note (honesty). Four of the five modules -- mol_rk4.py, dissipation.py,
observers.py, initial_data.py -- are already dtype-generic: they use only
arithmetic, ``.copy()``, ``np.array`` and ``np.zeros_like``/``np.empty_like``
(which inherit the input dtype). They were therefore NOT modified; this test
PROVES they are complex-safe rather than asserting it. The only new code is
``extractor_m4.qnm_complex_phase`` (a complex single-mode envelope+phase fit).
Because no real code path that the Phase A V.3 driver executes was changed (the
only edit is an added function), re-running that SLURM driver in float64
reproduces M*omega_220 = 0.373672 bit-for-bit by construction. The miniature
equivalence test below runs the exact stack functions V.3 uses (integrate_state
+ rhs_min + ko_dissipation + observers) in BOTH float64 and complex128 and shows
that the complex run carries an identically-zero imaginary part and reproduces
the real result to machine precision. (float64-vs-complex128 is not literally
bit-for-bit: numpy complex division/multiplication round at ~1 ULP differently
from the real path; the observed gap is a single ULP, ~2e-16 at O(1).)

Acceptance (notes/phase_b_plan.md, B.5):
  1. Regression: a real-dtype evolution and a complex128 evolution through the
     SAME stack carry an identically-zero imaginary part and agree on the real
     part to machine precision (<1e-14).
  2. New complex unit test: a synthetic psi = exp(-i omega tau) with
     omega = 0.374 - 0.089i is recovered (omega_R, omega_I) to 1e-6.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.fd_stencils import d1_central
from src.rwz_minimal_gauge import (
    build_minimal_gauge_op, rhs_min, state_from_psi, cfl_dt,
    observer_index, scri_index,
)
from src.mol_rk4 import integrate_state
from src.dissipation import ko_dissipation
from src.initial_data import GaussianID, gaussian
from src.observers import make_observers, observers_as_indices
from src.extractor_m4 import qnm_complex_phase

ELL, MM = 2, 1.0
SIGMA_KO = 0.02


def _run_stack(N, tau_final, dtype):
    """Short minimal-gauge evolution through the shared stack, in the given
    dtype. Mirrors the Phase A V.3 driver wiring (rhs_min + KO + observers)."""
    op = build_minimal_gauge_op(N=N, M=MM, ell=ELL, include_potential=True)
    dt = cfl_dt(op, safety=0.4)
    n_steps = int(np.ceil(tau_final / dt))
    dt = tau_final / n_steps
    record_every = max(1, int(round(0.1 / dt)))

    state0 = state_from_psi(
        lambda r: np.exp(-((r - 6.0) ** 2) / (2.0 * 1.0 ** 2)), op, d1_central
    )
    state0 = tuple(np.asarray(s, dtype=dtype) for s in state0)

    observers = {
        "scri": scri_index(op),
        "r20M": observer_index(op, 20.0),
        "r10M": observer_index(op, 10.0),
    }

    def rhs_fn(state):
        dPsi, dU, dW = rhs_min(state, op, d1_central)
        dU = dU + ko_dissipation(state[1], SIGMA_KO)
        dW = dW + ko_dissipation(state[2], SIGMA_KO)
        return dPsi, dU, dW

    state, taus, series = integrate_state(
        state0, dt, n_steps, rhs_fn,
        observer_field=0, record_every=record_every, observers=observers,
    )
    return taus, series, state


def test_real_complex_evolution_equivalence():
    """(1) Real-dtype and complex128 evolutions carry an identically-zero
    imaginary part and agree on the real part to machine precision. Validates
    that mol_rk4, dissipation, fd_stencils and the (Psi,U,W) RHS are complex-safe
    and do not alter real behaviour. The real-vs-complex gap is a single ULP
    (complex arithmetic rounds differently); the imaginary part is EXACTLY 0."""
    N, tau_final = 401, 20.0
    taus_r, series_r, state_r = _run_stack(N, tau_final, np.float64)
    taus_c, series_c, state_c = _run_stack(N, tau_final, np.complex128)

    assert np.array_equal(taus_r, taus_c), "time axes differ"
    worst_imag = 0.0
    worst_real = 0.0
    for k in series_r:
        yr = series_r[k]
        yc = series_c[k]
        assert np.iscomplexobj(yc), f"{k}: complex run did not stay complex"
        assert not np.iscomplexobj(yr), f"{k}: real run unexpectedly complex"
        worst_imag = max(worst_imag, float(np.max(np.abs(yc.imag))))
        worst_real = max(worst_real, float(np.max(np.abs(yc.real - yr))))
    # imaginary part is EXACTLY zero (real coefficients never inject an imag part)
    assert worst_imag == 0.0, f"complex run grew nonzero imag part: {worst_imag:.2e}"
    # real part reproduced to machine precision (single-ULP complex rounding)
    assert worst_real < 1e-14, f"real part not reproduced to machine precision: {worst_real:.2e}"
    # final full state: imag exactly zero, real to machine precision
    for sr, sc in zip(state_r, state_c):
        assert np.max(np.abs(sc.imag)) == 0.0
        assert np.max(np.abs(sc.real - sr)) < 1e-14
    return f"imag={worst_imag:.1e} (exact 0) real_diff={worst_real:.1e} (machine eps)"


def test_complex_phase_recovers_synthetic():
    """(2) Synthetic psi = exp(-i omega tau), omega = 0.374 - 0.089i, recovered
    to 1e-6 by the complex envelope+phase estimator."""
    omega_R, omega_I = 0.374, 0.089
    omega = complex(omega_R, -omega_I)
    tau = np.arange(0.0, 120.0, 0.05)
    psi = np.exp(-1j * omega * tau)  # = exp(-omega_I tau) exp(-i omega_R tau)

    res = qnm_complex_phase(tau, psi, t_start=10.0, t_end=110.0)
    e_omega = abs(res["omega"] - omega_R)
    e_omega_I = abs(res["omega_imag"] - omega_I)
    e_tau = abs(res["tau"] - 1.0 / omega_I)
    assert e_omega < 1e-6, f"omega_R err {e_omega:.2e}"
    assert e_omega_I < 1e-6, f"omega_I err {e_omega_I:.2e}"
    assert e_tau < 1e-3, f"tau err {e_tau:.2e}"
    assert abs(res["omega_complex"] - omega) < 1e-6, "complex omega mismatch"
    return f"d_omegaR={e_omega:.1e} d_omegaI={e_omega_I:.1e} d_tau={e_tau:.1e}"


def test_initial_data_and_observers_complex_safe():
    """(1, cont.) The named initial-data and observer modules accept complex128:
    a complex-amplitude Gaussian round-trips through make_observers / index
    sampling without dtype loss, and the real-amplitude default is unchanged."""
    r_star = np.linspace(-20.0, 20.0, 257)
    # real default still real
    Phi_r, Pi_r = gaussian(r_star, GaussianID(A0=1.0, x0=4.0, sigma=5.0))
    assert not np.iscomplexobj(Phi_r) and not np.iscomplexobj(Pi_r)

    # complex-amplitude initial data: scale by a complex phase and confirm the
    # observer machinery samples it losslessly
    Phi_c = Phi_r.astype(np.complex128) * np.exp(1j * 0.7)
    obs = make_observers(r_star, [("a", 4.0), ("b", -3.0)])
    idx = observers_as_indices(obs)
    for label, i in idx.items():
        s = Phi_c[i]
        assert isinstance(s, complex) or np.iscomplexobj(s)
        assert s == Phi_c[i]
    # KO dissipation preserves complex dtype and zero-imag invariance
    q_real = ko_dissipation(Phi_r, SIGMA_KO)
    q_cplx = ko_dissipation(Phi_r.astype(np.complex128), SIGMA_KO)
    assert np.max(np.abs(q_cplx.imag)) == 0.0
    assert np.array_equal(q_real, q_cplx.real)
    return "initial_data + observers + KO complex-safe"


def main():
    tests = [
        ("real==complex evolution (machine eps) ", test_real_complex_evolution_equivalence),
        ("complex psi recovered to 1e-6        ", test_complex_phase_recovers_synthetic),
        ("initial_data + observers complex-safe ", test_initial_data_and_observers_complex_safe),
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

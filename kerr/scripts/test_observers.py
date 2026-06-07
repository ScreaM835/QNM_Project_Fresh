"""A.7 acceptance: observer placement + time-series recording."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.observers import make_observers, observers_as_indices
from src.initial_data import GaussianID, gaussian
from src.mol_rk4 import integrate, cfl_dt
from src.rwz_hyperboloidal import build_operator, rhs
from src.fd_stencils import d1_central, d2_central


def test_nearest_index_picks_closest():
    r_star = np.linspace(-10.0, 20.0, 301)  # dx = 0.1
    obs = make_observers(r_star, [("inner", 2.0), ("outer", 10.0), ("offgrid", 3.07)])
    assert abs(obs["inner"].r_actual_M - 2.0) < 1e-6, obs["inner"]
    assert abs(obs["outer"].r_actual_M - 10.0) < 1e-6, obs["outer"]
    assert abs(obs["offgrid"].r_actual_M - 3.1) < 0.05, obs["offgrid"]
    return obs["inner"].r_actual_M, obs["outer"].r_actual_M, obs["offgrid"].r_actual_M


def test_observer_records_initial_condition():
    """At tau=0 the recorded series first value equals Phi0 at that index."""
    r_star = np.linspace(-15.0, 30.0, 301)
    op = build_operator(r_star, M=1.0, L=10.0, ell=2)
    Phi0, Pi0 = gaussian(r_star, GaussianID(A0=1.0, x0=4.0, sigma=5.0))

    obs = make_observers(r_star, [("r2", 2.0), ("r10", 10.0)])
    obs_idx = observers_as_indices(obs)

    def rhs_fn(Phi, Pi):
        return rhs(Phi, Pi, op, d1_central, d2_central)

    dt = cfl_dt(r_star[1] - r_star[0], float(np.max(np.abs(op.H))), safety=0.4)
    Phi_end, Pi_end, taus, series = integrate(Phi0, Pi0, dt, n_steps=200, rhs_fn=rhs_fn,
                                              record_every=10, observers=obs_idx)
    # first sample should be Phi0 at the observer index
    for k, idx in obs_idx.items():
        assert abs(series[k][0] - Phi0[idx]) < 1e-14, f"{k} t=0 mismatch"
    # sampling cadence
    assert taus[0] == 0.0
    assert abs(taus[1] - 10.0 * dt) < 1e-12, taus[:3]
    return len(taus), float(taus[-1])


def test_observer_records_evolve_in_time():
    """After many steps, the observer time series is non-constant."""
    r_star = np.linspace(-15.0, 30.0, 301)
    op = build_operator(r_star, M=1.0, L=10.0, ell=2)
    Phi0, Pi0 = gaussian(r_star, GaussianID(A0=1.0, x0=4.0, sigma=5.0))

    obs_idx = observers_as_indices(make_observers(r_star, [("r10", 10.0)]))

    def rhs_fn(Phi, Pi):
        return rhs(Phi, Pi, op, d1_central, d2_central)

    dt = cfl_dt(r_star[1] - r_star[0], float(np.max(np.abs(op.H))), safety=0.4)
    _, _, _, series = integrate(Phi0, Pi0, dt, n_steps=1500, rhs_fn=rhs_fn,
                                record_every=50, observers=obs_idx)
    s = series["r10"]
    variation = float(np.std(s))
    assert variation > 1e-6, f"observer series did not evolve: std={variation}"
    return variation


def main():
    tests = [
        ("nearest-index observer placement", test_nearest_index_picks_closest),
        ("observer records IC at tau=0    ", test_observer_records_initial_condition),
        ("observer series evolves         ", test_observer_records_evolve_in_time),
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

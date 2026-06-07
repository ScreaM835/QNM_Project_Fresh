"""Classical RK4 time stepper for the (Phi, Pi) MOL system.

The hyperboloidal RWZ system has principal symbol matrix

    [ 0         1                          ]
    [ d_x^2/(1-H^2)   -2 H d_x/(1-H^2)     ]

with characteristic speeds c_+ = 1/(1+H), c_- = -1/(1-H). On the bulk the
larger speed in magnitude is max(|c_+|, |c_-|) <= 1/(1 - |H|_max). For the
chosen gauge H = tanh(r_*/L) and an interior cap |H| <= H_cap < 1 the CFL
bound for RK4 on a 2nd-order spatial stencil is approximately

    dt <= C_cfl * dx * (1 - H_cap),     C_cfl ~ 1.0

We expose a helper that returns this advisory CFL dt; callers may pick a
smaller step.
"""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


def cfl_dt(dx: float, H_max: float, safety: float = 0.5) -> float:
    """Advisory dt for RK4 + 2nd-order central differences."""
    return safety * dx * (1.0 - min(abs(H_max), 0.9999))


def rk4_step(
    Phi: np.ndarray,
    Pi: np.ndarray,
    dt: float,
    rhs_fn: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """One classical RK4 step of (Phi, Pi)."""
    k1_Phi, k1_Pi = rhs_fn(Phi, Pi)
    k2_Phi, k2_Pi = rhs_fn(Phi + 0.5 * dt * k1_Phi, Pi + 0.5 * dt * k1_Pi)
    k3_Phi, k3_Pi = rhs_fn(Phi + 0.5 * dt * k2_Phi, Pi + 0.5 * dt * k2_Pi)
    k4_Phi, k4_Pi = rhs_fn(Phi + dt * k3_Phi, Pi + dt * k3_Pi)
    new_Phi = Phi + (dt / 6.0) * (k1_Phi + 2.0 * k2_Phi + 2.0 * k3_Phi + k4_Phi)
    new_Pi = Pi + (dt / 6.0) * (k1_Pi + 2.0 * k2_Pi + 2.0 * k3_Pi + k4_Pi)
    return new_Phi, new_Pi


def integrate(
    Phi0: np.ndarray,
    Pi0: np.ndarray,
    dt: float,
    n_steps: int,
    rhs_fn: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
    record_every: int = 1,
    observers: dict | None = None,
):
    """Time-march and optionally record observer time series.

    observers: dict of label -> int index into the spatial grid;
               returns dict of label -> (taus, Phi(tau, x_obs)).
    """
    Phi, Pi = Phi0.copy(), Pi0.copy()
    obs_keys = list(observers.keys()) if observers else []
    rec_tau = [0.0]
    rec_Phi = {k: [Phi0[observers[k]]] for k in obs_keys}
    for n in range(1, n_steps + 1):
        Phi, Pi = rk4_step(Phi, Pi, dt, rhs_fn)
        if n % record_every == 0:
            rec_tau.append(n * dt)
            for k in obs_keys:
                rec_Phi[k].append(Phi[observers[k]])
    return Phi, Pi, np.array(rec_tau), {k: np.array(v) for k, v in rec_Phi.items()}


# ----------------------------------------------------------------------
# Generic tuple-state RK4. Used by the (Psi, U, W) characteristic-form
# minimal-gauge system in rwz_minimal_gauge.py. The rhs_fn here takes and
# returns a tuple of arrays of identical shape.
# ----------------------------------------------------------------------


def _axpy_tuple(state, k, dt):
    return tuple(s + dt * ki for s, ki in zip(state, k))


def rk4_step_state(state, dt, rhs_fn):
    """One classical RK4 step on an arbitrary-arity tuple state."""
    k1 = rhs_fn(state)
    k2 = rhs_fn(_axpy_tuple(state, k1, 0.5 * dt))
    k3 = rhs_fn(_axpy_tuple(state, k2, 0.5 * dt))
    k4 = rhs_fn(_axpy_tuple(state, k3, dt))
    return tuple(
        s + (dt / 6.0) * (a + 2.0 * b + 2.0 * c + d)
        for s, a, b, c, d in zip(state, k1, k2, k3, k4)
    )


def integrate_state(
    state0,
    dt: float,
    n_steps: int,
    rhs_fn,
    observer_field: int = 0,
    record_every: int = 1,
    observers: dict | None = None,
    recorder=None,
):
    """Time-march a tuple-state system and record fields at observers.

    By default records state[observer_field] at the observer indices.
    If `recorder` is given it must map a state tuple to a single array of the
    spatial shape; that derived array is sampled at the observers instead
    (used to record Pi = (1-sigma^2) U + sigma^2 W).
    """
    state = tuple(s.copy() for s in state0)
    obs_keys = list(observers.keys()) if observers else []

    def sample(st):
        return recorder(st) if recorder is not None else st[observer_field]

    rec_tau = [0.0]
    field0 = sample(state)
    rec = {k: [field0[observers[k]]] for k in obs_keys}
    for n in range(1, n_steps + 1):
        state = rk4_step_state(state, dt, rhs_fn)
        if n % record_every == 0:
            rec_tau.append(n * dt)
            f = sample(state)
            for k in obs_keys:
                rec[k].append(f[observers[k]])
    return state, np.array(rec_tau), {k: np.array(v) for k, v in rec.items()}

"""Observer extraction at fixed tortoise-coordinate locations.

The integrator's `observers` arg takes integer grid indices. A.7 wraps that
with a "pick nearest grid index for desired r_obs (in tortoise coords)"
helper, and provides a small dataclass to carry the resulting time series.

Note on time variables. The integrator advances hyperboloidal time tau.
For a stationary observer at fixed r_*, the parent-paper time t and tau
are related by t = tau + h(r_*), so the recorded series is

    Phi(tau, r_obs) plotted against tau

If you want to compare directly against a Cauchy-time t series, add
h(r(r_obs)) to the tau axis. For QNM ringdown the frequency M*omega and
damping time tau_qnm are gauge-invariant, so extracting from tau is fine.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Observer:
    label: str
    r_obs_M: float  # desired r_* in units of M
    grid_index: int  # nearest grid index
    r_actual_M: float  # actual r_* at that grid index


def make_observers(r_star: np.ndarray, r_obs_list_M: list[tuple[str, float]]) -> dict[str, Observer]:
    """For each (label, r_obs in units of M), pick the nearest grid index."""
    out: dict[str, Observer] = {}
    for label, r_obs_M in r_obs_list_M:
        idx = int(np.argmin(np.abs(r_star - r_obs_M)))
        out[label] = Observer(label=label, r_obs_M=r_obs_M, grid_index=idx, r_actual_M=float(r_star[idx]))
    return out


def observers_as_indices(observers: dict[str, Observer]) -> dict[str, int]:
    """Convert to the {label: grid_index} dict the integrator expects."""
    return {k: v.grid_index for k, v in observers.items()}

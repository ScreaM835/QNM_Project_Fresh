"""Initial data for the hyperboloidal (Phi, Pi) MOL system.

Default initial data: time-symmetric Gaussian in r_*,

    Phi(0, r_*) = A0 * exp( - (r_* - x0)^2 / (2 sigma^2) )
    Pi (0, r_*) = 0

Matches the parent-paper convention so Schwarzschild QNMs compare directly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GaussianID:
    A0: float = 1.0
    x0: float = 4.0
    sigma: float = 5.0


def gaussian(r_star: np.ndarray, params: GaussianID) -> tuple[np.ndarray, np.ndarray]:
    Phi = params.A0 * np.exp(-((r_star - params.x0) ** 2) / (2.0 * params.sigma ** 2))
    Pi = np.zeros_like(r_star)
    return Phi, Pi

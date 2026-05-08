"""
PINN for the 1+1D Zerilli/Regge-Wheeler equation using DeepXDE.

PDE:  phi_tt - phi_xx + V(x*) phi = 0

Migrated from custom PyTorch to the DeepXDE framework.
Key improvements over the previous implementation:
  - Proper L-BFGS via PyTorch LBFGS with correct iteration management
    (DeepXDE handles the optimizer loop, no more max_iter=1 hack)
  - Exact autograd through V(x*) via pure-torch Lambert-W potential
    (no manual dV/dx correction needed for gradient-enhanced residuals)
  - Framework-standard IC/BC handling and collocation-point resampling
"""

from __future__ import annotations

import copy
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import deepxde as dde
import numpy as np
import torch

from .potentials import V_of_x_torch
from .initial_data import gaussian_phi, gaussian_phi_t


# ---------------------------------------------------------------------------
# Causal training (Wang, Perdikaris & Sifakis 2022, arXiv:2203.07404)
# ---------------------------------------------------------------------------

class CausalWeighter:
    """Apply causal weighting to PDE residuals (Wang et al. 2022).

    Divides the time domain into slices and weights residuals so that
    later times are penalised unless earlier times are already well-resolved.

        w_k = exp(-epsilon * sum_{j<k} L_j)

    where L_j is the mean squared PDE residual in time slice j.

    Light-cone masking (Patel et al. 2024): when computing L_j, only
    points inside the past light cone of the initial data are included.
    This prevents causally-empty regions (where psi=0 trivially) from
    diluting the per-slice loss estimate.  c=1 in tortoise coordinates,
    so the light cone is |x* - x0| <= t + 3*sigma.

    Residuals are multiplied by sqrt(w_k) so that MSE(sqrt(w)*r) = w*r**2,
    giving the causally weighted loss without modifying DeepXDE's loss
    pipeline.  The weights are detached so gradients flow only through the
    residuals, not through the weights themselves.
    """

    def __init__(
        self,
        tmin: float,
        tmax: float,
        epsilon: float = 10.0,
        num_slices: int = 20,
        x0: float = 0.0,
        sigma: float = 5.0,
        epsilon_max: Optional[float] = None,
        ramp_steps: int = 0,
    ):
        self.tmin = tmin
        self.tmax = tmax
        self.num_slices = num_slices
        self.w_min = 1.0  # min causal weight (approaches 1 as training converges)
        self.enabled = True  # can be disabled for L-BFGS phase

        # Epsilon schedule: exponential ramp from epsilon → epsilon_max
        # over ramp_steps.  If epsilon_max is None or ramp_steps <= 0,
        # epsilon stays fixed.
        self.epsilon_min = epsilon
        self.epsilon_max = epsilon_max if epsilon_max is not None else epsilon
        self.ramp_steps = max(ramp_steps, 0)
        self.epsilon = epsilon  # current value
        self._step = 0

        # Light-cone parameters: signal from Gaussian at x0 with width sigma
        # reaches x* when |x* - x0| <= t + 3*sigma  (c=1 in tortoise coords)
        self.x0 = x0
        self.sigma = sigma

    def step_epsilon(self):
        """Advance the epsilon schedule by one callback period."""
        if self.ramp_steps <= 0 or self.epsilon_min >= self.epsilon_max:
            return
        self._step += 1
        # Exponential interpolation: eps = eps_min * (eps_max/eps_min)^(t/T)
        frac = min(self._step / self.ramp_steps, 1.0)
        import math
        self.epsilon = self.epsilon_min * math.exp(
            frac * math.log(self.epsilon_max / self.epsilon_min)
        )

    def apply(self, x: torch.Tensor, *residuals: torch.Tensor) -> list:
        """Weight residuals by causal factor.

        Parameters
        ----------
        x : Tensor (N, 2)
            Input coordinates [x*, t].
        *residuals : Tensor (N, 1) each
            PDE residual components [r, r_x, r_t].

        Returns
        -------
        list of Tensor
            Weighted residuals, same shapes as inputs.
        """
        # When disabled, return residuals unchanged
        if not self.enabled:
            return list(residuals)

        xs = x[:, 0:1].detach()  # (N, 1)
        t = x[:, 1:2].detach()   # (N, 1)

        # Assign each point to a time slice
        dt = (self.tmax - self.tmin) / self.num_slices
        slice_idx = ((t - self.tmin) / dt).long().clamp(
            0, self.num_slices - 1
        ).squeeze(-1)  # (N,)

        # Light-cone mask: point is causally active if |x* - x0| <= t + 3*sigma
        in_light_cone = (
            torch.abs(xs - self.x0) <= t + 3.0 * self.sigma
        ).squeeze(-1)  # (N,)

        # Per-slice mean squared PDE residual (primary residual only)
        # Only include causally-active points to avoid dilution
        r_det = residuals[0].detach()  # (N, 1)
        per_slice_loss = torch.zeros(
            self.num_slices, device=r_det.device, dtype=r_det.dtype
        )
        for k in range(self.num_slices):
            mask = (slice_idx == k) & in_light_cone
            if mask.any():
                vals = r_det[mask] ** 2
                vals = torch.nan_to_num(vals, nan=0.0)
                per_slice_loss[k] = vals.mean()

        # Causal weights: w_k = exp(-epsilon * sum_{j<k} L_j)
        # w_0 = 1 (no prior losses), w_1 = exp(-eps*L_0), ...
        cumulative = torch.cumsum(per_slice_loss, dim=0)
        shifted = torch.cat(
            [torch.zeros(1, device=r_det.device, dtype=r_det.dtype),
             cumulative[:-1]]
        )
        w = torch.exp(-self.epsilon * shifted)

        # Track minimum weight for monitoring convergence
        self.w_min = w.min().item()

        # Map slice weights back to individual points
        # (weights apply to ALL points, including outside light cone)
        sqrt_w = torch.sqrt(w[slice_idx]).unsqueeze(1)  # (N, 1), detached

        return [res * sqrt_w for res in residuals]


class CausalTrainingMonitor(dde.callbacks.Callback):
    """Log the minimum causal weight for monitoring convergence."""

    def __init__(self, causal_weighter: CausalWeighter, period: int = 100):
        super().__init__()
        self.cw = causal_weighter
        self.period = period

    def on_epoch_end(self):
        step = self.model.train_state.step
        if step % self.period != 0:
            return
        # Advance epsilon schedule each monitoring period
        self.cw.step_epsilon()
        print(f"  [Causal] step {step}: epsilon={self.cw.epsilon:.2f}, "
              f"w_min={self.cw.w_min:.6f}")


# ---------------------------------------------------------------------------
# PDE residual
# ---------------------------------------------------------------------------

def _make_pde_func(cfg: Dict, tmax_override: Optional[float] = None):
    """Build the PDE residual function, closed over physics parameters.

    If causal training is enabled in the config, residuals are weighted by
    time-slice-dependent causal factors (Wang et al. 2022).
    """
    M = float(cfg["physics"]["M"])
    l = int(cfg["physics"]["l"])
    potential = cfg["physics"]["potential"]

    # --- optional causal weighting ---
    causal_cfg = cfg["pinn"].get("causal", {})
    causal_weighter = None
    if causal_cfg.get("enabled", False):
        tmin = float(cfg["domain"]["tmin"])
        tmax = tmax_override if tmax_override is not None else float(cfg["domain"]["tmax"])

        # Light-cone parameters from initial data
        x0_ic = float(cfg["initial_data"]["x0"])
        sigma_ic = float(cfg["initial_data"]["sigma"])

        eps_init = float(causal_cfg.get("epsilon", 10.0))
        eps_max = float(causal_cfg.get("epsilon_max", eps_init))
        ramp_steps = int(causal_cfg.get("ramp_steps", 0))

        causal_weighter = CausalWeighter(
            tmin=tmin,
            tmax=tmax,
            epsilon=eps_init,
            num_slices=int(causal_cfg.get("n_slices", 20)),
            x0=x0_ic,
            sigma=sigma_ic,
            epsilon_max=eps_max,
            ramp_steps=ramp_steps,
        )
        schedule_str = (
            f", schedule: {eps_init}→{eps_max} over {ramp_steps} steps"
            if ramp_steps > 0 else ""
        )
        print(f"[PINN] Causal training enabled: epsilon={eps_init}, "
              f"n_slices={causal_weighter.num_slices}, t=[{tmin},{tmax}], "
              f"light-cone: x0={x0_ic}, sigma={sigma_ic}{schedule_str}")

    def pde(x, y):
        """
        PDE residual for  phi_tt - phi_xx + V(x*) phi = 0.

        Returns [r, r_x, r_t] for gradient-enhanced training.

        Parameters
        ----------
        x : Tensor (N, 2)  -- columns are [x*, t]
        y : Tensor (N, 1)  -- network output phi
        """
        # Second derivatives via DeepXDE's cached Hessian
        phi_xx = dde.grad.hessian(y, x, i=0, j=0)
        phi_tt = dde.grad.hessian(y, x, i=1, j=1)

        # Potential -- fully inside the autograd graph (pure-torch Lambert-W)
        V = V_of_x_torch(x[:, 0:1], M, l, potential)

        # Standard PDE residual
        r = phi_tt - phi_xx + V * y

        # Gradient-enhanced residuals via autograd.
        # Because V(x) is differentiable, dr/dx automatically includes dV/dx * phi.
        dr = torch.autograd.grad(
            r, x,
            grad_outputs=torch.ones_like(r),
            create_graph=True,
            retain_graph=True,
        )[0]
        r_x = dr[:, 0:1]
        r_t = dr[:, 1:2]

        # Apply causal weighting if enabled
        if causal_weighter is not None:
            r, r_x, r_t = causal_weighter.apply(x, r, r_x, r_t)

        return [r, r_x, r_t]

    # Attach the causal weighter so callers can access it (e.g. for monitoring)
    pde._causal_weighter = causal_weighter
    return pde


def _make_pde_residual_only(cfg: Dict):
    """Build a lightweight PDE function returning only the primary residual r.

    Used by ResidualAdaptiveResampler (RAD/RAR-D) for evaluating candidate
    points.  Skips the gradient-enhanced terms (r_x, r_t) to avoid the
    expensive third-order autograd.
    """
    M = float(cfg["physics"]["M"])
    l = int(cfg["physics"]["l"])
    potential = cfg["physics"]["potential"]

    def pde_residual(x, y):
        phi_xx = dde.grad.hessian(y, x, i=0, j=0)
        phi_tt = dde.grad.hessian(y, x, i=1, j=1)
        V = V_of_x_torch(x[:, 0:1], M, l, potential)
        return phi_tt - phi_xx + V * y  # single tensor

    return pde_residual


# ---------------------------------------------------------------------------
# Residual-based Adaptive Sampling (Wu et al. 2023, arXiv:2207.10289)
# ---------------------------------------------------------------------------

class ResidualAdaptiveResampler(dde.callbacks.Callback):
    """Adaptive collocation-point resampling based on PDE residual magnitude.

    Implements both RAD and RAR-D from:
      Wu, Zhu, Tan, Kartha & Lu (2023). "A comprehensive study of
      non-adaptive and residual-based adaptive sampling for PINNs."
      Computer Methods in Applied Mechanics and Engineering, 403, 115671.

    Every ``period`` training steps:
      1. Generate ``num_candidates`` random points in the domain.
      2. Evaluate PDE residual r at each candidate (lightweight, no grad
         enhancement).
      3. Compute sampling probability  p ∝ |r|^k / mean(|r|^k) + c.
      4. RAD:   sample ``num_domain`` points → replace all domain points.
         RAR-D: sample ``num_add``    points → append to existing points.

    Parameters
    ----------
    pde_residual : callable
        Lightweight PDE function  (x, y) -> r  (single tensor, not list).
        Should be ``_make_pde_residual_only(cfg)`` — avoids computing r_x, r_t.
    period : int
        Resample every this many training steps.
    num_candidates : int
        Number of random candidate points to evaluate.
    k : float
        Power for residual weighting (0 = uniform, 1 = linear, 2 = quadratic).
        RAD default: k=1.  RAR-D default: k=0.
    c : float
        Additive constant ensuring minimum probability everywhere.
        RAD default: c=1.  RAR-D default: c=2.
    method : str
        'RAD' (replace all) or 'RAR_D' (append new).
    num_add : int
        Points to add per round (RAR-D only).
    eval_batch_size : int
        Batch size for residual evaluation (controls peak memory).
    """

    def __init__(self, pde_residual, period: int = 1000,
                 num_candidates: int = 50000, k: float = 1.0,
                 c: float = 1.0, method: str = 'RAD',
                 num_add: int = 160, eval_batch_size: int = 5000):
        super().__init__()
        self.pde_residual = pde_residual
        self.period = period
        self.num_candidates = num_candidates
        self.k = k
        self.c = c
        self.method = method
        self.num_add = num_add
        self.eval_batch_size = eval_batch_size

    def on_epoch_end(self):
        step = self.model.train_state.step
        if step == 0 or step % self.period != 0:
            return

        data = self.model.data
        geom = data.geom  # GeometryXTime

        # 1. Generate candidate points
        X_cand = geom.random_points(self.num_candidates)

        # 2. Evaluate PDE residual in batches (avoids large autograd graph)
        all_residuals = []
        for i in range(0, len(X_cand), self.eval_batch_size):
            batch = X_cand[i:i + self.eval_batch_size]
            res = self.model.predict(batch, operator=self.pde_residual)
            all_residuals.append(np.abs(res))
        Y = np.concatenate(all_residuals, axis=0).astype(np.float64)

        # 3. Compute sampling probability
        #    err_eq = |r|^k / mean(|r|^k) + c  (Wu et al. Eq. 2)
        if self.k == 0:
            Y_pow = np.ones_like(Y)
        else:
            Y_pow = np.power(Y, self.k)
        mean_Y = Y_pow.mean()
        if mean_Y < 1e-30:
            # All residuals near zero — fall back to uniform
            prob = np.ones(len(X_cand)) / len(X_cand)
        else:
            prob = (Y_pow / mean_Y + self.c).ravel()
            prob = prob / prob.sum()

        # 4. Sample and update training points
        if self.method == 'RAD':
            num_domain = data.num_domain
            do_replace = len(X_cand) >= num_domain
            ids = np.random.choice(len(X_cand), size=min(num_domain, len(X_cand)),
                                   replace=not do_replace, p=prob)
            data.replace_with_anchors(X_cand[ids])
            print(f"  [RAD] step {step}: replaced {len(ids)} pts "
                  f"(max|r|={Y.max():.4e}, mean|r|={Y.mean():.4e})")
        elif self.method == 'RAR_D':
            ids = np.random.choice(len(X_cand), size=self.num_add,
                                   replace=False, p=prob)
            data.add_anchors(X_cand[ids])
            n_total = len(data.train_x_all)
            print(f"  [RAR-D] step {step}: added {self.num_add} pts "
                  f"(total domain={n_total}, max|r|={Y.max():.4e})")
        else:
            raise ValueError(f"Unknown adaptive sampling method: {self.method}")

        # Sync train_state so other callbacks (e.g. GradientBalancing)
        # that use train_state.X_train see the updated collocation points.
        self.model.train_state.set_data_train(
            data.train_x, data.train_y,
        )

        # Sync test data so _test() uses consistent shapes
        data.test_x = data.train_x
        data.test_y = data.train_y
        self.model.train_state.set_data_test(
            data.test_x, data.test_y,
        )


class GreedyResampler(dde.callbacks.Callback):
    """Greedy collocation-point resampling by largest PDE residual."""

    def __init__(self, pde_residual, period: int = 1000,
                 num_candidates: int = 50000,
                 greedy_fraction: float = 0.5,
                 eval_batch_size: int = 5000):
        super().__init__()
        self.pde_residual = pde_residual
        self.period = period
        self.num_candidates = num_candidates
        self.greedy_fraction = greedy_fraction
        self.eval_batch_size = eval_batch_size

    def on_epoch_end(self):
        step = self.model.train_state.step
        if step == 0 or step % self.period != 0:
            return

        data = self.model.data
        geom = data.geom
        num_domain = data.num_domain

        X_cand = geom.random_points(self.num_candidates)
        residuals = []
        for i in range(0, len(X_cand), self.eval_batch_size):
            batch = X_cand[i:i + self.eval_batch_size]
            res = self.model.predict(batch, operator=self.pde_residual)
            residuals.append(np.abs(res))
        Y = np.concatenate(residuals, axis=0).ravel().astype(np.float64)

        n_greedy = int(num_domain * self.greedy_fraction)
        n_uniform = num_domain - n_greedy
        top_ids = np.argsort(Y)[-n_greedy:] if n_greedy > 0 else np.array([], dtype=int)
        X_greedy = X_cand[top_ids] if n_greedy > 0 else np.empty((0, X_cand.shape[1]))

        if n_uniform > 0:
            mask = np.ones(len(X_cand), dtype=bool)
            if n_greedy > 0:
                mask[top_ids] = False
            remaining_ids = np.where(mask)[0]
            uniform_ids = np.random.choice(
                remaining_ids,
                size=min(n_uniform, len(remaining_ids)),
                replace=False,
            )
            X_uniform = X_cand[uniform_ids]
        else:
            X_uniform = np.empty((0, X_cand.shape[1]))

        X_new = np.concatenate([X_greedy, X_uniform], axis=0)
        data.replace_with_anchors(X_new)

        greedy_min = Y[top_ids].min() if n_greedy > 0 else 0.0
        print(f"  [Greedy] step {step}: {n_greedy} greedy + {n_uniform} uniform pts "
              f"(max|r|={Y.max():.4e}, mean|r|={Y.mean():.4e}, "
              f"greedy min|r|={greedy_min:.4e})")

        self.model.train_state.set_data_train(data.train_x, data.train_y)
        data.test_x = data.train_x
        data.test_y = data.train_y
        self.model.train_state.set_data_test(data.test_x, data.test_y)


# ---------------------------------------------------------------------------
# Initial / boundary conditions
# ---------------------------------------------------------------------------

def _make_ic_bcs(cfg: Dict, geomtime):
    """Create the four IC/BC objects for the Zerilli equation."""
    A0 = float(cfg["initial_data"]["A"])
    x0_ic = float(cfg["initial_data"]["x0"])
    sigma = float(cfg["initial_data"]["sigma"])
    profile = cfg["initial_data"]["velocity_profile"]
    xmin = float(cfg["domain"]["xmin"])
    xmax = float(cfg["domain"]["xmax"])
    tmin = float(cfg["domain"]["tmin"])

    # ---- IC: phi(x, 0) = Gaussian ----
    def phi0_func(x):
        """x is numpy (N, 2). Return (N, 1)."""
        return gaussian_phi(x[:, 0:1], A=A0, x0=x0_ic, sigma=sigma)

    ic_disp = dde.icbc.IC(
        geomtime, phi0_func, lambda _, on_initial: on_initial
    )

    # Decay factoring: velocity IC becomes g_t(x,0) = Psi_t(x,0) + Psi(x,0)/tau
    df_cfg = cfg.get("pinn", {}).get("decay_factor", {})
    use_decay_factor = df_cfg.get("enabled", False)
    df_tau = float(df_cfg.get("tau", 0.0)) if use_decay_factor else 0.0

    # ---- IC: d_phi/dt(x, 0) = v0(x) ----
    #      (or d_g/dt(x, 0) = v0(x) + phi0(x)/tau for decay-factored PDE)
    def vel_func(inputs, outputs, X):
        phi_t = dde.grad.jacobian(outputs, inputs, i=0, j=1)
        # Use `inputs` (tensor) rather than `X` (numpy self.train_x) to
        # guarantee size consistency.  After PDEPointResampler fires,
        # self.train_x is regenerated but X_test may still hold old data,
        # causing a shape mismatch if we use X.
        x_np = inputs[:, 0:1].detach().cpu().numpy()
        v0 = gaussian_phi_t(
            x_np, A=A0, x0=x0_ic, sigma=sigma, profile=profile
        )
        if use_decay_factor:
            # g_t(x,0) = Psi_t(x,0) + Psi_0(x)/tau  (since e^0 = 1)
            phi0 = gaussian_phi(x_np, A=A0, x0=x0_ic, sigma=sigma)
            v0 = v0 + phi0 / df_tau
        v0_t = torch.as_tensor(v0, dtype=outputs.dtype, device=outputs.device)
        return phi_t - v0_t

    # Selects points at t = tmin from train_x_all (includes num_initial points)
    ic_vel = dde.icbc.OperatorBC(
        geomtime, vel_func,
        lambda x, on_boundary: np.isclose(x[1], tmin),
    )

    # ---- BC left: Sommerfeld ingoing at x = xmin ----
    #   Standard:       (d_t - d_x) Psi = 0
    #   Decay-factored: (d_t - d_x) g - g/tau = 0
    def bc_left_func(inputs, outputs, X):
        phi_x = dde.grad.jacobian(outputs, inputs, i=0, j=0)
        phi_t = dde.grad.jacobian(outputs, inputs, i=0, j=1)
        bc = phi_t - phi_x
        if use_decay_factor:
            bc = bc - outputs / df_tau
        return bc

    bc_left = dde.icbc.OperatorBC(
        geomtime, bc_left_func,
        lambda x, on_boundary: on_boundary and np.isclose(x[0], xmin),
    )

    # ---- BC right: Sommerfeld outgoing at x = xmax ----
    #   Standard:       (d_t + d_x) Psi = 0
    #   Decay-factored: (d_t + d_x) g - g/tau = 0
    def bc_right_func(inputs, outputs, X):
        phi_x = dde.grad.jacobian(outputs, inputs, i=0, j=0)
        phi_t = dde.grad.jacobian(outputs, inputs, i=0, j=1)
        bc = phi_t + phi_x
        if use_decay_factor:
            bc = bc - outputs / df_tau
        return bc

    bc_right = dde.icbc.OperatorBC(
        geomtime, bc_right_func,
        lambda x, on_boundary: on_boundary and np.isclose(x[0], xmax),
    )

    return [ic_disp, ic_vel, bc_left, bc_right]


def _make_numerical_ic_bcs(
    cfg: Dict,
    geomtime,
    phi_ic_func,
    phi_t_ic_func,
    tmin: float,
):
    """Create IC/BC objects using numerical IC callables at the window start."""
    xmin = float(cfg["domain"]["xmin"])
    xmax = float(cfg["domain"]["xmax"])

    def phi0_func(x):
        return np.asarray(phi_ic_func(x)).reshape(-1, 1)

    ic_disp = dde.icbc.IC(
        geomtime, phi0_func, lambda _, on_initial: on_initial
    )

    ic_vel = dde.icbc.OperatorBC(
        geomtime,
        phi_t_ic_func,
        lambda x, on_boundary: np.isclose(x[1], tmin),
    )

    def bc_left_func(inputs, outputs, X):
        phi_x = dde.grad.jacobian(outputs, inputs, i=0, j=0)
        phi_t = dde.grad.jacobian(outputs, inputs, i=0, j=1)
        return phi_t - phi_x

    bc_left = dde.icbc.OperatorBC(
        geomtime, bc_left_func,
        lambda x, on_boundary: on_boundary and np.isclose(x[0], xmin),
    )

    def bc_right_func(inputs, outputs, X):
        phi_x = dde.grad.jacobian(outputs, inputs, i=0, j=0)
        phi_t = dde.grad.jacobian(outputs, inputs, i=0, j=1)
        return phi_t + phi_x

    bc_right = dde.icbc.OperatorBC(
        geomtime, bc_right_func,
        lambda x, on_boundary: on_boundary and np.isclose(x[0], xmax),
    )

    return [ic_disp, ic_vel, bc_left, bc_right]


def build_model_numerical_ic(
    cfg: Dict,
    tmin_override: Optional[float] = None,
    tmax_override: Optional[float] = None,
    phi_ic_func=None,
    phi_t_ic_func=None,
    extra_bcs: Optional[List] = None,
) -> Tuple[dde.Model, dde.data.TimePDE]:
    """Construct a model whose initial conditions come from callables."""
    if phi_ic_func is None or phi_t_ic_func is None:
        raise ValueError("phi_ic_func and phi_t_ic_func are required")

    cfg_model = copy.deepcopy(cfg)
    if tmin_override is not None:
        cfg_model["domain"]["tmin"] = float(tmin_override)
    if tmax_override is not None:
        cfg_model["domain"]["tmax"] = float(tmax_override)

    xmin = float(cfg_model["domain"]["xmin"])
    xmax = float(cfg_model["domain"]["xmax"])
    tmin = float(cfg_model["domain"]["tmin"])
    tmax = float(cfg_model["domain"]["tmax"])

    geom = dde.geometry.Interval(xmin, xmax)
    timedomain = dde.geometry.TimeDomain(tmin, tmax)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    pde_func = _make_pde_func(cfg_model, tmax_override=tmax)
    ic_bcs = _make_numerical_ic_bcs(
        cfg_model, geomtime, phi_ic_func, phi_t_ic_func, tmin
    )
    if extra_bcs:
        ic_bcs.extend(extra_bcs)

    data = dde.data.TimePDE(
        geomtime,
        pde_func,
        ic_bcs,
        num_domain=int(cfg_model["pinn"]["Nr"]),
        num_boundary=int(cfg_model["pinn"]["Nb"]),
        num_initial=int(cfg_model["pinn"]["Ni"]),
        train_distribution="uniform",
    )

    base_model, _ = build_model(cfg_model)
    model = dde.Model(data, base_model.net)
    model._causal_weighter = getattr(pde_func, '_causal_weighter', None)
    return model, data


# ---------------------------------------------------------------------------
# Gradient balancing (Wang, Teng & Perdikaris 2021, Algorithm 1)
# ---------------------------------------------------------------------------

class GradientBalancing(dde.callbacks.Callback):
    """Dynamically adjust loss weights via Wang, Teng & Perdikaris 2021, Alg. 1.

    At every *period* Adam steps the callback:
      1. computes each unweighted loss L_i,
      2. backpropagates each independently,
      3. sets  λ̂_i = max_θ|∇L_r| / mean_θ|∇L_i|  (Alg. 1),
         where L_r (index 0) is the primary PDE residual,
      4. applies an exponential moving average with decay *alpha*.

    The PDE residual (index 0) always keeps weight 1.0 — it is the
    reference anchor.  All other terms (gradient-enhanced PDE residuals,
    IC, BC) are reweighted relative to it.

    The weights are frozen during L-BFGS (callback only fires in SGD loops).
    """

    def __init__(self, period: int = 100, alpha: float = 0.9):
        super().__init__()
        self.period = period
        self.alpha = alpha
        self._ema_weights: Optional[List[float]] = None

    def on_epoch_end(self):
        step = self.model.train_state.step
        if step == 0 or step % self.period != 0:
            return

        net = self.model.net
        n = len(self.model.loss_weights)

        # --- unweighted forward pass ---
        self.model.loss_weights = [1.0] * n

        _, losses = self.model.outputs_losses_train(
            self.model.train_state.X_train,
            self.model.train_state.y_train,
            self.model.train_state.train_aux_vars,
        )

        # --- per-term gradient statistics ---
        # max|grad|  : needed for the reference term (L_r, index 0)
        # mean|grad| : needed for the denominator of all other terms
        max_grads: List[float] = []
        mean_grads: List[float] = []
        for i, loss_i in enumerate(losses):
            net.zero_grad()
            loss_i.backward(retain_graph=(i < n - 1))
            mg = 0.0
            total_abs = 0.0
            count = 0
            for p in net.parameters():
                if p.grad is not None:
                    mg = max(mg, p.grad.abs().max().item())
                    total_abs += p.grad.abs().sum().item()
                    count += p.grad.numel()
            max_grads.append(mg + 1e-16)
            mean_grads.append(total_abs / max(count, 1) + 1e-16)

        net.zero_grad()

        # --- compute balanced weights (Algorithm 1) ---
        # Reference: max|∇_θ L_r| where L_r is the primary PDE residual (index 0).
        # Weight_i = max|∇L_r| / mean|∇L_i|  for i >= 1.
        # Weight_0 = 1.0 (PDE residual is the anchor).
        max_grad_r = max_grads[0]
        raw = [1.0]  # L_r weight
        for i in range(1, n):
            raw.append(max_grad_r / mean_grads[i])

        if self._ema_weights is None:
            self._ema_weights = raw
        else:
            # Standard EMA: alpha controls retention of history.
            # alpha=0.9 -> 90% old + 10% new (slow, smooth adaptation).
            self._ema_weights = [
                self.alpha * ew + (1.0 - self.alpha) * rw
                for ew, rw in zip(self._ema_weights, raw)
            ]

        self.model.loss_weights = list(self._ema_weights)
        # restore not needed — we just overwrote with the new balanced weights

        wstr = ", ".join(f"{w:.2f}" for w in self._ema_weights)
        print(f"  [GradBal] step {step}: weights=[{wstr}]")


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def _apply_hard_ic_output_transform(net: torch.nn.Module, cfg: Dict, tmin: float, tmax: float) -> None:
    """Apply a hard initial-condition ansatz when explicitly enabled."""
    A0 = float(cfg["initial_data"]["A"])
    x0_ic = float(cfg["initial_data"]["x0"])
    sigma = float(cfg["initial_data"]["sigma"])
    profile = cfg["initial_data"].get("velocity_profile", "paper")
    T = float(cfg["pinn"].get("hard_ic", {}).get("correction_time_scale", max(tmax - tmin, 1.0)))
    if T <= 0.0:
        raise ValueError("pinn.hard_ic.correction_time_scale must be positive")

    def transform(x, y):
        x_col = x[:, 0:1]
        dt = x[:, 1:2] - tmin
        phi0 = A0 * torch.exp(-((x_col - x0_ic) ** 2) / (sigma**2))
        if profile == "paper":
            v0 = 2.0 * ((x_col - x0_ic) ** 2) / (sigma**2) * phi0
        elif profile == "outgoing":
            v0 = 2.0 * (x_col - x0_ic) / (sigma**2) * phi0
        else:
            raise ValueError(f"Unknown velocity profile: {profile}")
        return phi0 + dt * v0 + (dt**2 / T) * A0 * torch.tanh(y)

    net.apply_output_transform(transform)

def build_model(cfg: Dict, tmax_override: Optional[float] = None, net_override: Optional[torch.nn.Module] = None) -> Tuple[dde.Model, dde.data.TimePDE]:
    """Construct the DeepXDE Model from the experiment config.
    """
    xmin = float(cfg["domain"]["xmin"])
    xmax = float(cfg["domain"]["xmax"])
    tmin = float(cfg["domain"]["tmin"])
    tmax = tmax_override if tmax_override is not None else float(cfg["domain"]["tmax"])

    geom = dde.geometry.Interval(xmin, xmax)
    timedomain = dde.geometry.TimeDomain(tmin, tmax)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    tmax_actual = tmax_override if tmax_override is not None else float(cfg["domain"]["tmax"])
    pde_func = _make_pde_func(cfg, tmax_override=tmax_actual)
    ic_bcs = _make_ic_bcs(cfg, geomtime)

    # Scale the number of points by the time domain fraction
    t_fraction = (tmax - tmin) / (float(cfg["domain"]["tmax"]) - tmin)
    Nr = int(int(cfg["pinn"]["Nr"]) * t_fraction)
    Ni = int(cfg["pinn"]["Ni"])
    Nb = int(int(cfg["pinn"]["Nb"]) * t_fraction)

    data = dde.data.TimePDE(
        geomtime,
        pde_func,
        ic_bcs,
        num_domain=Nr,
        num_boundary=Nb,
        num_initial=Ni,
        train_distribution="uniform",
    )

    if net_override is not None:
        net = net_override
        # Update input normalization for the new time window (curriculum)
        x_lo, x_hi = xmin, xmax
        t_lo, t_hi = tmin, tmax

        def _input_normalize_update(x, _xlo=x_lo, _xhi=x_hi, _tlo=t_lo, _thi=t_hi):
            x_n = 2.0 * (x[:, 0:1] - _xlo) / (_xhi - _xlo) - 1.0
            t_n = 2.0 * (x[:, 1:2] - _tlo) / (_thi - _tlo) - 1.0
            return torch.cat([x_n, t_n], dim=1)

        net.apply_feature_transform(_input_normalize_update)
    else:
        # --- Network architecture ---
        hidden = [int(w) for w in cfg["pinn"]["hidden_layers"]]
        activation = cfg["pinn"].get("activation", "tanh")
        arch = cfg["pinn"].get("architecture", "fnn")
        if arch != "fnn":
            raise ValueError(f"Unsupported architecture: {arch}")

        layer_size = [2] + hidden + [1]
        net = dde.nn.FNN(layer_size, activation, "Glorot uniform")
        print(f"[PINN] FNN: layers={layer_size}, activation={activation}, "
              "initializer=Glorot uniform")

        # Cast network to float64 if configured (nn.Linear defaults to float32)
        target_dtype = cfg["pinn"].get("dtype", "float64")
        if target_dtype == "float64":
            net = net.double()

        n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"[PINN] Trainable parameters: {n_params}")

        # Input normalization: map physical domain → [-1, 1]²
        x_lo, x_hi = xmin, xmax
        t_lo, t_hi = tmin, tmax   # uses tmax_override for curriculum windows

        def _input_normalize(x, _xlo=x_lo, _xhi=x_hi, _tlo=t_lo, _thi=t_hi):
            x_n = 2.0 * (x[:, 0:1] - _xlo) / (_xhi - _xlo) - 1.0
            t_n = 2.0 * (x[:, 1:2] - _tlo) / (_thi - _tlo) - 1.0
            return torch.cat([x_n, t_n], dim=1)

        net.apply_feature_transform(_input_normalize)
        print(f"[PINN] Input normalization: x*∈[{x_lo},{x_hi}]→[-1,1], "
              f"t∈[{t_lo},{t_hi}]→[-1,1]")

        # Output transform: A * tanh(y)
        # Bounding the output enforces the physical constraint of energy conservation
        # and prevents the network from adapting a blowing-up solution (Patel et al. 2024).
        A_bound = float(cfg["initial_data"]["A"])
        if cfg["pinn"].get("hard_ic", {}).get("enabled", False):
            _apply_hard_ic_output_transform(net, cfg, tmin=tmin, tmax=tmax)
            print("[PINN] Hard IC output transform enabled")
        else:
            net.apply_output_transform(lambda x, y: A_bound * torch.tanh(y))

    model = dde.Model(data, net)

    # Expose causal weighter on model for callback/monitoring access
    model._causal_weighter = getattr(pde_func, '_causal_weighter', None)

    return model, data


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train_pinn_curriculum(
    cfg: Dict,
    checkpoint_dir: Optional[str] = None,
    checkpoint_every: int = 500,
    resume: bool = False,
) -> Tuple[dde.Model, Dict[str, List[float]]]:
    """
    Train the PINN using Curriculum Learning (Expanding Time Windows).
    This is mathematically equivalent to Time-Marching but avoids error accumulation
    at window boundaries and bypasses PyTorch derivative extraction bugs.

    Resume logic per window:
      - model-final-*.pt exists  → skip entire window (weights-only restore)
      - model-adam_done-*.pt exists → skip Adam, resume/run L-BFGS
      - only model-*.pt exists   → resume Adam from latest checkpoint
      Weights-only restore is used when the optimizer type may differ between
      the checkpoint and the current compile phase.
    """
    seed = int(cfg["pinn"]["seed"])
    dde.config.set_random_seed(seed)
    dde.config.set_default_float(cfg["pinn"]["dtype"])

    loss_weights = [float(w) for w in cfg["pinn"]["lambda"]]

    curriculum_cfg = cfg["pinn"]["curriculum"]
    windows = curriculum_cfg["windows"]  # e.g., [10.0, 20.0, 30.0, 40.0, 50.0]

    adam_cfg = cfg["pinn"]["adam"]
    adam_iters = int(adam_cfg["iters"])
    lr = float(adam_cfg["lr"])
    resample_period = int(adam_cfg["resample_period"])

    net = None
    history_all = None

    for i, tmax in enumerate(windows):
        print(f"\n{'='*60}")
        print(f"[PINN] Curriculum Window {i+1}/{len(windows)}: t in [0, {tmax}]")
        print(f"{'='*60}\n")

        # Build model for this window (reuses network from previous window)
        model, data = build_model(cfg, tmax_override=tmax, net_override=net)
        net = model.net  # Keep the network reference for the next window

        # Set up checkpointing for this window
        window_ckpt_dir = None
        model_save_path = None
        if checkpoint_dir:
            window_ckpt_dir = os.path.join(checkpoint_dir, f"window_{i+1}")
            os.makedirs(window_ckpt_dir, exist_ok=True)
            model_save_path = os.path.join(window_ckpt_dir, "model")

        # ==============================================================
        # RESUME CHECK 1: Is this window already fully trained?
        # ==============================================================
        if resume and window_ckpt_dir:
            final_ckpt = _find_final_checkpoint(window_ckpt_dir)
            if final_ckpt is not None:
                print(f"[CKPT] Window {i+1} already completed. "
                      f"Restoring weights and skipping.")
                _restore_weights_only(net, final_ckpt, verbose=1)
                continue

        # ==============================================================
        # RESUME CHECK 2: Is Adam already done for this window?
        # ==============================================================
        skip_adam = False
        lbfgs_resume_ckpt = None

        if resume and window_ckpt_dir:
            adam_done_ckpt = _find_adam_done_checkpoint(window_ckpt_dir)
            if adam_done_ckpt is not None:
                skip_adam = True

        # ==============================================================
        # ADAM PHASE
        # ==============================================================
        losshistory_adam = None

        if not skip_adam:
            callbacks_adam: List = []
            callbacks_adam.append(
                dde.callbacks.PDEPointResampler(period=resample_period)
            )

            # Gradient balancing (Wang et al. 2021)
            grad_bal_cfg = cfg["pinn"].get("gradient_balancing", {})
            if grad_bal_cfg.get("enabled", False):
                gb_period = int(grad_bal_cfg.get("period", 100))
                gb_alpha = float(grad_bal_cfg.get("alpha", 0.9))
                callbacks_adam.append(
                    GradientBalancing(period=gb_period, alpha=gb_alpha)
                )

            # Causal training monitor (Adam only)
            if getattr(model, '_causal_weighter', None) is not None:
                callbacks_adam.append(
                    CausalTrainingMonitor(model._causal_weighter, period=100)
                )

            model_restore_path = None
            if resume and window_ckpt_dir:
                ckpt = _find_latest_checkpoint(window_ckpt_dir)
                if ckpt is not None:
                    model_restore_path = ckpt
                    print(f"[CKPT] Restoring from {ckpt}")

            if model_save_path:
                callbacks_adam.append(
                    dde.callbacks.ModelCheckpoint(
                        model_save_path,
                        save_better_only=False,
                        period=checkpoint_every,
                    )
                )

            model.compile("adam", lr=lr, loss_weights=loss_weights)

            print(f"[PINN] Adam: {adam_iters} iters, lr={lr}, "
                  f"resample every {resample_period}")
            losshistory_adam, _ = model.train(
                iterations=adam_iters,
                callbacks=callbacks_adam,
                display_every=100,
                model_save_path=model_save_path,
                model_restore_path=model_restore_path,
            )

            if model_save_path:
                model.save(model_save_path + "-adam_done")
                weights_file = os.path.join(
                    window_ckpt_dir, "loss_weights_adam.json"
                )
                with open(weights_file, "w") as f:
                    json.dump(list(model.loss_weights), f)

        else:
            # ---- Adam already done — restore and skip to L-BFGS ----
            print(f"[CKPT] Window {i+1} Adam already done — "
                  f"skipping to L-BFGS.")
            model.compile("adam", lr=lr, loss_weights=loss_weights)
            model.train(iterations=0, display_every=1)  # init train state
            model.restore(adam_done_ckpt, verbose=1)

            # Restore gradient-balanced weights
            weights_file = os.path.join(
                window_ckpt_dir, "loss_weights_adam.json"
            )
            if os.path.isfile(weights_file):
                with open(weights_file) as f:
                    saved_weights = json.load(f)
                model.loss_weights = saved_weights
                wstr = ", ".join(f"{w:.2f}" for w in saved_weights)
                print(f"[CKPT] Restored gradient-balanced weights: [{wstr}]")

            # Look for a more recent L-BFGS checkpoint
            lbfgs_resume_ckpt = _find_latest_checkpoint(
                window_ckpt_dir, exclude_prefix="model-adam_done"
            )
            if lbfgs_resume_ckpt is not None:
                step = _get_checkpoint_step(lbfgs_resume_ckpt)
                if step is not None and step <= adam_iters:
                    lbfgs_resume_ckpt = None  # Adam-phase ckpt, not L-BFGS

        # ==============================================================
        # L-BFGS PHASE
        # ==============================================================
        lbfgs_cfg = cfg["pinn"]["lbfgs"]
        lbfgs_iters = int(lbfgs_cfg["iters"])

        lbfgs_loss_weights = list(model.loss_weights)

        # Calculate remaining L-BFGS iterations
        lbfgs_remaining = lbfgs_iters
        if lbfgs_resume_ckpt is not None:
            step = _get_checkpoint_step(lbfgs_resume_ckpt)
            if step is not None:
                lbfgs_done = max(0, step - adam_iters)
                if lbfgs_done > 0:
                    lbfgs_remaining = lbfgs_iters - lbfgs_done
                    print(f"[CKPT] L-BFGS: {lbfgs_done}/{lbfgs_iters} done, "
                          f"{lbfgs_remaining} remaining")
                else:
                    lbfgs_resume_ckpt = None

        losshistory_lbfgs = None

        if lbfgs_remaining > 0:
            dde.optimizers.set_LBFGS_options(
                maxcor=100,
                maxiter=lbfgs_remaining,
                ftol=0,
                gtol=1e-8,
                maxls=50,
            )

            lbfgs_resample_period = int(lbfgs_cfg.get("resample_period", 0))
            if lbfgs_resample_period > 0:
                step_size = min(
                    checkpoint_every, lbfgs_resample_period, lbfgs_remaining
                )
            else:
                step_size = min(checkpoint_every, lbfgs_remaining)

            from deepxde.optimizers.config import LBFGS_options as _lbfgs_opts
            _lbfgs_opts["iter_per_step"] = step_size
            _lbfgs_opts["fun_per_step"] = int(step_size * 1.25)

            model.compile("L-BFGS", loss_weights=lbfgs_loss_weights)

            # Restore only model weights (ignore optimizer state — fresh
            # L-BFGS optimizer rebuilds curvature estimates from scratch)
            if lbfgs_resume_ckpt is not None:
                print(f"[CKPT] Restoring model weights from "
                      f"{lbfgs_resume_ckpt}")
                _restore_weights_only(net, lbfgs_resume_ckpt, verbose=1)

            callbacks_lbfgs = []
            if model_save_path:
                callbacks_lbfgs.append(
                    dde.callbacks.ModelCheckpoint(
                        model_save_path,
                        save_better_only=False,
                        period=checkpoint_every,
                    )
                )

            if lbfgs_resample_period > 0:
                callbacks_lbfgs.append(
                    dde.callbacks.PDEPointResampler(
                        period=lbfgs_resample_period
                    )
                )

            # Disable causal weighting during L-BFGS (per ScreaM835 finding)
            if getattr(model, '_causal_weighter', None) is not None:
                model._causal_weighter.enabled = False
                print("[Causal] Disabled causal weighting for L-BFGS phase")

            losshistory_lbfgs, _ = model.train(
                iterations=lbfgs_remaining,
                callbacks=callbacks_lbfgs,
                display_every=100,
            )

            if model_save_path:
                model.save(model_save_path + "-final")

        else:
            print(f"[PINN] L-BFGS already completed for window {i+1}.")
            # Make sure the network has the latest L-BFGS weights
            if lbfgs_resume_ckpt:
                _restore_weights_only(net, lbfgs_resume_ckpt, verbose=1)

        # ==============================================================
        # COMBINE HISTORY FOR THIS WINDOW
        # ==============================================================
        if losshistory_adam is not None and losshistory_lbfgs is not None:
            history = _combine_loss_histories(
                losshistory_adam, losshistory_lbfgs,
                loss_weights_adam=loss_weights,
                loss_weights_lbfgs=lbfgs_loss_weights,
            )
        elif losshistory_lbfgs is not None:
            history = _convert_loss_history(
                losshistory_lbfgs,
                loss_weights=lbfgs_loss_weights,
                phase="lbfgs",
            )
        elif losshistory_adam is not None:
            history = _convert_loss_history(
                losshistory_adam,
                loss_weights=loss_weights,
                phase="adam",
            )
        else:
            history = None  # Fully skipped or fully resumed window

        if history is not None:
            if history_all is None:
                history_all = history
            else:
                for key in history_all:
                    history_all[key].extend(history[key])

    # If all windows were skipped (full resume), return empty history
    if history_all is None:
        history_all = {
            "L_total": [], "Lr": [], "Lrx": [], "Lrt": [],
            "Lic": [], "Liv": [], "Lbl": [], "Lbr": [], "w_min": [],
        }

    return model, history_all


def train_pinn(
    cfg: Dict,
    checkpoint_dir: Optional[str] = None,
    checkpoint_every: int = 500,
    resume: bool = False,
) -> Tuple[dde.Model, Dict[str, List[float]]]:
    """
    Train the PINN with DeepXDE: Adam -> L-BFGS.

    Parameters
    ----------
    cfg : dict
        Full experiment config.
    checkpoint_dir : str or None
        Directory for checkpoints (None disables).
    checkpoint_every : int
        Checkpoint interval during Adam phase.
    resume : bool
        If True, restore from the latest checkpoint.

    Returns
    -------
    model : dde.Model
    history : dict  -- per-step loss components
        Keys: L_total, Lr, Lrx, Lrt, Lic, Liv, Lbl, Lbr, w_min
    """
    # Check if curriculum learning is enabled
    curriculum_cfg = cfg["pinn"].get("curriculum", {})
    if curriculum_cfg.get("enabled", False):
        return _train_pinn_curriculum(cfg, checkpoint_dir, checkpoint_every, resume)

    seed = int(cfg["pinn"]["seed"])
    dde.config.set_random_seed(seed)
    dde.config.set_default_float(cfg["pinn"]["dtype"])

    model, data = build_model(cfg)
    loss_weights = [float(w) for w in cfg["pinn"]["lambda"]]

    print(f"[PINN] Loss weights: {loss_weights}")

    model_save_path = None
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_save_path = os.path.join(checkpoint_dir, "model")

    # ---- Check if Adam already completed (for resume) ----
    # DeepXDE appends the step number: model-adam_done-{step}.pt
    adam_done_ckpt = _find_adam_done_checkpoint(checkpoint_dir) if checkpoint_dir else None
    adam_already_done = resume and adam_done_ckpt is not None

    losshistory_adam = None
    lbfgs_resume_ckpt = None  # deferred to after L-BFGS compile

    # Total Adam iterations (needed for global step numbering in L-BFGS)
    adam_iters_total = int(cfg["pinn"]["adam"]["iters"])

    if adam_already_done:
        # ---- Skip Adam: restore weights from adam_done checkpoint ----
        print("[CKPT] Adam already completed — skipping to L-BFGS")
        restore_path = adam_done_ckpt
        model.compile("adam", lr=1e-3, loss_weights=loss_weights)
        # Need one dummy call to initialise train_state before restore
        model.train(iterations=0, display_every=1)
        model.restore(restore_path, verbose=1)

        # Restore gradient-balanced weights saved after Adam
        weights_file = os.path.join(checkpoint_dir, "loss_weights_adam.json")
        if os.path.isfile(weights_file):
            with open(weights_file) as f:
                saved_weights = json.load(f)
            model.loss_weights = saved_weights
            wstr = ", ".join(f"{w:.4f}" for w in saved_weights)
            print(f"[CKPT] Restored gradient-balanced weights: [{wstr}]")

        # Check for a newer L-BFGS checkpoint to continue from.
        # Defer the actual restore to after L-BFGS compile so the
        # optimizer state types match.
        lbfgs_resume_ckpt = _find_latest_checkpoint(
            checkpoint_dir, exclude_prefix="model-adam_done"
        )
        if lbfgs_resume_ckpt is not None:
            print(f"[CKPT] Will restore L-BFGS checkpoint: {lbfgs_resume_ckpt}")

    else:
        # ---- Adam phase ----
        adam_cfg = cfg["pinn"]["adam"]
        adam_iters = int(adam_cfg["iters"])
        lr = float(adam_cfg["lr"])
        resample_period = int(adam_cfg["resample_period"])

        callbacks_adam: List = []

        # Adaptive sampling (RAD/RAR-D, Wu et al. 2023) or uniform resampling
        adaptive_cfg = cfg["pinn"].get("adaptive_sampling", {})
        if adaptive_cfg.get("enabled", False):
            method = adaptive_cfg.get("method", "RAD")
            rad_period = int(adaptive_cfg.get("period", 1000))
            num_cand = int(adaptive_cfg.get("num_candidates", 50000))
            k = float(adaptive_cfg.get("k", 1.0))
            c = float(adaptive_cfg.get("c", 1.0))
            num_add = int(adaptive_cfg.get("num_add", 160))
            eval_bs = int(adaptive_cfg.get("eval_batch_size", 5000))

            pde_residual = _make_pde_residual_only(cfg)
            if method == "greedy":
                greedy_frac = float(adaptive_cfg.get("greedy_fraction", 0.5))
                callbacks_adam.append(
                    GreedyResampler(
                        pde_residual=pde_residual,
                        period=rad_period,
                        num_candidates=num_cand,
                        greedy_fraction=greedy_frac,
                        eval_batch_size=eval_bs,
                    )
                )
                print(f"[PINN] Adaptive sampling: greedy "
                      f"(period={rad_period}, greedy_fraction={greedy_frac}, "
                      f"candidates={num_cand})")
            else:
                callbacks_adam.append(
                    ResidualAdaptiveResampler(
                        pde_residual=pde_residual,
                        period=rad_period,
                        num_candidates=num_cand,
                        k=k, c=c,
                        method=method,
                        num_add=num_add,
                        eval_batch_size=eval_bs,
                    )
                )
                print(f"[PINN] Adaptive sampling: {method} "
                      f"(period={rad_period}, k={k}, c={c}, "
                      f"candidates={num_cand})")
        else:
            callbacks_adam.append(
                dde.callbacks.PDEPointResampler(period=resample_period)
            )

        # Gradient balancing (Wang et al. 2021)
        grad_bal_cfg = cfg["pinn"].get("gradient_balancing", {})
        if grad_bal_cfg.get("enabled", False):
            gb_period = int(grad_bal_cfg.get("period", 100))
            gb_alpha = float(grad_bal_cfg.get("alpha", 0.9))
            callbacks_adam.append(
                GradientBalancing(period=gb_period, alpha=gb_alpha)
            )
            print(f"[PINN] Gradient balancing enabled "
                  f"(period={gb_period}, alpha={gb_alpha})")

        # Causal training monitor (Adam only)
        if getattr(model, '_causal_weighter', None) is not None:
            callbacks_adam.append(
                CausalTrainingMonitor(model._causal_weighter, period=100)
            )

        if model_save_path:
            callbacks_adam.append(
                dde.callbacks.ModelCheckpoint(
                    model_save_path,
                    save_better_only=False,
                    period=checkpoint_every,
                )
            )

        # Resume from mid-Adam checkpoint if available
        model_restore_path = None
        if resume and checkpoint_dir:
            ckpt = _find_latest_checkpoint(checkpoint_dir)
            if ckpt is not None:
                model_restore_path = ckpt
                print(f"[CKPT] Restoring from {ckpt}")

        model.compile("adam", lr=lr, loss_weights=loss_weights)

        print(f"[PINN] Adam: {adam_iters} iters, lr={lr}, "
              f"resample every {resample_period}")
        losshistory_adam, _ = model.train(
            iterations=adam_iters,
            callbacks=callbacks_adam,
            display_every=100,
            model_save_path=model_save_path,
            model_restore_path=model_restore_path,
        )

        if model_save_path:
            model.save(model_save_path + "-adam_done")
            print("[CKPT] Adam complete -- checkpoint saved")
            # Persist gradient-balanced weights so L-BFGS resume can use them
            weights_file = os.path.join(checkpoint_dir, "loss_weights_adam.json")
            with open(weights_file, "w") as f:
                json.dump(list(model.loss_weights), f)
            print(f"[CKPT] Gradient-balanced weights saved to {weights_file}")

    # ---- L-BFGS phase ----
    lbfgs_cfg = cfg["pinn"]["lbfgs"]
    lbfgs_iters = int(lbfgs_cfg["iters"])

    # Freeze the gradient-balanced weights from Adam for the L-BFGS phase.
    lbfgs_loss_weights = list(model.loss_weights)
    wstr = ", ".join(f"{w:.4f}" for w in lbfgs_loss_weights)
    print(f"[PINN] L-BFGS loss weights (frozen from Adam): [{wstr}]")

    # Determine how many L-BFGS iterations have already been completed
    lbfgs_iters_done = 0
    if lbfgs_resume_ckpt is not None:
        m = re.search(r"model-(\d+)\.pt$", lbfgs_resume_ckpt)
        if m:
            ckpt_step = int(m.group(1))
            lbfgs_iters_done = max(0, ckpt_step - adam_iters_total)
            if lbfgs_iters_done > 0:
                print(f"[CKPT] L-BFGS iterations already done: {lbfgs_iters_done}")
            else:
                print(f"[CKPT] Checkpoint {lbfgs_resume_ckpt} is from Adam phase — skipping")
                lbfgs_resume_ckpt = None

    iters_remaining = lbfgs_iters - lbfgs_iters_done
    if iters_remaining <= 0:
        print("[PINN] L-BFGS already completed.")
        return model, _convert_loss_history(
            losshistory_adam,
            loss_weights=loss_weights,
            phase="adam",
        ) if losshistory_adam else {}

    print(f"[PINN] L-BFGS: {iters_remaining} iterations remaining")

    # Set L-BFGS options
    dde.optimizers.set_LBFGS_options(
        maxcor=100,
        maxiter=iters_remaining,
        ftol=0,
        gtol=1e-8,
        maxls=50,
    )
    
    # DeepXDE executes L-BFGS within a single closure. To enable periodic callbacks
    # (such as ModelCheckpoint and PDEPointResampler) without resetting the optimizer
    # state and losing the Hessian approximation history, we configure iter_per_step.
    # We must set iter_per_step to the greatest common divisor of our callback periods
    # (or simply the minimum period) so that the closure yields control back to the
    # callback loop frequently enough.
    lbfgs_resample_period = int(lbfgs_cfg.get("resample_period", 0))
    if lbfgs_resample_period > 0:
        step_size = min(checkpoint_every, lbfgs_resample_period, iters_remaining)
    else:
        step_size = min(checkpoint_every, iters_remaining)

    from deepxde.optimizers.config import LBFGS_options as _lbfgs_opts
    _lbfgs_opts["iter_per_step"] = step_size
    _lbfgs_opts["fun_per_step"] = int(_lbfgs_opts["iter_per_step"] * 1.25)

    model.compile("L-BFGS", loss_weights=lbfgs_loss_weights)

    if lbfgs_resume_ckpt is not None:
        print(f"[CKPT] Restoring model weights from {lbfgs_resume_ckpt}")
        checkpoint = torch.load(lbfgs_resume_ckpt, weights_only=True)
        model.net.load_state_dict(checkpoint["model_state_dict"])

    callbacks_lbfgs = []
    if model_save_path:
        callbacks_lbfgs.append(
            dde.callbacks.ModelCheckpoint(
                model_save_path,
                save_better_only=False,
                period=checkpoint_every,
            )
        )
    
    # Add PDEPointResampler for L-BFGS if configured
    if lbfgs_resample_period > 0:
        callbacks_lbfgs.append(
            dde.callbacks.PDEPointResampler(period=lbfgs_resample_period)
        )
        print(f"[PINN] L-BFGS: resampling collocation points every {lbfgs_resample_period} iterations")

    # Disable causal weighting during L-BFGS (per ScreaM835 finding)
    if getattr(model, '_causal_weighter', None) is not None:
        model._causal_weighter.enabled = False
        print("[Causal] Disabled causal weighting for L-BFGS phase")

    losshistory_lbfgs, _ = model.train(
        iterations=iters_remaining,
        callbacks=callbacks_lbfgs,
        display_every=100
    )

    if model_save_path:
        model.save(model_save_path + "-final")
        print("[CKPT] Training complete -- final checkpoint saved")

    # ---- Combine loss histories ----
    if losshistory_adam is not None:
        history = _combine_loss_histories(
            losshistory_adam, losshistory_lbfgs,
            loss_weights_adam=loss_weights,
            loss_weights_lbfgs=lbfgs_loss_weights,
        )
    else:
        history = _convert_loss_history(
            losshistory_lbfgs,
            loss_weights=lbfgs_loss_weights,
            phase="lbfgs",
        )

    return model, history


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _find_latest_checkpoint(
    checkpoint_dir: str, exclude_prefix: Optional[str] = None
) -> Optional[str]:
    """Find the latest DeepXDE checkpoint in the directory.

    Parameters
    ----------
    checkpoint_dir : str
        Directory to search.
    exclude_prefix : str or None
        If given, skip checkpoints whose filename starts with this prefix.
        Useful for skipping the adam_done marker when looking for L-BFGS
        checkpoints.
    """
    if not os.path.isdir(checkpoint_dir):
        return None
    candidates = []
    for f in os.listdir(checkpoint_dir):
        if not f.endswith(".pt"):
            continue
        if exclude_prefix and f.startswith(exclude_prefix):
            continue
        candidates.append(os.path.join(checkpoint_dir, f))
    if not candidates:
        return None
    latest = max(candidates, key=os.path.getmtime)
    return latest


def _find_adam_done_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the model-adam_done-*.pt checkpoint if it exists.

    Returns the full path (with .pt extension) or None.
    """
    if not os.path.isdir(checkpoint_dir):
        return None
    for f in os.listdir(checkpoint_dir):
        if f.startswith("model-adam_done") and f.endswith(".pt"):
            return os.path.join(checkpoint_dir, f)
    return None


def _find_final_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the model-final-*.pt checkpoint if it exists.

    DeepXDE appends the step number when saving, so the actual file is
    ``model-final-{step}.pt``, not ``model-final.pt``.

    Returns the full path or None.
    """
    if not os.path.isdir(checkpoint_dir):
        return None
    for f in os.listdir(checkpoint_dir):
        if f.startswith("model-final-") and f.endswith(".pt"):
            return os.path.join(checkpoint_dir, f)
    return None


def _restore_weights_only(net: torch.nn.Module, checkpoint_path: str,
                          verbose: int = 0) -> None:
    """Load only network weights from a DeepXDE checkpoint.

    Unlike ``Model.restore()``, this ignores the optimizer state dict,
    which avoids crashes when the checkpoint was saved with a different
    optimizer (e.g. restoring an L-BFGS checkpoint into an Adam-compiled
    model).
    """
    if verbose > 0:
        print(f"  Loading weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu",
                            weights_only=True)
    net.load_state_dict(checkpoint["model_state_dict"])


def _get_checkpoint_step(filepath: str) -> Optional[int]:
    """Extract the step number from a DeepXDE checkpoint filename.

    Examples
    --------
    >>> _get_checkpoint_step("model-500.pt")
    500
    >>> _get_checkpoint_step("model-adam_done-2000.pt")
    2000
    >>> _get_checkpoint_step("model-final-5000.pt")
    5000
    """
    basename = os.path.basename(filepath)
    m = re.search(r"-(\d+)\.pt$", basename)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Loss-history conversion
# ---------------------------------------------------------------------------

def _convert_loss_history(
    losshistory,
    loss_weights: Optional[List[float]] = None,
    phase: str = "adam",
) -> Dict[str, List]:
    """Convert a single DeepXDE LossHistory to our dict format.

    DeepXDE's losshistory.loss_train stores **weighted** losses (multiplied
    by model.loss_weights).  We divide them out here so that the saved JSON
    and plots always show unweighted per-component MSEs, consistent with
    what DeepXDE prints to stdout.

    Order: [PDE outputs ..., IC/BCs ...] = [r, r_x, r_t, ic, iv, bl, br].
    """
    loss_names = ["Lr", "Lrx", "Lrt", "Lic", "Liv", "Lbl", "Lbr"]
    history: Dict[str, List] = {name: [] for name in loss_names}
    history["L_total"] = []
    history["w_min"] = []
    history["steps"] = []
    history["phase"] = []

    steps = losshistory.steps
    losses = np.array(losshistory.loss_train)

    # Build per-component divisor to un-weight the losses
    n_components = losses.shape[1] if losses.ndim > 1 else len(loss_names)
    if loss_weights is not None and len(loss_weights) >= n_components:
        weights = np.array(loss_weights[:n_components], dtype=float)
        weights[weights == 0] = 1.0  # avoid division by zero
    else:
        weights = np.ones(n_components)

    for i in range(len(steps)):
        total = 0.0
        for j, name in enumerate(loss_names):
            if j < losses.shape[1]:
                val = float(losses[i, j]) / weights[j]
            else:
                val = 0.0
            history[name].append(val)
            total += val
        history["L_total"].append(total)
        history["w_min"].append(1.0)
        history["steps"].append(int(steps[i]))
        history["phase"].append(phase)

    return history


def _combine_loss_histories(
    lh_adam,
    lh_lbfgs,
    loss_weights_adam: Optional[List[float]] = None,
    loss_weights_lbfgs: Optional[List[float]] = None,
) -> Dict[str, List]:
    """Concatenate Adam and L-BFGS loss histories.

    Fixes non-monotonic step numbering that occurs when RAD/resampling
    inflates Adam's step counter beyond the requested iteration count.
    """
    h1 = _convert_loss_history(lh_adam, loss_weights=loss_weights_adam, phase="adam")
    h2 = _convert_loss_history(lh_lbfgs, loss_weights=loss_weights_lbfgs, phase="lbfgs")

    # Fix step numbering: if L-BFGS first step <= Adam last step,
    # offset L-BFGS steps to continue monotonically after Adam.
    if h1["steps"] and h2["steps"] and h2["steps"][0] <= h1["steps"][-1]:
        offset = h1["steps"][-1] - h2["steps"][0] + 1
        h2["steps"] = [s + offset for s in h2["steps"]]

    combined: Dict[str, List] = {}
    for key in h1:
        combined[key] = h1[key] + h2[key]
    return combined


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def eval_on_grid(
    model, x: np.ndarray, t: np.ndarray, dtype: str = "float64",
    decay_factor_tau: float = 0.0,
) -> np.ndarray:
    """
    Evaluate the model on a full space-time grid.

    Returns phi[t_index, x_index].
    """
    X_list = []
    for ti in t:
        X_list.append(np.stack([x, np.full_like(x, ti)], axis=1))
    X = np.concatenate(X_list, axis=0)

    if isinstance(model, dde.Model):
        y = model.predict(X)
        y = y.reshape(len(t), len(x))
        if decay_factor_tau > 0.0:
            y = y * np.exp(-t / decay_factor_tau)[:, None]
        return y

    # Backward compatibility for raw PyTorch model
    device = next(model.parameters()).device
    tdtype = torch.float64 if dtype == "float64" else torch.float32
    X_t = torch.tensor(X, dtype=tdtype, device=device)
    with torch.no_grad():
        y = model(X_t).cpu().numpy()
    y = y.reshape(len(t), len(x))
    if decay_factor_tau > 0.0:
        y = y * np.exp(-t / decay_factor_tau)[:, None]
    return y

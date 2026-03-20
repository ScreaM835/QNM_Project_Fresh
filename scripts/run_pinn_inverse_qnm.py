"""
Inverse PINN + QNM extraction: Learn black hole mass M AND
quasi-normal mode parameters (ω, τ) from noisy observations.

This combines two ideas:
  Idea 1: Treat M as a learnable parameter in the PDE (V depends on M).
  Idea 4: Treat (ω, τ, A_ring, φ₀) as learnable parameters and add a
           ringdown-template loss that constrains the PINN output to
           match A·exp(−t/τ)·cos(ωt + φ₀) at a fixed observation point
           during the ringdown phase.

The combined loss is:

    L = λ_pde · L_pde(φ; M)
      + λ_data · L_data(φ, φ_obs)
      + λ_ring · L_ring(φ; ω, τ, A_ring, φ₀)
      + λ_ic · L_ic  +  λ_iv · L_iv  +  λ_bl · L_bl  +  λ_br · L_br

Because the PDE constrains the physics, the observation data constrains
the waveform, and the ringdown template constrains the late-time
behaviour, only the physically correct values of (M, ω, τ) can satisfy
all three simultaneously.

Usage:
    python scripts/run_pinn_inverse_qnm.py \\
        --config configs/zerilli_l2_inverse_qnm.yaml

Requires ``inverse`` and ``ringdown`` sections in config.
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import numpy as np
import torch

import deepxde as dde

from src.config import load_config
from src.fd_solver import solve_fd
from src.pinn import eval_on_grid
from src.potentials import _lambert_w_torch
from src.initial_data import gaussian_phi, gaussian_phi_t
from src.utils import ensure_dir, save_json, rmsd, mad, rl2
from src.plotting import (
    plot_snapshots, plot_abs_diff_snapshots,
    plot_snapshots_zoomed, plot_error_heatmap, plot_ringdown_overlay,
)


# ------------------------------------------------------------------
# Zerilli potential with tensor M (fully differentiable)
# ------------------------------------------------------------------

def _r_of_x_torch(x: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    z = torch.exp(x / (2.0 * M) - 1.0)
    y = _lambert_w_torch(z)
    return 2.0 * M * (1.0 + y)


def _V_zerilli_torch(x: torch.Tensor, M: torch.Tensor, l: int) -> torch.Tensor:
    r = _r_of_x_torch(x, M)
    n = 0.5 * (l - 1) * (l + 2)
    f = 1.0 - 2.0 * M / r
    num = (
        2.0 * n**2 * (n + 1.0) * r**3
        + 6.0 * n**2 * M * r**2
        + 18.0 * n * (M**2) * r
        + 18.0 * (M**3)
    )
    den = r**3 * (n * r + 3.0 * M) ** 2
    return f * (num / den)


# ------------------------------------------------------------------
# Build inverse + QNM model
# ------------------------------------------------------------------

def build_inverse_qnm_model(cfg, M_init, omega_init, tau_init,
                              A_ring_init, phi0_init):
    """Build model with learnable M, ω, τ, A_ring, φ₀.

    M enters the PDE (Zerilli potential depends on M).
    (ω, τ, A_ring, φ₀) enter a ringdown template loss that constrains
    the PINN output at a fixed spatial point for late times.

    Returns:
        model, data, M_var, omega_var, tau_var, A_ring_var, phi0_var, fd
    """
    l = int(cfg["physics"]["l"])

    xmin = float(cfg["domain"]["xmin"])
    xmax = float(cfg["domain"]["xmax"])
    tmin = float(cfg["domain"]["tmin"])
    tmax = float(cfg["domain"]["tmax"])

    geom = dde.geometry.Interval(xmin, xmax)
    timedomain = dde.geometry.TimeDomain(tmin, tmax)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    # Learnable parameters
    M_var = dde.Variable(M_init, dtype=torch.float64)
    omega_var = dde.Variable(omega_init, dtype=torch.float64)
    tau_var = dde.Variable(tau_init, dtype=torch.float64)
    A_ring_var = dde.Variable(A_ring_init, dtype=torch.float64)
    phi0_var = dde.Variable(phi0_init, dtype=torch.float64)

    # PDE with learnable M
    def pde(x, y):
        y_xx = dde.grad.hessian(y, x, i=0, j=0)
        y_tt = dde.grad.hessian(y, x, i=1, j=1)
        V = _V_zerilli_torch(x[:, 0:1], M_var, l)
        r = y_tt - y_xx + V * y

        dr = torch.autograd.grad(
            r, x,
            grad_outputs=torch.ones_like(r),
            create_graph=True,
            retain_graph=True,
        )[0]
        r_x = dr[:, 0:1]
        r_t = dr[:, 1:2]
        return [r, r_x, r_t]

    # IC / BCs (identical to forward/inverse problem)
    A_ic = float(cfg["initial_data"]["A"])
    x0_ic = float(cfg["initial_data"]["x0"])
    sigma_ic = float(cfg["initial_data"]["sigma"])
    vel_profile = cfg["initial_data"].get("velocity_profile", "paper")

    def ic_func(x):
        return gaussian_phi(x[:, 0:1], A_ic, x0_ic, sigma_ic)

    ic_disp = dde.icbc.IC(geomtime, ic_func, lambda _, on_initial: on_initial)

    def ic_vel_func(inputs, outputs, X):
        phi_t = dde.grad.jacobian(outputs, inputs, i=0, j=1)
        x_np = inputs[:, 0:1].detach().cpu().numpy()
        target = gaussian_phi_t(x_np, A_ic, x0_ic, sigma_ic, vel_profile)
        target_t = torch.as_tensor(target, dtype=outputs.dtype, device=outputs.device)
        return phi_t - target_t

    ic_vel = dde.icbc.OperatorBC(
        geomtime, ic_vel_func, lambda x, on_initial: on_initial,
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

    ic_bcs = [ic_disp, ic_vel, bc_left, bc_right]

    # --- Observation data (noisy FD solution) ---
    inv_cfg = cfg["inverse"]
    n_obs = int(inv_cfg["n_obs"])
    noise_level = float(inv_cfg["noise_level"])
    obs_tmin = float(inv_cfg.get("obs_t_min", tmin))
    obs_tmax = float(inv_cfg.get("obs_t_max", tmax))

    print("[INV] Generating FD reference for observation data...")
    fd = solve_fd(cfg)
    x_fd, t_fd, phi_fd = fd["x"], fd["t"], fd["phi"]

    rng = np.random.RandomState(42)
    obs_x = rng.uniform(xmin, xmax, size=n_obs)
    obs_t = rng.uniform(obs_tmin, obs_tmax, size=n_obs)

    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((t_fd, x_fd), phi_fd, method="linear",
                                      bounds_error=False, fill_value=0.0)
    phi_obs_clean = interp(np.stack([obs_t, obs_x], axis=1))

    noise = rng.normal(0, noise_level * np.abs(phi_obs_clean).max(), size=n_obs)
    phi_obs = phi_obs_clean + noise

    X_obs = np.stack([obs_x, obs_t], axis=1)

    print(f"[INV] Observations: {n_obs} points, noise σ={noise_level:.2%} "
          f"(abs σ={noise_level * np.abs(phi_obs_clean).max():.6f})")

    observe = dde.icbc.PointSetBC(X_obs, phi_obs.reshape(-1, 1), component=0)
    ic_bcs.append(observe)

    # --- Ringdown template as PointSetBC ---
    ring_cfg = cfg["ringdown"]
    x_ring = float(ring_cfg["x_obs"])            # spatial observation point
    t_ring_min = float(ring_cfg["t_ring_min"])    # start of ringdown regime
    n_ring = int(ring_cfg.get("n_ring_points", 200))

    # Ringdown evaluation times: uniformly spaced in [t_ring_min, tmax]
    t_ring_pts = np.linspace(t_ring_min, tmax, n_ring)
    X_ring = np.stack([np.full(n_ring, x_ring), t_ring_pts], axis=1)

    # The "observed" values for this PointSetBC are zero because we
    # implement the ringdown constraint as an OperatorBC instead.
    # OperatorBC returns the residual: φ_NN(x₀,t) - template(t).

    def ringdown_residual(inputs, outputs, X):
        """Return φ_NN(x₀, t) − A·exp(−t/τ)·cos(ω·t + φ₀)."""
        t = inputs[:, 1:2]
        # Use absolute values for ω, τ to keep them physically meaningful
        # (tau must be positive; omega is a frequency so positive)
        tau_abs = torch.abs(tau_var)
        omega_abs = torch.abs(omega_var)
        template = A_ring_var * torch.exp(-t / tau_abs) * torch.cos(omega_abs * t + phi0_var)
        return outputs - template

    # We use OperatorBC on a fixed set of points during the ringdown phase.
    # The boundary selector picks only points at x=x_ring and t >= t_ring_min.
    # However, OperatorBC uses random sampling from the geometry, which won't
    # reliably hit our observation point. Instead, use PointSetOperatorBC
    # (if available) or implement via callback.

    # Approach: use a custom OperatorBC with fixed evaluation points.
    # DeepXDE's PointSetBC only compares against fixed values, but we need
    # the template to be a function of learnable parameters. So we use
    # PointSetOperatorBC if available, otherwise a callback.

    # Check if PointSetOperatorBC exists in dde.icbc
    if hasattr(dde.icbc, "PointSetOperatorBC"):
        def _ring_op(inputs, outputs, X):
            t = inputs[:, 1:2]
            tau_abs = torch.abs(tau_var)
            omega_abs = torch.abs(omega_var)
            template = A_ring_var * torch.exp(-t / tau_abs) * torch.cos(omega_abs * t + phi0_var)
            return outputs - template

        ring_bc = dde.icbc.PointSetOperatorBC(X_ring, np.zeros((n_ring, 1)), _ring_op)
        ic_bcs.append(ring_bc)
        print(f"[RING] Using PointSetOperatorBC: {n_ring} points at x={x_ring}, "
              f"t=[{t_ring_min}, {tmax}]")
        use_ring_callback = False
    else:
        # Fallback: implement ringdown loss purely through a callback
        print(f"[RING] PointSetOperatorBC not available; using callback approach")
        print(f"[RING] Ringdown constraint: {n_ring} points at x={x_ring}, "
              f"t=[{t_ring_min}, {tmax}]")
        use_ring_callback = True

    Nr = int(cfg["pinn"]["Nr"])
    Ni = int(cfg["pinn"]["Ni"])
    Nb = int(cfg["pinn"]["Nb"])

    data = dde.data.TimePDE(
        geomtime,
        pde,
        ic_bcs,
        num_domain=Nr,
        num_boundary=Nb,
        num_initial=Ni,
    )

    layers = [2] + [int(w) for w in cfg["pinn"]["hidden_layers"]] + [1]
    net = dde.nn.FNN(layers, "tanh", "Glorot uniform")

    A_bound = float(cfg["initial_data"]["A"])
    net.apply_output_transform(lambda x, y: A_bound * torch.tanh(y))

    model = dde.Model(data, net)

    return (model, data, M_var, omega_var, tau_var, A_ring_var, phi0_var,
            fd, X_ring, use_ring_callback)


# ------------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------------

class ParamTracker(dde.callbacks.Callback):
    """Track all 5 learnable parameters during training."""

    def __init__(self, M_var, omega_var, tau_var, A_ring_var, phi0_var,
                 M_true, omega_true, tau_true, log_every=500):
        super().__init__()
        self.M_var = M_var
        self.omega_var = omega_var
        self.tau_var = tau_var
        self.A_ring_var = A_ring_var
        self.phi0_var = phi0_var
        self.M_true = M_true
        self.omega_true = omega_true
        self.tau_true = tau_true
        self.log_every = log_every

        self.M_history = []
        self.omega_history = []
        self.tau_history = []
        self.A_ring_history = []
        self.phi0_history = []

    def on_epoch_end(self):
        M_val = self.M_var.item()
        omega_val = abs(self.omega_var.item())
        tau_val = abs(self.tau_var.item())
        A_val = self.A_ring_var.item()
        phi0_val = self.phi0_var.item()

        self.M_history.append(M_val)
        self.omega_history.append(omega_val)
        self.tau_history.append(tau_val)
        self.A_ring_history.append(A_val)
        self.phi0_history.append(phi0_val)

        step = self.model.train_state.step
        if step % self.log_every == 0:
            M_err = abs(M_val - self.M_true) / self.M_true * 100
            omega_err = abs(omega_val - self.omega_true) / self.omega_true * 100
            tau_err = abs(tau_val - self.tau_true) / self.tau_true * 100
            print(f"  [PARAMS] step {step}: "
                  f"M={M_val:.6f} ({M_err:.2f}%), "
                  f"ω={omega_val:.6f} ({omega_err:.2f}%), "
                  f"τ={tau_val:.4f} ({tau_err:.2f}%), "
                  f"A={A_val:.6f}, φ₀={phi0_val:.4f}")


class RingdownLossCallback(dde.callbacks.Callback):
    """Compute ringdown template loss and push gradients to (ω, τ, A, φ₀).

    Used as fallback when PointSetOperatorBC is not available.
    """

    def __init__(self, X_ring_np, omega_var, tau_var, A_ring_var, phi0_var,
                 lambda_ring, log_every=500):
        super().__init__()
        self.X_ring_np = X_ring_np
        self.omega_var = omega_var
        self.tau_var = tau_var
        self.A_ring_var = A_ring_var
        self.phi0_var = phi0_var
        self.lambda_ring = lambda_ring
        self.log_every = log_every
        self.ring_losses = []

    def on_epoch_end(self):
        step = self.model.train_state.step

        # Predict at ringdown points
        phi_pred = self.model.predict(self.X_ring_np)
        phi_pred_t = torch.as_tensor(phi_pred, dtype=torch.float64)

        t_vals = torch.as_tensor(self.X_ring_np[:, 1:2], dtype=torch.float64)

        tau_abs = torch.abs(self.tau_var)
        omega_abs = torch.abs(self.omega_var)
        template = (self.A_ring_var
                     * torch.exp(-t_vals / tau_abs)
                     * torch.cos(omega_abs * t_vals + self.phi0_var))

        ring_loss = torch.mean((phi_pred_t - template) ** 2)
        scaled_loss = self.lambda_ring * ring_loss

        # Backward pass on ringdown params
        for v in [self.omega_var, self.tau_var, self.A_ring_var, self.phi0_var]:
            if v.grad is not None:
                v.grad.zero_()
        scaled_loss.backward()

        lr = 1e-4
        with torch.no_grad():
            for v in [self.omega_var, self.tau_var, self.A_ring_var, self.phi0_var]:
                if v.grad is not None:
                    v.data -= lr * v.grad
                    v.grad.zero_()

        self.ring_losses.append(ring_loss.item())

        if step % self.log_every == 0:
            print(f"  [RING] step {step}: ring_loss={ring_loss.item():.6e}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    name = cfg["experiment"]["name"]

    inv_cfg = cfg["inverse"]
    if not inv_cfg.get("enabled", False):
        raise ValueError("inverse.enabled must be true in config")

    ring_cfg = cfg["ringdown"]

    # True values
    M_true = float(cfg["physics"]["M"])
    omega_true = float(cfg["qnm"]["omega_theory"])
    tau_true = float(cfg["qnm"]["tau_theory"])

    # Initial guesses
    M_init = float(inv_cfg["M_init"])
    omega_init = float(ring_cfg["omega_init"])
    tau_init = float(ring_cfg["tau_init"])
    A_ring_init = float(ring_cfg["A_ring_init"])
    phi0_init = float(ring_cfg["phi0_init"])
    lambda_data = float(inv_cfg.get("lambda_data", 10.0))
    lambda_ring = float(ring_cfg.get("lambda_ring", 1.0))

    noise_level = float(inv_cfg["noise_level"])

    seed = int(cfg["pinn"]["seed"])
    dde.config.set_random_seed(seed)
    dde.config.set_default_float(cfg["pinn"]["dtype"])

    print("=" * 70)
    print("INVERSE PINN + QNM EXTRACTION")
    print(f"  True M     = {M_true}")
    print(f"  Initial M  = {M_init}")
    print(f"  True ω     = {omega_true}")
    print(f"  Initial ω  = {omega_init}")
    print(f"  True τ     = {tau_true}")
    print(f"  Initial τ  = {tau_init}")
    print(f"  A_ring₀    = {A_ring_init}")
    print(f"  φ₀         = {phi0_init}")
    print(f"  Noise level = {noise_level:.2%}")
    print(f"  λ_data     = {lambda_data}")
    print(f"  λ_ring     = {lambda_ring}")
    print("=" * 70)

    # Build model
    (model, data, M_var, omega_var, tau_var, A_ring_var, phi0_var,
     fd, X_ring, use_ring_callback) = build_inverse_qnm_model(
        cfg, M_init, omega_init, tau_init, A_ring_init, phi0_init)

    x_fd, t_fd, phi_fd = fd["x"], fd["t"], fd["phi"]

    # All 5 learnable variables
    ext_vars = [M_var, omega_var, tau_var, A_ring_var, phi0_var]

    # Loss weights
    base_lambda = [float(w) for w in cfg["pinn"]["lambda"]]
    # If PointSetOperatorBC is used, we have an extra BC term for ringdown:
    #   [r, r_x, r_t, ic, iv, bl, br, data_obs, ring]
    if not use_ring_callback:
        loss_weights = base_lambda + [lambda_data, lambda_ring]
    else:
        # Without PointSetOperatorBC, ringdown is handled in callback:
        #   [r, r_x, r_t, ic, iv, bl, br, data_obs]
        loss_weights = base_lambda + [lambda_data]

    # Tracking callback
    tracker = ParamTracker(M_var, omega_var, tau_var, A_ring_var, phi0_var,
                           M_true, omega_true, tau_true)

    # Adaptive sampling
    adaptive_cfg = cfg["pinn"].get("adaptive_sampling", {})
    callbacks = [tracker]

    if use_ring_callback:
        ring_cb = RingdownLossCallback(
            X_ring, omega_var, tau_var, A_ring_var, phi0_var, lambda_ring)
        callbacks.append(ring_cb)

    if adaptive_cfg.get("enabled", False):
        from src.pinn import GreedyResampler, _make_pde_residual_only
        method = adaptive_cfg.get("method", "RAD")
        period = int(adaptive_cfg.get("period", 1000))
        if method == "greedy":
            pde_func = _make_pde_residual_only(cfg)
            sampler = GreedyResampler(
                pde_residual=pde_func,
                period=period,
                num_candidates=int(adaptive_cfg.get("num_candidates", 50000)),
                greedy_fraction=float(adaptive_cfg.get("greedy_fraction", 0.5)),
                eval_batch_size=int(adaptive_cfg.get("eval_batch_size", 5000)),
            )
            callbacks.append(sampler)
    else:
        resample_period = int(cfg["pinn"]["adam"]["resample_period"])
        callbacks.append(dde.callbacks.PDEPointResampler(period=resample_period))

    # --- Adam phase ---
    adam_cfg = cfg["pinn"]["adam"]
    adam_iters = int(adam_cfg["iters"])
    lr = float(adam_cfg["lr"])

    print(f"\n[ADAM] Training {adam_iters} iterations, lr={lr}")
    model.compile(
        "adam", lr=lr,
        loss_weights=loss_weights,
        external_trainable_variables=ext_vars,
    )
    losshistory_adam, train_state = model.train(
        iterations=adam_iters,
        display_every=500,
        callbacks=callbacks,
    )

    # Report after Adam
    M_after = M_var.item()
    omega_after = abs(omega_var.item())
    tau_after = abs(tau_var.item())
    print(f"\n[ADAM] Done.")
    print(f"  M     = {M_after:.6f}  (err = {abs(M_after - M_true)/M_true*100:.4f}%)")
    print(f"  ω     = {omega_after:.6f} (err = {abs(omega_after - omega_true)/omega_true*100:.4f}%)")
    print(f"  τ     = {tau_after:.4f}  (err = {abs(tau_after - tau_true)/tau_true*100:.4f}%)")

    # --- L-BFGS phase ---
    lbfgs_cfg = cfg["pinn"]["lbfgs"]
    lbfgs_iters = int(lbfgs_cfg["iters"])

    tracker_lb = ParamTracker(M_var, omega_var, tau_var, A_ring_var, phi0_var,
                              M_true, omega_true, tau_true)
    callbacks_lb = [tracker_lb]

    if use_ring_callback:
        ring_cb_lb = RingdownLossCallback(
            X_ring, omega_var, tau_var, A_ring_var, phi0_var, lambda_ring)
        callbacks_lb.append(ring_cb_lb)

    if adaptive_cfg.get("enabled", False) and adaptive_cfg.get("method") == "greedy":
        pde_func = _make_pde_residual_only(cfg)
        sampler_lb = GreedyResampler(
            pde_residual=pde_func,
            period=int(adaptive_cfg.get("period", 1000)),
            num_candidates=int(adaptive_cfg.get("num_candidates", 50000)),
            greedy_fraction=float(adaptive_cfg.get("greedy_fraction", 0.5)),
            eval_batch_size=int(adaptive_cfg.get("eval_batch_size", 5000)),
        )
        callbacks_lb.append(sampler_lb)

    print(f"\n[L-BFGS] Training {lbfgs_iters} iterations")
    dde.optimizers.config.set_LBFGS_options(maxcor=100, gtol=1e-08)
    model.compile(
        "L-BFGS",
        loss_weights=loss_weights,
        external_trainable_variables=ext_vars,
    )
    losshistory_lbfgs, _ = model.train(
        iterations=lbfgs_iters,
        display_every=500,
        callbacks=callbacks_lb,
    )

    # Final values
    M_final = M_var.item()
    omega_final = abs(omega_var.item())
    tau_final = abs(tau_var.item())
    A_ring_final = A_ring_var.item()
    phi0_final = phi0_var.item()

    M_err = abs(M_final - M_true) / M_true * 100
    omega_err = abs(omega_final - omega_true) / omega_true * 100
    tau_err = abs(tau_final - tau_true) / tau_true * 100

    print(f"\n[L-BFGS] Done.")
    print(f"  M     = {M_final:.6f}  (err = {M_err:.4f}%)")
    print(f"  ω     = {omega_final:.6f} (err = {omega_err:.4f}%)")
    print(f"  τ     = {tau_final:.4f}  (err = {tau_err:.4f}%)")
    print(f"  A     = {A_ring_final:.6f}")
    print(f"  φ₀    = {phi0_final:.4f}")

    # --- Evaluation ---
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    phi_pinn = eval_on_grid(model, x=x_fd, t=t_fd, dtype=cfg["pinn"]["dtype"])

    metrics = {
        "RMSD": rmsd(phi_fd, phi_pinn),
        "MAD": mad(phi_fd, phi_pinn),
        "RL2": rl2(phi_fd, phi_pinn),
        "M_true": M_true,
        "M_init": M_init,
        "M_learned": M_final,
        "M_err_pct": M_err,
        "omega_true": omega_true,
        "omega_init": omega_init,
        "omega_learned": omega_final,
        "omega_err_pct": omega_err,
        "tau_true": tau_true,
        "tau_init": tau_init,
        "tau_learned": tau_final,
        "tau_err_pct": tau_err,
        "A_ring_learned": A_ring_final,
        "phi0_learned": phi0_final,
        "noise_level": noise_level,
    }

    print(f"  RMSD:  {metrics['RMSD']:.6f}")
    print(f"  MAD:   {metrics['MAD']:.6f}")
    print(f"  RL2:   {metrics['RL2']:.6f}")
    print(f"  M:     {M_final:.6f}  (true: {M_true}, err: {M_err:.4f}%)")
    print(f"  ω:     {omega_final:.6f} (true: {omega_true}, err: {omega_err:.4f}%)")
    print(f"  τ:     {tau_final:.4f}  (true: {tau_true}, err: {tau_err:.4f}%)")

    # --- Save outputs ---
    outdir = os.path.join("outputs", "pinn", name)
    ensure_dir(outdir)

    np.savez_compressed(os.path.join(outdir, f"{name}_fd.npz"), **fd)
    np.savez_compressed(os.path.join(outdir, f"{name}_pinn.npz"),
                        x=x_fd, t=t_fd, phi=phi_pinn)
    save_json(os.path.join(outdir, "metrics.json"), metrics)

    # Save parameter histories
    M_hist = tracker.M_history + tracker_lb.M_history
    omega_hist = tracker.omega_history + tracker_lb.omega_history
    tau_hist = tracker.tau_history + tracker_lb.tau_history
    A_hist = tracker.A_ring_history + tracker_lb.A_ring_history
    phi0_hist = tracker.phi0_history + tracker_lb.phi0_history

    save_json(os.path.join(outdir, "param_history.json"), {
        "M_history": M_hist,
        "omega_history": omega_hist,
        "tau_history": tau_hist,
        "A_ring_history": A_hist,
        "phi0_history": phi0_hist,
        "M_true": M_true,
        "M_init": M_init,
        "omega_true": omega_true,
        "omega_init": omega_init,
        "tau_true": tau_true,
        "tau_init": tau_init,
    })

    # --- Standard PINN plots ---
    times = [10.0, 20.0, 30.0, 40.0]
    pot = cfg["physics"]["potential"].title()
    l_val = cfg["physics"]["l"]

    plot_snapshots(
        x_fd, t_fd, phi_fd, phi_pinn, times,
        outpath=os.path.join(outdir, "snapshots.png"),
        title=f"Inverse+QNM PINN — {pot} (l={l_val}), M={M_final:.4f}",
    )
    plot_abs_diff_snapshots(
        x_fd, t_fd, phi_fd, phi_pinn, times,
        outpath=os.path.join(outdir, "abs_diff_snapshots.png"),
        title=f"Abs difference — {pot} (l={l_val})",
    )
    plot_snapshots_zoomed(
        x_fd, t_fd, phi_fd, phi_pinn, times,
        outpath=os.path.join(outdir, "snapshots_zoomed.png"),
        title=f"Snapshots (zoomed) — {pot} (l={l_val})",
    )
    plot_error_heatmap(
        x_fd, t_fd, phi_fd, phi_pinn,
        outpath=os.path.join(outdir, "error_heatmap.png"),
        title=f"Pointwise error — {pot} (l={l_val})",
    )
    plot_error_heatmap(
        x_fd, t_fd, phi_fd, phi_pinn,
        outpath=os.path.join(outdir, "error_heatmap_zoomed.png"),
        title=f"Pointwise error (zoomed) — {pot} (l={l_val})",
        xlim=(-20.0, 60.0),
    )

    xq = float(cfg["evaluation"]["xq"])
    ix = int(np.argmin(np.abs(x_fd - xq)))
    plot_ringdown_overlay(
        t_fd, phi_fd[:, ix], phi_pinn[:, ix],
        outpath=os.path.join(outdir, "ringdown_overlay.png"),
        title=f"Ringdown — {pot} (l={l_val})",
        xq=xq,
    )

    # --- Parameter convergence plots ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # M convergence
    ax = axes[0, 0]
    ax.plot(M_hist, "b-", linewidth=0.8, label="Learned M")
    ax.axhline(M_true, color="r", linestyle="--", linewidth=1.5,
               label=f"True M = {M_true}")
    ax.axhline(M_init, color="gray", linestyle=":", linewidth=1.0,
               label=f"Init M = {M_init}")
    ax.set_ylabel("M")
    ax.set_title(f"Mass convergence (err = {M_err:.3f}%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ω convergence
    ax = axes[0, 1]
    ax.plot(omega_hist, "b-", linewidth=0.8, label="Learned ω")
    ax.axhline(omega_true, color="r", linestyle="--", linewidth=1.5,
               label=f"True ω = {omega_true}")
    ax.axhline(omega_init, color="gray", linestyle=":", linewidth=1.0,
               label=f"Init ω = {omega_init}")
    ax.set_ylabel("ω")
    ax.set_title(f"Frequency convergence (err = {omega_err:.3f}%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # τ convergence
    ax = axes[1, 0]
    ax.plot(tau_hist, "b-", linewidth=0.8, label="Learned τ")
    ax.axhline(tau_true, color="r", linestyle="--", linewidth=1.5,
               label=f"True τ = {tau_true}")
    ax.axhline(tau_init, color="gray", linestyle=":", linewidth=1.0,
               label=f"Init τ = {tau_init}")
    ax.set_ylabel("τ")
    ax.set_title(f"Damping time convergence (err = {tau_err:.3f}%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # A_ring convergence
    ax = axes[1, 1]
    ax.plot(A_hist, "b-", linewidth=0.8, label="Learned A")
    ax.axhline(A_ring_init, color="gray", linestyle=":", linewidth=1.0,
               label=f"Init A = {A_ring_init}")
    ax.set_ylabel("A")
    ax.set_title("Ringdown amplitude convergence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # φ₀ convergence
    ax = axes[2, 0]
    ax.plot(phi0_hist, "b-", linewidth=0.8, label="Learned φ₀")
    ax.axhline(phi0_init, color="gray", linestyle=":", linewidth=1.0,
               label=f"Init φ₀ = {phi0_init}")
    ax.set_ylabel("φ₀")
    ax.set_title("Phase convergence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Ringdown fit overlay
    ax = axes[2, 1]
    t_plot = np.linspace(float(ring_cfg["t_ring_min"]),
                         float(cfg["domain"]["tmax"]), 500)
    template_plot = (A_ring_final
                     * np.exp(-t_plot / tau_final)
                     * np.cos(omega_final * t_plot + phi0_final))
    # FD ringdown at x=xq
    ax.plot(t_fd, phi_fd[:, ix], "k-", linewidth=0.8, label="FD (truth)")
    ax.plot(t_plot, template_plot, "r--", linewidth=1.2,
            label=f"Template: A·e^(-t/τ)·cos(ωt+φ₀)")
    ax.set_xlabel("t")
    ax.set_ylabel("φ")
    ax.set_title(f"Ringdown template fit at x*={xq}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    for ax in axes[-1]:
        ax.set_xlabel("Training step (logged)")

    fig.suptitle(f"Inverse+QNM Parameter Convergence (noise={noise_level:.1%})",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "param_convergence.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[INV+QNM] Outputs saved to: {outdir}")
    print(f"[INV+QNM] Done!")


if __name__ == "__main__":
    main()

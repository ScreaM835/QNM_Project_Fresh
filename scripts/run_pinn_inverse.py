"""
Inverse PINN: Learn black hole mass M from noisy observations.

Instead of fixing M and solving the forward problem, this script treats M
as a learnable parameter.  The PINN simultaneously learns φ(x,t) and M
by minimising:

    L = λ_pde · L_pde(φ; M)  +  λ_data · L_data(φ, φ_obs)
        + λ_ic · L_ic  +  λ_iv · L_iv  +  λ_bl · L_bl  +  λ_br · L_br

The Zerilli potential V(x; M) depends on M, so the PDE itself changes as
M is optimised.  At convergence, the learned M should match the true value.

Usage:
    python scripts/run_pinn_inverse.py --config configs/zerilli_l2_inverse.yaml

Requires ``inverse`` section in config:
    inverse:
      enabled: true
      M_init: 0.8          # initial guess for M (true = 1.0)
      noise_level: 0.01    # std of Gaussian noise added to FD data
      n_obs: 500           # number of observation points
      obs_t_min: 0.0       # observation time range
      obs_t_max: 50.0
      lambda_data: 10.0    # weight for data-fitting loss
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
# Build inverse model
# ------------------------------------------------------------------

def build_inverse_model(cfg, M_param):
    """Build a DeepXDE model where the PDE uses a learnable M.

    Parameters
    ----------
    cfg : dict
        Full experiment config.
    M_param : torch.nn.Parameter
        Learnable mass parameter (scalar).

    Returns
    -------
    model : dde.Model
    data : dde.data.TimePDE
    """
    l = int(cfg["physics"]["l"])

    xmin = float(cfg["domain"]["xmin"])
    xmax = float(cfg["domain"]["xmax"])
    tmin = float(cfg["domain"]["tmin"])
    tmax = float(cfg["domain"]["tmax"])

    geom = dde.geometry.Interval(xmin, xmax)
    timedomain = dde.geometry.TimeDomain(tmin, tmax)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    # --- PDE with learnable M ---
    def pde(x, y):
        y_xx = dde.grad.hessian(y, x, i=0, j=0)
        y_tt = dde.grad.hessian(y, x, i=1, j=1)
        V = _V_zerilli_torch(x[:, 0:1], M_param, l)
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

    # --- IC / BCs (use config M_true for these -- they are "known" data) ---
    M_true = float(cfg["physics"]["M"])
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
        target_t = torch.as_tensor(
            target, dtype=outputs.dtype, device=outputs.device
        )
        return phi_t - target_t

    ic_vel = dde.icbc.OperatorBC(
        geomtime, ic_vel_func,
        lambda x, on_initial: on_initial,
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
    return model, data


# ------------------------------------------------------------------
# Observation data callback
# ------------------------------------------------------------------

class ObservationLoss(dde.callbacks.Callback):
    """Adds a data-fitting loss term to the PINN training.

    At each training step, evaluates the network at observation points
    and computes MSE against (noisy) observed values.  The gradient of
    this loss is added to the parameter updates via a manual backward pass.
    """

    def __init__(self, X_obs, phi_obs, M_param, lambda_data=10.0, log_every=500):
        super().__init__()
        self.X_obs_np = X_obs
        self.phi_obs_t = torch.as_tensor(phi_obs, dtype=torch.float64)
        self.M_param = M_param
        self.lambda_data = lambda_data
        self.log_every = log_every
        self.data_losses = []
        self.M_history = []

    def on_train_begin(self):
        print(f"[OBS] {len(self.X_obs_np)} observation points, λ_data={self.lambda_data}")
        print(f"[OBS] M_init = {self.M_param.item():.6f}")

    def on_epoch_end(self):
        step = self.model.train_state.step

        # Predict at observation points
        phi_pred = self.model.predict(self.X_obs_np)
        phi_pred_t = torch.as_tensor(phi_pred, dtype=torch.float64)

        # Data loss
        data_loss = torch.mean((phi_pred_t - self.phi_obs_t) ** 2)
        scaled_loss = self.lambda_data * data_loss

        # Backward pass on M_param only (network grads handled by DeepXDE)
        if self.M_param.grad is not None:
            self.M_param.grad.zero_()
        scaled_loss.backward()

        # Update M using simple gradient descent
        with torch.no_grad():
            lr_M = 1e-4  # small learning rate for M
            if self.M_param.grad is not None:
                self.M_param.data -= lr_M * self.M_param.grad
                self.M_param.grad.zero_()

        self.data_losses.append(data_loss.item())
        self.M_history.append(self.M_param.item())

        if step % self.log_every == 0:
            print(f"  [OBS] step {step}: data_loss={data_loss.item():.6e}, "
                  f"M={self.M_param.item():.6f}")


# ------------------------------------------------------------------
# Alternative: External variable approach (simpler, more robust)
# ------------------------------------------------------------------

def build_inverse_model_extvar(cfg, M_init):
    """Build model using DeepXDE's external trainable variable for M.

    This is cleaner: M is registered as an external variable and
    DeepXDE's optimizer updates it alongside network weights.
    """
    l = int(cfg["physics"]["l"])

    xmin = float(cfg["domain"]["xmin"])
    xmax = float(cfg["domain"]["xmax"])
    tmin = float(cfg["domain"]["tmin"])
    tmax = float(cfg["domain"]["tmax"])

    geom = dde.geometry.Interval(xmin, xmax)
    timedomain = dde.geometry.TimeDomain(tmin, tmax)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    # Learnable M as external variable (dde.Variable returns a tensor directly)
    M_var = dde.Variable(M_init, dtype=torch.float64)

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

    # IC / BCs
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

    # Add observation data (noisy FD solution) as PointSetBC
    inv_cfg = cfg["inverse"]
    n_obs = int(inv_cfg["n_obs"])
    noise_level = float(inv_cfg["noise_level"])
    obs_tmin = float(inv_cfg.get("obs_t_min", tmin))
    obs_tmax = float(inv_cfg.get("obs_t_max", tmax))

    # Generate FD solution for observation data
    print("[INV] Generating FD reference for observation data...")
    fd = solve_fd(cfg)
    x_fd, t_fd, phi_fd = fd["x"], fd["t"], fd["phi"]

    # Sample observation points randomly within the domain
    rng = np.random.RandomState(42)
    obs_x = rng.uniform(xmin, xmax, size=n_obs)
    obs_t = rng.uniform(obs_tmin, obs_tmax, size=n_obs)

    # Interpolate FD solution at observation points
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((t_fd, x_fd), phi_fd, method="linear",
                                      bounds_error=False, fill_value=0.0)
    phi_obs_clean = interp(np.stack([obs_t, obs_x], axis=1))

    # Add noise
    noise = rng.normal(0, noise_level * np.abs(phi_obs_clean).max(), size=n_obs)
    phi_obs = phi_obs_clean + noise

    X_obs = np.stack([obs_x, obs_t], axis=1)

    print(f"[INV] Observations: {n_obs} points, noise σ={noise_level:.2%} "
          f"(abs σ={noise_level * np.abs(phi_obs_clean).max():.6f})")
    print(f"[INV] Obs range: x=[{obs_x.min():.1f}, {obs_x.max():.1f}], "
          f"t=[{obs_t.min():.1f}, {obs_t.max():.1f}]")

    # Add as PointSetBC
    observe = dde.icbc.PointSetBC(X_obs, phi_obs.reshape(-1, 1), component=0)
    ic_bcs.append(observe)

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

    return model, data, M_var, fd


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

    M_true = float(cfg["physics"]["M"])
    M_init = float(inv_cfg["M_init"])
    noise_level = float(inv_cfg["noise_level"])
    lambda_data = float(inv_cfg.get("lambda_data", 10.0))

    seed = int(cfg["pinn"]["seed"])
    dde.config.set_random_seed(seed)
    dde.config.set_default_float(cfg["pinn"]["dtype"])

    print("=" * 60)
    print(f"INVERSE PINN: Learning M from noisy observations")
    print(f"  True M     = {M_true}")
    print(f"  Initial M  = {M_init}")
    print(f"  Noise level = {noise_level:.2%}")
    print(f"  λ_data     = {lambda_data}")
    print("=" * 60)

    # Build model with external variable M
    model, data, M_var, fd = build_inverse_model_extvar(cfg, M_init)
    x_fd, t_fd, phi_fd = fd["x"], fd["t"], fd["phi"]

    # Loss weights: [r, r_x, r_t, ic, iv, bl, br, data_obs]
    base_lambda = [float(w) for w in cfg["pinn"]["lambda"]]
    loss_weights = base_lambda + [lambda_data]

    # Tracking callback for M
    class MTracker(dde.callbacks.Callback):
        def __init__(self, M_var, M_true, log_every=500):
            super().__init__()
            self.M_var = M_var
            self.M_true = M_true
            self.log_every = log_every
            self.M_history = []

        def on_epoch_end(self):
            M_val = self.M_var.item()
            self.M_history.append(M_val)
            step = self.model.train_state.step
            if step % self.log_every == 0:
                err = abs(M_val - self.M_true) / self.M_true * 100
                print(f"  [M] step {step}: M = {M_val:.6f}  "
                      f"(true = {self.M_true}, err = {err:.4f}%)")

    tracker = MTracker(M_var, M_true)

    # Adaptive sampling
    adaptive_cfg = cfg["pinn"].get("adaptive_sampling", {})
    callbacks = [tracker]

    if adaptive_cfg.get("enabled", False):
        from src.pinn import GreedyResampler, _make_pde_residual_only
        method = adaptive_cfg.get("method", "RAD")
        period = int(adaptive_cfg.get("period", 1000))
        if method == "greedy":
            # Note: the greedy resampler uses fixed M from config for residual
            # evaluation. This is OK — it's just for point selection, not training.
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
        external_trainable_variables=[M_var],
    )
    losshistory_adam, train_state = model.train(
        iterations=adam_iters,
        display_every=500,
        callbacks=callbacks,
    )

    M_after_adam = M_var.item()
    err_adam = abs(M_after_adam - M_true) / M_true * 100
    print(f"\n[ADAM] Done. M = {M_after_adam:.6f} (err = {err_adam:.4f}%)")

    # --- L-BFGS phase ---
    lbfgs_cfg = cfg["pinn"]["lbfgs"]
    lbfgs_iters = int(lbfgs_cfg["iters"])

    # Fresh callbacks for L-BFGS
    tracker_lb = MTracker(M_var, M_true)
    callbacks_lb = [tracker_lb]

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
        external_trainable_variables=[M_var],
    )
    losshistory_lbfgs, _ = model.train(
        iterations=lbfgs_iters,
        display_every=500,
        callbacks=callbacks_lb,
    )

    M_final = M_var.item()
    err_final = abs(M_final - M_true) / M_true * 100
    print(f"\n[L-BFGS] Done. M = {M_final:.6f} (err = {err_final:.4f}%)")

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
        "M_err_pct": err_final,
        "noise_level": noise_level,
    }

    print(f"  RMSD: {metrics['RMSD']:.6f}")
    print(f"  MAD:  {metrics['MAD']:.6f}")
    print(f"  RL2:  {metrics['RL2']:.6f}")
    print(f"  M_learned: {M_final:.6f} (true: {M_true}, err: {err_final:.4f}%)")

    # --- Save outputs ---
    outdir = os.path.join("outputs", "pinn", name)
    ensure_dir(outdir)

    np.savez_compressed(os.path.join(outdir, f"{name}_fd.npz"), **fd)
    np.savez_compressed(os.path.join(outdir, f"{name}_pinn.npz"),
                        x=x_fd, t=t_fd, phi=phi_pinn)
    save_json(os.path.join(outdir, "metrics.json"), metrics)

    # Save M history
    M_hist = tracker.M_history + tracker_lb.M_history
    save_json(os.path.join(outdir, "M_history.json"), {
        "M_history": M_hist,
        "M_true": M_true,
        "M_init": M_init,
    })

    # --- Plots ---
    times = [10.0, 20.0, 30.0, 40.0]
    pot = cfg["physics"]["potential"].title()
    l_val = cfg["physics"]["l"]

    plot_snapshots(
        x_fd, t_fd, phi_fd, phi_pinn, times,
        outpath=os.path.join(outdir, "snapshots.png"),
        title=f"Inverse PINN — {pot} (l={l_val}), M_learned={M_final:.4f}",
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

    # M convergence plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(M_hist, "b-", linewidth=0.8, label="Learned M")
    ax.axhline(M_true, color="r", linestyle="--", linewidth=1.5,
               label=f"True M = {M_true}")
    ax.axhline(M_init, color="gray", linestyle=":", linewidth=1.0,
               label=f"Initial M = {M_init}")
    ax.set_xlabel("Training step (logged)")
    ax.set_ylabel("M")
    ax.set_title(f"Mass convergence (noise={noise_level:.1%}, final err={err_final:.3f}%)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "M_convergence.png"), dpi=150)
    plt.close(fig)

    print(f"\n[INV] Outputs saved to: {outdir}")
    print(f"[INV] Done!")


if __name__ == "__main__":
    main()

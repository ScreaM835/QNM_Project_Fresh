"""
Inverse PINN + QNM extraction (TWO-MODE TEMPLATE variant).

Same as run_pinn_inverse_qnm.py except the ringdown template is

    template(t) = A0·exp(-t/τ0)·cos(ω0·t + φ0)
                + A1·exp(-t/τ1)·cos(ω1·t + φ1)

i.e. fundamental + first overtone. Nine learnable parameters in total:
    M, ω0, τ0, A0, φ0, ω1, τ1, A1, φ1.

Motivation: single-mode template was found to bias ω because the
fundamental and first overtone interfere in the early ringdown. M4
(extract_qnm.py --two-mode) showed that adding the overtone removes
this bias on the forward field; this script ports the same idea into
the inverse PINN loss.

Config requirements: same as run_pinn_inverse_qnm.py plus optional
ringdown.overtone_init block:
    ringdown:
      omega1_init: 0.3467
      tau1_init:   3.674
      A1_init:     0.5
      phi1_init:   0.0

If overtone_init values are absent, defaults from
src.qnm._THEORY_OVERTONE are used.

Usage:
    python scripts/run_pinn_inverse_qnm_2mode.py \\
        --config configs/zerilli_l2_inverse_qnm_2mode.yaml
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
from src.qnm import _THEORY_OVERTONE


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
# Build inverse + 2-mode QNM model
# ------------------------------------------------------------------

def build_inverse_qnm_model_2mode(
    cfg,
    M_init,
    omega0_init, tau0_init, A0_init, phi0_init,
    omega1_init, tau1_init, A1_init, phi1_init,
):
    """Build model with 9 learnable params:
       M, (ω0,τ0,A0,φ0) fundamental, (ω1,τ1,A1,φ1) overtone."""
    l = int(cfg["physics"]["l"])

    xmin = float(cfg["domain"]["xmin"])
    xmax = float(cfg["domain"]["xmax"])
    tmin = float(cfg["domain"]["tmin"])
    tmax = float(cfg["domain"]["tmax"])

    geom = dde.geometry.Interval(xmin, xmax)
    timedomain = dde.geometry.TimeDomain(tmin, tmax)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    M_var       = dde.Variable(M_init,       dtype=torch.float64)
    omega0_var  = dde.Variable(omega0_init,  dtype=torch.float64)
    tau0_var    = dde.Variable(tau0_init,    dtype=torch.float64)
    A0_var      = dde.Variable(A0_init,      dtype=torch.float64)
    phi0_var    = dde.Variable(phi0_init,    dtype=torch.float64)
    omega1_var  = dde.Variable(omega1_init,  dtype=torch.float64)
    tau1_var    = dde.Variable(tau1_init,    dtype=torch.float64)
    A1_var      = dde.Variable(A1_init,      dtype=torch.float64)
    phi1_var    = dde.Variable(phi1_init,    dtype=torch.float64)

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

    # --- Two-mode ringdown template ---
    ring_cfg = cfg["ringdown"]
    x_ring = float(ring_cfg["x_obs"])
    t_ring_min = float(ring_cfg["t_ring_min"])
    n_ring = int(ring_cfg.get("n_ring_points", 200))

    t_ring_pts = np.linspace(t_ring_min, tmax, n_ring)
    X_ring = np.stack([np.full(n_ring, x_ring), t_ring_pts], axis=1)

    if hasattr(dde.icbc, "PointSetOperatorBC"):
        def _ring_op(inputs, outputs, X):
            t = inputs[:, 1:2]
            tau0_abs   = torch.abs(tau0_var)
            omega0_abs = torch.abs(omega0_var)
            tau1_abs   = torch.abs(tau1_var)
            omega1_abs = torch.abs(omega1_var)
            mode0 = A0_var * torch.exp(-t / tau0_abs) * torch.cos(omega0_abs * t + phi0_var)
            mode1 = A1_var * torch.exp(-t / tau1_abs) * torch.cos(omega1_abs * t + phi1_var)
            template = mode0 + mode1
            return outputs - template

        ring_bc = dde.icbc.PointSetOperatorBC(X_ring, np.zeros((n_ring, 1)), _ring_op)
        ic_bcs.append(ring_bc)
        print(f"[RING-2MODE] Using PointSetOperatorBC: {n_ring} points at "
              f"x={x_ring}, t=[{t_ring_min}, {tmax}]")
        use_ring_callback = False
    else:
        print(f"[RING-2MODE] PointSetOperatorBC not available; "
              f"callback fallback ({n_ring} pts, x={x_ring})")
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

    return (model, data,
            M_var, omega0_var, tau0_var, A0_var, phi0_var,
            omega1_var, tau1_var, A1_var, phi1_var,
            fd, X_ring, use_ring_callback)


# ------------------------------------------------------------------
# Param tracker (9 params)
# ------------------------------------------------------------------

class ParamTracker2Mode(dde.callbacks.Callback):
    def __init__(self, M_var,
                 omega0_var, tau0_var, A0_var, phi0_var,
                 omega1_var, tau1_var, A1_var, phi1_var,
                 M_true, omega_true, tau_true, log_every=500):
        super().__init__()
        self.vars = dict(
            M=M_var,
            omega0=omega0_var, tau0=tau0_var, A0=A0_var, phi0=phi0_var,
            omega1=omega1_var, tau1=tau1_var, A1=A1_var, phi1=phi1_var,
        )
        self.M_true = M_true
        self.omega_true = omega_true
        self.tau_true = tau_true
        self.log_every = log_every
        self.history = {k: [] for k in self.vars}

    def on_epoch_end(self):
        snap = {}
        for k, v in self.vars.items():
            val = v.item()
            if k.startswith("omega") or k.startswith("tau"):
                val = abs(val)
            snap[k] = val
            self.history[k].append(val)

        step = self.model.train_state.step
        if step % self.log_every == 0:
            M_err  = abs(snap["M"]      - self.M_true)     / self.M_true     * 100
            o0_err = abs(snap["omega0"] - self.omega_true) / self.omega_true * 100
            t0_err = abs(snap["tau0"]   - self.tau_true)   / self.tau_true   * 100
            print(f"  [PARAMS] step {step}: "
                  f"M={snap['M']:.6f} ({M_err:.2f}%), "
                  f"ω0={snap['omega0']:.6f} ({o0_err:.2f}%), "
                  f"τ0={snap['tau0']:.4f} ({t0_err:.2f}%), "
                  f"ω1={snap['omega1']:.4f}, τ1={snap['tau1']:.3f}, "
                  f"A0={snap['A0']:.3f}, A1={snap['A1']:.3f}")


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

    M_true = float(cfg["physics"]["M"])
    omega_true = float(cfg["qnm"]["omega_theory"])
    tau_true = float(cfg["qnm"]["tau_theory"])

    M_init = float(inv_cfg["M_init"])

    # Fundamental mode init (same fields as single-mode script)
    omega0_init = float(ring_cfg["omega_init"])
    tau0_init   = float(ring_cfg["tau_init"])
    A0_init     = float(ring_cfg["A_ring_init"])
    phi0_init   = float(ring_cfg["phi0_init"])

    # Overtone init: explicit config values override theory defaults
    potential = cfg["physics"]["potential"].lower()
    ell = int(cfg["physics"]["l"])
    ovt = _THEORY_OVERTONE.get(potential, {}).get(ell)
    if ovt is None:
        raise ValueError(
            f"No theory overtone available for potential={potential}, l={ell}; "
            f"set ringdown.omega1_init and ringdown.tau1_init explicitly")
    omega1_init = float(ring_cfg.get("omega1_init", ovt["omega"]))
    tau1_init   = float(ring_cfg.get("tau1_init",   ovt["tau"]))
    A1_init     = float(ring_cfg.get("A1_init",     0.5 * A0_init))
    phi1_init   = float(ring_cfg.get("phi1_init",   0.0))

    lambda_data = float(inv_cfg.get("lambda_data", 10.0))
    lambda_ring = float(ring_cfg.get("lambda_ring", 1.0))

    noise_level = float(inv_cfg["noise_level"])

    seed = int(cfg["pinn"]["seed"])
    dde.config.set_random_seed(seed)
    dde.config.set_default_float(cfg["pinn"]["dtype"])

    print("=" * 70)
    print("INVERSE PINN + 2-MODE QNM EXTRACTION")
    print(f"  True M  = {M_true}    Init M  = {M_init}")
    print(f"  True ω  = {omega_true}  Init ω0 = {omega0_init}  "
          f"Init ω1 = {omega1_init}")
    print(f"  True τ  = {tau_true}  Init τ0 = {tau0_init}  "
          f"Init τ1 = {tau1_init}")
    print(f"  Init A0 = {A0_init}   Init A1 = {A1_init}")
    print(f"  Init φ0 = {phi0_init}   Init φ1 = {phi1_init}")
    print(f"  Noise   = {noise_level:.2%}")
    print(f"  λ_data  = {lambda_data}   λ_ring = {lambda_ring}")
    print("=" * 70)

    (model, data,
     M_var, omega0_var, tau0_var, A0_var, phi0_var_t,
     omega1_var, tau1_var, A1_var, phi1_var,
     fd, X_ring, use_ring_callback) = build_inverse_qnm_model_2mode(
        cfg, M_init,
        omega0_init, tau0_init, A0_init, phi0_init,
        omega1_init, tau1_init, A1_init, phi1_init,
    )

    if use_ring_callback:
        raise RuntimeError(
            "PointSetOperatorBC missing from this DeepXDE install; "
            "the 2-mode template has no callback fallback yet.")

    x_fd, t_fd, phi_fd = fd["x"], fd["t"], fd["phi"]

    ext_vars = [M_var,
                omega0_var, tau0_var, A0_var, phi0_var_t,
                omega1_var, tau1_var, A1_var, phi1_var]

    base_lambda = [float(w) for w in cfg["pinn"]["lambda"]]
    # [r, r_x, r_t, ic, iv, bl, br, data_obs, ring]
    loss_weights = base_lambda + [lambda_data, lambda_ring]

    tracker = ParamTracker2Mode(
        M_var,
        omega0_var, tau0_var, A0_var, phi0_var_t,
        omega1_var, tau1_var, A1_var, phi1_var,
        M_true, omega_true, tau_true)

    adaptive_cfg = cfg["pinn"].get("adaptive_sampling", {})
    callbacks = [tracker]

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

    adam_cfg = cfg["pinn"]["adam"]
    adam_iters = int(adam_cfg["iters"])
    lr = float(adam_cfg["lr"])

    print(f"\n[ADAM] Training {adam_iters} iterations, lr={lr}")
    model.compile(
        "adam", lr=lr,
        loss_weights=loss_weights,
        external_trainable_variables=ext_vars,
    )
    model.train(
        iterations=adam_iters,
        display_every=500,
        callbacks=callbacks,
    )

    M_after = M_var.item()
    o0_after = abs(omega0_var.item())
    t0_after = abs(tau0_var.item())
    print(f"\n[ADAM] Done.")
    print(f"  M  = {M_after:.6f}  (err = {abs(M_after - M_true)/M_true*100:.4f}%)")
    print(f"  ω0 = {o0_after:.6f}  (err = {abs(o0_after - omega_true)/omega_true*100:.4f}%)")
    print(f"  τ0 = {t0_after:.4f}  (err = {abs(t0_after - tau_true)/tau_true*100:.4f}%)")

    lbfgs_cfg = cfg["pinn"]["lbfgs"]
    lbfgs_iters = int(lbfgs_cfg["iters"])

    tracker_lb = ParamTracker2Mode(
        M_var,
        omega0_var, tau0_var, A0_var, phi0_var_t,
        omega1_var, tau1_var, A1_var, phi1_var,
        M_true, omega_true, tau_true)
    callbacks_lb = [tracker_lb]

    if adaptive_cfg.get("enabled", False) and adaptive_cfg.get("method") == "greedy":
        from src.pinn import GreedyResampler, _make_pde_residual_only
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
    model.train(
        iterations=lbfgs_iters,
        display_every=500,
        callbacks=callbacks_lb,
    )

    M_final  = M_var.item()
    o0_final = abs(omega0_var.item())
    t0_final = abs(tau0_var.item())
    A0_final = A0_var.item()
    p0_final = phi0_var_t.item()
    o1_final = abs(omega1_var.item())
    t1_final = abs(tau1_var.item())
    A1_final = A1_var.item()
    p1_final = phi1_var.item()

    M_err  = abs(M_final  - M_true)     / M_true     * 100
    o0_err = abs(o0_final - omega_true) / omega_true * 100
    t0_err = abs(t0_final - tau_true)   / tau_true   * 100

    print(f"\n[L-BFGS] Done.")
    print(f"  M  = {M_final:.6f}  (err = {M_err:.4f}%)")
    print(f"  ω0 = {o0_final:.6f}  (err = {o0_err:.4f}%)")
    print(f"  τ0 = {t0_final:.4f}  (err = {t0_err:.4f}%)")
    print(f"  ω1 = {o1_final:.6f}  (theory ovt ≈ {omega1_init})")
    print(f"  τ1 = {t1_final:.4f}  (theory ovt ≈ {tau1_init})")
    print(f"  A0 = {A0_final:.4f}  A1 = {A1_final:.4f}")
    print(f"  φ0 = {p0_final:.4f}  φ1 = {p1_final:.4f}")

    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    phi_pinn = eval_on_grid(model, x=x_fd, t=t_fd, dtype=cfg["pinn"]["dtype"])

    metrics = {
        "RMSD": rmsd(phi_fd, phi_pinn),
        "MAD":  mad(phi_fd, phi_pinn),
        "RL2":  rl2(phi_fd, phi_pinn),
        "M_true": M_true,
        "M_init": M_init,
        "M_learned": M_final,
        "M_err_pct": M_err,
        "omega_true": omega_true,
        "omega0_init": omega0_init,
        "omega0_learned": o0_final,
        "omega0_err_pct": o0_err,
        "tau_true": tau_true,
        "tau0_init": tau0_init,
        "tau0_learned": t0_final,
        "tau0_err_pct": t0_err,
        "A0_learned":   A0_final,
        "phi0_learned": p0_final,
        "omega1_init":  omega1_init,
        "omega1_learned": o1_final,
        "tau1_init":    tau1_init,
        "tau1_learned": t1_final,
        "A1_init":      A1_init,
        "A1_learned":   A1_final,
        "phi1_init":    phi1_init,
        "phi1_learned": p1_final,
        "noise_level":  noise_level,
    }

    print(f"  RMSD: {metrics['RMSD']:.6f}")
    print(f"  MAD:  {metrics['MAD']:.6f}")
    print(f"  RL2:  {metrics['RL2']:.6f}")

    outdir = os.path.join("outputs", "pinn", name)
    ensure_dir(outdir)

    np.savez_compressed(os.path.join(outdir, f"{name}_fd.npz"), **fd)
    np.savez_compressed(os.path.join(outdir, f"{name}_pinn.npz"),
                        x=x_fd, t=t_fd, phi=phi_pinn)
    save_json(os.path.join(outdir, "metrics.json"), metrics)

    history = {f"{k}_history": tracker.history[k] + tracker_lb.history[k]
               for k in tracker.history}
    history.update({
        "M_true": M_true, "M_init": M_init,
        "omega_true": omega_true,
        "omega0_init": omega0_init, "omega1_init": omega1_init,
        "tau_true": tau_true,
        "tau0_init": tau0_init, "tau1_init": tau1_init,
    })
    save_json(os.path.join(outdir, "param_history.json"), history)

    times = [10.0, 20.0, 30.0, 40.0]
    pot = cfg["physics"]["potential"].title()
    l_val = cfg["physics"]["l"]

    plot_snapshots(
        x_fd, t_fd, phi_fd, phi_pinn, times,
        outpath=os.path.join(outdir, "snapshots.png"),
        title=f"Inverse+2-mode QNM PINN — {pot} (l={l_val}), M={M_final:.4f}",
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

    # --- 2-mode parameter convergence + template overlay ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H = history
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))

    def _hline(ax, val, color, ls, label):
        ax.axhline(val, color=color, linestyle=ls, linewidth=1.2, label=label)

    ax = axes[0, 0]
    ax.plot(H["M_history"], "b-", linewidth=0.8, label="Learned M")
    _hline(ax, M_true, "r", "--", f"True M = {M_true}")
    _hline(ax, M_init, "gray", ":", f"Init M = {M_init}")
    ax.set_title(f"M (err = {M_err:.3f}%)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(H["omega0_history"], "b-", linewidth=0.8, label="Learned ω0")
    _hline(ax, omega_true, "r", "--", f"True ω = {omega_true}")
    _hline(ax, omega0_init, "gray", ":", f"Init ω0 = {omega0_init}")
    ax.set_title(f"ω0 (err = {o0_err:.3f}%)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(H["tau0_history"], "b-", linewidth=0.8, label="Learned τ0")
    _hline(ax, tau_true, "r", "--", f"True τ = {tau_true}")
    _hline(ax, tau0_init, "gray", ":", f"Init τ0 = {tau0_init}")
    ax.set_title(f"τ0 (err = {t0_err:.3f}%)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(H["omega1_history"], "g-", linewidth=0.8, label="Learned ω1")
    _hline(ax, omega1_init, "gray", ":", f"Init ω1 = {omega1_init}")
    ax.set_title("ω1 (overtone)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(H["tau1_history"], "g-", linewidth=0.8, label="Learned τ1")
    _hline(ax, tau1_init, "gray", ":", f"Init τ1 = {tau1_init}")
    ax.set_title("τ1 (overtone)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(H["A0_history"], "b-", linewidth=0.8, label="A0")
    ax.plot(H["A1_history"], "g-", linewidth=0.8, label="A1")
    ax.set_title("Amplitudes")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    ax.plot(H["phi0_history"], "b-", linewidth=0.8, label="φ0")
    ax.plot(H["phi1_history"], "g-", linewidth=0.8, label="φ1")
    ax.set_title("Phases")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    t_plot = np.linspace(float(ring_cfg["t_ring_min"]),
                         float(cfg["domain"]["tmax"]), 500)
    mode0 = A0_final * np.exp(-t_plot / t0_final) * np.cos(o0_final * t_plot + p0_final)
    mode1 = A1_final * np.exp(-t_plot / t1_final) * np.cos(o1_final * t_plot + p1_final)
    template_plot = mode0 + mode1
    ax.plot(t_fd, phi_fd[:, ix], "k-", linewidth=0.8, label="FD truth")
    ax.plot(t_plot, template_plot, "r--", linewidth=1.2, label="Template (2-mode)")
    ax.plot(t_plot, mode0, "b:", linewidth=0.8, label="mode 0")
    ax.plot(t_plot, mode1, "g:", linewidth=0.8, label="mode 1")
    ax.set_xlabel("t"); ax.set_ylabel("φ")
    ax.set_title(f"2-mode template fit at x*={xq}")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[2, 2]
    ax.plot(t_fd, phi_fd[:, ix] - np.interp(t_fd, t_plot, template_plot),
            "k-", linewidth=0.6)
    ax.set_xlabel("t"); ax.set_ylabel("FD − template")
    ax.set_title("Template residual at x*")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Inverse+2-mode QNM Convergence (noise={noise_level:.1%})",
                 fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "param_convergence.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[INV+QNM 2MODE] Outputs saved to: {outdir}")
    print(f"[INV+QNM 2MODE] Done!")


if __name__ == "__main__":
    main()

"""PINN training with Levenberg-Marquardt (LM) optimizer.

Workflow
--------
1. Adam warm-up phase (via DeepXDE, same as standard pipeline)
2. LM phase (replaces L-BFGS) — operates directly on PyTorch network
3. Evaluation on FD grid + QNM extraction

The LM phase reformulates the PINN loss as ||r(θ)||² where r is the
vector of weighted residuals (PDE + IC + IV + BC), then solves
    (JᵀJ + μI) δ = −Jᵀr
at each iteration using Cholesky factorisation of the 5k×5k system.

References
----------
- Taylor et al. (2022), arXiv:2205.07430
- Shahab et al. (2026), arXiv:2602.08515
"""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import copy
import json
import time
import numpy as np
import torch

import deepxde as dde

from src.config import load_config
from src.fd_solver import solve_fd
from src.pinn import build_model, eval_on_grid
from src.initial_data import gaussian_phi, gaussian_phi_t
from src.potentials import V_of_x_torch
from src.utils import ensure_dir, save_json, rmsd, mad, rl2
from src.plotting import (
    plot_snapshots, plot_abs_diff_snapshots, plot_loss,
    plot_snapshots_zoomed, plot_error_heatmap, plot_ringdown_overlay,
)


# ======================================================================
# Residual computation (pure PyTorch, no DeepXDE dependency)
# ======================================================================

def _sample_domain_points(xmin, xmax, tmin, tmax, N, dtype=torch.float64):
    """Uniformly sample N interior collocation points."""
    x = torch.rand(N, dtype=dtype) * (xmax - xmin) + xmin
    t = torch.rand(N, dtype=dtype) * (tmax - tmin) + tmin
    return torch.stack([x, t], dim=1)


def _sample_ic_points(xmin, xmax, tmin, N, dtype=torch.float64):
    """Uniformly sample N points on the initial time slice."""
    x = torch.rand(N, dtype=dtype) * (xmax - xmin) + xmin
    t = torch.full((N,), tmin, dtype=dtype)
    return torch.stack([x, t], dim=1)


def _sample_bc_points(x_val, tmin, tmax, N, dtype=torch.float64):
    """Uniformly sample N points on a boundary x = x_val."""
    x = torch.full((N,), x_val, dtype=dtype)
    t = torch.rand(N, dtype=dtype) * (tmax - tmin) + tmin
    return torch.stack([x, t], dim=1)


def compute_pde_residual(net, X, M, l, potential):
    """PDE residual: y_tt - y_xx + V(x)*y at collocation points X.

    X : (N, 2) tensor with columns [x, t], requires_grad will be set.
    Returns (N,) tensor with autograd graph attached.
    """
    X = X.detach().requires_grad_(True)
    y = net(X)                                       # (N, 1)

    # First derivatives via autograd
    dy = torch.autograd.grad(y.sum(), X, create_graph=True)[0]
    y_x = dy[:, 0:1]
    y_t = dy[:, 1:2]

    # Second derivatives
    dy_x = torch.autograd.grad(y_x.sum(), X, create_graph=True)[0]
    y_xx = dy_x[:, 0:1]

    dy_t = torch.autograd.grad(y_t.sum(), X, create_graph=True)[0]
    y_tt = dy_t[:, 1:2]

    V = V_of_x_torch(X[:, 0:1], M, l, potential)
    r = (y_tt - y_xx + V * y).squeeze(-1)            # (N,)
    return r


def compute_ic_residual(net, X_ic, A, x0, sigma):
    """IC residual: y(x, 0) - phi_0(x)."""
    y = net(X_ic)                                     # (N, 1)
    phi0_np = gaussian_phi(X_ic[:, 0:1].detach().cpu().numpy(), A=A, x0=x0, sigma=sigma)
    phi0 = torch.as_tensor(phi0_np, dtype=y.dtype, device=y.device).squeeze(-1)
    return y.squeeze(-1) - phi0


def compute_iv_residual(net, X_ic, A, x0, sigma, profile):
    """IV residual: y_t(x, 0) - v_0(x)."""
    X = X_ic.detach().requires_grad_(True)
    y = net(X)
    dy = torch.autograd.grad(y.sum(), X, create_graph=True)[0]
    y_t = dy[:, 1:2].squeeze(-1)                     # (N,)

    v0_np = gaussian_phi_t(X[:, 0:1].detach().cpu().numpy(),
                           A=A, x0=x0, sigma=sigma, profile=profile)
    v0 = torch.as_tensor(v0_np, dtype=y.dtype, device=y.device).squeeze(-1)
    return y_t - v0


def compute_bc_sommerfeld_left(net, X_bl):
    """Left BC (Sommerfeld ingoing): (d_t - d_x) y = 0."""
    X = X_bl.detach().requires_grad_(True)
    y = net(X)
    dy = torch.autograd.grad(y.sum(), X, create_graph=True)[0]
    y_x = dy[:, 0:1].squeeze(-1)
    y_t = dy[:, 1:2].squeeze(-1)
    return y_t - y_x


def compute_bc_sommerfeld_right(net, X_br):
    """Right BC (Sommerfeld outgoing): (d_t + d_x) y = 0."""
    X = X_br.detach().requires_grad_(True)
    y = net(X)
    dy = torch.autograd.grad(y.sum(), X, create_graph=True)[0]
    y_x = dy[:, 0:1].squeeze(-1)
    y_t = dy[:, 1:2].squeeze(-1)
    return y_t + y_x


# ======================================================================
# LM core
# ======================================================================

def _flatten_params(net):
    """Flatten all parameters into a single 1-D tensor."""
    return torch.cat([p.detach().flatten() for p in net.parameters()])


def _unflatten_and_update(net, delta):
    """Apply parameter update δ in-place."""
    offset = 0
    with torch.no_grad():
        for p in net.parameters():
            n = p.numel()
            p.add_(delta[offset:offset + n].view_as(p))
            offset += n


def _unflatten_and_set(net, flat):
    """Set parameters from a flat tensor."""
    offset = 0
    with torch.no_grad():
        for p in net.parameters():
            n = p.numel()
            p.copy_(flat[offset:offset + n].view_as(p))
            offset += n


def accumulate_JtJ_Jtr(net, residual_groups, chunk_size=500):
    """Compute JᵀJ and Jᵀr by accumulating per-sample gradients in chunks.

    Parameters
    ----------
    net : torch.nn.Module
    residual_groups : list of (residuals_tensor, weight)
        Each residuals_tensor has shape (N_k,) with autograd graph.
        weight is the scalar loss weight λ_k / N_k.
    chunk_size : int
        Process this many residuals before releasing the graph.

    Returns
    -------
    JtJ : (P, P)  Gauss-Newton Hessian approximation
    Jtr : (P,)    gradient of ½||r̃||²
    total_loss : float
    """
    params = [p for p in net.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    dtype = params[0].dtype

    JtJ = torch.zeros(n_params, n_params, dtype=dtype)
    Jtr = torch.zeros(n_params, dtype=dtype)
    total_loss = 0.0

    for residuals, w in residual_groups:
        N = residuals.shape[0]
        sqrt_w = w ** 0.5

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            r_chunk = residuals[start:end]
            B = r_chunk.shape[0]

            # Collect per-sample gradients for this chunk
            grads_list = []
            for i in range(B):
                retain = (i < B - 1) or (end < N)
                g = torch.autograd.grad(
                    r_chunk[i], params,
                    retain_graph=retain,
                    allow_unused=True,
                )
                g_flat = torch.cat([
                    (gi.flatten() if gi is not None else torch.zeros(p.numel(), dtype=dtype))
                    for gi, p in zip(g, params)
                ])
                grads_list.append(g_flat)

            # Stack into (B, P) sub-Jacobian and accumulate
            J_chunk = torch.stack(grads_list) * sqrt_w   # (B, P)
            r_vals = r_chunk.detach() * sqrt_w            # (B,)

            JtJ += J_chunk.T @ J_chunk
            Jtr += J_chunk.T @ r_vals
            total_loss += (r_vals ** 2).sum().item()

    return JtJ, Jtr, total_loss


def lm_solve(JtJ, Jtr, mu):
    """Solve (JᵀJ + μI)δ = −Jᵀr via Cholesky."""
    n = JtJ.shape[0]
    A = JtJ + mu * torch.eye(n, dtype=JtJ.dtype)
    try:
        L = torch.linalg.cholesky(A)
        delta = torch.cholesky_solve(-Jtr.unsqueeze(1), L).squeeze(1)
    except torch.linalg.LinAlgError:
        # Fallback to general solver if not positive-definite
        delta = torch.linalg.solve(A, -Jtr)
    return delta


# ======================================================================
# Training loop
# ======================================================================

def run_lm_phase(net, cfg, lm_cfg, outdir, adam_loss_history=None):
    """Run the Levenberg-Marquardt training phase.

    Parameters
    ----------
    net : the PyTorch network (already Adam-warmed)
    cfg : full experiment config dict
    lm_cfg : the ``lm`` section of the config
    outdir : output directory for checkpoints and logs
    adam_loss_history : optional dict with Adam phase loss history

    Returns
    -------
    loss_history : dict with keys "steps", "loss", "phase"
    """
    # ---- Config ----
    max_iters = int(lm_cfg.get("iters", 500))
    mu = float(lm_cfg.get("mu_init", 1.0))
    mu_factor = float(lm_cfg.get("mu_factor", 10.0))
    mu_min = float(lm_cfg.get("mu_min", 1e-12))
    mu_max = float(lm_cfg.get("mu_max", 1e12))
    Nr_lm = int(lm_cfg.get("Nr", 10000))
    Ni_lm = int(lm_cfg.get("Ni", 800))
    Nb_lm = int(lm_cfg.get("Nb", 400))
    resample_every = int(lm_cfg.get("resample_every", 50))
    chunk_size = int(lm_cfg.get("jacobian_chunk", 500))
    log_every = int(lm_cfg.get("log_every", 1))
    ckpt_every = int(lm_cfg.get("checkpoint_every", 50))

    # Loss weights: [r_pde, ic, iv, bl, br]
    # For LM we use 5 residual groups (no gradient-enhanced r_x, r_t)
    lm_weights = lm_cfg.get("lambda", [100.0, 1.0, 100.0, 1.0, 1.0])
    w_pde, w_ic, w_iv, w_bl, w_br = [float(w) for w in lm_weights]

    # Physics
    M = float(cfg["physics"]["M"])
    l_phys = int(cfg["physics"]["l"])
    potential = cfg["physics"]["potential"]

    # Domain
    xmin = float(cfg["domain"]["xmin"])
    xmax = float(cfg["domain"]["xmax"])
    tmin = float(cfg["domain"]["tmin"])
    tmax = float(cfg["domain"]["tmax"])

    # Initial data
    A0 = float(cfg["initial_data"]["A"])
    x0_ic = float(cfg["initial_data"]["x0"])
    sigma = float(cfg["initial_data"]["sigma"])
    profile = cfg["initial_data"]["velocity_profile"]

    dtype = torch.float64 if cfg["pinn"]["dtype"] == "float64" else torch.float32

    # ---- Checkpoint directory ----
    ckpt_dir = os.path.join(outdir, "checkpoints_lm")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- LM iteration ----
    n_params = sum(p.numel() for p in net.parameters())
    print(f"[LM] Parameters: {n_params}")
    print(f"[LM] Collocation: Nr={Nr_lm}, Ni={Ni_lm}, Nb={Nb_lm}")
    print(f"[LM] Weights: PDE={w_pde}, IC={w_ic}, IV={w_iv}, BL={w_bl}, BR={w_br}")
    print(f"[LM] max_iters={max_iters}, μ₀={mu}, factor={mu_factor}")
    print(f"[LM] Jacobian chunk size: {chunk_size}")
    print(f"[LM] Resample every {resample_every} iterations")
    print()

    loss_log = {"steps": [], "loss": [], "mu": [], "accepted": [], "phase": []}
    adam_offset = int(cfg["pinn"]["adam"]["iters"])

    # Sample initial collocation points
    X_pde = _sample_domain_points(xmin, xmax, tmin, tmax, Nr_lm, dtype=dtype)
    X_ic = _sample_ic_points(xmin, xmax, tmin, Ni_lm, dtype=dtype)
    X_bl = _sample_bc_points(xmin, tmin, tmax, Nb_lm // 2, dtype=dtype)
    X_br = _sample_bc_points(xmax, tmin, tmax, Nb_lm // 2, dtype=dtype)

    prev_loss = None
    consecutive_rejects = 0

    for it in range(1, max_iters + 1):
        t_start = time.time()

        # Resample collocation points periodically
        if resample_every > 0 and it > 1 and (it - 1) % resample_every == 0:
            X_pde = _sample_domain_points(xmin, xmax, tmin, tmax, Nr_lm, dtype=dtype)
            X_ic = _sample_ic_points(xmin, xmax, tmin, Ni_lm, dtype=dtype)
            X_bl = _sample_bc_points(xmin, tmin, tmax, Nb_lm // 2, dtype=dtype)
            X_br = _sample_bc_points(xmax, tmin, tmax, Nb_lm // 2, dtype=dtype)
            # Recompute prev_loss on new points so accept/reject is apples-to-apples
            prev_loss = _eval_loss_no_grad(
                net, X_pde, X_ic, X_bl, X_br,
                M, l_phys, potential, A0, x0_ic, sigma, profile,
                w_pde, w_ic, w_iv, w_bl, w_br,
                Nr_lm, Ni_lm, Nb_lm,
            )
            mu = float(lm_cfg.get("mu_init", 1.0))   # reset damping
            consecutive_rejects = 0
            print(f"  [resample] New collocation points at iter {it}, "
                  f"prev_loss={prev_loss:.6e}, μ reset to {mu:.2e}")

        # Compute residuals (with autograd graph)
        r_pde = compute_pde_residual(net, X_pde, M, l_phys, potential)
        r_ic = compute_ic_residual(net, X_ic, A0, x0_ic, sigma)
        r_iv = compute_iv_residual(net, X_ic, A0, x0_ic, sigma, profile)
        r_bl = compute_bc_sommerfeld_left(net, X_bl)
        r_br = compute_bc_sommerfeld_right(net, X_br)

        # Per-sample weight = λ_k / N_k (MSE convention)
        residual_groups = [
            (r_pde, w_pde / Nr_lm),
            (r_ic,  w_ic  / Ni_lm),
            (r_iv,  w_iv  / Ni_lm),
            (r_bl,  w_bl  / (Nb_lm // 2)),
            (r_br,  w_br  / (Nb_lm // 2)),
        ]

        # Accumulate JᵀJ and Jᵀr
        JtJ, Jtr, current_loss = accumulate_JtJ_Jtr(net, residual_groups, chunk_size)

        if prev_loss is None:
            prev_loss = current_loss

        # Save current parameters (for potential rollback)
        theta_old = _flatten_params(net)

        # Solve LM system and apply update
        delta = lm_solve(JtJ, Jtr, mu)
        _unflatten_and_update(net, delta)

        # Recompute loss at new parameters (forward only, no Jacobian)
        new_loss = _eval_loss_no_grad(
            net, X_pde, X_ic, X_bl, X_br,
            M, l_phys, potential, A0, x0_ic, sigma, profile,
            w_pde, w_ic, w_iv, w_bl, w_br,
            Nr_lm, Ni_lm, Nb_lm,
        )

        # Gain ratio and acceptance
        predicted_reduction = -(delta @ Jtr + 0.5 * delta @ (JtJ @ delta)).item()
        actual_reduction = prev_loss - new_loss

        accepted = new_loss < prev_loss
        if accepted:
            prev_loss = new_loss
            mu = max(mu / mu_factor, mu_min)
            consecutive_rejects = 0
        else:
            # Reject: rollback parameters, increase damping
            _unflatten_and_set(net, theta_old)
            mu = min(mu * mu_factor, mu_max)
            consecutive_rejects += 1

        t_iter = time.time() - t_start

        # Log
        loss_log["steps"].append(adam_offset + it)
        loss_log["loss"].append(prev_loss)
        loss_log["mu"].append(mu)
        loss_log["accepted"].append(accepted)
        loss_log["phase"].append("lm")

        if it % log_every == 0 or it == 1:
            status = "✓" if accepted else "✗"
            print(
                f"  [{status}] iter {it:5d}/{max_iters}  "
                f"loss={prev_loss:.6e}  μ={mu:.2e}  "
                f"Δpred={predicted_reduction:+.2e}  Δactual={actual_reduction:+.2e}  "
                f"({t_iter:.1f}s)"
            )

        # Checkpoint
        if it % ckpt_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"model-lm-{it}.pt")
            torch.save({"model_state_dict": net.state_dict(), "mu": mu, "iter": it}, ckpt_path)

        # Early stopping: mu too large means we're stuck
        if mu >= mu_max:
            print(f"[LM] μ reached maximum ({mu_max}). Stopping.")
            break

        if consecutive_rejects >= 20:
            print(f"[LM] 20 consecutive rejections. Stopping.")
            break

    # Final checkpoint
    ckpt_path = os.path.join(ckpt_dir, f"model-lm-final.pt")
    torch.save({"model_state_dict": net.state_dict(), "mu": mu, "iter": it}, ckpt_path)
    print(f"[LM] Final loss: {prev_loss:.6e}  (after {it} iterations)")

    return loss_log


def _eval_loss_no_grad(net, X_pde, X_ic, X_bl, X_br,
                       M, l_phys, potential, A0, x0_ic, sigma, profile,
                       w_pde, w_ic, w_iv, w_bl, w_br,
                       Nr_lm, Ni_lm, Nb_lm):
    """Compute total weighted MSE loss (forward only, no parameter gradients)."""
    # Disable parameter gradients so autograd only tracks input derivatives
    for p in net.parameters():
        p.requires_grad_(False)

    r_pde = compute_pde_residual(net, X_pde, M, l_phys, potential)
    r_ic  = compute_ic_residual(net, X_ic, A0, x0_ic, sigma)
    r_iv  = compute_iv_residual(net, X_ic, A0, x0_ic, sigma, profile)
    r_bl  = compute_bc_sommerfeld_left(net, X_bl)
    r_br  = compute_bc_sommerfeld_right(net, X_br)

    # Re-enable parameter gradients
    for p in net.parameters():
        p.requires_grad_(True)

    loss = (
        (w_pde / Nr_lm)       * (r_pde.detach() ** 2).sum().item()
        + (w_ic / Ni_lm)      * (r_ic.detach() ** 2).sum().item()
        + (w_iv / Ni_lm)      * (r_iv.detach() ** 2).sum().item()
        + (w_bl / (Nb_lm//2)) * (r_bl.detach() ** 2).sum().item()
        + (w_br / (Nb_lm//2)) * (r_br.detach() ** 2).sum().item()
    )
    return loss


# ======================================================================
# Main
# ======================================================================

def main():
    ap = argparse.ArgumentParser(description="PINN training: Adam + LM")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint-every", type=int, default=500,
                    help="Adam checkpoint interval")
    args = ap.parse_args()

    cfg = load_config(args.config)
    name = cfg["experiment"]["name"]

    # Validate LM config exists
    lm_cfg = cfg.get("lm")
    if lm_cfg is None or not lm_cfg.get("enabled", False):
        print("[ERROR] Config must have lm.enabled: true")
        sys.exit(1)

    # --- FD baseline ---
    print("=" * 60)
    print("FD BASELINE")
    print("=" * 60)
    fd = solve_fd(cfg)
    x, t, phi_fd = fd["x"], fd["t"], fd["phi"]

    # --- Adam warm-up (via DeepXDE) ---
    print()
    print("=" * 60)
    print("PHASE 1: ADAM WARM-UP")
    print("=" * 60)

    seed = int(cfg["pinn"]["seed"])
    dde.config.set_random_seed(seed)
    dde.config.set_default_float(cfg["pinn"]["dtype"])

    model, data = build_model(cfg)
    loss_weights = [float(w) for w in cfg["pinn"]["lambda"]]

    adam_cfg = cfg["pinn"]["adam"]
    adam_iters = int(adam_cfg["iters"])
    lr = float(adam_cfg["lr"])
    resample_period = int(adam_cfg["resample_period"])

    outdir = os.path.join("outputs", "pinn", name)
    ensure_dir(outdir)
    ckpt_dir = os.path.join(outdir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    model_save_path = os.path.join(ckpt_dir, "model")

    callbacks_adam = [
        dde.callbacks.PDEPointResampler(period=resample_period),
        dde.callbacks.ModelCheckpoint(
            model_save_path, save_better_only=False,
            period=args.checkpoint_every,
        ),
    ]

    # Check if Adam already done (from a previous run)
    import glob
    adam_done_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "model-adam_done*.pt")))
    if adam_done_ckpts:
        print(f"[ADAM] Found completed checkpoint: {adam_done_ckpts[-1]}")
        print("[ADAM] Skipping Adam — loading weights")
        model.compile("adam", lr=lr, loss_weights=loss_weights)
        model.train(iterations=0, display_every=1)
        ckpt = torch.load(adam_done_ckpts[-1], map_location="cpu")
        model.net.load_state_dict(ckpt["model_state_dict"])
    else:
        model.compile("adam", lr=lr, loss_weights=loss_weights)
        print(f"[ADAM] {adam_iters} iters, lr={lr}")
        model.train(
            iterations=adam_iters,
            callbacks=callbacks_adam,
            display_every=100,
            model_save_path=model_save_path,
        )
        model.save(model_save_path + "-adam_done")
        print("[ADAM] Complete — checkpoint saved")

    # --- LM phase ---
    print()
    print("=" * 60)
    print("PHASE 2: LEVENBERG-MARQUARDT")
    print("=" * 60)

    net = model.net  # Extract the PyTorch network
    loss_log = run_lm_phase(net, cfg, lm_cfg, outdir)

    # --- Evaluate ---
    print()
    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)

    df_cfg = cfg["pinn"].get("decay_factor", {})
    df_tau = float(df_cfg.get("tau", 0.0)) if df_cfg.get("enabled", False) else 0.0

    # eval_on_grid works with either dde.Model or raw nn.Module
    phi_pinn = eval_on_grid(model, x=x, t=t, dtype=cfg["pinn"]["dtype"],
                            decay_factor_tau=df_tau)

    metrics = {
        "RMSD": rmsd(phi_fd, phi_pinn),
        "MAD":  mad(phi_fd, phi_pinn),
        "RL2":  rl2(phi_fd, phi_pinn),
    }

    # Save outputs
    np.savez_compressed(os.path.join(outdir, f"{name}_fd.npz"), **fd)
    np.savez_compressed(os.path.join(outdir, f"{name}_pinn.npz"),
                        x=x, t=t, phi=phi_pinn)
    save_json(os.path.join(outdir, "metrics.json"), metrics)
    save_json(os.path.join(outdir, "loss_history.json"), loss_log)

    # Plots
    times = [10.0, 20.0, 30.0, 40.0]
    title_base = f"{cfg['physics']['potential'].title()} potential (l={cfg['physics']['l']})"

    plot_snapshots(
        x, t, phi_fd, phi_pinn, times,
        outpath=os.path.join(outdir, "snapshots.png"),
        title=f"Snapshots — {title_base}",
    )
    plot_abs_diff_snapshots(
        x, t, phi_fd, phi_pinn, times,
        outpath=os.path.join(outdir, "abs_diff_snapshots.png"),
        title=f"Absolute difference — {title_base}",
    )
    plot_snapshots_zoomed(
        x, t, phi_fd, phi_pinn, times,
        outpath=os.path.join(outdir, "snapshots_zoomed.png"),
        title=f"Snapshots (zoomed) — {title_base}",
    )
    plot_error_heatmap(
        x, t, phi_fd, phi_pinn,
        outpath=os.path.join(outdir, "error_heatmap.png"),
        title=f"Pointwise error — {title_base}",
    )
    plot_error_heatmap(
        x, t, phi_fd, phi_pinn,
        outpath=os.path.join(outdir, "error_heatmap_zoomed.png"),
        title=f"Pointwise error (zoomed) — {title_base}",
        xlim=(-20.0, 60.0),
    )

    xq = float(cfg["evaluation"]["xq"])
    ix = int(np.argmin(np.abs(x - xq)))
    plot_ringdown_overlay(
        t, phi_fd[:, ix], phi_pinn[:, ix],
        outpath=os.path.join(outdir, "ringdown_overlay.png"),
        title=f"Ringdown — {title_base}",
        xq=xq,
    )

    # LM-specific loss plot
    _plot_lm_loss(loss_log, os.path.join(outdir, "loss_lm.png"))

    print(f"\n[PINN-LM] Metrics: {metrics}")
    print(f"[PINN-LM] Outputs in: {outdir}")


def _plot_lm_loss(loss_log, outpath):
    """Plot LM loss curve with accepted/rejected markers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = loss_log["steps"]
    losses = loss_log["loss"]
    accepted = loss_log.get("accepted", [True] * len(steps))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Loss
    ax1.semilogy(steps, losses, "b-", linewidth=0.8)
    ax1.set_ylabel("Total loss")
    ax1.set_title("Levenberg-Marquardt Training")
    ax1.grid(True, alpha=0.3)

    # μ
    if "mu" in loss_log:
        ax2.semilogy(steps, loss_log["mu"], "r-", linewidth=0.8)
        ax2.set_ylabel("Damping μ")
    ax2.set_xlabel("Iteration")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] LM loss curve: {outpath}")


if __name__ == "__main__":
    main()

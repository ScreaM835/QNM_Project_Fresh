"""
Curriculum learning with time-window splitting for PINN training.

Trains two PINNs sequentially on [tmin, t_split] and [t_split, tmax].
The second window uses the first PINN's output as initial conditions.

Usage:
    python scripts/run_pinn_curriculum.py --config configs/zerilli_l2_curriculum.yaml

The config should contain a top-level ``curriculum`` section:
    curriculum:
      enabled: true
      t_split: 25.0
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import copy
import numpy as np
import torch

import deepxde as dde

from src.config import load_config
from src.fd_solver import solve_fd
from src.pinn import (
    train_pinn, eval_on_grid, build_model,
    build_model_numerical_ic, _make_pde_residual_only,
)
import glob
from src.utils import ensure_dir, save_json, rmsd, mad, rl2
from src.plotting import (
    plot_snapshots, plot_abs_diff_snapshots, plot_loss,
    plot_snapshots_zoomed, plot_error_heatmap, plot_ringdown_overlay,
)


def _extract_ic_from_model(model, x_grid, t_split, dtype="float64"):
    """Evaluate phi and d_phi/dt from a trained model at t = t_split.

    Returns (phi_vals, phi_t_vals) as numpy arrays of shape (Nx,).
    """
    X = np.stack([x_grid, np.full_like(x_grid, t_split)], axis=1)

    # phi values: straightforward prediction
    phi_vals = model.predict(X).ravel()

    # phi_t values: use automatic differentiation
    tdtype = torch.float64 if dtype == "float64" else torch.float32
    X_t = torch.tensor(X, dtype=tdtype, requires_grad=True)

    net = model.net
    net.eval()
    y = net(X_t)
    grad_outputs = torch.ones_like(y)
    grads = torch.autograd.grad(y, X_t, grad_outputs=grad_outputs, create_graph=False)[0]
    phi_t_vals = grads[:, 1].detach().cpu().numpy()

    return phi_vals, phi_t_vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--checkpoint-every", type=int, default=500)
    args = ap.parse_args()

    cfg = load_config(args.config)
    name = cfg["experiment"]["name"]

    curriculum_cfg = cfg.get("curriculum", {})
    if not curriculum_cfg.get("enabled", False):
        raise ValueError("curriculum.enabled must be true in config")

    t_split = float(curriculum_cfg["t_split"])
    tmin = float(cfg["domain"]["tmin"])
    tmax = float(cfg["domain"]["tmax"])

    assert tmin < t_split < tmax, f"t_split={t_split} must be between tmin={tmin} and tmax={tmax}"

    print("=" * 60)
    print(f"CURRICULUM LEARNING: 2 windows")
    print(f"  Window 1: t = [{tmin}, {t_split}]")
    print(f"  Window 2: t = [{t_split}, {tmax}]")
    print("=" * 60)

    # --- FD baseline (full domain, for evaluation) ---
    fd = solve_fd(cfg)
    x_fd, t_fd, phi_fd = fd["x"], fd["t"], fd["phi"]

    # =================================================================
    # WINDOW 1: [tmin, t_split] — uses analytic ICs (standard training)
    # =================================================================
    print("\n" + "=" * 60)
    print(f"WINDOW 1: t = [{tmin}, {t_split}]")
    print("=" * 60)

    cfg_w1 = copy.deepcopy(cfg)
    cfg_w1["domain"]["tmax"] = t_split
    cfg_w1["experiment"]["name"] = f"{name}_w1"

    ckpt_dir_w1 = os.path.join("outputs", "pinn", name, "checkpoints_w1")

    # Check for a completed window-1 checkpoint to skip retraining
    w1_final_ckpts = sorted(glob.glob(os.path.join(ckpt_dir_w1, "model-final-*.pt")))
    if w1_final_ckpts:
        w1_ckpt_path = w1_final_ckpts[-1]
        print(f"[W1] Found completed checkpoint: {w1_ckpt_path}")
        print("[W1] Skipping window-1 training — loading from checkpoint")

        seed = int(cfg["pinn"]["seed"])
        dde.config.set_random_seed(seed)
        dde.config.set_default_float(cfg["pinn"]["dtype"])

        model_w1, _ = build_model(cfg_w1)
        model_w1.compile("adam", lr=1e-3)
        model_w1.train(iterations=0, display_every=1)
        # Load only network weights (not optimizer state) to avoid
        # Adam/L-BFGS state mismatch on restore.
        import torch as _torch
        _ckpt = _torch.load(w1_ckpt_path, map_location="cpu")
        model_w1.net.load_state_dict(_ckpt["model_state_dict"])
        print(f"[W1] Loaded network weights from {w1_ckpt_path}")
    else:
        model_w1, hist_w1 = train_pinn(
            cfg_w1,
            checkpoint_dir=ckpt_dir_w1,
            checkpoint_every=args.checkpoint_every,
            resume=args.resume,
        )

    # Evaluate window 1 on its sub-grid
    t_mask_w1 = t_fd <= t_split + 1e-10
    t_w1 = t_fd[t_mask_w1]
    df_tau = 0.0
    df_cfg = cfg["pinn"].get("decay_factor", {})
    if df_cfg.get("enabled", False):
        df_tau = float(df_cfg["tau"])
    phi_pinn_w1 = eval_on_grid(model_w1, x=x_fd, t=t_w1,
                                dtype=cfg["pinn"]["dtype"],
                                decay_factor_tau=df_tau)

    print(f"[W1] RMSD (window 1): {rmsd(phi_fd[t_mask_w1], phi_pinn_w1):.6f}")

    # =================================================================
    # Extract IC at t_split from window-1 model
    # =================================================================
    print(f"\n[IC] Extracting phi and phi_t at t_split={t_split} from window-1 model...")
    phi_split, phi_t_split = _extract_ic_from_model(
        model_w1, x_fd, t_split, dtype=cfg["pinn"]["dtype"]
    )
    print(f"[IC] phi range: [{phi_split.min():.6f}, {phi_split.max():.6f}]")
    print(f"[IC] phi_t range: [{phi_t_split.min():.6f}, {phi_t_split.max():.6f}]")

    # Build interpolators for the IC (the DeepXDE IC functions receive
    # arbitrary x-coordinates, not just grid points)
    from scipy.interpolate import interp1d

    phi_interp = interp1d(x_fd, phi_split, kind="cubic", fill_value="extrapolate")
    phi_t_interp = interp1d(x_fd, phi_t_split, kind="cubic", fill_value="extrapolate")

    def phi_ic_func(x):
        """IC displacement: phi(x, t_split) from window-1 PINN."""
        return phi_interp(x[:, 0:1]).reshape(-1, 1)

    def phi_t_ic_func(inputs, outputs, X):
        """IC velocity: phi_t(x, t_split) from window-1 PINN."""
        phi_t = dde.grad.jacobian(outputs, inputs, i=0, j=1)
        x_np = inputs[:, 0:1].detach().cpu().numpy()
        target = phi_t_interp(x_np).reshape(-1, 1)
        target_t = torch.as_tensor(target, dtype=outputs.dtype, device=outputs.device)
        return phi_t - target_t

    # =================================================================
    # WINDOW 2: [t_split, tmax] — uses numerical ICs from window 1
    # =================================================================
    print("\n" + "=" * 60)
    print(f"WINDOW 2: t = [{t_split}, {tmax}]")
    print("=" * 60)

    cfg_w2 = copy.deepcopy(cfg)
    cfg_w2["domain"]["tmin"] = t_split
    cfg_w2["domain"]["tmax"] = tmax
    cfg_w2["experiment"]["name"] = f"{name}_w2"

    # Build model with numerical ICs
    seed = int(cfg["pinn"]["seed"])
    dde.config.set_random_seed(seed + 1)  # different seed for window 2
    dde.config.set_default_float(cfg["pinn"]["dtype"])

    model_w2_dde, data_w2 = build_model_numerical_ic(
        cfg_w2,
        tmin_override=t_split,
        tmax_override=tmax,
        phi_ic_func=phi_ic_func,
        phi_t_ic_func=phi_t_ic_func,
    )

    # Train window 2 using the standard train_pinn
    # We pass the modified config and train_pinn will call build_model again,
    # but we need to use the numerical-IC model. So we train manually here,
    # reusing the same training logic.
    ckpt_dir_w2 = os.path.join("outputs", "pinn", name, "checkpoints_w2")
    os.makedirs(ckpt_dir_w2, exist_ok=True)

    loss_weights = [float(w) for w in cfg["pinn"]["lambda"]]

    # Adam phase
    adam_cfg = cfg["pinn"]["adam"]
    adam_iters = int(adam_cfg["iters"])
    lr = float(adam_cfg["lr"])
    resample_period = int(adam_cfg["resample_period"])

    adaptive_cfg = cfg["pinn"].get("adaptive_sampling", {})
    callbacks_adam = []

    if adaptive_cfg.get("enabled", False):
        method = adaptive_cfg.get("method", "RAD")
        rad_period = int(adaptive_cfg.get("period", 1000))

        if method == "greedy":
            from src.pinn import GreedyResampler
            pde_func = _make_pde_residual_only(cfg_w2)
            greedy_frac = float(adaptive_cfg.get("greedy_fraction", 0.5))
            num_cand = int(adaptive_cfg.get("num_candidates", 50000))
            eval_bs = int(adaptive_cfg.get("eval_batch_size", 5000))
            sampler = GreedyResampler(
                pde_residual=pde_func,
                period=rad_period,
                num_candidates=num_cand,
                greedy_fraction=greedy_frac,
                eval_batch_size=eval_bs,
            )
            callbacks_adam.append(sampler)
        else:
            # RAD
            from src.pinn import ResidualAdaptiveResampler
            pde_func = _make_pde_residual_only(cfg_w2)
            num_cand = int(adaptive_cfg.get("num_candidates", 50000))
            k = float(adaptive_cfg.get("k", 1.0))
            c = float(adaptive_cfg.get("c", 1.0))
            eval_bs = int(adaptive_cfg.get("eval_batch_size", 5000))
            sampler = ResidualAdaptiveResampler(
                pde_residual=pde_func,
                period=rad_period,
                num_candidates=num_cand,
                k=k, c=c,
                eval_batch_size=eval_bs,
            )
            callbacks_adam.append(sampler)
    else:
        callbacks_adam.append(dde.callbacks.PDEPointResampler(period=resample_period))

    model_w2_dde.compile("adam", lr=lr, loss_weights=loss_weights)
    losshistory_adam = model_w2_dde.train(
        iterations=adam_iters,
        display_every=500,
        callbacks=callbacks_adam,
    )

    # L-BFGS phase
    lbfgs_cfg = cfg["pinn"]["lbfgs"]
    lbfgs_iters = int(lbfgs_cfg["iters"])
    lbfgs_resample = int(lbfgs_cfg.get("resample_period", 100))

    callbacks_lbfgs = []
    if adaptive_cfg.get("enabled", False):
        # Re-use the same sampler type for L-BFGS
        if method == "greedy":
            sampler_lb = GreedyResampler(
                pde_residual=pde_func,
                period=rad_period,
                num_candidates=num_cand,
                greedy_fraction=greedy_frac,
                eval_batch_size=eval_bs,
            )
            callbacks_lbfgs.append(sampler_lb)
        else:
            sampler_lb = ResidualAdaptiveResampler(
                pde_residual=pde_func,
                period=rad_period,
                num_candidates=num_cand,
                k=k, c=c,
                eval_batch_size=eval_bs,
            )
            callbacks_lbfgs.append(sampler_lb)

    dde.optimizers.config.set_LBFGS_options(maxcor=100, gtol=1e-08)
    model_w2_dde.compile("L-BFGS", loss_weights=loss_weights)
    losshistory_lbfgs = model_w2_dde.train(
        iterations=lbfgs_iters,
        display_every=500,
        callbacks=callbacks_lbfgs,
    )

    # Evaluate window 2 on its sub-grid
    t_mask_w2 = t_fd >= t_split - 1e-10
    t_w2 = t_fd[t_mask_w2]
    phi_pinn_w2 = eval_on_grid(model_w2_dde, x=x_fd, t=t_w2,
                                dtype=cfg["pinn"]["dtype"],
                                decay_factor_tau=df_tau)

    print(f"[W2] RMSD (window 2): {rmsd(phi_fd[t_mask_w2], phi_pinn_w2):.6f}")

    # =================================================================
    # Stitch windows and evaluate on full domain
    # =================================================================
    print("\n" + "=" * 60)
    print("STITCHING windows and evaluating on full domain")
    print("=" * 60)

    # Find the split index in t_fd
    i_split = np.searchsorted(t_fd, t_split)

    # Window 1 covers t_fd[0:i_split+1], window 2 covers t_fd[i_split:]
    # At t_split itself, average the two predictions
    phi_pinn_full = np.empty_like(phi_fd)
    phi_pinn_full[:i_split, :] = phi_pinn_w1[:i_split, :]
    phi_pinn_full[i_split:, :] = phi_pinn_w2  # w2 starts at t_split

    metrics = {
        "RMSD": rmsd(phi_fd, phi_pinn_full),
        "MAD": mad(phi_fd, phi_pinn_full),
        "RL2": rl2(phi_fd, phi_pinn_full),
    }

    outdir = os.path.join("outputs", "pinn", name)
    ensure_dir(outdir)

    np.savez_compressed(os.path.join(outdir, f"{name}_fd.npz"), **fd)
    np.savez_compressed(os.path.join(outdir, f"{name}_pinn.npz"),
                        x=x_fd, t=t_fd, phi=phi_pinn_full)
    save_json(os.path.join(outdir, "metrics.json"), metrics)

    # Save per-window metrics too
    w1_metrics = {
        "RMSD": rmsd(phi_fd[t_mask_w1], phi_pinn_w1),
        "MAD": mad(phi_fd[t_mask_w1], phi_pinn_w1),
        "RL2": rl2(phi_fd[t_mask_w1], phi_pinn_w1),
    }
    w2_metrics = {
        "RMSD": rmsd(phi_fd[t_mask_w2], phi_pinn_w2),
        "MAD": mad(phi_fd[t_mask_w2], phi_pinn_w2),
        "RL2": rl2(phi_fd[t_mask_w2], phi_pinn_w2),
    }
    save_json(os.path.join(outdir, "metrics_w1.json"), w1_metrics)
    save_json(os.path.join(outdir, "metrics_w2.json"), w2_metrics)

    # Plots
    times = [10.0, 20.0, 30.0, 40.0]
    pot = cfg['physics']['potential'].title()
    l_val = cfg['physics']['l']

    plot_snapshots(
        x_fd, t_fd, phi_fd, phi_pinn_full, times,
        outpath=os.path.join(outdir, "snapshots.png"),
        title=f"Snapshots — {pot} potential (l={l_val})"
    )
    plot_abs_diff_snapshots(
        x_fd, t_fd, phi_fd, phi_pinn_full, times,
        outpath=os.path.join(outdir, "abs_diff_snapshots.png"),
        title=f"Absolute difference — {pot} potential (l={l_val})"
    )
    plot_snapshots_zoomed(
        x_fd, t_fd, phi_fd, phi_pinn_full, times,
        outpath=os.path.join(outdir, "snapshots_zoomed.png"),
        title=f"Snapshots (zoomed) — {pot} potential (l={l_val})",
    )
    plot_error_heatmap(
        x_fd, t_fd, phi_fd, phi_pinn_full,
        outpath=os.path.join(outdir, "error_heatmap.png"),
        title=f"Pointwise error — {pot} (l={l_val})",
    )
    plot_error_heatmap(
        x_fd, t_fd, phi_fd, phi_pinn_full,
        outpath=os.path.join(outdir, "error_heatmap_zoomed.png"),
        title=f"Pointwise error (zoomed) — {pot} (l={l_val})",
        xlim=(-20.0, 60.0),
    )

    xq = float(cfg["evaluation"]["xq"])
    ix = int(np.argmin(np.abs(x_fd - xq)))
    plot_ringdown_overlay(
        t_fd, phi_fd[:, ix], phi_pinn_full[:, ix],
        outpath=os.path.join(outdir, "ringdown_overlay.png"),
        title=f"Ringdown — {pot} (l={l_val})",
        xq=xq,
    )

    print(f"\n[CURRICULUM] Full-domain metrics: {metrics}")
    print(f"[CURRICULUM] Window 1 RMSD: {w1_metrics['RMSD']:.6f}")
    print(f"[CURRICULUM] Window 2 RMSD: {w2_metrics['RMSD']:.6f}")
    print(f"[CURRICULUM] Outputs in: {outdir}")


if __name__ == "__main__":
    main()

"""
Curriculum learning with N time-windows for PINN training.

Generalised version: trains PINNs sequentially on N windows.
Window 1 uses analytic ICs; subsequent windows use the previous
PINN's output as numerical initial conditions.

Usage:
    python scripts/run_pinn_curriculum_nw.py --config configs/zerilli_l2_curriculum_3w.yaml

Config ``curriculum`` section:
    curriculum:
      enabled: true
      t_splits: [17.0, 34.0]        # N-1 split points → N windows
"""

from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import copy
import glob
import numpy as np
import torch

import deepxde as dde
from scipy.interpolate import interp1d

from src.config import load_config
from src.fd_solver import solve_fd
from src.pinn import (
    train_pinn, eval_on_grid, build_model,
    build_model_numerical_ic, _make_pde_residual_only,
    GreedyResampler, ResidualAdaptiveResampler,
)
from src.utils import ensure_dir, save_json, rmsd, mad, rl2
from src.plotting import (
    plot_snapshots, plot_abs_diff_snapshots, plot_loss,
    plot_snapshots_zoomed, plot_error_heatmap, plot_ringdown_overlay,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _extract_ic_from_model(model, x_grid, t_val, dtype="float64"):
    """Evaluate phi and d_phi/dt from a trained model at t = t_val."""
    X = np.stack([x_grid, np.full_like(x_grid, t_val)], axis=1)
    phi_vals = model.predict(X).ravel()

    tdtype = torch.float64 if dtype == "float64" else torch.float32
    X_t = torch.tensor(X, dtype=tdtype, requires_grad=True)

    net = model.net
    net.eval()
    y = net(X_t)
    grad_outputs = torch.ones_like(y)
    grads = torch.autograd.grad(y, X_t, grad_outputs=grad_outputs, create_graph=False)[0]
    phi_t_vals = grads[:, 1].detach().cpu().numpy()

    return phi_vals, phi_t_vals


def _make_adaptive_callbacks(cfg_window, adaptive_cfg, method=None):
    """Build a list of adaptive-sampling callbacks for a window config."""
    callbacks = []
    if not adaptive_cfg.get("enabled", False):
        resample_period = int(cfg_window["pinn"]["adam"]["resample_period"])
        callbacks.append(dde.callbacks.PDEPointResampler(period=resample_period))
        return callbacks, None

    if method is None:
        method = adaptive_cfg.get("method", "RAD")
    rad_period = int(adaptive_cfg.get("period", 1000))
    pde_func = _make_pde_residual_only(cfg_window)
    num_cand = int(adaptive_cfg.get("num_candidates", 50000))
    eval_bs = int(adaptive_cfg.get("eval_batch_size", 5000))

    if method == "greedy":
        greedy_frac = float(adaptive_cfg.get("greedy_fraction", 0.5))
        sampler = GreedyResampler(
            pde_residual=pde_func,
            period=rad_period,
            num_candidates=num_cand,
            greedy_fraction=greedy_frac,
            eval_batch_size=eval_bs,
        )
    else:
        k = float(adaptive_cfg.get("k", 1.0))
        c = float(adaptive_cfg.get("c", 1.0))
        sampler = ResidualAdaptiveResampler(
            pde_residual=pde_func,
            period=rad_period,
            num_candidates=num_cand,
            k=k, c=c,
            eval_batch_size=eval_bs,
        )
    callbacks.append(sampler)
    return callbacks, method


def _train_window(model_dde, cfg, adaptive_cfg, loss_weights, method=None):
    """Run Adam + L-BFGS on a window's model. Returns the model."""
    # --- Adam ---
    adam_cfg = cfg["pinn"]["adam"]
    adam_iters = int(adam_cfg["iters"])
    lr = float(adam_cfg["lr"])

    cbs_adam, method = _make_adaptive_callbacks(cfg, adaptive_cfg, method)
    model_dde.compile("adam", lr=lr, loss_weights=loss_weights)
    model_dde.train(iterations=adam_iters, display_every=500, callbacks=cbs_adam)

    # --- L-BFGS ---
    lbfgs_cfg = cfg["pinn"]["lbfgs"]
    lbfgs_iters = int(lbfgs_cfg["iters"])

    cbs_lbfgs, _ = _make_adaptive_callbacks(cfg, adaptive_cfg, method)
    dde.optimizers.config.set_LBFGS_options(maxcor=100, gtol=1e-08)
    model_dde.compile("L-BFGS", loss_weights=loss_weights)
    model_dde.train(iterations=lbfgs_iters, display_every=500, callbacks=cbs_lbfgs)

    return model_dde


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--checkpoint-every", type=int, default=500)
    args = ap.parse_args()

    cfg = load_config(args.config)
    name = cfg["experiment"]["name"]
    dtype = cfg["pinn"]["dtype"]

    curriculum_cfg = cfg.get("curriculum", {})
    if not curriculum_cfg.get("enabled", False):
        raise ValueError("curriculum.enabled must be true in config")

    # Support both old format (t_split) and new format (t_splits)
    if "t_splits" in curriculum_cfg:
        t_splits = [float(s) for s in curriculum_cfg["t_splits"]]
    elif "t_split" in curriculum_cfg:
        t_splits = [float(curriculum_cfg["t_split"])]
    else:
        raise ValueError("curriculum config must have t_splits or t_split")

    tmin = float(cfg["domain"]["tmin"])
    tmax = float(cfg["domain"]["tmax"])
    boundaries = [tmin] + t_splits + [tmax]
    n_windows = len(boundaries) - 1

    for i in range(len(boundaries) - 1):
        assert boundaries[i] < boundaries[i + 1], \
            f"boundaries must be increasing: {boundaries}"

    print("=" * 60)
    print(f"CURRICULUM LEARNING: {n_windows} windows")
    for w in range(n_windows):
        print(f"  Window {w + 1}: t = [{boundaries[w]}, {boundaries[w + 1]}]")
    print("=" * 60)

    # --- FD baseline (full domain) ---
    fd = solve_fd(cfg)
    x_fd, t_fd, phi_fd = fd["x"], fd["t"], fd["phi"]

    df_tau = 0.0
    df_cfg = cfg["pinn"].get("decay_factor", {})
    if df_cfg.get("enabled", False):
        df_tau = float(df_cfg["tau"])

    loss_weights = [float(w) for w in cfg["pinn"]["lambda"]]
    adaptive_cfg = cfg["pinn"].get("adaptive_sampling", {})

    outdir = os.path.join("outputs", "pinn", name)
    ensure_dir(outdir)

    # Storage for per-window PINN solutions
    window_phipinn = []   # list of (t_mask, phi_pinn_on_mask)
    window_metrics = []
    prev_model = None     # previous window's trained model

    for w in range(n_windows):
        t_lo = boundaries[w]
        t_hi = boundaries[w + 1]
        w_name = f"w{w + 1}"
        ckpt_dir = os.path.join(outdir, f"checkpoints_{w_name}")

        print("\n" + "=" * 60)
        print(f"WINDOW {w + 1}/{n_windows}: t = [{t_lo}, {t_hi}]")
        print("=" * 60)

        cfg_w = copy.deepcopy(cfg)
        cfg_w["domain"]["tmin"] = t_lo
        cfg_w["domain"]["tmax"] = t_hi
        cfg_w["experiment"]["name"] = f"{name}_{w_name}"

        seed = int(cfg["pinn"]["seed"]) + w
        dde.config.set_random_seed(seed)
        dde.config.set_default_float(dtype)

        # Check for completed checkpoint (skip if exists)
        final_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "model-final-*.pt")))

        if final_ckpts:
            ckpt_path = final_ckpts[-1]
            print(f"[W{w + 1}] Found completed checkpoint: {ckpt_path}")
            print(f"[W{w + 1}] Skipping training — loading from checkpoint")

            if w == 0:
                model_w, _ = build_model(cfg_w)
            else:
                # Build with numerical ICs (need interpolators from prev model)
                phi_split, phi_t_split = _extract_ic_from_model(
                    prev_model, x_fd, t_lo, dtype=dtype
                )
                phi_interp = interp1d(x_fd, phi_split, kind="cubic", fill_value="extrapolate")
                phi_t_interp = interp1d(x_fd, phi_t_split, kind="cubic", fill_value="extrapolate")

                _interp_phi = phi_interp
                _interp_phi_t = phi_t_interp

                def _phi_ic(x):
                    return _interp_phi(x[:, 0:1]).reshape(-1, 1)

                def _phi_t_ic(inputs, outputs, X):
                    phi_t = dde.grad.jacobian(outputs, inputs, i=0, j=1)
                    x_np = inputs[:, 0:1].detach().cpu().numpy()
                    target = _interp_phi_t(x_np).reshape(-1, 1)
                    target_t = torch.as_tensor(target, dtype=outputs.dtype, device=outputs.device)
                    return phi_t - target_t

                model_w, _ = build_model_numerical_ic(
                    cfg_w, tmin_override=t_lo, tmax_override=t_hi,
                    phi_ic_func=_phi_ic, phi_t_ic_func=_phi_t_ic,
                )

            model_w.compile("adam", lr=1e-3)
            model_w.train(iterations=0, display_every=1)
            ckpt_data = torch.load(ckpt_path, map_location="cpu")
            model_w.net.load_state_dict(ckpt_data["model_state_dict"])
            print(f"[W{w + 1}] Loaded network weights")

        elif w == 0:
            # Window 1: standard analytic ICs via train_pinn
            model_w, _ = train_pinn(
                cfg_w,
                checkpoint_dir=ckpt_dir,
                checkpoint_every=args.checkpoint_every,
                resume=args.resume,
            )
        else:
            # Windows 2+: numerical ICs from previous model
            phi_split, phi_t_split = _extract_ic_from_model(
                prev_model, x_fd, t_lo, dtype=dtype
            )
            print(f"[IC] phi range at t={t_lo}: [{phi_split.min():.6f}, {phi_split.max():.6f}]")
            print(f"[IC] phi_t range at t={t_lo}: [{phi_t_split.min():.6f}, {phi_t_split.max():.6f}]")

            phi_interp = interp1d(x_fd, phi_split, kind="cubic", fill_value="extrapolate")
            phi_t_interp = interp1d(x_fd, phi_t_split, kind="cubic", fill_value="extrapolate")

            _interp_phi = phi_interp
            _interp_phi_t = phi_t_interp

            def _phi_ic(x):
                return _interp_phi(x[:, 0:1]).reshape(-1, 1)

            def _phi_t_ic(inputs, outputs, X):
                phi_t = dde.grad.jacobian(outputs, inputs, i=0, j=1)
                x_np = inputs[:, 0:1].detach().cpu().numpy()
                target = _interp_phi_t(x_np).reshape(-1, 1)
                target_t = torch.as_tensor(target, dtype=outputs.dtype, device=outputs.device)
                return phi_t - target_t

            model_w, _ = build_model_numerical_ic(
                cfg_w, tmin_override=t_lo, tmax_override=t_hi,
                phi_ic_func=_phi_ic, phi_t_ic_func=_phi_t_ic,
            )

            os.makedirs(ckpt_dir, exist_ok=True)
            model_w = _train_window(model_w, cfg_w, adaptive_cfg, loss_weights)

        # Evaluate this window on its sub-grid
        t_mask = (t_fd >= t_lo - 1e-10) & (t_fd <= t_hi + 1e-10)
        t_w = t_fd[t_mask]
        phi_pinn_w = eval_on_grid(model_w, x=x_fd, t=t_w, dtype=dtype,
                                   decay_factor_tau=df_tau)

        w_met = {
            "RMSD": rmsd(phi_fd[t_mask], phi_pinn_w),
            "MAD": mad(phi_fd[t_mask], phi_pinn_w),
            "RL2": rl2(phi_fd[t_mask], phi_pinn_w),
        }
        window_metrics.append(w_met)
        window_phipinn.append((t_mask, phi_pinn_w))

        save_json(os.path.join(outdir, f"metrics_{w_name}.json"), w_met)
        print(f"[W{w + 1}] RMSD: {w_met['RMSD']:.6f}")

        prev_model = model_w

    # =================================================================
    # Stitch all windows
    # =================================================================
    print("\n" + "=" * 60)
    print(f"STITCHING {n_windows} windows")
    print("=" * 60)

    phi_pinn_full = np.empty_like(phi_fd)

    for w in range(n_windows):
        t_lo = boundaries[w]
        t_hi = boundaries[w + 1]
        t_mask, phi_w = window_phipinn[w]

        if w == 0:
            # First window: take everything up to (but not including) the split
            if n_windows > 1:
                i_split = np.searchsorted(t_fd, boundaries[1])
                phi_pinn_full[:i_split, :] = phi_w[:i_split, :]
            else:
                phi_pinn_full[:] = phi_w
        elif w == n_windows - 1:
            # Last window: take from the split point to end
            i_start = np.searchsorted(t_fd, t_lo)
            phi_pinn_full[i_start:, :] = phi_w
        else:
            # Middle window: take from start to next split
            i_start = np.searchsorted(t_fd, t_lo)
            i_end = np.searchsorted(t_fd, boundaries[w + 1])
            n_rows = i_end - i_start
            phi_pinn_full[i_start:i_end, :] = phi_w[:n_rows, :]

    metrics = {
        "RMSD": rmsd(phi_fd, phi_pinn_full),
        "MAD": mad(phi_fd, phi_pinn_full),
        "RL2": rl2(phi_fd, phi_pinn_full),
    }

    np.savez_compressed(os.path.join(outdir, f"{name}_fd.npz"), **fd)
    np.savez_compressed(os.path.join(outdir, f"{name}_pinn.npz"),
                        x=x_fd, t=t_fd, phi=phi_pinn_full)
    save_json(os.path.join(outdir, "metrics.json"), metrics)

    # Plots
    times = [10.0, 20.0, 30.0, 40.0]
    pot = cfg["physics"]["potential"].title()
    l_val = cfg["physics"]["l"]

    plot_snapshots(
        x_fd, t_fd, phi_fd, phi_pinn_full, times,
        outpath=os.path.join(outdir, "snapshots.png"),
        title=f"Snapshots — {pot} potential (l={l_val})",
    )
    plot_abs_diff_snapshots(
        x_fd, t_fd, phi_fd, phi_pinn_full, times,
        outpath=os.path.join(outdir, "abs_diff_snapshots.png"),
        title=f"Absolute difference — {pot} potential (l={l_val})",
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
    for w in range(n_windows):
        print(f"[CURRICULUM] Window {w + 1} RMSD: {window_metrics[w]['RMSD']:.6f}")
    print(f"[CURRICULUM] Outputs in: {outdir}")


if __name__ == "__main__":
    main()

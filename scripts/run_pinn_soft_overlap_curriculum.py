"""
Soft-overlap temporal curriculum for the Zerilli PINN.

Trains two windows around the old t=25 split:
  - W1: [tmin, t_split + overlap_width/2]
  - W2: [t_split - overlap_width/2, tmax]

W2 starts from numerical ICs extracted from W1 at the overlap start, and
also receives fixed value-continuity anchors from W1 across the overlap.
The final waveform is blended with a smoothstep ramp over the shared band.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import deepxde as dde
import numpy as np
import torch
from scipy.interpolate import interp1d

from src.config import load_config
from src.fd_solver import solve_fd
from src.pinn import (
    GreedyResampler,
    ResidualAdaptiveResampler,
    _combine_loss_histories,
    _convert_loss_history,
    _find_adam_done_checkpoint,
    _find_final_checkpoint,
    _find_latest_checkpoint,
    _get_checkpoint_step,
    _make_pde_residual_only,
    _restore_weights_only,
    build_model_numerical_ic,
    eval_on_grid,
    train_pinn,
)
from src.plotting import (
    plot_abs_diff_snapshots,
    plot_error_heatmap,
    plot_ringdown_overlay,
    plot_snapshots,
    plot_snapshots_zoomed,
)
from src.utils import ensure_dir, mad, rl2, rmsd, save_json


def _extract_ic_from_model(model, x_grid, t_val, dtype="float64"):
    X = np.stack([x_grid, np.full_like(x_grid, t_val)], axis=1)
    phi_vals = model.predict(X).ravel()

    tdtype = torch.float64 if dtype == "float64" else torch.float32
    X_t = torch.tensor(X, dtype=tdtype, requires_grad=True)
    net = model.net
    net.eval()
    y = net(X_t)
    grads = torch.autograd.grad(
        y, X_t, grad_outputs=torch.ones_like(y), create_graph=False
    )[0]
    phi_t_vals = grads[:, 1].detach().cpu().numpy()
    return phi_vals, phi_t_vals


def _make_adam_callbacks(cfg: Dict, checkpoint_path: Optional[str], checkpoint_every: int):
    callbacks: List = []
    adaptive_cfg = cfg["pinn"].get("adaptive_sampling", {})

    if adaptive_cfg.get("enabled", False):
        method = adaptive_cfg.get("method", "RAD")
        period = int(adaptive_cfg.get("period", 1000))
        num_candidates = int(adaptive_cfg.get("num_candidates", 50000))
        eval_batch_size = int(adaptive_cfg.get("eval_batch_size", 5000))
        pde_residual = _make_pde_residual_only(cfg)

        if method == "greedy":
            greedy_fraction = float(adaptive_cfg.get("greedy_fraction", 0.5))
            callbacks.append(
                GreedyResampler(
                    pde_residual=pde_residual,
                    period=period,
                    num_candidates=num_candidates,
                    greedy_fraction=greedy_fraction,
                    eval_batch_size=eval_batch_size,
                )
            )
            print(f"[PINN] Adaptive sampling: greedy f={greedy_fraction}")
        else:
            callbacks.append(
                ResidualAdaptiveResampler(
                    pde_residual=pde_residual,
                    period=period,
                    num_candidates=num_candidates,
                    k=float(adaptive_cfg.get("k", 1.0)),
                    c=float(adaptive_cfg.get("c", 1.0)),
                    method=method,
                    num_add=int(adaptive_cfg.get("num_add", 160)),
                    eval_batch_size=eval_batch_size,
                )
            )
            print(f"[PINN] Adaptive sampling: {method}")
    else:
        callbacks.append(
            dde.callbacks.PDEPointResampler(
                period=int(cfg["pinn"]["adam"].get("resample_period", 100))
            )
        )

    if checkpoint_path:
        callbacks.append(
            dde.callbacks.ModelCheckpoint(
                checkpoint_path, save_better_only=False, period=checkpoint_every
            )
        )
    return callbacks


def _train_existing_model(
    model,
    cfg: Dict,
    checkpoint_dir: str,
    checkpoint_every: int,
    resume: bool,
    loss_weights: List[float],
) -> Tuple[object, Dict]:
    ensure_dir(checkpoint_dir)
    model_save_path = os.path.join(checkpoint_dir, "model")

    final_ckpt = _find_final_checkpoint(checkpoint_dir) if resume else None
    if final_ckpt is not None:
        print(f"[CKPT] Final checkpoint exists: {final_ckpt}")
        print("[CKPT] Loading weights and skipping this window")
        model.compile("adam", lr=float(cfg["pinn"]["adam"]["lr"]), loss_weights=loss_weights)
        model.train(iterations=0, display_every=1)
        _restore_weights_only(model.net, final_ckpt, verbose=1)
        return model, {}

    adam_cfg = cfg["pinn"]["adam"]
    adam_iters = int(adam_cfg["iters"])
    lr = float(adam_cfg["lr"])

    adam_done_ckpt = _find_adam_done_checkpoint(checkpoint_dir) if resume else None
    losshistory_adam = None
    lbfgs_resume_ckpt = None

    if adam_done_ckpt is None:
        callbacks_adam = _make_adam_callbacks(cfg, model_save_path, checkpoint_every)
        model_restore_path = None
        if resume:
            latest = _find_latest_checkpoint(checkpoint_dir)
            if latest is not None:
                model_restore_path = latest
                print(f"[CKPT] Restoring Adam checkpoint: {latest}")

        model.compile("adam", lr=lr, loss_weights=loss_weights)
        print(f"[PINN] Adam: {adam_iters} iters, lr={lr}")
        losshistory_adam, _ = model.train(
            iterations=adam_iters,
            callbacks=callbacks_adam,
            display_every=100,
            model_save_path=model_save_path,
            model_restore_path=model_restore_path,
        )
        model.save(model_save_path + "-adam_done")
        with open(os.path.join(checkpoint_dir, "loss_weights_adam.json"), "w") as f:
            json.dump(list(model.loss_weights), f)
    else:
        print(f"[CKPT] Adam already completed: {adam_done_ckpt}")
        model.compile("adam", lr=lr, loss_weights=loss_weights)
        model.train(iterations=0, display_every=1)
        model.restore(adam_done_ckpt, verbose=1)

        weights_file = os.path.join(checkpoint_dir, "loss_weights_adam.json")
        if os.path.isfile(weights_file):
            with open(weights_file) as f:
                model.loss_weights = json.load(f)

        lbfgs_resume_ckpt = _find_latest_checkpoint(
            checkpoint_dir, exclude_prefix="model-adam_done"
        )
        if lbfgs_resume_ckpt is not None:
            step = _get_checkpoint_step(lbfgs_resume_ckpt)
            if step is not None and step <= adam_iters:
                lbfgs_resume_ckpt = None

    lbfgs_cfg = cfg["pinn"]["lbfgs"]
    lbfgs_iters = int(lbfgs_cfg["iters"])
    lbfgs_done = 0
    if lbfgs_resume_ckpt is not None:
        step = _get_checkpoint_step(lbfgs_resume_ckpt)
        if step is not None:
            lbfgs_done = max(0, step - adam_iters)

    lbfgs_remaining = lbfgs_iters - lbfgs_done
    if lbfgs_remaining <= 0:
        print("[PINN] L-BFGS already completed")
        return model, _convert_loss_history(
            losshistory_adam, loss_weights=loss_weights, phase="adam"
        ) if losshistory_adam is not None else {}

    print(f"[PINN] L-BFGS: {lbfgs_remaining} iterations remaining")
    dde.optimizers.set_LBFGS_options(
        maxcor=100, maxiter=lbfgs_remaining, ftol=0, gtol=1e-8, maxls=50
    )
    lbfgs_resample_period = int(lbfgs_cfg.get("resample_period", 0))
    step_size = min(
        checkpoint_every,
        lbfgs_resample_period if lbfgs_resample_period > 0 else checkpoint_every,
        lbfgs_remaining,
    )
    from deepxde.optimizers.config import LBFGS_options as _lbfgs_opts
    _lbfgs_opts["iter_per_step"] = step_size
    _lbfgs_opts["fun_per_step"] = int(step_size * 1.25)

    lbfgs_loss_weights = list(model.loss_weights)
    model.compile("L-BFGS", loss_weights=lbfgs_loss_weights)
    if lbfgs_resume_ckpt is not None:
        print(f"[CKPT] Restoring L-BFGS weights: {lbfgs_resume_ckpt}")
        _restore_weights_only(model.net, lbfgs_resume_ckpt, verbose=1)

    callbacks_lbfgs = [
        dde.callbacks.ModelCheckpoint(
            model_save_path, save_better_only=False, period=checkpoint_every
        )
    ]
    if lbfgs_resample_period > 0:
        callbacks_lbfgs.append(
            dde.callbacks.PDEPointResampler(period=lbfgs_resample_period)
        )

    losshistory_lbfgs, _ = model.train(
        iterations=lbfgs_remaining, callbacks=callbacks_lbfgs, display_every=100
    )
    model.save(model_save_path + "-final")

    if losshistory_adam is not None:
        history = _combine_loss_histories(
            losshistory_adam,
            losshistory_lbfgs,
            loss_weights_adam=loss_weights,
            loss_weights_lbfgs=lbfgs_loss_weights,
        )
    else:
        history = _convert_loss_history(
            losshistory_lbfgs, loss_weights=lbfgs_loss_weights, phase="lbfgs"
        )
    return model, history


def _smoothstep(s):
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=500)
    args = parser.parse_args()

    cfg = load_config(args.config)
    name = cfg["experiment"]["name"]
    dtype = cfg["pinn"]["dtype"]

    curriculum_cfg = cfg.get("curriculum", {})
    if not curriculum_cfg.get("enabled", False):
        raise ValueError("curriculum.enabled must be true")
    if curriculum_cfg.get("mode") != "soft_overlap":
        raise ValueError("curriculum.mode must be soft_overlap")

    tmin = float(cfg["domain"]["tmin"])
    tmax = float(cfg["domain"]["tmax"])
    t_split = float(curriculum_cfg.get("t_split", 25.0))
    overlap_width = float(curriculum_cfg.get("overlap_width", 10.0))
    overlap_start = t_split - 0.5 * overlap_width
    overlap_end = t_split + 0.5 * overlap_width
    if not (tmin < overlap_start < overlap_end < tmax):
        raise ValueError("Soft-overlap bounds must satisfy tmin < start < end < tmax")

    print("=" * 60)
    print("SOFT-OVERLAP TEMPORAL CURRICULUM")
    print(f"  W1: [{tmin}, {overlap_end}]")
    print(f"  W2: [{overlap_start}, {tmax}]")
    print(f"  Blend: [{overlap_start}, {overlap_end}]")
    print("=" * 60)

    fd = solve_fd(cfg)
    x_fd, t_fd, phi_fd = fd["x"], fd["t"], fd["phi"]
    outdir = os.path.join("outputs", "pinn", name)
    ensure_dir(outdir)

    df_cfg = cfg["pinn"].get("decay_factor", {})
    df_tau = float(df_cfg.get("tau", 0.0)) if df_cfg.get("enabled", False) else 0.0

    cfg_w1 = copy.deepcopy(cfg)
    cfg_w1["domain"]["tmax"] = overlap_end
    cfg_w1["experiment"]["name"] = f"{name}_w1"
    ckpt_w1 = os.path.join(outdir, "checkpoints_w1")

    print("\n" + "=" * 60)
    print(f"WINDOW 1: t=[{tmin}, {overlap_end}] analytic IC")
    print("=" * 60)
    model_w1, hist_w1 = train_pinn(
        cfg_w1,
        checkpoint_dir=ckpt_w1,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
    )

    t_mask_w1 = t_fd <= overlap_end + 1e-10
    phi_w1 = eval_on_grid(
        model_w1, x=x_fd, t=t_fd[t_mask_w1], dtype=dtype,
        decay_factor_tau=df_tau,
    )

    print(f"\n[IC] Extracting W2 IC from W1 at t={overlap_start}")
    phi_start, phi_t_start = _extract_ic_from_model(
        model_w1, x_fd, overlap_start, dtype=dtype
    )
    phi_interp = interp1d(x_fd, phi_start, kind="cubic", fill_value="extrapolate")
    phi_t_interp = interp1d(x_fd, phi_t_start, kind="cubic", fill_value="extrapolate")

    def phi_ic_func(x):
        return phi_interp(x[:, 0:1]).reshape(-1, 1)

    def phi_t_ic_func(inputs, outputs, X):
        phi_t = dde.grad.jacobian(outputs, inputs, i=0, j=1)
        x_np = inputs[:, 0:1].detach().cpu().numpy()
        target = phi_t_interp(x_np).reshape(-1, 1)
        target_t = torch.as_tensor(target, dtype=outputs.dtype, device=outputs.device)
        return phi_t - target_t

    continuity_cfg = curriculum_cfg.get("overlap_continuity", {})
    extra_bcs = []
    loss_weights_w2 = [float(w) for w in cfg["pinn"]["lambda"]]
    if continuity_cfg.get("enabled", True):
        n_x = int(continuity_cfg.get("n_x", 201))
        n_t = int(continuity_cfg.get("n_t", 11))
        weight = float(continuity_cfg.get("weight", 10.0))
        x_anchor = np.linspace(float(cfg["domain"]["xmin"]), float(cfg["domain"]["xmax"]), n_x)
        t_anchor = np.linspace(overlap_start, overlap_end, n_t)
        points = np.concatenate(
            [np.stack([x_anchor, np.full_like(x_anchor, tv)], axis=1) for tv in t_anchor],
            axis=0,
        )
        values = model_w1.predict(points).reshape(-1, 1).astype(np.float64)
        extra_bcs.append(dde.icbc.PointSetBC(points, values, component=0))
        loss_weights_w2.append(weight)
        print(f"[Overlap] Added {len(points)} W1 continuity anchors, weight={weight}")

    cfg_w2 = copy.deepcopy(cfg)
    cfg_w2["domain"]["tmin"] = overlap_start
    cfg_w2["domain"]["tmax"] = tmax
    cfg_w2["experiment"]["name"] = f"{name}_w2"
    ckpt_w2 = os.path.join(outdir, "checkpoints_w2")

    seed = int(cfg["pinn"]["seed"]) + 1
    dde.config.set_random_seed(seed)
    dde.config.set_default_float(dtype)

    print("\n" + "=" * 60)
    print(f"WINDOW 2: t=[{overlap_start}, {tmax}] numerical IC + overlap anchors")
    print("=" * 60)
    model_w2, _ = build_model_numerical_ic(
        cfg_w2,
        tmin_override=overlap_start,
        tmax_override=tmax,
        phi_ic_func=phi_ic_func,
        phi_t_ic_func=phi_t_ic_func,
        extra_bcs=extra_bcs,
    )
    model_w2, hist_w2 = _train_existing_model(
        model_w2,
        cfg_w2,
        checkpoint_dir=ckpt_w2,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        loss_weights=loss_weights_w2,
    )

    t_mask_w2 = t_fd >= overlap_start - 1e-10
    phi_w2 = eval_on_grid(
        model_w2, x=x_fd, t=t_fd[t_mask_w2], dtype=dtype,
        decay_factor_tau=df_tau,
    )

    w1_full = np.full_like(phi_fd, np.nan)
    w2_full = np.full_like(phi_fd, np.nan)
    w1_full[t_mask_w1, :] = phi_w1
    w2_full[t_mask_w2, :] = phi_w2

    pre_mask = t_fd < overlap_start - 1e-10
    post_mask = t_fd > overlap_end + 1e-10
    overlap_mask = (t_fd >= overlap_start - 1e-10) & (t_fd <= overlap_end + 1e-10)

    phi_full = np.empty_like(phi_fd)
    phi_full[pre_mask, :] = w1_full[pre_mask, :]
    phi_full[post_mask, :] = w2_full[post_mask, :]
    s = (t_fd[overlap_mask] - overlap_start) / (overlap_end - overlap_start)
    alpha = _smoothstep(s)[:, None]
    phi_full[overlap_mask, :] = (
        (1.0 - alpha) * w1_full[overlap_mask, :]
        + alpha * w2_full[overlap_mask, :]
    )

    metrics = {
        "RMSD": rmsd(phi_fd, phi_full),
        "MAD": mad(phi_fd, phi_full),
        "RL2": rl2(phi_fd, phi_full),
    }
    metrics_w1 = {
        "RMSD": rmsd(phi_fd[t_mask_w1], phi_w1),
        "MAD": mad(phi_fd[t_mask_w1], phi_w1),
        "RL2": rl2(phi_fd[t_mask_w1], phi_w1),
    }
    metrics_w2 = {
        "RMSD": rmsd(phi_fd[t_mask_w2], phi_w2),
        "MAD": mad(phi_fd[t_mask_w2], phi_w2),
        "RL2": rl2(phi_fd[t_mask_w2], phi_w2),
    }
    metrics_overlap = {
        "RMSD_blend": rmsd(phi_fd[overlap_mask], phi_full[overlap_mask]),
        "RMSD_w1": rmsd(phi_fd[overlap_mask], w1_full[overlap_mask]),
        "RMSD_w2": rmsd(phi_fd[overlap_mask], w2_full[overlap_mask]),
    }

    np.savez_compressed(os.path.join(outdir, f"{name}_fd.npz"), **fd)
    np.savez_compressed(
        os.path.join(outdir, f"{name}_pinn.npz"), x=x_fd, t=t_fd, phi=phi_full
    )
    save_json(os.path.join(outdir, "metrics.json"), metrics)
    save_json(os.path.join(outdir, "metrics_w1.json"), metrics_w1)
    save_json(os.path.join(outdir, "metrics_w2.json"), metrics_w2)
    save_json(os.path.join(outdir, "metrics_overlap.json"), metrics_overlap)
    save_json(
        os.path.join(outdir, "curriculum_info.json"),
        {
            "mode": "soft_overlap",
            "t_split": t_split,
            "overlap_width": overlap_width,
            "overlap_start": overlap_start,
            "overlap_end": overlap_end,
            "continuity_anchors": len(extra_bcs) > 0,
        },
    )
    if hist_w1:
        save_json(os.path.join(outdir, "loss_history_w1.json"), hist_w1)
    if hist_w2:
        save_json(os.path.join(outdir, "loss_history_w2.json"), hist_w2)

    times = [10.0, 20.0, 30.0, 40.0]
    pot = cfg["physics"]["potential"].title()
    l_val = cfg["physics"]["l"]
    plot_snapshots(
        x_fd, t_fd, phi_fd, phi_full, times,
        outpath=os.path.join(outdir, "snapshots.png"),
        title=f"Snapshots - {pot} potential (l={l_val})",
    )
    plot_abs_diff_snapshots(
        x_fd, t_fd, phi_fd, phi_full, times,
        outpath=os.path.join(outdir, "abs_diff_snapshots.png"),
        title=f"Absolute difference - {pot} potential (l={l_val})",
    )
    plot_snapshots_zoomed(
        x_fd, t_fd, phi_fd, phi_full, times,
        outpath=os.path.join(outdir, "snapshots_zoomed.png"),
        title=f"Snapshots (zoomed) - {pot} potential (l={l_val})",
    )
    plot_error_heatmap(
        x_fd, t_fd, phi_fd, phi_full,
        outpath=os.path.join(outdir, "error_heatmap.png"),
        title=f"Pointwise error - {pot} (l={l_val})",
    )
    plot_error_heatmap(
        x_fd, t_fd, phi_fd, phi_full,
        outpath=os.path.join(outdir, "error_heatmap_zoomed.png"),
        title=f"Pointwise error (zoomed) - {pot} (l={l_val})",
        xlim=(-20.0, 60.0),
    )
    xq = float(cfg["evaluation"]["xq"])
    ix = int(np.argmin(np.abs(x_fd - xq)))
    plot_ringdown_overlay(
        t_fd, phi_fd[:, ix], phi_full[:, ix],
        outpath=os.path.join(outdir, "ringdown_overlay.png"),
        title=f"Ringdown - {pot} (l={l_val})",
        xq=xq,
    )

    print(f"\n[CURRICULUM] Full metrics: {metrics}")
    print(f"[CURRICULUM] W1 metrics: {metrics_w1}")
    print(f"[CURRICULUM] W2 metrics: {metrics_w2}")
    print(f"[CURRICULUM] Overlap metrics: {metrics_overlap}")
    print(f"[CURRICULUM] Outputs in: {outdir}")


if __name__ == "__main__":
    main()
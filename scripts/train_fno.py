"""Train an FNO surrogate for the Zerilli operator.

Usage:
    python scripts/train_fno.py --config configs/fno_zerilli_l2.yaml
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
import random
import time
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import load_config
from src.utils import ensure_dir, save_json, rmsd, rl2
from src.fno_dataset import load_dataset
from src.fno_model import (
    build_fno,
    build_model,
    count_parameters,
    loss_field,
    loss_h1,
    loss_obs_slice,
    loss_pde_residual,
    loss_ringdown,
    loss_time_weighted,
)


def _device(spec: str) -> torch.device:
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_loader(split, batch_size: int, shuffle: bool, V_per_sample: np.ndarray):
    X = torch.from_numpy(split["X"])
    Y = torch.from_numpy(split["Y"])
    V = torch.from_numpy(V_per_sample)
    ds = TensorDataset(X, Y, V)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _evaluate(model, loader, device, dx, dt, t_grid, x_grid, lcfg, qnm_cfg, xq):
    model.eval()
    tot = {"field": 0.0, "rmsd": 0.0, "rl2": 0.0, "n": 0}
    with torch.no_grad():
        for X, Y, V in loader:
            X = X.to(device); Y = Y.to(device); V = V.to(device)
            P = model(X)
            tot["field"] += loss_field(P, Y).item() * X.shape[0]
            tot["rmsd"] += float(torch.sqrt(((P - Y) ** 2).mean()).item()) * X.shape[0]
            den = float(torch.sqrt((Y ** 2).mean()).item())
            tot["rl2"] += (float(torch.sqrt(((P - Y) ** 2).mean()).item()) / max(den, 1e-30)) * X.shape[0]
            tot["n"] += X.shape[0]
    n = max(tot["n"], 1)
    return {"val_field_mse": tot["field"] / n,
            "val_rmsd": tot["rmsd"] / n,
            "val_rl2": tot["rl2"] / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--quick", action="store_true",
                    help="Override epochs to 5 for a smoke test.")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from <out_dir>/model.pt if present (model+opt+sched+epoch+RNG).")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override training.epochs from CLI (useful when extending a finished run).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg["training"]
    name = cfg["experiment"]["name"]

    seed = int(tcfg.get("seed", 1234))
    torch.manual_seed(seed); np.random.seed(seed)

    device = _device(tcfg.get("device", "auto"))
    dtype = torch.float64 if str(tcfg.get("dtype", "float32")) == "float64" else torch.float32
    torch.set_default_dtype(dtype)

    data_path = tcfg["data_path"]
    out_dir = tcfg["out_dir"]
    ensure_dir(out_dir)

    splits, grid, meta = load_dataset(data_path)
    if "train" not in splits:
        raise SystemExit(f"No 'train' split in {data_path}.  Run generate_fno_dataset.py first.")

    x_grid = torch.from_numpy(grid.x).to(device)
    t_grid = torch.from_numpy(grid.t).to(device)
    dx = float(meta["fd_dx"]) * int(meta.get("stride_x", 1))
    dt = float(meta["fd_dt"]) * int(meta.get("stride_t", 1))
    xq = float(meta["xq"])
    qnm_cfg = {"t_start": float(meta["qnm_t_start"]),
               "t_end": float(meta["qnm_t_end"])}
    lcfg = tcfg["loss"]

    bs = int(tcfg.get("batch_size", 4))
    train_loader = _make_loader(splits["train"], bs, True, splits["train"]["V"])
    val_loader = _make_loader(splits["val"], bs, False, splits["val"]["V"]) \
        if "val" in splits else None

    model = build_model(cfg, t_grid=t_grid, x_grid=x_grid).to(device)
    n_params = count_parameters(model)
    print(f"[fno-train] device={device} dtype={dtype} params={n_params:,}")

    if args.epochs is not None:
        epochs = int(args.epochs)
    else:
        epochs = 5 if args.quick else int(tcfg.get("epochs", 200))
    opt = torch.optim.AdamW(model.parameters(),
                            lr=float(tcfg.get("lr", 1e-3)),
                            weight_decay=float(tcfg.get("weight_decay", 1e-6)))
    if str(tcfg.get("scheduler", "cosine")) == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    else:
        sched = None
    grad_clip = float(tcfg.get("grad_clip", 0.0))

    log_every = int(tcfg.get("log_every", 10))
    ckpt_every = int(tcfg.get("ckpt_every", 50))

    # --- Resume-safe checkpoint load ---------------------------------
    ckpt_path = os.path.join(out_dir, "model.pt")
    best_path = os.path.join(out_dir, "best.pt")
    start_epoch = 1
    best_val = float("inf")
    history = []
    if args.resume and os.path.isfile(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        if "optimizer_state" in ck:
            opt.load_state_dict(ck["optimizer_state"])
        if sched is not None and "scheduler_state" in ck:
            sched.load_state_dict(ck["scheduler_state"])
        start_epoch = int(ck.get("epoch", 0)) + 1
        # Legacy checkpoints without scheduler_state: fast-forward cosine
        # by (start_epoch - 1) steps so LR matches what it would have been.
        if sched is not None and "scheduler_state" not in ck:
            for _ in range(start_epoch - 1):
                sched.step()
        best_val = float(ck.get("best_val", float("inf")))
        rng = ck.get("rng_state")
        if rng is not None:
            torch.set_rng_state(rng["torch"])
            if torch.cuda.is_available() and rng.get("cuda") is not None:
                torch.cuda.set_rng_state_all(rng["cuda"])
            np.random.set_state(rng["numpy"])
            random.setstate(rng["python"])
        hist_path = os.path.join(out_dir, "history.json")
        if os.path.isfile(hist_path):
            try:
                history = json.load(open(hist_path))["history"]
            except Exception:
                history = []
        print(f"[fno-train] RESUMED from {ckpt_path} at epoch {start_epoch-1} "
              f"-> continuing to {epochs}; best_val={best_val:.3e}")
    elif args.resume:
        print(f"[fno-train] --resume requested but no checkpoint at {ckpt_path}; starting fresh.")
    t_train0 = time.time()
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        ep_t0 = time.time()
        # v4 new-loss config knobs (back-compat: default to 0 / no-op)
        lam_obs = float(lcfg.get("lam_obs", 0.0))
        lam_tw  = float(lcfg.get("lam_tw",  0.0))
        tw_t_ramp_start = float(lcfg.get("tw_t_ramp_start", qnm_cfg["t_start"]))
        tw_t_ramp_end   = float(lcfg.get("tw_t_ramp_end",   qnm_cfg["t_start"] + 5.0))
        tw_beta         = float(lcfg.get("tw_beta", 0.0))
        running: Dict[str, float] = {"loss": 0.0, "field": 0.0, "h1": 0.0,
                                      "ring": 0.0, "pde": 0.0,
                                      "obs": 0.0, "tw": 0.0, "n": 0}
        for X, Y, V in train_loader:
            X = X.to(device); Y = Y.to(device); V = V.to(device)
            P = model(X)
            l_field = loss_field(P, Y)
            l_h1 = loss_h1(P, Y, dx=dx, dt=dt) if lcfg["lam_h1"] > 0 else P.new_tensor(0.0)
            l_ring = loss_ringdown(P, Y, t_grid, x_grid,
                                   t_start=qnm_cfg["t_start"], t_end=qnm_cfg["t_end"],
                                   xq=xq) if lcfg["lam_ring"] > 0 else P.new_tensor(0.0)
            l_pde = loss_pde_residual(P, V, dx=dx, dt=dt) if lcfg["lam_pde"] > 0 else P.new_tensor(0.0)
            l_obs = loss_obs_slice(P, Y, x_grid, xq=xq) if lam_obs > 0 else P.new_tensor(0.0)
            l_tw  = loss_time_weighted(P, Y, t_grid,
                                       t_ramp_start=tw_t_ramp_start,
                                       t_ramp_end=tw_t_ramp_end,
                                       beta=tw_beta) if lam_tw > 0 else P.new_tensor(0.0)
            loss = (lcfg["lam_field"] * l_field
                    + lcfg["lam_h1"] * l_h1
                    + lcfg["lam_ring"] * l_ring
                    + lcfg["lam_pde"] * l_pde
                    + lam_obs * l_obs
                    + lam_tw  * l_tw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            b = X.shape[0]
            running["loss"] += loss.item() * b
            running["field"] += l_field.item() * b
            running["h1"] += float(l_h1.item()) * b
            running["ring"] += float(l_ring.item()) * b
            running["pde"] += float(l_pde.item()) * b
            running["obs"] += float(l_obs.item()) * b
            running["tw"]  += float(l_tw.item()) * b
            running["n"] += b
        if sched is not None:
            sched.step()
        n = max(running["n"], 1)
        rec = {"epoch": epoch,
               "loss": running["loss"] / n,
               "field": running["field"] / n,
               "h1": running["h1"] / n,
               "ring": running["ring"] / n,
               "pde": running["pde"] / n,
               "obs": running["obs"] / n,
               "tw":  running["tw"]  / n,
               "lr": opt.param_groups[0]["lr"],
               "epoch_sec": time.time() - ep_t0}
        if val_loader is not None and (epoch % log_every == 0 or epoch == epochs):
            rec.update(_evaluate(model, val_loader, device, dx, dt, t_grid, x_grid,
                                 lcfg, qnm_cfg, xq))
        history.append(rec)
        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            extras = (f"  val_rmsd={rec.get('val_rmsd', float('nan')):.3e}"
                      if "val_rmsd" in rec else "")
            print(f"[fno-train] ep {epoch:4d}/{epochs}  "
                  f"loss={rec['loss']:.3e}  field={rec['field']:.3e}  "
                  f"ring={rec['ring']:.3e}  obs={rec['obs']:.3e}  "
                  f"tw={rec['tw']:.3e}  pde={rec['pde']:.3e}{extras}",
                  flush=True)

        # --- Best-val checkpoint (whenever val improves) ---
        if "val_rmsd" in rec and rec["val_rmsd"] < best_val:
            best_val = float(rec["val_rmsd"])
            torch.save({"model_state": model.state_dict(),
                        "optimizer_state": opt.state_dict(),
                        "scheduler_state": sched.state_dict() if sched is not None else None,
                        "epoch": epoch,
                        "best_val": best_val,
                        "rng_state": {
                            "torch": torch.get_rng_state(),
                            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                            "numpy": np.random.get_state(),
                            "python": random.getstate(),
                        },
                        "config": cfg},
                       best_path)

        # --- Rolling checkpoint (every ckpt_every + at end) ---
        if epoch % ckpt_every == 0 or epoch == epochs:
            torch.save({"model_state": model.state_dict(),
                        "optimizer_state": opt.state_dict(),
                        "scheduler_state": sched.state_dict() if sched is not None else None,
                        "epoch": epoch,
                        "best_val": best_val,
                        "rng_state": {
                            "torch": torch.get_rng_state(),
                            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                            "numpy": np.random.get_state(),
                            "python": random.getstate(),
                        },
                        "config": cfg},
                       ckpt_path)
            save_json(os.path.join(out_dir, "history.json"), {"history": history})

    elapsed = time.time() - t_train0
    print(f"[fno-train] done in {elapsed:.1f}s")

    save_json(os.path.join(out_dir, "history.json"), {"history": history})

    # Final test-set metrics
    if "test" in splits:
        test_loader = _make_loader(splits["test"], bs, False, splits["test"]["V"])
        test_metrics = _evaluate(model, test_loader, device, dx, dt, t_grid, x_grid,
                                  lcfg, qnm_cfg, xq)
        save_json(os.path.join(out_dir, "metrics.json"),
                  {"n_params": n_params, "elapsed_sec": elapsed, **test_metrics})
        print(f"[fno-train] test  field_mse={test_metrics['val_field_mse']:.3e}  "
              f"rmsd={test_metrics['val_rmsd']:.3e}  rl2={test_metrics['val_rl2']:.3e}")


if __name__ == "__main__":
    main()

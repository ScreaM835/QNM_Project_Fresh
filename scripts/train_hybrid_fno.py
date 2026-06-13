"""Train the hybrid coarse-FD + FNO residual surrogate.

Usage:
    python scripts/train_hybrid_fno.py --config configs/hybrid_sw_train_k2.yaml
    python scripts/train_hybrid_fno.py --config configs/hybrid_sw_train_k4.yaml [--smoke]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.hybrid_dataset import load_dataset
from src.hybrid_data_pipe import assemble_split, to_torch
from src.hybrid_fno import build_hybrid_fno, count_parameters
from src.fno_model import loss_pde_residual


def _resolve_device(want: str) -> str:
    if want == "cuda" and not torch.cuda.is_available():
        print("[HYBRID-TRAIN] cuda requested but unavailable; using cpu")
        return "cpu"
    return want


def _make_loader(X: np.ndarray, Y: np.ndarray, batch_size: int, shuffle: bool,
                 device: str) -> DataLoader:
    Xt = torch.from_numpy(X)
    Yt = torch.from_numpy(Y)
    ds = TensorDataset(Xt, Yt)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=(device == "cuda"))


def _compose_loss(model: nn.Module, Xb: torch.Tensor, Yb: torch.Tensor,
                  dx: float, dt: float,
                  weights: Tuple[float, float, float]
                  ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Composite hybrid loss. ``weights = (data, physics, anchor)``.

    * data    : supervised MSE on the fine-FD delta target ``Yb``  [needs fine FD]
    * physics : Zerilli PDE residual of the RECONSTRUCTED field
                ``u = coarse_prior + correction``                  [LABEL-FREE]
    * anchor  : ``||correction||^2 = ||u - coarse_prior||^2``      [LABEL-FREE]

    The coarse prior (``Xb[:, 0:1]``) is the structural anchor: the model output
    is the additive correction and the field is ``prior + correction``, so a
    small correction keeps ``u`` near the (cheap, correct-IC/BC) prior. The PDE
    residual is the only training signal that does not need fine-FD labels and so
    is the piece that transfers to higher dimensions. ``V(x)`` is channel 2.

    **Hard-IC gating (label-free only).** The Zerilli equation is linear &
    homogeneous, so ``u = 0`` (correction = -prior) trivially satisfies the PDE
    residual -> a from-scratch physics loss can collapse to zero (the Kerr C.3
    failure). When ``w_data == 0`` (no supervised pin) and ``w_phys > 0`` we gate
    the correction by ``s^2 = (t/T)^2`` so the reconstructed field obeys
    ``u(.,0) = prior(.,0)`` EXACTLY for any network output. The prior carries the
    correct initial pulse, so this locks the IC to the right value and makes the
    zero solution structurally unreachable. ``s^2`` (and its t-derivative) vanish
    at ``t=0`` so the initial data is untouched. In supervised mode this is a
    no-op (gating off), preserving byte-identical legacy behaviour.
    """
    w_data, w_phys, w_anchor = weights
    raw = model(Xb)                                   # raw network correction
    hard_ic = (w_data == 0.0 and w_phys > 0.0)
    if hard_ic:
        Nt = Xb.shape[2]
        t = torch.arange(Nt, dtype=Xb.dtype, device=Xb.device) * dt
        tmax = t[-1].clamp(min=1e-12)
        gate = (t / tmax).pow(2).view(1, 1, Nt, 1)    # s^2, =0 at t=0
        corr = gate * raw
    else:
        corr = raw

    loss = Xb.new_zeros(())
    terms = {"data": 0.0, "physics": 0.0, "anchor": 0.0}
    if w_data > 0.0:
        l_data = torch.mean((corr - Yb) ** 2)
        loss = loss + w_data * l_data
        terms["data"] = float(l_data.item())
    if w_phys > 0.0:
        u = Xb[:, 0:1] + corr                         # reconstructed fine field
        V = Xb[:, 2, 0, :]                            # V(x): channel 2 (bcast over t)
        l_phys = loss_pde_residual(u, V, dx, dt)
        loss = loss + w_phys * l_phys
        terms["physics"] = float(l_phys.item())
    if w_anchor > 0.0:
        l_anchor = torch.mean(corr ** 2)
        loss = loss + w_anchor * l_anchor
        terms["anchor"] = float(l_anchor.item())
    return loss, corr, terms


def _epoch_pass(model: nn.Module, loader: DataLoader, device: str,
                optimizer: torch.optim.Optimizer | None,
                dx: float, dt: float,
                weights: Tuple[float, float, float]
                ) -> Tuple[float, float, Dict[str, float]]:
    """Returns (mean composite loss, mean field rel-L2 vs fine FD, mean terms).

    The reported L2 ratio is ALWAYS the field rel-L2 against the held-out fine
    field (``Yb`` only enters the *metric*, not the loss, in label-free mode), so
    it remains a valid eval number whatever the training weights.
    """
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_n = 0
    total_l2_ratio = 0.0
    n_batches = 0
    term_sums = {"data": 0.0, "physics": 0.0, "anchor": 0.0}
    for Xb, Yb in loader:
        Xb = Xb.to(device, non_blocking=True)
        Yb = Yb.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad()
        loss, corr, terms = _compose_loss(model, Xb, Yb, dx, dt, weights)
        if training:
            loss.backward()
            optimizer.step()
        bs = Xb.shape[0]
        total_loss += float(loss.item()) * bs
        total_n += bs
        for k in term_sums:
            term_sums[k] += terms[k]
        # eval metric: field rel-L2 vs fine FD (valid in every training mode).
        # corr is the EFFECTIVE correction (hard-IC gated when label-free), so
        # u = prior + corr and field error = ||corr - Yb|| / ||fine field||.
        field = Yb + Xb[:, 0:1]
        resid_after = corr.detach() - Yb
        l2_field = torch.sqrt((field ** 2).mean()).item()
        l2_err   = torch.sqrt((resid_after ** 2).mean()).item()
        total_l2_ratio += l2_err / max(l2_field, 1e-12)
        n_batches += 1
    nb = max(n_batches, 1)
    mean_terms = {k: v / nb for k, v in term_sums.items()}
    return total_loss / total_n, total_l2_ratio / max(n_batches, 1), mean_terms


def _save_ckpt(path: str, *, model, optimizer, epoch: int, phase: str,
               best_val: float, history: list, n_params: int,
               config_path: str, meta: dict) -> None:
    """Atomic checkpoint save: write to .tmp then rename."""
    tmp = path + ".tmp"
    torch.save({
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "epoch": epoch,
        "phase": phase,
        "best_val": best_val,
        "history": history,
        "n_params": n_params,
        "config_path": config_path,
        "dataset_meta": meta,
    }, tmp)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="Train for a few epochs on tiny tensors for pipeline check.")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from <out_dir>/ckpt.pt if it exists.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg["logging"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    device = _resolve_device(str(cfg["train"].get("device", "cuda")).lower())
    torch.manual_seed(int(cfg["train"].get("seed", 1234)))

    # ---- data ----------------------------------------------------------------
    ds_path = cfg["dataset"]["path"]
    print(f"[HYBRID-TRAIN] loading {ds_path}")
    splits, grid, meta = load_dataset(ds_path)
    print(f"[HYBRID-TRAIN] meta: {meta}")

    t0 = time.time()
    X_tr, Y_tr, _ = assemble_split(
        splits["train"], grid.x_coarse, grid.t_coarse, grid.x_fine, grid.t_fine,
    )
    X_va, Y_va, _ = assemble_split(
        splits["val"],   grid.x_coarse, grid.t_coarse, grid.x_fine, grid.t_fine,
    )
    print(f"[HYBRID-TRAIN] assembled in {time.time()-t0:.1f}s: "
          f"X_tr {X_tr.shape}, Y_tr {Y_tr.shape}, X_va {X_va.shape}")

    if args.smoke:
        X_tr, Y_tr = X_tr[:2], Y_tr[:2]
        X_va, Y_va = X_va[:2], Y_va[:2]
        cfg["train"]["epochs_adam"] = 3
        cfg["train"]["epochs_lbfgs"] = 0

    bs = int(cfg["train"]["batch_size"])
    loader_tr = _make_loader(X_tr, Y_tr, bs, True, device)
    loader_va = _make_loader(X_va, Y_va, bs, False, device)

    # ---- loss weights (default = pure supervised, == old behaviour) ----------
    lcfg = cfg["train"].get("loss", {})
    weights = (float(lcfg.get("data_weight", 1.0)),
               float(lcfg.get("physics_weight", 0.0)),
               float(lcfg.get("anchor_weight", 0.0)))
    dx_f = float(grid.x_fine[1] - grid.x_fine[0])
    dt_f = float(grid.t_fine[1] - grid.t_fine[0])
    print(f"[HYBRID-TRAIN] loss weights (data,physics,anchor)={weights}  "
          f"dx_f={dx_f:.3f} dt_f={dt_f:.3f}")

    # ---- model ---------------------------------------------------------------
    model = build_hybrid_fno(cfg).to(device)
    n_params = count_parameters(model)
    print(f"[HYBRID-TRAIN] model params: {n_params}")

    # ---- Adam ----------------------------------------------------------------
    n_ep_adam = int(cfg["train"]["epochs_adam"])
    opt = torch.optim.AdamW(model.parameters(),
                            lr=float(cfg["train"]["lr_adam"]),
                            weight_decay=float(cfg["train"].get("weight_decay", 0.0)))
    history: list = []
    best_val = float("inf")
    log_every = int(cfg["logging"].get("log_every", 10))
    start_ep_adam = 1
    start_ep_lbfgs = 1
    resume_phase = "adam"

    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        try:
            opt.load_state_dict(ck["optim_state"])
        except Exception as exc:  # phase changed (Adam->LBFGS) so optim shape differs
            print(f"[RESUME] could not restore Adam optim state: {exc}; reinitialised")
        history = list(ck.get("history", []))
        best_val = float(ck.get("best_val", float("inf")))
        resume_phase = str(ck.get("phase", "adam"))
        last_ep = int(ck.get("epoch", 0))
        if resume_phase == "adam":
            start_ep_adam = last_ep + 1
        else:
            start_ep_adam = n_ep_adam + 1  # skip Adam, jump to L-BFGS
            start_ep_lbfgs = last_ep + 1
        print(f"[RESUME] loaded {ckpt_path}: phase={resume_phase} "
              f"epoch={last_ep} best_val={best_val:.3e}; "
              f"continue Adam@{start_ep_adam}, LBFGS@{start_ep_lbfgs}")

    for ep in range(start_ep_adam, n_ep_adam + 1):
        t_ep = time.time()
        tr_loss, tr_ratio, tr_terms = _epoch_pass(model, loader_tr, device, opt,
                                                  dx_f, dt_f, weights)
        va_loss, va_ratio, _ = _epoch_pass(model, loader_va, device, None,
                                           dx_f, dt_f, weights)
        history.append({
            "epoch": ep, "phase": "adam",
            "train_mse": tr_loss, "val_mse": va_loss,
            "train_l2_ratio": tr_ratio, "val_l2_ratio": va_ratio,
            "terms": tr_terms,
            "wall_s": time.time() - t_ep,
        })
        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), os.path.join(out_dir, "model_best.pt"))
        _save_ckpt(ckpt_path, model=model, optimizer=opt, epoch=ep, phase="adam",
                   best_val=best_val, history=history, n_params=n_params,
                   config_path=args.config, meta=meta)
        if ep == start_ep_adam or ep % log_every == 0 or ep == n_ep_adam:
            print(f"[ADAM] ep {ep:4d}/{n_ep_adam}  "
                  f"train MSE {tr_loss:.3e} (L2 {tr_ratio:.3e}) | "
                  f"val MSE {va_loss:.3e} (L2 {va_ratio:.3e}) | "
                  f"terms d/p/a {tr_terms['data']:.2e}/{tr_terms['physics']:.2e}/"
                  f"{tr_terms['anchor']:.2e} | "
                  f"{history[-1]['wall_s']:.1f}s")

    # ---- L-BFGS finetune -----------------------------------------------------
    n_ep_lbfgs = int(cfg["train"].get("epochs_lbfgs", 0))
    if n_ep_lbfgs > 0:
        # L-BFGS needs a *scalar* loss per closure call, but the full training
        # set does not fit on one A100 (~119 GiB of activations for 1000 samples
        # at (501, 1001) fields). We accumulate over chunks of `lbfgs_chunk_size`
        # samples, sum the (un-normalised) MSE contributions, and divide once at
        # the end; backward() inside each chunk frees that chunk's activations.
        X_full_cpu = torch.from_numpy(X_tr)        # stay on CPU; ship per-chunk
        Y_full_cpu = torch.from_numpy(Y_tr)
        n_full = X_full_cpu.shape[0]
        chunk = int(cfg["train"].get("lbfgs_chunk_size", 8))
        n_elems = float(Y_full_cpu.numel())        # for MSE normalisation
        print(f"[LBFGS] chunked closure: chunk={chunk}, n_full={n_full}, "
              f"n_elems={n_elems:.0f}")

        opt = torch.optim.LBFGS(
            model.parameters(),
            max_iter=int(cfg["train"].get("lbfgs_max_iter", 20)),
            history_size=20,
            line_search_fn="strong_wolfe",
        )
        if args.resume and resume_phase == "lbfgs" and os.path.exists(ckpt_path):
            try:
                ck = torch.load(ckpt_path, map_location=device, weights_only=False)
                opt.load_state_dict(ck["optim_state"])
                print(f"[RESUME] restored L-BFGS optim state at epoch {start_ep_lbfgs - 1}")
            except Exception as exc:
                print(f"[RESUME] could not restore L-BFGS optim state: {exc}")

        def closure() -> torch.Tensor:
            opt.zero_grad()
            total = torch.zeros((), device=device)
            for i in range(0, n_full, chunk):
                Xc = X_full_cpu[i:i + chunk].to(device, non_blocking=True)
                Yc = Y_full_cpu[i:i + chunk].to(device, non_blocking=True)
                # per-chunk composite loss, weighted by chunk fraction so the
                # accumulated scalar is the dataset-mean composite loss.
                part, _, _ = _compose_loss(model, Xc, Yc, dx_f, dt_f, weights)
                part = part * (Xc.shape[0] / float(n_full))
                part.backward()        # frees this chunk's activations
                total = total + part.detach()
            return total

        for ep in range(start_ep_lbfgs, n_ep_lbfgs + 1):
            t_ep = time.time()
            model.train()
            opt.step(closure)
            va_loss, va_ratio, _ = _epoch_pass(model, loader_va, device, None,
                                               dx_f, dt_f, weights)
            history.append({
                "epoch": ep, "phase": "lbfgs",
                "val_mse": va_loss, "val_l2_ratio": va_ratio,
                "wall_s": time.time() - t_ep,
            })
            if va_loss < best_val:
                best_val = va_loss
                torch.save(model.state_dict(), os.path.join(out_dir, "model_best.pt"))
            _save_ckpt(ckpt_path, model=model, optimizer=opt, epoch=ep, phase="lbfgs",
                       best_val=best_val, history=history, n_params=n_params,
                       config_path=args.config, meta=meta)
            print(f"[LBFGS] ep {ep:3d}/{n_ep_lbfgs}  "
                  f"val MSE {va_loss:.3e} (L2 {va_ratio:.3e}) | "
                  f"{history[-1]['wall_s']:.1f}s")

    # ---- save artefacts ------------------------------------------------------
    torch.save(model.state_dict(), os.path.join(out_dir, "model_last.pt"))
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump({
            "history": history,
            "best_val_mse": best_val,
            "model_params": n_params,
            "config_path": args.config,
            "dataset_meta": meta,
        }, f, indent=2)
    print(f"[HYBRID-TRAIN] best val MSE = {best_val:.3e}; wrote {out_dir}")

if __name__ == "__main__":
    main()

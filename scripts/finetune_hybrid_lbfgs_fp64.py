"""Float64 L-BFGS finetune on top of a trained hybrid FNO.

Starts from `<out_dir>/model_best.pt` (assumed fp32 weights), upcasts model
and data to float64, runs chunked L-BFGS, and writes:
    <out_dir>/model_best_fp64.pt   (best fp64 weights by val MSE)
    <out_dir>/model_last_fp64.pt
    <out_dir>/ckpt_fp64.pt         (resume-safe)
    <out_dir>/history_fp64.json

Usage:
    python scripts/finetune_hybrid_lbfgs_fp64.py --config configs/hybrid_sw_train_k2.yaml
        [--epochs 30] [--max_iter 20] [--chunk 4] [--resume]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.hybrid_dataset import load_dataset
from src.hybrid_data_pipe import assemble_split
from src.hybrid_fno import build_hybrid_fno


def _save_ckpt(path: str, *, model, optimizer, epoch: int, best_val: float,
               history: list) -> None:
    tmp = path + ".tmp"
    torch.save({
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "epoch": epoch,
        "best_val": best_val,
        "history": history,
    }, tmp)
    os.replace(tmp, path)


_CAST_MAP = {
    torch.float32: torch.float64,
    torch.float16: torch.float64,
    torch.bfloat16: torch.float64,
    torch.complex64: torch.complex128,
}


def _upcast_module_(module: torch.nn.Module) -> None:
    """In-place type-aware upcast: float32->float64, complex64->complex128.

    Naive `model.to(torch.float64)` casts complex tensors to float64 and
    discards their imaginary parts — the FNO's spectral convolution stores
    its mode weights as complex64, so that would silently destroy the model.
    """
    def _cast(t: torch.Tensor) -> torch.Tensor:
        return t.to(_CAST_MAP[t.dtype]) if t.dtype in _CAST_MAP else t
    for name, param in list(module.named_parameters(recurse=True)):
        new = _cast(param.data)
        if new.dtype != param.dtype:
            owner_name, _, leaf = name.rpartition(".")
            owner = module.get_submodule(owner_name) if owner_name else module
            setattr(owner, leaf, torch.nn.Parameter(new, requires_grad=param.requires_grad))
    for name, buf in list(module.named_buffers(recurse=True)):
        new = _cast(buf)
        if new.dtype != buf.dtype:
            owner_name, _, leaf = name.rpartition(".")
            owner = module.get_submodule(owner_name) if owner_name else module
            owner.register_buffer(leaf, new)


def _upcast_state_dict(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        if torch.is_tensor(v) and v.dtype in _CAST_MAP:
            out[k] = v.to(_CAST_MAP[v.dtype])
        else:
            out[k] = v
    return out


def _val_mse_fp64(model, X_va_cpu: torch.Tensor, Y_va_cpu: torch.Tensor,
                  device: str, chunk: int) -> tuple[float, float]:
    """Returns (val_mse, val_l2_ratio) in fp64."""
    model.eval()
    n_full = X_va_cpu.shape[0]
    n_elems = float(Y_va_cpu.numel())
    sse = torch.zeros((), dtype=torch.float64, device=device)
    field_sq = torch.zeros((), dtype=torch.float64, device=device)
    with torch.no_grad():
        for i in range(0, n_full, chunk):
            Xc = X_va_cpu[i:i + chunk].to(device, dtype=torch.float64, non_blocking=True)
            Yc = Y_va_cpu[i:i + chunk].to(device, dtype=torch.float64, non_blocking=True)
            pred = model(Xc)
            err = pred - Yc
            sse = sse + (err ** 2).sum()
            # field magnitude = Y + ch0 of X (i.e. upsampled coarse)
            field = Yc + Xc[:, 0:1]
            field_sq = field_sq + (field ** 2).sum()
    val_mse = (sse / n_elems).item()
    val_l2_ratio = (torch.sqrt(sse) / torch.sqrt(field_sq)).item()
    return val_mse, val_l2_ratio


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--max_iter", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=4,
                    help="L-BFGS closure chunk size (fp64 doubles memory vs fp32 run)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg["logging"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(int(cfg["train"].get("seed", 1234)))

    # ---- data ----------------------------------------------------------------
    ds_path = cfg["dataset"]["path"]
    print(f"[FP64-LBFGS] loading {ds_path}")
    splits, grid, meta = load_dataset(ds_path)
    t0 = time.time()
    X_tr_np, Y_tr_np, _ = assemble_split(
        splits["train"], grid.x_coarse, grid.t_coarse, grid.x_fine, grid.t_fine,
    )
    X_va_np, Y_va_np, _ = assemble_split(
        splits["val"], grid.x_coarse, grid.t_coarse, grid.x_fine, grid.t_fine,
    )
    print(f"[FP64-LBFGS] assembled in {time.time()-t0:.1f}s: "
          f"X_tr {X_tr_np.shape}, Y_tr {Y_tr_np.shape}")
    X_tr_cpu = torch.from_numpy(X_tr_np)  # still fp32 on CPU; cast per-chunk
    Y_tr_cpu = torch.from_numpy(Y_tr_np)
    X_va_cpu = torch.from_numpy(X_va_np)
    Y_va_cpu = torch.from_numpy(Y_va_np)

    # ---- model: build then type-aware upcast to fp64 ------------------------
    model = build_hybrid_fno(cfg).to(device)
    _upcast_module_(model)
    # quick sanity check
    n_f64 = sum(1 for p in model.parameters() if p.dtype == torch.float64)
    n_c128 = sum(1 for p in model.parameters() if p.dtype == torch.complex128)
    n_other = sum(1 for p in model.parameters() if p.dtype not in (torch.float64, torch.complex128))
    print(f"[FP64-LBFGS] params after upcast: float64={n_f64}, complex128={n_c128}, other={n_other}")

    # ---- seed weights from fp32 checkpoint or resume ckpt -------------------
    ckpt_path = os.path.join(out_dir, "ckpt_fp64.pt")
    history: list = []
    best_val = float("inf")
    start_ep = 1
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        history = list(ck.get("history", []))
        best_val = float(ck.get("best_val", float("inf")))
        start_ep = int(ck.get("epoch", 0)) + 1
        print(f"[FP64-LBFGS] RESUME from ckpt: ep {start_ep-1}, best_val {best_val:.3e}")
    else:
        fp32_path = os.path.join(out_dir, "model_best.pt")
        if not os.path.exists(fp32_path):
            raise FileNotFoundError(f"need {fp32_path} as starting point")
        sd_fp32 = torch.load(fp32_path, map_location=device, weights_only=False)
        sd_fp64 = _upcast_state_dict(sd_fp32)
        model.load_state_dict(sd_fp64)
        print(f"[FP64-LBFGS] loaded fp32 weights from {fp32_path}, upcast to fp64")

    # baseline (fp64) val loss at the loaded weights
    va_mse0, va_l2_0 = _val_mse_fp64(model, X_va_cpu, Y_va_cpu, device, args.chunk)
    print(f"[FP64-LBFGS] baseline val MSE (fp64) = {va_mse0:.6e}  L2 = {va_l2_0:.3e}")
    if start_ep == 1:
        best_val = va_mse0
        torch.save(model.state_dict(), os.path.join(out_dir, "model_best_fp64.pt"))
        history.append({"epoch": 0, "phase": "baseline_fp64",
                        "val_mse": va_mse0, "val_l2_ratio": va_l2_0, "wall_s": 0.0})

    # ---- L-BFGS in fp64, chunked closure -------------------------------------
    n_full = X_tr_cpu.shape[0]
    n_elems = float(Y_tr_cpu.numel())
    chunk = int(args.chunk)
    print(f"[FP64-LBFGS] chunked closure: chunk={chunk}, n_full={n_full}, "
          f"n_elems={n_elems:.0f}")

    opt = torch.optim.LBFGS(
        model.parameters(),
        max_iter=int(args.max_iter),
        history_size=20,
        tolerance_grad=1e-12,         # don't bail early on fp32-scale tolerances
        tolerance_change=1e-14,
        line_search_fn="strong_wolfe",
    )
    if args.resume and os.path.exists(ckpt_path):
        try:
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            opt.load_state_dict(ck["optim_state"])
            print("[FP64-LBFGS] restored L-BFGS optim state from ckpt")
        except Exception as exc:
            print(f"[FP64-LBFGS] could not restore optim state: {exc}")

    def closure() -> torch.Tensor:
        opt.zero_grad()
        total = torch.zeros((), dtype=torch.float64, device=device)
        for i in range(0, n_full, chunk):
            Xc = X_tr_cpu[i:i + chunk].to(device, dtype=torch.float64, non_blocking=True)
            Yc = Y_tr_cpu[i:i + chunk].to(device, dtype=torch.float64, non_blocking=True)
            pred = model(Xc)
            part = ((pred - Yc) ** 2).sum() / n_elems
            part.backward()
            total = total + part.detach()
        return total

    for ep in range(start_ep, args.epochs + 1):
        t_ep = time.time()
        model.train()
        loss_tr = opt.step(closure).item()
        va_mse, va_l2 = _val_mse_fp64(model, X_va_cpu, Y_va_cpu, device, chunk)
        wall = time.time() - t_ep
        history.append({"epoch": ep, "phase": "lbfgs_fp64",
                        "train_mse": float(loss_tr),
                        "val_mse": va_mse, "val_l2_ratio": va_l2,
                        "wall_s": wall})
        if va_mse < best_val:
            best_val = va_mse
            torch.save(model.state_dict(), os.path.join(out_dir, "model_best_fp64.pt"))
        _save_ckpt(ckpt_path, model=model, optimizer=opt, epoch=ep,
                   best_val=best_val, history=history)
        print(f"[LBFGS-FP64] ep {ep:3d}/{args.epochs}  "
              f"train {loss_tr:.6e} | val {va_mse:.6e} (L2 {va_l2:.3e}) | {wall:.1f}s")

    torch.save(model.state_dict(), os.path.join(out_dir, "model_last_fp64.pt"))
    with open(os.path.join(out_dir, "history_fp64.json"), "w") as f:
        json.dump({"history": history, "best_val_mse": best_val,
                   "config_path": args.config, "dataset_meta": meta,
                   "epochs": args.epochs, "max_iter": args.max_iter, "chunk": chunk},
                  f, indent=2)
    print(f"[FP64-LBFGS] DONE: best val MSE = {best_val:.6e}; "
          f"start was {va_mse0:.6e}  ({va_mse0 / best_val:.2f}x improvement)")


if __name__ == "__main__":
    main()

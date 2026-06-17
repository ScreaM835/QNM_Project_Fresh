"""Diagnostic: compare the loss gradient at model_best.pt in fp32 vs fp64.

If fp32 noise dominates the true gradient near the minimum, the two grads will
disagree (large angle / very different magnitude). If they agree to 4+ decimals
on both magnitude and direction, precision is NOT the bottleneck.
"""
from __future__ import annotations

import argparse
import math
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
from scripts.finetune_hybrid_lbfgs_fp64 import _upcast_module_, _upcast_state_dict


def _flat_grad(model: torch.nn.Module) -> torch.Tensor:
    parts = []
    for p in model.parameters():
        if p.grad is None:
            parts.append(torch.zeros(p.numel(), dtype=p.dtype, device=p.device))
        else:
            parts.append(p.grad.detach().reshape(-1))
    return torch.cat(parts)


def _full_grad(model: torch.nn.Module, X: torch.Tensor, Y: torch.Tensor,
               dtype: torch.dtype, device: str, chunk: int) -> tuple[float, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    n_full = X.shape[0]
    n_elems = float(Y.numel())
    total = torch.zeros((), dtype=dtype, device=device)
    for i in range(0, n_full, chunk):
        Xc = X[i:i + chunk].to(device, dtype=dtype, non_blocking=True)
        Yc = Y[i:i + chunk].to(device, dtype=dtype, non_blocking=True)
        pred = model(Xc)
        part = ((pred - Yc) ** 2).sum() / n_elems
        part.backward()
        total = total + part.detach()
    g = _flat_grad(model)
    return float(total.item()), g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--n_samples", type=int, default=64,
                    help="subset of training samples used to compute the gradient")
    ap.add_argument("--device", default="cpu",
                    help="cuda or cpu; login-node check usually cpu")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg["logging"]["out_dir"]
    ckpt_path = os.path.join(out_dir, "model_best.pt")

    print(f"[DIAG] dataset {cfg['dataset']['path']}")
    splits, grid, _ = load_dataset(cfg["dataset"]["path"])
    t0 = time.time()
    X_tr, Y_tr, _ = assemble_split(
        splits["train"], grid.x_coarse, grid.t_coarse, grid.x_fine, grid.t_fine,
    )
    X_tr = X_tr[:args.n_samples]
    Y_tr = Y_tr[:args.n_samples]
    print(f"[DIAG] assembled subset {X_tr.shape} in {time.time()-t0:.1f}s")
    X_cpu = torch.from_numpy(X_tr)
    Y_cpu = torch.from_numpy(Y_tr)

    device = args.device

    # ---- fp32 grad -----------------------------------------------------------
    model32 = build_hybrid_fno(cfg).to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    model32.load_state_dict(sd)
    t0 = time.time()
    loss32, g32 = _full_grad(model32, X_cpu, Y_cpu, torch.float32, device, args.chunk)
    print(f"[DIAG] fp32 loss {loss32:.10e}  grad ‖g‖₂ {g32.norm().item():.6e}  "
          f"({time.time()-t0:.1f}s)")

    # ---- fp64 grad -----------------------------------------------------------
    model64 = build_hybrid_fno(cfg).to(device)
    _upcast_module_(model64)
    model64.load_state_dict(_upcast_state_dict(sd))
    t0 = time.time()
    loss64, g64 = _full_grad(model64, X_cpu, Y_cpu, torch.float64, device, args.chunk)
    print(f"[DIAG] fp64 loss {loss64:.10e}  grad ‖g‖₂ {g64.norm().item():.6e}  "
          f"({time.time()-t0:.1f}s)")

    # ---- compare -------------------------------------------------------------
    # fp64 cast on g32 in the same parameter ordering. The two grads should be
    # over identical parameter slices because model32/model64 are built from
    # the same cfg in the same order.
    g32_64 = g32.to(torch.float64)
    if g32_64.numel() != g64.numel():
        print(f"[DIAG] PARAM COUNT MISMATCH: fp32 n={g32_64.numel()} fp64 n={g64.numel()}")
        return
    diff = (g64 - g32_64).norm().item()
    n64 = g64.norm().item()
    n32 = g32_64.norm().item()
    cos = float((g64 @ g32_64) / max(n64 * n32, 1e-30))
    cos = max(-1.0, min(1.0, cos))
    angle_deg = math.degrees(math.acos(cos))
    print(f"[DIAG] ‖g64 - g32‖₂ = {diff:.6e}")
    print(f"[DIAG] rel diff     = {diff / max(n64, 1e-30):.6e}")
    print(f"[DIAG] mag ratio    = fp32/fp64 = {n32 / max(n64, 1e-30):.6f}")
    print(f"[DIAG] cos(g32,g64) = {cos:.8f}  →  angle = {angle_deg:.4f}°")
    print(f"[DIAG] loss diff    = {loss64 - loss32:.3e}")

    if angle_deg > 10.0 or n32 / max(n64, 1e-30) > 10.0 or n32 / max(n64, 1e-30) < 0.1:
        print("[DIAG] VERDICT: fp32 gradient is NOISE-DOMINATED → fp64 finetune is justified")
    elif angle_deg < 1.0 and abs(n32 / max(n64, 1e-30) - 1.0) < 0.01:
        print("[DIAG] VERDICT: fp32 and fp64 grads AGREE → precision is NOT the bottleneck")
    else:
        print(f"[DIAG] VERDICT: fp32 grad has mild contamination "
              f"(angle {angle_deg:.2f}°, mag ratio {n32 / max(n64, 1e-30):.3f}) "
              f"— modest fp64 benefit expected")


if __name__ == "__main__":
    main()

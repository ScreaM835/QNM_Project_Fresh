"""Inverse problem: recover M from a noisy ringdown observation by
back-propagating through a frozen, trained FNO.

This is the FNO analogue of scripts/run_pinn_inverse.py.  Each step is a
single FNO forward+backward (~ms) instead of a full PINN re-train.

Usage:
    python scripts/inverse_fno.py --config configs/fno_zerilli_l2.yaml
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import numpy as np
import torch

from src.config import load_config
from src.fd_solver import solve_fd
from src.utils import ensure_dir, save_json
from src.fno_dataset import IN_CHANNELS, _channels_from_solution
from src.fno_model import build_fno
from src.potentials import V_zerilli_torch, V_regge_wheeler_torch


def _device(spec: str) -> torch.device:
    if spec == "cpu": return torch.device("cpu")
    if spec == "cuda": return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _V_torch(potential: str, x: torch.Tensor, M: torch.Tensor, l: int) -> torch.Tensor:
    if potential == "zerilli":
        return V_zerilli_torch(x, M, l)
    if potential == "regge-wheeler":
        return V_regge_wheeler_torch(x, M, l)
    raise ValueError(f"unknown potential {potential}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    name = cfg["experiment"]["name"]
    tcfg = cfg["training"]
    icfg = cfg["inverse"]
    out_dir = tcfg["out_dir"]
    ensure_dir(out_dir)

    device = _device(tcfg.get("device", "auto"))
    ckpt = os.path.join(out_dir, "model.pt")
    if not os.path.exists(ckpt):
        raise SystemExit(f"No checkpoint at {ckpt}.  Train first.")

    # 1) Build truth observation: run the FD solver at M_true with the
    #    *base* initial-data parameters, then add Gaussian noise.
    M_true = float(icfg["M_true"])
    cfg_true = dict(cfg)
    cfg_true["physics"] = {**cfg["physics"], "M": M_true}
    sol = solve_fd(cfg_true)
    x_np = sol["x"]; t_np = sol["t"]
    phi_obs_full = sol["phi"]                                  # (Nt, Nx)
    rng = np.random.default_rng(int(tcfg.get("seed", 1234)))
    sigma_n = float(icfg["noise_sigma"])
    obs_x_targets = list(icfg["obs_x"])
    obs_idx = [int(np.argmin(np.abs(x_np - xq))) for xq in obs_x_targets]
    phi_obs = phi_obs_full[:, obs_idx] + sigma_n * rng.standard_normal((phi_obs_full.shape[0], len(obs_idx)))
    phi_obs_t = torch.from_numpy(phi_obs.astype(np.float32)).to(device)

    # Pi0 used to build the FNO input must match what the FD solver used
    # at t=0; we compute it the same way as src/fno_dataset.py.
    Phi0 = sol["phi"][0]
    Pi0 = (sol["phi"][1] - sol["phi"][0]) / float(sol["dt"])
    Phi0_t = torch.from_numpy(Phi0.astype(np.float32)).to(device)
    Pi0_t = torch.from_numpy(Pi0.astype(np.float32)).to(device)

    # 2) Frozen FNO
    model = build_fno(cfg).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model_state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # 3) Optimisable scalar M.  Kept in fp64 because the Lambert-W backward
    # used inside V_zerilli_torch overflows in fp32 at large positive x.
    M_hat = torch.nn.Parameter(torch.tensor(float(icfg["M_init"]),
                                            dtype=torch.float64, device=device))
    opt = torch.optim.Adam([M_hat], lr=float(icfg["lr"]))

    Nt, Nx = phi_obs_full.shape
    x_t64 = torch.from_numpy(x_np.astype(np.float64)).to(device)
    obs_idx_t = torch.tensor(obs_idx, dtype=torch.long, device=device)

    history = []
    steps = int(icfg["steps"])
    potential = cfg["physics"]["potential"]
    l = int(cfg["physics"]["l"])

    for step in range(1, steps + 1):
        # Channel 2 (V) and channel 3 (M scalar) depend on M_hat.
        # Compute V in fp64 (for stable Lambert-W backward) then cast.
        V_hat64 = _V_torch(potential, x_t64, M_hat, l)         # (Nx,) fp64
        V_hat = V_hat64.to(torch.float32)
        M_hat32 = M_hat.to(torch.float32)
        chans = torch.empty((1, IN_CHANNELS, Nt, Nx), device=device)
        chans[0, 0] = Phi0_t.unsqueeze(0).expand(Nt, -1)
        chans[0, 1] = Pi0_t.unsqueeze(0).expand(Nt, -1)
        chans[0, 2] = V_hat.unsqueeze(0).expand(Nt, -1)
        chans[0, 3] = M_hat32 * torch.ones((Nt, Nx), device=device)
        pred = model(chans)                                    # (1, 1, Nt, Nx)
        pred_obs = pred[0, 0][:, obs_idx_t]                    # (Nt, n_obs)
        loss = torch.mean((pred_obs - phi_obs_t) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # Clip the scalar gradient so an under-trained surrogate can't blow up M_hat.
        torch.nn.utils.clip_grad_norm_([M_hat], max_norm=1.0)
        opt.step()
        # Keep M_hat physically meaningful (Schwarzschild mass > 0 and within
        # the FNO training range to avoid extrapolation pathologies).
        with torch.no_grad():
            M_hat.clamp_(0.5, 1.5)
        if step == 1 or step % max(1, steps // 20) == 0 or step == steps:
            print(f"[fno-inv] step {step:4d}/{steps}  M_hat={M_hat.item():.6f}  "
                  f"|err|={abs(M_hat.item()-M_true):.2e}  loss={loss.item():.3e}",
                  flush=True)
            history.append({"step": step,
                            "M_hat": float(M_hat.item()),
                            "loss": float(loss.item())})

    save_json(os.path.join(out_dir, "inverse.json"),
              {"M_true": M_true, "M_init": float(icfg["M_init"]),
               "M_hat_final": float(M_hat.item()),
               "abs_error": float(abs(M_hat.item() - M_true)),
               "history": history,
               "noise_sigma": sigma_n,
               "obs_x_targets": obs_x_targets,
               "obs_x_used": [float(x_np[i]) for i in obs_idx]})
    print(f"[fno-inv] final  M_hat={M_hat.item():.6f}  M_true={M_true:.6f}")


if __name__ == "__main__":
    main()

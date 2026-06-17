"""Evaluate a trained FNO on the test split and emit FD-schema .npz files
that the existing `scripts/extract_qnm.py` can consume unchanged.

Usage:
    python scripts/eval_fno.py --config configs/fno_zerilli_l2.yaml [--n 4]
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import numpy as np
import torch

from src.config import load_config
from src.utils import ensure_dir, save_json, rmsd, rl2
from src.fno_dataset import load_dataset
from src.fno_model import build_fno, build_model


def _device(spec: str) -> torch.device:
    if spec == "cpu": return torch.device("cpu")
    if spec == "cuda": return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=4,
                    help="Number of test samples to predict + dump as .npz.")
    ap.add_argument("--split", choices=["test", "val", "train"], default="test")
    args = ap.parse_args()

    cfg = load_config(args.config)
    name = cfg["experiment"]["name"]
    tcfg = cfg["training"]
    out_dir = tcfg["out_dir"]
    ckpt_path = os.path.join(out_dir, "model.pt")
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"No checkpoint at {ckpt_path}. Train first.")

    device = _device(tcfg.get("device", "auto"))
    splits, grid, meta = load_dataset(tcfg["data_path"])
    if args.split not in splits:
        raise SystemExit(f"split '{args.split}' not in dataset.")

    # QNM-head wrapper needs the grids at construction time
    x_grid_t = torch.from_numpy(grid.x).to(device)
    t_grid_t = torch.from_numpy(grid.t).to(device)
    model = build_model(cfg, t_grid=t_grid_t, x_grid=x_grid_t).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()

    X = torch.from_numpy(splits[args.split]["X"]).to(device)
    Y = splits[args.split]["Y"]
    P = splits[args.split]["P"]
    V = splits[args.split]["V"]

    n = min(int(args.n), X.shape[0])
    pred_dir = os.path.join(out_dir, "predictions", args.split)
    ensure_dir(pred_dir)

    summary = []
    with torch.no_grad():
        for i in range(n):
            yhat = model(X[i:i+1]).cpu().numpy()[0, 0]   # (Nt, Nx)
            # If the model has a QNM head, grab its predicted (omega, tau) too.
            qnm_pred = None
            if hasattr(model, "last_qnm_params"):
                qp = model.last_qnm_params()
                if qp is not None:
                    qnm_pred = {
                        "omega": qp["omega"][0].numpy().tolist(),
                        "tau":   qp["tau"][0].numpy().tolist(),
                    }
            ytrue = Y[i, 0]
            M_i, x0_i, sigma_i = [float(v) for v in P[i]]
            sample_name = f"{name}_pred_{i:03d}_M{M_i:.3f}"
            # Per-sample subdir mirrors the FD pipeline so extract_qnm.py works.
            sub = os.path.join("outputs", "fno", name, "as_fd", sample_name)
            ensure_dir(sub)
            # Mirror the FD .npz schema (x, t, phi, V, dx, dt).
            fd_like_path = os.path.join("outputs", "fd", f"{sample_name}_fd.npz")
            ensure_dir(os.path.dirname(fd_like_path))
            np.savez_compressed(
                fd_like_path,
                x=grid.x, t=grid.t, phi=yhat.astype(np.float64),
                V=V[i].astype(np.float64),
                dx=float(meta["fd_dx"]) * int(meta.get("stride_x", 1)),
                dt=float(meta["fd_dt"]) * int(meta.get("stride_t", 1)),
            )
            # Also write the FD ground truth for direct comparison.
            np.savez_compressed(
                os.path.join(pred_dir, f"sample_{i:03d}.npz"),
                x=grid.x, t=grid.t,
                phi_pred=yhat, phi_true=ytrue,
                V=V[i], M=M_i, x0=x0_i, sigma=sigma_i,
            )
            r = rmsd(yhat, ytrue)
            l = rl2(ytrue, yhat)   # rl2 takes (true, pred) per src/utils.py
            row = {
                "i": i, "M": M_i, "x0": x0_i, "sigma": sigma_i,
                "rmsd": r, "rl2": l, "fd_like_npz": fd_like_path,
            }
            if qnm_pred is not None:
                row["qnm_head_omega"] = qnm_pred["omega"]
                row["qnm_head_tau"]   = qnm_pred["tau"]
            summary.append(row)
            qmsg = (f"  qnm_head: omega={qnm_pred['omega']} tau={qnm_pred['tau']}"
                    if qnm_pred is not None else "")
            print(f"[fno-eval] {i:3d}  M={M_i:.3f} x0={x0_i:+.2f} sigma={sigma_i:.2f}"
                  f"   RMSD={r:.3e}  rL2={l:.3e}{qmsg}")

    save_json(os.path.join(out_dir, f"eval_{args.split}.json"),
              {"split": args.split, "n": n, "samples": summary})
    print(f"[fno-eval] wrote {n} predictions to {pred_dir}")
    print(f"[fno-eval] FD-schema .npz files in outputs/fd/ "
          f"(rerun extract_qnm.py with the appropriate config name)")


if __name__ == "__main__":
    main()

"""Head-to-head FNO vs PINN on the canonical fixed-BH setup.

Build inputs for (M=1, x0=4, sigma=5) — the same physical setup the
zerilli_l2_curriculum_3w PINN was trained on — feed them to the trained
FNO v4 model, and extract the QNM from the predicted waveform at xq=2.0.
Also run FD for ground truth and write everything to a results JSON.

Usage:
    python scripts/fno_fixed_bh_qnm.py --config configs/fno_zerilli_l2_v4.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.fno_dataset import load_dataset
from src.fno_model import build_model
from src.potentials import V_zerilli
from src.qnm import (
    qnm_method_1, qnm_method_2, qnm_method_3_esprit,
    qnm_method_4_window_scan, percentage_errors,
)


def _safe(result, M):
    try:
        pct = percentage_errors(result, potential="zerilli", ell=2, M=M)
        return {
            "omega_M": pct.get("omega_dim"),
            "tau_over_M": pct.get("tau_dim"),
            "omega_pct_err": pct.get("omega_pct_err"),
            "tau_pct_err": pct.get("tau_pct_err"),
        }
    except Exception:
        return {"omega_M": None, "tau_over_M": None,
                "omega_pct_err": None, "tau_pct_err": None}


def _clean(d):
    out = {}
    for k, v in d.items():
        if v is None:
            out[k] = None
        else:
            try:
                f = float(v)
                out[k] = f if np.isfinite(f) else None
            except Exception:
                out[k] = None
    return out


def extract_all(t, y, M, t_start, t_end):
    try: r1 = qnm_method_1(t, y, t_start, t_end)
    except Exception: r1 = {"omega": float("nan"), "tau": float("nan")}
    try: r2 = qnm_method_2(t, y, t_start, t_end)
    except Exception: r2 = {"omega": float("nan"), "tau": float("nan")}
    try: r3 = qnm_method_3_esprit(t, y, t_start, t_end, K=4)
    except Exception: r3 = {"omega": float("nan"), "tau": float("nan")}
    try:
        r4 = qnm_method_4_window_scan(
            t, y, t_start_min=t_start, t_start_max=t_start+10,
            t_end=t_end, n_starts=12,
        )
    except Exception:
        r4 = {"omega": float("nan"), "tau": float("nan")}
    return {
        "m1": _clean(_safe(r1, M)),
        "m2": _clean(_safe(r2, M)),
        "m3": _clean(_safe(r3, M)),
        "m4": _clean(_safe(r4, M)),
    }


def build_input(x, t, M, x0, sigma, ell, profile="outgoing"):
    """Build (4, Nt, Nx) input matching the FNO channel layout.

    Uses the same Gaussian convention as src/initial_data.py:
        Phi0 = exp(-((x-x0)^2)/sigma^2)        (no 1/2 factor)
        Pi0  = gaussian_phi_t(...)             (depends on profile)
    Then advances one FD step on the same (dx, dt) used at training, so the
    Pi0 channel matches the training pipeline's (phi[1]-phi[0])/dt feature.
    """
    from src.initial_data import gaussian_phi, gaussian_phi_t
    from src.potentials import V_of_x
    Nx, Nt = x.size, t.size
    Phi0 = gaussian_phi(x, A=1.0, x0=x0, sigma=sigma).astype(np.float32)
    phi_t0 = gaussian_phi_t(x, A=1.0, x0=x0, sigma=sigma, profile=profile)
    Pi0 = phi_t0.astype(np.float32)
    V = V_of_x(x.astype(np.float64), M=M, l=ell, potential="zerilli").astype(np.float32)
    chans = np.empty((4, Nt, Nx), dtype=np.float32)
    chans[0] = np.broadcast_to(Phi0, (Nt, Nx))
    chans[1] = np.broadcast_to(Pi0, (Nt, Nx))
    chans[2] = np.broadcast_to(V, (Nt, Nx))
    chans[3] = np.full((Nt, Nx), float(M), dtype=np.float32)
    return chans, V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--M", type=float, default=1.0)
    ap.add_argument("--x0", type=float, default=4.0)
    ap.add_argument("--sigma", type=float, default=5.0)
    ap.add_argument("--xq", type=float, default=2.0)
    ap.add_argument("--t_start", type=float, default=10.0)
    ap.add_argument("--t_end", type=float, default=100.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    name = cfg["experiment"]["name"]
    out_dir = cfg["training"]["out_dir"]
    ckpt = os.path.join(out_dir, "model.pt")
    if not os.path.exists(ckpt):
        raise SystemExit(f"No checkpoint at {ckpt}")

    # Load grid + meta from the run's dataset.
    splits, grid, meta = load_dataset(cfg["training"]["data_path"])
    x = grid.x.astype(np.float64)
    t = grid.t.astype(np.float64)
    print(f"[fno-fixed] grid x=[{x.min()},{x.max()}] Nx={x.size}  "
          f"t=[{t.min()},{t.max()}] Nt={t.size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_grid_t = torch.from_numpy(grid.x).to(device)
    t_grid_t = torch.from_numpy(grid.t).to(device)
    model = build_model(cfg, t_grid=t_grid_t, x_grid=x_grid_t).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()

    # Build canonical input
    chans, V = build_input(x, t, args.M, args.x0, args.sigma, ell=2)
    X = torch.from_numpy(chans[None]).to(device)  # (1, 4, Nt, Nx)
    with torch.no_grad():
        phi_pred = model(X).cpu().numpy()[0, 0]   # (Nt, Nx)
    print(f"[fno-fixed] FNO prediction: shape={phi_pred.shape}  "
          f"|phi_pred|.max={np.abs(phi_pred).max():.3e}")

    # Ground truth: FD solve on the same grid as the FNO
    try:
        from src.fd_solver import solve_fd
    except Exception:
        solve_fd = None

    phi_true = None
    if solve_fd is not None:
        # Build a minimal cfg dict for run_fd to mirror the v4 setup
        fd_cfg = {
            "physics": {"M": args.M, "potential": "zerilli", "l": 2},
            "initial_data": {"A": 1.0, "x0": args.x0, "sigma": args.sigma,
                             "velocity_profile": "outgoing"},
            "domain": {"xmin": float(x.min()), "xmax": float(x.max()),
                       "tmin": 0.0, "tmax": float(t.max())},
            "fd": {"dx": float(meta["fd_dx"]), "dt": float(meta["fd_dt"])},
        }
        try:
            sol = solve_fd(fd_cfg)
            ph = sol["phi"]
            xs = sol["x"]; ts = sol["t"]
            # Downsample to FNO grid if needed
            if ph.shape != phi_pred.shape:
                # interp t then x
                from scipy.interpolate import RegularGridInterpolator
                f = RegularGridInterpolator((ts, xs), ph, bounds_error=False,
                                            fill_value=0.0)
                tt, xx = np.meshgrid(t, x, indexing="ij")
                phi_true = f((tt, xx))
            else:
                phi_true = ph
            print(f"[fno-fixed] FD ground truth solved.")
        except Exception as e:
            print(f"[fno-fixed] FD solve failed: {e}")

    # Extract waveform at xq
    ix = int(np.argmin(np.abs(x - args.xq)))
    print(f"[fno-fixed] xq={args.xq} -> x[{ix}]={x[ix]:.3f}")
    y_pred = phi_pred[:, ix].astype(np.float64)
    print(f"[fno-fixed] FNO @xq peak |y|={np.abs(y_pred).max():.3e}")

    results = {
        "config": args.config,
        "run": name,
        "M": args.M, "x0": args.x0, "sigma": args.sigma,
        "xq": args.xq, "x_idx": ix, "x_actual": float(x[ix]),
        "t_start": args.t_start, "t_end": args.t_end,
        "theory_omega_M": 0.3737, "theory_tau_over_M": 11.241,
        "FNO": extract_all(t, y_pred, args.M, args.t_start, args.t_end),
    }
    # QNM head readout if available
    if hasattr(model, "last_qnm_params"):
        qp = model.last_qnm_params()
        if qp is not None:
            results["FNO_qnm_head"] = {
                "omega": qp["omega"][0].cpu().numpy().tolist(),
                "tau":   qp["tau"][0].cpu().numpy().tolist(),
            }

    if phi_true is not None:
        y_true = phi_true[:, ix].astype(np.float64)
        print(f"[fno-fixed] GT  @xq peak |y|={np.abs(y_true).max():.3e}")
        results["GT"] = extract_all(t, y_true, args.M, args.t_start, args.t_end)
        # Field error
        results["field_rmsd"] = float(np.sqrt(np.mean((phi_pred - phi_true)**2)))

    # Save
    out_path = args.out or f"outputs/qnm/fno_xq2_comparison/fixed_BH_{name}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[fno-fixed] wrote {out_path}")

    # Pretty print
    print("\n=== FNO QNM extraction (fixed BH: M=%s, x0=%s, sigma=%s) ===" %
          (args.M, args.x0, args.sigma))
    for m in ("m1", "m2", "m3", "m4"):
        f = results["FNO"][m]
        g = results.get("GT", {}).get(m, {})
        print(f"  {m}:  FNO Mω={f['omega_M']}  τ/M={f['tau_over_M']}  "
              f"({f['omega_pct_err']}% / {f['tau_pct_err']}%)")
        if g:
            print(f"       GT  Mω={g['omega_M']}  τ/M={g['tau_over_M']}  "
                  f"({g['omega_pct_err']}% / {g['tau_pct_err']}%)")
    if "FNO_qnm_head" in results:
        print(f"  QNM head: omega={results['FNO_qnm_head']['omega']}, "
              f"tau={results['FNO_qnm_head']['tau']}")


if __name__ == "__main__":
    main()

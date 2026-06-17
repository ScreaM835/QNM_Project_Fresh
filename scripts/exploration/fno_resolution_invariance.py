"""FNO zero-shot resolution invariance demo.

Loads the v4 FNO trained on the (Nt=401, Nx=401) grid and evaluates it at
several other grid resolutions on the same physical domain
(x in [-50, 150] M, t in [0, 100] M), with no retraining and no
fine-tuning. Compares the output to a FD reference solution computed
independently at each resolution.

Outputs:
    outputs/qnm/fno_resolution_invariance/summary.json
    outputs/qnm/fno_resolution_invariance/resolution_curves.png
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.fd_solver import solve_fd
from src.fno_model import build_model
from src.qnm import qnm_method_4_window_scan
from scripts.fno_fixed_bh_qnm import build_input

CFG = "configs/fno_zerilli_l2_v4.yaml"
M_BH, X0, SIGMA = 1.0, 4.0, 5.0
XQ = 2.0
T_START, T_END = 10.0, 100.0
THEORY_W = 0.3737
XMIN, XMAX = -50.0, 150.0
TMIN, TMAX = 0.0, 100.0

# (Nt, Nx) test grids — training was (401, 401). dt/dx <= 1 for FD stability.
GRIDS = [
    (201, 201),
    (301, 301),
    (401, 401),   # training resolution
    (601, 601),
    (801, 801),
]

OUT_DIR = ROOT / "outputs/qnm/fno_resolution_invariance"


def load_v4_model():
    cfg = load_config(CFG)
    ckpt = os.path.join(cfg["training"]["out_dir"], "model.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # build_model only needs t_grid/x_grid for the qnm_head variant; v4 is plain
    model = build_model(cfg).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model, device


def run_one(Nt, Nx, model, device):
    x = np.linspace(XMIN, XMAX, Nx, dtype=np.float64)
    t = np.linspace(TMIN, TMAX, Nt, dtype=np.float64)
    dx = float(x[1] - x[0])
    dt = float(t[1] - t[0])

    chans, V = build_input(x, t, M_BH, X0, SIGMA, ell=2)
    X = torch.from_numpy(chans[None]).to(device)
    t0 = time.time()
    with torch.no_grad():
        phi_pred = model(X).cpu().numpy()[0, 0]
    t_fno = time.time() - t0

    fd_cfg = {
        "physics": {"M": M_BH, "potential": "zerilli", "l": 2},
        "initial_data": {"A": 1.0, "x0": X0, "sigma": SIGMA,
                         "velocity_profile": "outgoing"},
        "domain": {"xmin": XMIN, "xmax": XMAX, "tmin": TMIN, "tmax": TMAX},
        "fd": {"dx": dx, "dt": dt},
    }
    t0 = time.time()
    sol = solve_fd(fd_cfg)
    t_fd = time.time() - t0
    phi_true = sol["phi"]
    # FD solver may produce a slightly off-by-one grid — pad/truncate as needed
    if phi_true.shape != phi_pred.shape:
        from scipy.interpolate import RegularGridInterpolator
        f = RegularGridInterpolator((sol["t"], sol["x"]), phi_true,
                                    bounds_error=False, fill_value=0.0)
        tt, xx = np.meshgrid(t, x, indexing="ij")
        phi_true = f((tt, xx))

    rmsd_full = float(np.sqrt(np.mean((phi_pred - phi_true) ** 2)))
    rmsd_late = float(np.sqrt(np.mean(
        (phi_pred[Nt // 2 :] - phi_true[Nt // 2 :]) ** 2)))

    # M3 (ESPRIT at xq=2) on the FNO field
    from src.qnm import qnm_method_3_esprit
    ix = int(np.argmin(np.abs(x - XQ)))
    sig = phi_pred[:, ix]
    res = qnm_method_3_esprit(t, sig, t_start=T_START, t_end=T_END, K=4)
    om = res.get("omega")
    if om is None or not np.isfinite(om):
        om_fno = np.nan
    else:
        # convert angular frequency to M*omega (M=1 here so identity)
        om_fno = float(om) * M_BH
    om_err = 100.0 * abs(om_fno - THEORY_W) / THEORY_W if np.isfinite(om_fno) else np.nan

    return dict(
        Nt=Nt, Nx=Nx, dt=dt, dx=dx,
        rmsd_full=rmsd_full, rmsd_late=rmsd_late,
        omega_M_fno=float(om_fno) if np.isfinite(om_fno) else None,
        omega_pct_err=float(om_err) if np.isfinite(om_err) else None,
        t_fno_s=t_fno, t_fd_s=t_fd,
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model, device = load_v4_model()
    print(f"[setup] device={device}, model loaded")

    rows = []
    for Nt, Nx in GRIDS:
        print(f"[run] Nt={Nt} Nx={Nx} ...")
        r = run_one(Nt, Nx, model, device)
        rows.append(r)
        print(f"    rmsd_late={r['rmsd_late']:.3e}  "
              f"omega_pct_err={r['omega_pct_err']}  "
              f"t_fno={r['t_fno_s']:.2f}s  t_fd={r['t_fd_s']:.2f}s")

    summary = dict(
        config=CFG,
        canonical_BH=dict(M=M_BH, x0=X0, sigma=SIGMA),
        domain=dict(xmin=XMIN, xmax=XMAX, tmin=TMIN, tmax=TMAX),
        training_grid=dict(Nt=401, Nx=401),
        theory_omega_M=THEORY_W,
        rows=rows,
    )
    p = OUT_DIR / "summary.json"
    with open(p, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[json] {p}")

    # Plot: dual y-axis vs Nx, training resolution highlighted
    Nx_arr = np.array([r["Nx"] for r in rows])
    rmsd_arr = np.array([r["rmsd_late"] for r in rows])
    om_arr = np.array([r["omega_pct_err"] if r["omega_pct_err"] is not None else np.nan for r in rows])

    fig, ax1 = plt.subplots(figsize=(7.0, 4.0))
    color1 = "C0"
    ax1.plot(Nx_arr, rmsd_arr, "o-", color=color1, lw=1.6, ms=7,
             label="late-time field RMSD")
    ax1.set_xlabel(r"evaluation grid resolution $N_t = N_x$")
    ax1.set_ylabel(r"$\sqrt{\langle(\Phi_{\rm FNO}-\Phi_{\rm FD})^{2}\rangle}_{t>T/2}$",
                   color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    ax2 = ax1.twinx()
    color2 = "C3"
    ax2.plot(Nx_arr, om_arr, "s--", color=color2, lw=1.4, ms=6,
             label="M3 $\\omega$ \\% err at $x_q=2$")
    ax2.set_ylabel(r"M3 $\omega$ \% error", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_yscale("log")

    # mark training resolution
    ax1.axvline(401, color="grey", ls=":", lw=1.0)
    ax1.text(401, ax1.get_ylim()[1] * 0.6, "training\n$401\\times 401$",
             ha="center", va="top", fontsize=8, color="grey")

    fig.suptitle("FNO v4 zero-shot resolution transfer: trained at $401^{2}$, "
                 "evaluated at $201^{2}$–$801^{2}$  (canonical BH)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p_png = OUT_DIR / "resolution_curves.png"
    fig.savefig(p_png, dpi=150)
    plt.close(fig)
    print(f"[png] {p_png}")


if __name__ == "__main__":
    main()

"""Curve-fitting overlay plot: log|Phi| vs t at the QNM observer, with the
extracted damped-cosine reconstructions (Method 1 FFT+envelope, Method 2
nonlinear curve fit) overlaid on the raw FD and PINN waveforms.

Reproduces the "Curve Fitting" slide from the presentation for any config.

Usage:
    python scripts/plot_curve_fitting.py --config configs/regge_wheeler_l2_paper.yaml
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import load_config
from src.utils import ensure_dir


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _y_at_xq(npz_path, xq):
    d = np.load(npz_path)
    x, t, phi = d["x"], d["t"], d["phi"]
    ix = int(np.argmin(np.abs(x - xq)))
    return t, phi[:, ix]


def _damped_cosine(t, A, tau, omega, phi):
    return A * np.exp(-t / tau) * np.cos(omega * t + phi)


def _fit_amp_phase(t, y, omega, tau):
    """Given fixed omega, tau (e.g. from Method 1), recover (A, phi) by linear
    least squares: A e^{-t/tau} cos(omega t + phi) = e^{-t/tau}(a cos + b sin).
    """
    env = np.exp(-t / tau)
    basis = np.stack([env * np.cos(omega * t), env * np.sin(omega * t)], axis=1)
    coef, *_ = np.linalg.lstsq(basis, y, rcond=None)
    a, b = coef
    A = float(np.hypot(a, b))
    phi = float(np.arctan2(-b, a))
    return A, phi


# Colour scheme reproduced from Patel et al. Figure 6 ("Curve Fitting"):
# TRUE blue solid, FD1 crimson, PINN1 purple, FD2 green, PINN2 orange (all dashed).
PATEL_COLORS = {
    "TRUE": "#4C72B0",
    "FD1": "#A61C3C",
    "PINN1": "#8172B3",
    "FD2": "#55A868",
    "PINN2": "#DD7E1E",
}


def _plot_patel_style(ax, sources, t_start, t_end):
    """Patel Figure 6 reproduction: analytic TRUE waveform (from the theoretical
    QNM) overlaid with the Method 1 and Method 2 damped-cosine reconstructions of
    both the FD and PINN solutions."""
    src = {tag: (t, y, m1, m2) for tag, t, y, m1, m2 in sources}

    # TRUE: analytic damped cosine from the theoretical (omega, tau); amplitude and
    # phase anchored to the FD waveform (falls back to PINN if FD is absent).
    anchor_tag = "FD" if "FD" in src else next(iter(src))
    at, ay, am1, _ = src[anchor_tag]
    m = (at >= t_start) & (at <= t_end)
    att, ayy = at[m], ay[m]
    w_th, tau_th = am1["omega_theory"], am1["tau_theory"]
    A_th, phi_th = _fit_amp_phase(att, ayy, w_th, tau_th)
    true_fit = _damped_cosine(att, A_th, tau_th, w_th, phi_th)
    ax.semilogy(att, np.abs(true_fit), "-", color=PATEL_COLORS["TRUE"], lw=1.8,
                label="TRUE")

    method_style = {
        ("FD", 1): ("FD1", r"$\mathrm{FD}_1$"),
        ("PINN", 1): ("PINN1", r"$\mathrm{PINN}_1$"),
        ("FD", 2): ("FD2", r"$\mathrm{FD}_2$"),
        ("PINN", 2): ("PINN2", r"$\mathrm{PINN}_2$"),
    }
    for (tag, method), (ckey, label) in method_style.items():
        if tag not in src:
            continue
        t, y, m1, m2 = src[tag]
        mask = (t >= t_start) & (t <= t_end)
        tt, yy = t[mask], y[mask]
        if method == 1:
            A1, phi1 = _fit_amp_phase(tt, yy, m1["omega"], m1["tau"])
            fit = _damped_cosine(tt, A1, m1["tau"], m1["omega"], phi1)
        else:
            fit = _damped_cosine(tt, m2["A"], m2["tau"], m2["omega"], m2["phi"])
        ax.semilogy(tt, np.abs(fit), "--", color=PATEL_COLORS[ckey], lw=1.4,
                    label=label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--patel-style", action="store_true",
                    help="Reproduce Patel Fig. 6 colour scheme/overlay (repro runs).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    name = cfg["experiment"]["name"]
    xq = float(cfg["evaluation"]["xq"])
    potential = cfg["physics"]["potential"]
    t_start = float(cfg["qnm"]["t_start"])
    t_end = float(cfg["qnm"]["t_end"])

    pinn_npz = os.path.join("outputs", "pinn", name, f"{name}_pinn.npz")
    fd_npz = os.path.join("outputs", "fd", f"{name}_fd.npz")
    if not os.path.exists(fd_npz):
        fd_npz = os.path.join("outputs", "pinn", name, f"{name}_fd.npz")

    qdir = os.path.join("outputs", "qnm", name)

    sources = []
    for tag, npz_path in (("FD", fd_npz), ("PINN", pinn_npz)):
        if not os.path.exists(npz_path):
            continue
        t, y = _y_at_xq(npz_path, xq)
        m1 = _load_json(os.path.join(qdir, f"{tag.lower()}_method1.json"))
        m2 = _load_json(os.path.join(qdir, f"{tag.lower()}_method2.json"))
        sources.append((tag, t, y, m1, m2))

    if not sources:
        raise SystemExit("No waveform/QNM outputs found; run run_pinn.py and extract_qnm.py first.")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"FD": ("tab:blue", "tab:blue"), "PINN": ("tab:orange", "tab:orange")}

    if args.patel_style:
        _plot_patel_style(ax, sources, t_start, t_end)
    else:
        for tag, t, y, m1, m2 in sources:
            m = (t >= t_start) & (t <= t_end)
            tt, yy = t[m], y[m]
            c_raw, c_fit = colors.get(tag, ("gray", "black"))

            ax.semilogy(tt, np.abs(yy), color=c_raw, lw=1.6, alpha=0.9,
                        label=f"{tag} (raw)")

            # Method 1: omega, tau fixed; recover amplitude/phase for display.
            A1, phi1 = _fit_amp_phase(tt, yy, m1["omega"], m1["tau"])
            fit1 = _damped_cosine(tt, A1, m1["tau"], m1["omega"], phi1)
            ax.semilogy(tt, np.abs(fit1), "--", color=c_fit, lw=1.1, alpha=0.75,
                        label=f"{tag} M1 fit")

            # Method 2: full nonlinear fit (A, tau, omega, phi all in json).
            fit2 = _damped_cosine(tt, m2["A"], m2["tau"], m2["omega"], m2["phi"])
            ax.semilogy(tt, np.abs(fit2), ":", color=c_fit, lw=1.8,
                        label=f"{tag} M2 fit")

    ax.set_xlabel(r"$t/M$")
    ax.set_ylabel(r"$\log|\Phi(x_q, t)|$" if args.patel_style
                  else r"$|\Phi(x_q, t)|$")
    pot_label = potential.replace("-", " ").replace("_", " ").title()
    ax.set_title(f"Curve Fitting — {pot_label} $\\ell=2$  ($x_q={xq:g}\\,M$)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()

    ensure_dir(qdir)
    out1 = os.path.join(qdir, "curve_fitting.png")
    out2 = os.path.join("outputs", "pinn", name, "curve_fitting.png")
    fig.savefig(out1, dpi=150)
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    print(f"[curve_fitting] wrote {out1}")
    print(f"[curve_fitting] wrote {out2}")


if __name__ == "__main__":
    main()

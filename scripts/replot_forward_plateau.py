"""Re-plot the forward-PINN M4/M5 plateau diagnostics from saved scan JSON.

The original PNGs were produced by an older pipeline whose titles carried the
internal run name (e.g. ``zerilli_l2_greedy_f03_lbfgs30k``) plus debug text.
The scan arrays are fully preserved in
``outputs/qnm/<run>/pinn_method{4,5}_*.json``, so we redraw both figures with
neutral titles and the same visual convention as the hybrid plateau figures
(``scripts/make_hybrid_figs_xq10.py``) for cross-model consistency. No field
data or model is required.

Usage:
    python scripts/replot_forward_plateau.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
QDIR = ROOT / "outputs" / "qnm" / "zerilli_l2_greedy_f03_lbfgs30k"
XQ = 10.0


def replot_m4() -> None:
    m4 = json.loads((QDIR / "pinn_method4_two_mode.json").read_text())
    ts = np.asarray(m4["t_starts"]); os_ = np.asarray(m4["omegas"])
    tas = np.asarray(m4["taus"]); pidx = m4.get("plateau_idx") or []
    tw, tt = m4["omega_theory"], m4["tau_theory"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(ts, os_, "o-", label="per-window fit")
    if pidx:
        axes[0].plot(ts[pidx], os_[pidx], "o", color="C3", label="plateau")
        axes[0].axhline(m4["omega"], color="C3", ls="--", lw=0.8)
    axes[0].axhline(tw, color="k", ls=":", lw=0.8, label="theory")
    axes[0].set_xlabel("start time t0"); axes[0].set_ylabel(r"$\omega M$")
    axes[0].legend(loc="best", fontsize=8)
    axes[1].plot(ts, tas, "o-")
    if pidx:
        axes[1].plot(ts[pidx], tas[pidx], "o", color="C3")
        axes[1].axhline(m4["tau"], color="C3", ls="--", lw=0.8)
    axes[1].axhline(tt, color="k", ls=":", lw=0.8)
    axes[1].set_xlabel("start time t0"); axes[1].set_ylabel(r"$\tau / M$")
    fig.suptitle(rf"Method 4 stability scan ($x_q = {XQ:g}\,M$)")
    fig.tight_layout()
    out = QDIR / "pinn_method4_stability.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"[fig] {out}")


def replot_m5() -> None:
    m5 = json.loads((QDIR / "pinn_method5_2d_scan.json").read_text())
    ts = np.asarray(m5["t_starts"]); tes = np.asarray(m5["t_ends"])
    og = np.asarray(m5["omegas_grid"]); tg = np.asarray(m5["taus_grid"])
    extent = [ts[0], ts[-1], tes[0], tes[-1]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    im0 = axes[0].imshow(og, origin="lower", aspect="auto", extent=extent,
                         cmap="viridis", vmin=0.30, vmax=0.50)
    axes[0].set_xlabel(r"$t_0 / M$"); axes[0].set_ylabel(r"$t_{\rm end} / M$")
    axes[0].set_title(rf"$\omega M$  (theory$={m5['omega_theory']}$)")
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(tg, origin="lower", aspect="auto", extent=extent,
                         cmap="viridis", vmin=5.0, vmax=20.0)
    axes[1].set_xlabel(r"$t_0 / M$"); axes[1].set_ylabel(r"$t_{\rm end} / M$")
    axes[1].set_title(rf"$\tau / M$  (theory$={m5['tau_theory']}$)")
    plt.colorbar(im1, ax=axes[1])
    t0_lo, t0_hi = m5["t0_plateau_min"], m5["t0_plateau_max"]
    te_lo, te_hi = m5["te_plateau_min"], m5["te_plateau_max"]
    for ax in axes:
        ax.add_patch(plt.Rectangle(
            (t0_lo, te_lo), t0_hi - t0_lo, te_hi - te_lo,
            fill=False, edgecolor="red", lw=1.5,
        ))
    fig.suptitle(rf"Method 5 2-D stability scan ($x_q = {XQ:g}\,M$)")
    fig.tight_layout()
    out = QDIR / "pinn_method5_2d_heatmap.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"[fig] {out}")


if __name__ == "__main__":
    replot_m4()
    replot_m5()

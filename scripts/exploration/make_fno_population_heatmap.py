"""2D parameter-space error heatmap for the 100-BH FNO test population.

Plots per-BH FNO field RMSD as colored scatter in the three pairwise
projections of the (M, x0, sigma) initial-data manifold, so the reader
can see *where* in IC space the surrogate degrades. Zero compute: reuses
cached predictions in outputs/fno/fno_zerilli_l2_v4/predictions/test/.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "outputs/fno/fno_zerilli_l2_v4/predictions/test"
OUT_DIR = ROOT / "outputs/qnm/fno_xq2_comparison/figs_v4"


def collect():
    rows = []
    for fn in sorted(os.listdir(PRED_DIR)):
        if not fn.endswith(".npz"):
            continue
        z = np.load(PRED_DIR / fn)
        pp = z["phi_pred"]
        pt = z["phi_true"]
        rmsd = float(np.sqrt(np.mean((pp - pt) ** 2)))
        # late-time-only RMSD: t in last half (ringdown-dominated window)
        Nt = pp.shape[0]
        rmsd_late = float(np.sqrt(np.mean((pp[Nt // 2 :] - pt[Nt // 2 :]) ** 2)))
        rows.append(dict(
            M=float(z["M"]),
            x0=float(z["x0"]),
            sigma=float(z["sigma"]),
            rmsd=rmsd,
            rmsd_late=rmsd_late,
        ))
    return rows


def make_figure(rows):
    M = np.array([r["M"] for r in rows])
    x0 = np.array([r["x0"] for r in rows])
    sg = np.array([r["sigma"] for r in rows])
    err = np.array([r["rmsd_late"] for r in rows])

    vmin = max(err.min(), 1e-6)
    vmax = err.max()
    norm = LogNorm(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    panels = [
        (axes[0], M, x0, r"$M$", r"$x_{0}/M$"),
        (axes[1], M, sg, r"$M$", r"$\sigma/M$"),
        (axes[2], x0, sg, r"$x_{0}/M$", r"$\sigma/M$"),
    ]
    for ax, xs, ys, xl, yl in panels:
        sc = ax.scatter(xs, ys, c=err, cmap="viridis", norm=norm,
                        s=55, edgecolor="k", linewidth=0.4)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(True, alpha=0.3)

    cax = fig.add_axes((0.92, 0.18, 0.012, 0.7))
    cb = fig.colorbar(sc, cax=cax)
    cb.set_label(r"FNO late-time field RMSD "
                 r"$\sqrt{\langle(\Phi_{\rm FNO}-\Phi_{\rm FD})^{2}\rangle}_{t > T/2}$",
                 fontsize=9)
    fig.suptitle(
        f"FNO v4 error across the $100$-BH test population, projected onto each "
        f"pair of initial-data parameters  "
        f"(median$={np.median(err):.2e}$, $95$th pct$={np.percentile(err, 95):.2e}$)",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 0.91, 0.94))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / "fno_v4_population_heatmap.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[fig] {p}")
    print(f"[stats] field RMSD over 100 BHs: "
          f"min={err.min():.2e}  median={np.median(err):.2e}  "
          f"mean={err.mean():.2e}  max={err.max():.2e}")


if __name__ == "__main__":
    rows = collect()
    print(f"[load] {len(rows)} BH predictions")
    make_figure(rows)

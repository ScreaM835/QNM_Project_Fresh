"""Re-render the uniform-baseline error heatmap (from the sibling repo
project32_qnm_pinn/outputs/pinn/zerilli_l2_paper) AND the greedy run
(this repo, zerilli_l2_greedy_f03_lbfgs30k) using a SHARED log colour
scale tuned to the greedy run, so the slide
"Greedy resampling --- error map: uniform vs. greedy" shows two genuinely
comparable images side by side.

Greedy result is also written to ``error_heatmap_compare.png`` so the
existing wider-scale ``error_heatmap.png`` (used elsewhere) is not
overwritten.
"""
from __future__ import annotations
import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIB  = os.path.normpath(os.path.join(ROOT, "..", "project32_qnm_pinn"))

def load(d):
    p = glob.glob(os.path.join(d, "*pinn.npz"))
    f = glob.glob(os.path.join(d, "*fd.npz"))
    pp = np.load(p[0]); ff = np.load(f[0])
    return pp["x"], pp["t"], ff["phi"], pp["phi"]

# Greedy run (improved repo) --- shared log-scale max
gx, gt, gfd, gpinn = load(os.path.join(ROOT, "outputs/pinn/zerilli_l2_greedy_f03_lbfgs30k"))
greedy_err = np.abs(gfd - gpinn)
abs_max = float(greedy_err.max())
vmin = max(abs_max * 1e-4, 1e-6)
print(f"shared abs_max = {abs_max:.4e}, vmin = {vmin:.4e}")


def render(x, t, err, title, outpath, zoom=False):
    err_clip = np.maximum(err, vmin)
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.pcolormesh(x, t, err_clip, shading="auto", cmap="magma_r",
                       norm=LogNorm(vmin=vmin, vmax=abs_max))
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r"$|\Phi_{\mathrm{FD}} - \Phi_{\mathrm{PINN}}|$"
                   f"  (shared log scale, max={abs_max:.2e})")
    ax.set_xlabel(r"$x_* / M$"); ax.set_ylabel("t / M")
    ax.set_title(title)
    if zoom:
        ax.set_xlim(-20.0, 60.0)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200); plt.close(fig)
    print(f"  wrote {outpath}")


# Baseline run (sibling repo)
bdir = os.path.join(SIB, "outputs/pinn/zerilli_l2_paper")
bx, bt, bfd, bpinn = load(bdir)
berr = np.abs(bfd - bpinn)
render(bx, bt, berr, "Pointwise error  (uniform baseline, zerilli_l2_paper)",
       os.path.join(bdir, "error_heatmap.png"))
render(bx, bt, berr, "Pointwise error  (uniform baseline)  --  zoom $x_*/M\\in[-20,60]$",
       os.path.join(bdir, "error_heatmap_zoomed.png"), zoom=True)

# Greedy run, written to a separate "compare" filename so we don't clobber
# the wider-scale plot used by other slides.
gdir = os.path.join(ROOT, "outputs/pinn/zerilli_l2_greedy_f03_lbfgs30k")
render(gx, gt, greedy_err, "Pointwise error  (greedy $f=0.3$ + L-BFGS 30k)",
       os.path.join(gdir, "error_heatmap_compare.png"))


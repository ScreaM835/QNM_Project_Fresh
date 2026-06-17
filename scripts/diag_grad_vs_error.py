#!/usr/bin/env python
# ---------------------------------------------------------------------------
# diag_grad_vs_error.py  --  VERIFY the grad-channel hypothesis.
#
# Hypothesis (to be tested, NOT assumed): the regions of high pointwise error
# coincide with regions of high |grad(prior)|. If true, |grad(prior)| is a
# legitimate "where the prior fails" map and is worth feeding to the FNO as an
# input channel. If false, the grad channel is useless and we stop.
#
# This renders three (x,t) heatmaps on the IDENTICAL colour scale used by the
# paper pointwise-error figures (magma_r + LogNorm, vmax=1.11e-1, vmin=1.11e-5)
# so the spatial patterns can be compared directly by eye:
#     (1) baseline pointwise error   |psi_fine - psi_coarse_up|
#     (2) |grad(prior)|              the actual FNO feature (unit-spacing grad
#                                    of channel 0, == src/hybrid_fno wrapper)
#     (3) hybrid pointwise error     |psi_fine - psi_hybrid|
# plus QUANTITATIVE checks (log-log correlation; per-decile cross-tabulation).
#
# Login-safe (venv_csd3, numpy+matplotlib only; reads an existing canonical.npz
# cache, no FD solve, no model). Run:
#   venv_csd3/bin/python scripts/diag_grad_vs_error.py \
#       --run outputs/hybrid/fno_sw_obsloss_derisk
# ---------------------------------------------------------------------------
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Shared colour-scale convention copied verbatim from make_hybrid_paper_figs.py
VMAX = 1.11e-1
VMIN = VMAX * 1e-4   # = 1.11e-5


def grad_mag(prior: np.ndarray) -> np.ndarray:
    """|grad(prior)| with UNIT spacing -- identical to the _PriorFeatureFNO
    wrapper feature (torch.gradient over dims t,x with default spacing=1)."""
    g_t = np.gradient(prior, axis=0)   # d/dt  (axis 0 = t)
    g_x = np.gradient(prior, axis=1)   # d/dx  (axis 1 = x)
    return np.sqrt(g_t * g_t + g_x * g_x + 1e-24)


def _panel(ax, x, t, field, title, cmap="magma_r"):
    fc = np.maximum(field, VMIN)
    im = ax.pcolormesh(x, t, fc, shading="auto", cmap=cmap,
                       norm=LogNorm(vmin=VMIN, vmax=VMAX))
    ax.set_xlabel(r"$x_* / M$")
    ax.set_title(title, fontsize=10)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/hybrid/fno_sw_obsloss_derisk",
                    help="run dir containing field_cache/canonical.npz")
    args = ap.parse_args()

    cache = os.path.join(args.run, "field_cache", "canonical.npz")
    z = np.load(cache)
    x = z["x"]; t = z["t"]
    fine = z["psi_fine"]
    prior = z["psi_coarse_up"]
    hyb = z["psi_hybrid"]

    err_base = np.abs(fine - prior)     # what the grad map is supposed to predict
    err_hyb = np.abs(fine - hyb)
    g = grad_mag(prior)                 # the FNO feature

    # ---- figure: 3 panels, identical scale, one shared colourbar -----------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    _panel(axes[0], x, t, err_base,
           r"(1) baseline error  $|\Phi_{\rm fine}-\Phi_{\rm coarse\text{-}up}|$")
    im = _panel(axes[1], x, t, g,
                r"(2) FNO feature  $|\nabla\,\Phi_{\rm prior}|$  (unit spacing)")
    _panel(axes[2], x, t, err_hyb,
           r"(3) hybrid error  $|\Phi_{\rm fine}-\Phi_{\rm hybrid}|$")
    axes[0].set_ylabel("t / M")
    cbar = fig.colorbar(im, ax=axes, pad=0.015, fraction=0.025)
    cbar.set_label(f"shared log scale  [{VMIN:.2e}, {VMAX:.2e}]")
    fig.suptitle(
        "grad-channel hypothesis check: does high pointwise error coincide "
        "with high |grad(prior)|?  (canonical BH)", fontsize=12)
    out = os.path.join(args.run, "figs", "grad_vs_error.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {out}")

    # ---- quantitative verification ----------------------------------------
    # (a) log-log correlation of grad vs baseline error over the whole domain
    lg = np.log10(np.maximum(g, 1e-30)).ravel()
    le = np.log10(np.maximum(err_base, 1e-30)).ravel()
    r = float(np.corrcoef(lg, le)[0, 1])
    print("\n========== HYPOTHESIS: high baseline error <-> high |grad prior| ==========")
    print(f"  Pearson r( log|grad prior| , log|baseline err| ) = {r:+.3f}")

    # (b) per-|grad| decile: mean baseline error (should rise monotonically)
    gflat = g.ravel(); ebflat = err_base.ravel(); ehflat = err_hyb.ravel()
    order = np.argsort(gflat)
    q = np.linspace(0, gflat.size, 11).astype(int)
    print("\n  decile of |grad prior|   <|grad|>      <base err>     <hybrid err>")
    for d in range(10):
        idx = order[q[d]:q[d + 1]]
        print(f"   {10*d:3d}-{10*(d+1):3d}%            "
              f"{gflat[idx].mean():.3e}   {ebflat[idx].mean():.3e}   "
              f"{ehflat[idx].mean():.3e}")

    # (c) cross-tab: of the worst-1% baseline-error points, what frac sit in the
    #     top-decile of |grad|?  (and where does the hybrid speckle live?)
    n = gflat.size
    top_err = set(np.argsort(ebflat)[-n // 100:])          # worst 1% baseline err
    top_grad = set(order[-n // 10:])                       # top 10% grad
    hit = len(top_err & top_grad) / len(top_err)
    print(f"\n  of worst-1% BASELINE-error points, {100*hit:.1f}% are in the "
          f"top-10% |grad| (prior fails where grad is high?)")
    # where does the hybrid error live by grad decile (the speckle question)
    bot_grad = set(order[:n // 2])                         # smoothest 50%
    worst_hyb = set(np.argsort(ehflat)[-n // 100:])        # worst 1% hybrid err
    sphit = len(worst_hyb & bot_grad) / len(worst_hyb)
    print(f"  of worst-1% HYBRID-error points,   {100*sphit:.1f}% are in the "
          f"smoothest-50% |grad| (= speckle on good prior, the defect)")


if __name__ == "__main__":
    main()

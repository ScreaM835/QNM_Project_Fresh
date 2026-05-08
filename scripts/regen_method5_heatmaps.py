"""Regenerate Method-5 2-D heatmaps for all variants from saved
*_method5_2d_scan.json, fixing three readability problems of the original
plot (linear viridis dominated by outliers, NaNs invisible, truth value
buried in the title):

  - plot signed % error from theory on a SymLogNorm so the plateau (~0%)
    is unambiguous and corner outliers do not compress the dynamic range
  - explicit hatched/grey overlay for NaN cells
  - colorbar centred on truth (zero error)
  - keep the red plateau rectangle exactly as in the original

Pure post-processing: reads JSON only, no retraining or re-extraction.
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

# Cap the colour scale at this % deviation. Anything worse saturates to
# yellow. Chosen so that a genuine plateau (sub-%) appears uniformly dark
# and a non-plateau region (10s of %) saturates and looks obviously bad.
VMAX_PCT = 5.0

ROOT = os.path.join(os.path.dirname(__file__), "..", "outputs", "qnm")
PATTERN = os.path.join(ROOT, "*", "*_method5_2d_scan.json")


def render(json_path):
    with open(json_path) as f:
        d = json.load(f)

    ts = np.asarray(d["t_starts"])
    tes = np.asarray(d["t_ends"])
    og = np.asarray(d["omegas_grid"], dtype=float)
    tg = np.asarray(d["taus_grid"], dtype=float)
    o_true = float(d["omega_theory"])
    t_true = float(d["tau_theory"])

    # Reference value for the colour scale: the median of the plateau
    # block. This makes the heatmap answer the question "is this region
    # flat?" rather than "is this region close to truth?". A genuine
    # plateau then appears uniformly dark regardless of any systematic
    # bias from truth, and that bias is annotated separately.
    if d["plateau_t0_idx"] and d["plateau_te_idx"]:
        i_lo = d["plateau_t0_idx"][0]
        i_hi = d["plateau_t0_idx"][-1] + 1
        j_lo = d["plateau_te_idx"][0]
        j_hi = d["plateau_te_idx"][-1] + 1
        o_blk = og[j_lo:j_hi, i_lo:i_hi]
        t_blk = tg[j_lo:j_hi, i_lo:i_hi]
        o_ref = float(np.nanmedian(o_blk)) if np.isfinite(o_blk).any() else o_true
        t_ref = float(np.nanmedian(t_blk)) if np.isfinite(t_blk).any() else t_true
    else:
        o_ref, t_ref = o_true, t_true

    # % deviation from the plateau median (flatness diagnostic)
    o_err = (og - o_ref) / o_ref * 100.0
    t_err = (tg - t_ref) / t_ref * 100.0

    # extent for imshow with origin="lower":
    # cell-centred -> shift by half a step so axes align with cell centres
    dt0 = ts[1] - ts[0]
    dte = tes[1] - tes[0]
    extent = [ts[0] - dt0 / 2, ts[-1] + dt0 / 2,
              tes[0] - dte / 2, tes[-1] + dte / 2]

    # Use absolute % error (low=good=dark, high=bad=yellow) so the viridis
    # sequential colour scheme (dark blue -> yellow) is meaningful.
    o_abs = np.abs(o_err)
    t_abs = np.abs(t_err)

    fig, axes = plt.subplots(2, 1, figsize=(7, 9))

    # Linear scale, shared, hard-capped: a real plateau (<<1%) appears
    # uniformly dark; non-plateau regions saturate to yellow.
    vmax = VMAX_PCT

    for ax, data, title, label, ref, truth in (
        (axes[0], o_abs,
         rf"$|\omega - \omega_{{\rm plat}}|/\omega_{{\rm plat}}$  "
         rf"(plateau median $M\omega={o_ref:.4f}$, truth ${o_true}$)",
         "omega", o_ref, o_true),
        (axes[1], t_abs,
         rf"$|\tau - \tau_{{\rm plat}}|/\tau_{{\rm plat}}$  "
         rf"(plateau median $\tau/M={t_ref:.3f}$, truth ${t_true}$)",
         "tau", t_ref, t_true),
    ):
        finite = np.isfinite(data)

        # Linear scale capped at vmax %. Everything <= vmax shows fine
        # gradient; everything > vmax saturates to yellow ("extend='max'").
        plot_data = np.where(finite, data, np.nan)
        if not finite.all():
            from scipy.ndimage import distance_transform_edt
            idx = distance_transform_edt(~finite, return_distances=False,
                                         return_indices=True)
            plot_data = np.where(finite, plot_data, data[tuple(idx)])

        # Bicubic-upsample to a fine mesh so the heatmap is smooth rather
        # than a 10x6 grid of polygons.
        from scipy.interpolate import RectBivariateSpline
        nx_fine, ny_fine = 200, 200
        ts_fine = np.linspace(ts[0], ts[-1], nx_fine)
        tes_fine = np.linspace(tes[0], tes[-1], ny_fine)
        spline = RectBivariateSpline(tes, ts, plot_data,
                                     kx=min(3, len(tes) - 1),
                                     ky=min(3, len(ts) - 1))
        plot_fine = spline(tes_fine, ts_fine)
        plot_fine = np.clip(plot_fine, 0.0, vmax)
        T0F, TEF = np.meshgrid(ts_fine, tes_fine)

        norm = Normalize(vmin=0.0, vmax=vmax)
        levels = np.linspace(0.0, vmax, 80)
        im = ax.contourf(T0F, TEF, plot_fine, levels=levels,
                         cmap="viridis", norm=norm, extend="max")

        # NaN overlay: grey patches at the actual NaN cell footprints
        nan_mask = ~finite
        if nan_mask.any():
            grey = np.where(nan_mask, 1.0, np.nan)
            ax.imshow(grey, origin="lower", aspect="auto", extent=extent,
                      cmap=plt.matplotlib.colors.ListedColormap(["#888888"]),
                      vmin=0, vmax=1, interpolation="nearest", zorder=2)
        ax.set_xlim(ts[0], ts[-1])
        ax.set_ylim(tes[0], tes[-1])

        ax.set_xlabel(r"$t_0\ /\ M$")
        ax.set_ylabel(r"$t_{\mathrm{end}}\ /\ M$")
        ax.set_title(title)
        cb = plt.colorbar(im, ax=ax, extend="max")
        cb.set_label(rf"absolute % deviation from plateau median (capped at {vmax:.0f}%)")

        # plateau rectangle (always drawn from JSON metadata)
        t0_lo, t0_hi = d["t0_plateau_min"], d["t0_plateau_max"]
        te_lo, te_hi = d["te_plateau_min"], d["te_plateau_max"]
        # rectangle drawn on cell centres (matches contourf footprint)
        ax.add_patch(plt.Rectangle(
            (t0_lo, te_lo),
            (t0_hi - t0_lo),
            (te_hi - te_lo),
            fill=False, edgecolor="red", lw=2.5,
            label="plateau",
            zorder=5,
        ))

        # In-block relative scatter: this is the actual quantity the
        # plateau rule minimises. Annotated so the reader can see that the
        # plateau is about flatness, not closeness to truth.
        if d["plateau_t0_idx"] and d["plateau_te_idx"]:
            i_lo = d["plateau_t0_idx"][0]
            i_hi = d["plateau_t0_idx"][-1] + 1
            j_lo = d["plateau_te_idx"][0]
            j_hi = d["plateau_te_idx"][-1] + 1
            if label == "omega":
                blk = np.asarray(d["omegas_grid"])[j_lo:j_hi, i_lo:i_hi]
            else:
                blk = np.asarray(d["taus_grid"])[j_lo:j_hi, i_lo:i_hi]
            if np.all(np.isfinite(blk)) and abs(np.mean(blk)) > 0:
                rel = np.std(blk) / abs(np.mean(blk)) * 100.0
                bias = (np.mean(blk) - truth) / truth * 100.0
                ax.text(0.02, 0.02,
                        f"plateau std/mean = {rel:.2f}%\n"
                        f"bias from truth   = {bias:+.2f}%",
                        transform=ax.transAxes, fontsize=9,
                        color="red", family="monospace",
                        bbox=dict(facecolor="white", edgecolor="red",
                                  alpha=0.85, pad=2))

    # NaN legend if any NaN cells anywhere
    if (~np.isfinite(og)).any() or (~np.isfinite(tg)).any():
        from matplotlib.patches import Patch
        axes[0].legend(handles=[Patch(facecolor="#888888",
                                      label="NaN (fit failed)"),
                                plt.Rectangle((0, 0), 1, 1, fill=False,
                                              edgecolor="red", lw=2.5,
                                              label="plateau (min joint scatter)")],
                       loc="upper right", fontsize=8, framealpha=0.9)
    else:
        axes[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, fill=False,
                                              edgecolor="red", lw=2.5,
                                              label="plateau (min joint scatter)")],
                       loc="upper right", fontsize=8, framealpha=0.9)

    tag = os.path.basename(os.path.dirname(json_path))
    fig.suptitle(f"Method 5 2-D plateau scan ({tag})", fontsize=11)
    fig.tight_layout()

    out = json_path.replace("_method5_2d_scan.json",
                            "_method5_2d_heatmap.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    paths = sorted(glob.glob(PATTERN))
    if not paths:
        print(f"no scan json found under {PATTERN}")
        return
    for p in paths:
        out = render(p)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

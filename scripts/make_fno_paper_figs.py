"""Generate the FNO figures for the paper.

Outputs go to ``outputs/qnm/fno_xq2_comparison/figs_v4/``.

Stages:
    A) field cache   — load model + run FNO forward + FD GT for v4 (and v2/v3
                       if requested), save (phi_pred, phi_true) as .npz.
                       Skipped if cache exists.
    B) plots         — read cache + summary JSONs and write the six PNGs:
                       1. snapshots + residual (v4)
                       2. xq sweep (v4 FNO vs GT, all four methods)
                       3. ringdown overlay at xq=2 (v4)
                       4. 100-BH population scatter (v4 FNO/m4 vs GT/m4)
                       5. v2/v3/v4 ablation (field RMSD + m4 %err)
                       6. M4 1-D plateau diagnostic at xq=2 (v4 FNO)

Usage:
    python scripts/make_fno_paper_figs.py
    python scripts/make_fno_paper_figs.py --versions v2 v3 v4    # ablation needs all three
    python scripts/make_fno_paper_figs.py --recompute             # force re-cache
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.fno_dataset import load_dataset
from src.fno_model import build_model
from src.fd_solver import solve_fd
from src.qnm import (
    qnm_method_4_window_scan,
    qnm_method_5_2d_scan,
    percentage_errors,
)
from src import plotting as pinn_plot
from scripts.fno_fixed_bh_qnm import build_input

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "qnm" / "fno_xq2_comparison" / "figs_v4"
CACHE_DIR = ROOT / "outputs" / "qnm" / "fno_xq2_comparison" / "field_cache"
CONFIGS = {
    "v2": "configs/fno_zerilli_l2_v2.yaml",
    "v3": "configs/fno_zerilli_l2_v3.yaml",
    "v4": "configs/fno_zerilli_l2_v4.yaml",
}
M_CANON, X0_CANON, SIGMA_CANON = 1.0, 4.0, 5.0
THEORY_W = 0.3737
THEORY_T = 11.241


# --------------------------------------------------------------------------
# Stage A: field cache
# --------------------------------------------------------------------------
def _solve_one(cfg_path: str):
    """Return (x, t, phi_pred, phi_true, field_rmsd) for the canonical BH."""
    cfg = load_config(cfg_path)
    name = cfg["experiment"]["name"]
    out_dir = cfg["training"]["out_dir"]
    ckpt = os.path.join(out_dir, "model.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"No checkpoint at {ckpt}")

    splits, grid, meta = load_dataset(cfg["training"]["data_path"])
    x = grid.x.astype(np.float64)
    t = grid.t.astype(np.float64)
    print(f"[{name}] grid Nx={x.size} Nt={t.size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg,
                        t_grid=torch.from_numpy(grid.t).to(device),
                        x_grid=torch.from_numpy(grid.x).to(device)).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()

    chans, _ = build_input(x, t, M_CANON, X0_CANON, SIGMA_CANON, ell=2)
    X = torch.from_numpy(chans[None]).to(device)
    print(f"[{name}] running FNO forward on {device} ...")
    with torch.no_grad():
        phi_pred = model(X).cpu().numpy()[0, 0]
    print(f"[{name}] FNO done  |phi|.max={np.abs(phi_pred).max():.3e}")

    fd_cfg = {
        "physics": {"M": M_CANON, "potential": "zerilli", "l": 2},
        "initial_data": {"A": 1.0, "x0": X0_CANON, "sigma": SIGMA_CANON,
                         "velocity_profile": "outgoing"},
        "domain": {"xmin": float(x.min()), "xmax": float(x.max()),
                   "tmin": 0.0, "tmax": float(t.max())},
        "fd": {"dx": float(meta["fd_dx"]), "dt": float(meta["fd_dt"])},
    }
    sol = solve_fd(fd_cfg)
    ph = sol["phi"]; xs = sol["x"]; ts = sol["t"]
    if ph.shape != phi_pred.shape:
        from scipy.interpolate import RegularGridInterpolator
        f = RegularGridInterpolator((ts, xs), ph, bounds_error=False,
                                    fill_value=0.0)
        tt, xx = np.meshgrid(t, x, indexing="ij")
        phi_true = f((tt, xx))
    else:
        phi_true = ph
    field_rmsd = float(np.sqrt(np.mean((phi_pred - phi_true) ** 2)))
    print(f"[{name}] FD done  field_rmsd={field_rmsd:.3e}")
    return x, t, phi_pred, phi_true, field_rmsd


def ensure_cache(version: str, recompute: bool = False):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = CACHE_DIR / f"{version}_canonical.npz"
    if p.exists() and not recompute:
        print(f"[cache] {p} exists, skipping recompute")
        return p
    cfg_path = CONFIGS[version]
    x, t, phi_pred, phi_true, field_rmsd = _solve_one(cfg_path)
    np.savez(p, x=x, t=t, phi_pred=phi_pred, phi_true=phi_true,
             field_rmsd=field_rmsd)
    print(f"[cache] wrote {p}  field_rmsd={field_rmsd:.3e}")
    return p


# --------------------------------------------------------------------------
# Stage B helpers
# --------------------------------------------------------------------------
def load_cache(version: str):
    p = CACHE_DIR / f"{version}_canonical.npz"
    z = np.load(p)
    return dict(x=z["x"], t=z["t"], phi_pred=z["phi_pred"],
                phi_true=z["phi_true"], field_rmsd=float(z["field_rmsd"]))


# --------------------------------------------------------------------------
# Figure 1a: snapshots overlay (FD vs FNO at four times) - split from residual
# --------------------------------------------------------------------------
def fig_snapshots(version="v4"):
    """FNO/FD snapshot overlay at four times; 2x2 grid matching the
    forward-PINN snapshots.png (Fig.~fwd-field upper panel)."""
    c = load_cache(version)
    x, t, pp, pt = c["x"], c["t"], c["phi_pred"], c["phi_true"]
    times = [10.0, 30.0, 50.0, 100.0]
    times = [tj for tj in times if tj <= t.max() + 1e-6]
    out = OUT_DIR / f"fno_{version}_snapshots.png"
    pinn_plot.plot_snapshots(
        x=x, t=t,
        phi_fd=pt, phi_pinn=pp,
        times=times,
        outpath=str(out),
        model_label=f"FNO {version}",
        title=(f"FNO {version} vs FD reference, canonical "
               f"$M=1,\\,x_0=4,\\,\\sigma=5$ "
               f"(field RMSD $= {c['field_rmsd']:.2e}$)"),
    )
    print(f"[fig] {out}")


# --------------------------------------------------------------------------
# Figure 1a-companion: pointwise |FNO - FD| at the same snapshot times
# --------------------------------------------------------------------------
def fig_abs_diff_snapshots(version="v4"):
    """Pointwise |FNO - FD| at the four snapshot times. 2x2 grid,
    linear y-axis -- matches src/plotting.py::plot_abs_diff_snapshots used
    for all PINN runs (Fig.~fwd-field lower panel)."""
    c = load_cache(version)
    x, t, pp, pt = c["x"], c["t"], c["phi_pred"], c["phi_true"]
    times = [10.0, 30.0, 50.0, 100.0]
    times = [tj for tj in times if tj <= t.max() + 1e-6]
    out = OUT_DIR / f"fno_{version}_abs_diff_snapshots.png"
    pinn_plot.plot_abs_diff_snapshots(
        x=x, t=t,
        phi_fd=pt, phi_pinn=pp,
        times=times,
        outpath=str(out),
        title=(f"|FNO {version} - FD| at the snapshot times, canonical "
               f"$M=1,\\,x_0=4,\\,\\sigma=5$"),
    )
    print(f"[fig] {out}")


# --------------------------------------------------------------------------
# Figure 1b: global pointwise error heatmap on (x, t)
# --------------------------------------------------------------------------
def fig_pointwise_error(version="v4"):
    """Pointwise-error heatmap. Uses the same colour-scale convention as
    the PINN figures (magma_r + LogNorm, vmax = global PINN abs_max
    ~1.11e-1, vmin = vmax/1e4) so the two are directly comparable in
    magnitude, but the time axis spans the FNO's own training window
    t in [0, 100] M (the PINN figures stop at 50 M because that is the
    PINN's training window)."""
    from matplotlib.colors import LogNorm
    c = load_cache(version)
    x, t, pp, pt = c["x"], c["t"], c["phi_pred"], c["phi_true"]
    t_c = t
    err = np.abs(pt - pp)
    # Shared colour scale: vmax = 1.11e-1 is the global PINN abs-error max
    # printed by replot_consistent_scales.py; same vmin floor of vmax/1e4.
    vmax = 1.11e-1
    vmin = vmax * 1e-4
    err_clip = np.maximum(err, vmin)
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.pcolormesh(x, t_c, err_clip, shading="auto", cmap="magma_r",
                       norm=LogNorm(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r"$|\Phi_{\mathrm{FD}} - \Phi_{\mathrm{FNO}}|$"
                   f"  (shared log scale, max={vmax:.2e})")
    ax.set_xlabel(r"$x_* / M$"); ax.set_ylabel("t / M")
    ax.set_title(f"Pointwise error  (fno_zerilli_l2_{version})")
    fig.tight_layout()
    out = OUT_DIR / f"fno_{version}_pointwise_error.png"
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"[fig] {out}")


# --------------------------------------------------------------------------
# Figure 2: xq sweep — FNO vs GT across xq for v4 (all four methods)
# --------------------------------------------------------------------------
def fig_xq_sweep():
    p = ROOT / "outputs/qnm/fno_xq2_comparison/fixed_BH_sweep.json"
    d = json.load(open(p))
    v4 = d["runs"]["fno_zerilli_l2_v4"]["xq_sweep"]
    xqs = sorted([float(k.split("_")[1]) for k in v4.keys()])
    methods = ["m1", "m2", "m3", "m4"]
    method_labels = ["M1 FFT+log", "M2 1-mode NLS", "M3 ESPRIT",
                     "M4 2-mode plateau"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    width = 0.10
    colors_fno = ["#1f77b4", "#1f77b4", "#1f77b4", "#1f77b4"]
    colors_gt = ["#d62728", "#d62728", "#d62728", "#d62728"]
    hatches = ["", "//", "..", "xx"]

    for ax, metric, label in [(axes[0], "omega_pct_err", r"$\omega$ % err"),
                              (axes[1], "tau_pct_err", r"$\tau$ % err")]:
        xpos = np.arange(len(xqs))
        for k, m in enumerate(methods):
            fno_vals = []
            gt_vals = []
            for xq in xqs:
                key = f"xq_{xq}"
                r = v4[key]
                fv = r["FNO"][m].get(metric)
                gv = r["GT"][m].get(metric)
                fno_vals.append(np.nan if fv is None else fv)
                gt_vals.append(np.nan if gv is None else gv)
            off_f = (k - 1.5) * 2 * width
            off_g = off_f + width
            bars_f = ax.bar(xpos + off_f, fno_vals, width,
                            color=colors_fno[k], hatch=hatches[k],
                            edgecolor="white", linewidth=0.5,
                            label=f"FNO {method_labels[k]}" if metric == "omega_pct_err" else None)
            bars_g = ax.bar(xpos + off_g, gt_vals, width,
                            color=colors_gt[k], hatch=hatches[k],
                            edgecolor="white", linewidth=0.5,
                            label=f"GT  {method_labels[k]}" if metric == "omega_pct_err" else None)
            # annotate "no plateau" where m4 is NaN (single marker per xq)
            if m == "m4":
                for i, (fv, gv) in enumerate(zip(fno_vals, gt_vals)):
                    if np.isnan(fv) and np.isnan(gv):
                        ax.annotate(
                            "M4: no plateau\n(FNO & FD)",
                            xy=(xpos[i] + (off_f + off_g) / 2, 0.05),
                            xytext=(xpos[i], 30), ha="center", va="center",
                            fontsize=7, color="#444",
                            arrowprops=dict(arrowstyle="->", color="#444", lw=0.7))

        ax.set_yscale("symlog", linthresh=0.01)
        ax.set_xticks(xpos)
        ax.set_xticklabels([f"{xq:g}" for xq in xqs])
        ax.set_xlabel(r"observer location $x_q\,/\,M$")
        ax.set_ylabel(label)
        ax.grid(True, axis="y", alpha=0.3)
        ax.axhline(1.0, color="grey", lw=0.5, ls=":")
        ax.set_ylim(0.001, 200)
    axes[0].legend(loc="upper left", fontsize=7, ncol=2, frameon=False)
    fig.suptitle(
        "v4 FNO and FD-GT QNM extraction across $x_q$, canonical "
        "$M=1,\\,x_0=4,\\,\\sigma=5$, $t\\in[10,100]\\,M$", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p_out = OUT_DIR / "fno_v4_xq_sweep.png"
    fig.savefig(p_out, dpi=150)
    plt.close(fig)
    print(f"[fig] {p_out}")


# --------------------------------------------------------------------------
# Figure 3: ringdown overlay at xq=2 (v4)
# --------------------------------------------------------------------------
def fig_ringdown_xq2(version="v4"):
    c = load_cache(version)
    x, t, pp, pt = c["x"], c["t"], c["phi_pred"], c["phi_true"]
    ix = int(np.argmin(np.abs(x - 2.0)))
    y_p = pp[:, ix]
    y_t = pt[:, ix]

    # Match the PINN ringdown convention (Fig.~fwd-ringdown etc.): plot only
    # t in [0, 50] M.  Beyond t ~ 50 M the QNM amplitude is at the field-error
    # floor for both PINN and FNO, and PINN runs only train to t = 50 M, so
    # using the same display window makes the FNO and PINN ringdowns directly
    # comparable.
    tmask = t <= 50.0
    t_c = t[tmask]; y_p = y_p[tmask]; y_t = y_t[tmask]

    out = OUT_DIR / f"fno_{version}_ringdown_xq2.png"
    pinn_plot.plot_ringdown_overlay(
        t=t_c, y_fd=y_t, y_pinn=y_p,
        outpath=str(out),
        model_label=f"FNO {version}",
        xq=2.0,
        title=(f"Ringdown at $x_q=2\\,M$: FNO {version} vs FD reference, "
               f"canonical BH"),
    )
    print(f"[fig] {out}")


# --------------------------------------------------------------------------
# Figure 4: 100-BH population scatter
# --------------------------------------------------------------------------
def fig_pop_scatter(version="v4"):
    p_in = ROOT / f"outputs/qnm/fno_xq2_comparison/full100_qnm_xq2_{version}.json"
    if not p_in.exists() and version == "v2":
        p_in = ROOT / "outputs/qnm/fno_xq2_comparison/full100_qnm_xq2.json"
    d = json.load(open(p_in))
    recs = d["records"]
    M_arr = np.array([r["M"] for r in recs])

    def collect(key, metric):
        return np.array([
            (r[key]["m4"].get(metric) if r[key]["m4"].get(metric) is not None
             else np.nan)
            for r in recs])

    fno_w = collect("FNO", "omega_pct_err")
    fno_t = collect("FNO", "tau_pct_err")
    gt_w = collect("GT", "omega_pct_err")
    gt_t = collect("GT", "tau_pct_err")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax, vals_f, vals_g, label in [
        (axes[0], fno_w, gt_w, r"$\omega$ % err"),
        (axes[1], fno_t, gt_t, r"$\tau$ % err"),
    ]:
        valid = np.isfinite(vals_f) & np.isfinite(vals_g)
        sc = ax.scatter(vals_g[valid], vals_f[valid],
                        c=M_arr[valid], cmap="viridis", s=22,
                        edgecolor="k", linewidth=0.3)
        lim_hi = max(np.nanmax(vals_f[valid]), np.nanmax(vals_g[valid])) * 1.1
        lim_lo = 1e-3
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=0.8,
                label="y = x (FNO matches FD-GT)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lim_lo, lim_hi); ax.set_ylim(lim_lo, lim_hi)
        ax.set_xlabel(f"FD-GT/M4 {label}")
        ax.set_ylabel(f"FNO/M4 {label}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        ax.set_title(
            f"valid: FNO {valid.sum()}/{len(recs)},  "
            f"med FNO={np.nanmedian(vals_f):.3g}%  med GT={np.nanmedian(vals_g):.3g}%",
            fontsize=9)
        plt.colorbar(sc, ax=ax, label=r"BH mass $M$")
    fig.suptitle(f"FNO {version} vs FD-GT, M4 plateau on 100-BH population at "
                 f"$x_q=2\\,M$, $t\\in[10,100]\\,M$", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p_out = OUT_DIR / f"fno_{version}_population_scatter.png"
    fig.savefig(p_out, dpi=150)
    plt.close(fig)
    print(f"[fig] {p_out}")


# --------------------------------------------------------------------------
# Figure 5: v2/v3/v4 ablation
# --------------------------------------------------------------------------
def fig_ablation():
    p = ROOT / "outputs/qnm/fno_xq2_comparison/summary_v2_v3_v4.json"
    d = json.load(open(p))
    sweep = d["fixed_BH_sweep"]
    runs = d["runs"]

    versions = ["v2", "v3", "v4"]
    rmsd_by_ver = {}
    canon_m4_w = {}
    canon_m4_t = {}

    # Source 1: fixed_BH_sweep section (v3, v4 today)
    for k, v in sweep["runs"].items():
        ver = k.split("_")[-1]
        rmsd_by_ver[ver] = v["field_rmsd"]
        rec = v["xq_sweep"].get("xq_2.0")
        if rec is not None:
            m4 = rec["FNO"]["m4"]
            canon_m4_w[ver] = m4.get("omega_pct_err")
            canon_m4_t[ver] = m4.get("tau_pct_err")

    # Source 2: cached canonical field (works for any version) — overrides if
    # present so the three panels are populated from the same canonical run.
    for ver in versions:
        p_cache = CACHE_DIR / f"{ver}_canonical.npz"
        if not p_cache.exists():
            continue
        z = np.load(p_cache)
        rmsd_by_ver[ver] = float(z["field_rmsd"])
        x = z["x"]; t = z["t"]; pp = z["phi_pred"]
        ix = int(np.argmin(np.abs(x - 2.0)))
        y_p = pp[:, ix].astype(np.float64)
        r = qnm_method_4_window_scan(t, y_p, t_start_min=10.0,
                                     t_start_max=20.0, t_end=100.0, n_starts=12)
        # Convert to dimensionless and to percent error
        w = r.get("omega", float("nan"))
        ta = r.get("tau", float("nan"))
        if np.isfinite(w) and np.isfinite(ta):
            canon_m4_w[ver] = abs(w - THEORY_W) / THEORY_W * 100.0
            canon_m4_t[ver] = abs(ta - THEORY_T) / THEORY_T * 100.0

    rmsds = [rmsd_by_ver.get(v, np.nan) for v in versions]

    # Population medians (FNO m4) from each run
    pop_w, pop_t = {}, {}
    pop_pass1 = {}
    for r in runs:
        b = r["by"].get("FNO/m4", {})
        pop_w[r["label"]] = b.get("median_omega_pct", np.nan)
        pop_t[r["label"]] = b.get("median_tau_pct", np.nan)
        pop_pass1[r["label"]] = b.get("pass_lt_1pct", np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))

    # Panel 1: field RMSD
    ax = axes[0]
    bars = ax.bar(versions, rmsds, color="#4c72b0", edgecolor="black", lw=0.5)
    ax.set_ylabel("canonical-BH field RMSD")
    ax.set_yscale("log")
    ax.set_title("Field accuracy on canonical BH")
    ax.set_ylim(min(rmsds) * 0.55, max(rmsds) * 2.5)
    for b, v in zip(bars, rmsds):
        if np.isfinite(v):
            ax.text(b.get_x() + b.get_width() / 2,
                    v * 1.10, f"{v:.2e}",
                    ha="center", va="bottom", fontsize=9)

    # Panel 2: canonical FNO/m4 % err
    ax = axes[1]
    have_canon = [v for v in versions if canon_m4_w.get(v) is not None]
    xpos = np.arange(len(have_canon))
    w = 0.35
    ws = [canon_m4_w[v] for v in have_canon]
    ts = [canon_m4_t[v] for v in have_canon]
    ax.bar(xpos - w / 2, ws, w, color="#4c72b0", edgecolor="black", lw=0.5,
           label=r"$\omega$ % err")
    ax.bar(xpos + w / 2, ts, w, color="#c44e52", edgecolor="black", lw=0.5,
           label=r"$\tau$ % err")
    ax.set_xticks(xpos); ax.set_xticklabels(have_canon)
    ax.set_ylabel(r"FNO M4 % err (canonical BH, $x_q=2$)")
    ax.set_yscale("log")
    ax.set_title("Canonical-BH QNM accuracy (FNO/M4)")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.set_ylim(min(ws + ts) * 0.4, max(ws + ts) * 5.0)
    for i, (vw, vt) in enumerate(zip(ws, ts)):
        ax.text(i - w / 2, vw * 1.15, f"{vw:.3g}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w / 2, vt * 1.15, f"{vt:.3g}", ha="center", va="bottom", fontsize=8)

    # Panel 3: population 100-BH FNO/m4 median
    ax = axes[2]
    xpos = np.arange(len(versions))
    ws = [pop_w[v] for v in versions]
    ts = [pop_t[v] for v in versions]
    ax.bar(xpos - w / 2, ws, w, color="#4c72b0", edgecolor="black", lw=0.5,
           label=r"$\omega$ % err (med)")
    ax.bar(xpos + w / 2, ts, w, color="#c44e52", edgecolor="black", lw=0.5,
           label=r"$\tau$ % err (med)")
    ax.set_xticks(xpos); ax.set_xticklabels(versions)
    ax.set_ylabel(r"FNO M4 % err (100-BH median, $x_q=2$)")
    ax.set_yscale("log")
    ax.set_title("Population-median QNM accuracy")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.set_ylim(min(ws + ts) * 0.4, max(ws + ts) * 5.0)
    for i, (vw, vt) in enumerate(zip(ws, ts)):
        ax.text(i - w / 2, vw * 1.15, f"{vw:.3g}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w / 2, vt * 1.15, f"{vt:.3g}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(r"FNO version ablation: v2 baseline, v3 ($t_{\max}{=}100$, "
                 r"$n_t{=}64$), v4 (slice-loss, time-weighted loss)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p_out = OUT_DIR / "fno_ablation_v2v3v4.png"
    fig.savefig(p_out, dpi=150)
    plt.close(fig)
    print(f"[fig] {p_out}")


# --------------------------------------------------------------------------
# Figure 6: M4 1-D plateau diagnostic at xq=2 (v4 FNO)
# --------------------------------------------------------------------------
def fig_plateau_m4(version="v4"):
    """M4 1-D plateau scan: 1x2 layout, per-window dots in C0, plateau
    indices highlighted in C3, theory dotted black -- identical visual
    convention to scripts/extract_qnm.py and the *_method4_stability.png
    used by every PINN run (Fig.~fwd-plateau-m4 etc.)."""
    c = load_cache(version)
    x, t, pp = c["x"], c["t"], c["phi_pred"]
    ix = int(np.argmin(np.abs(x - 2.0)))
    y_p = pp[:, ix].astype(np.float64)

    r = qnm_method_4_window_scan(
        t, y_p, t_start_min=10.0, t_start_max=24.0,
        t_end=50.0, n_starts=15, potential="zerilli", ell=2,
    )
    # Persist for reproducibility / the LaTeX table.
    scan_path = OUT_DIR / f"fno_{version}_method4_two_mode.json"
    e4 = percentage_errors({"omega": r["omega"], "tau": r["tau"]},
                           potential="zerilli", ell=2, M=1.0)
    with open(scan_path, "w") as f:
        s = {}
        for k, v in {**r, **e4}.items():
            if isinstance(v, np.ndarray):
                s[k] = v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                s[k] = float(v)
            else:
                s[k] = v
        json.dump(s, f, indent=2)

    ts = np.asarray(r["t_starts"])
    os_ = np.asarray(r["omegas"])
    tas = np.asarray(r["taus"])
    pidx = r.get("plateau_idx") or []

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(ts, os_, "o-", label="per-window fit")
    if pidx:
        axes[0].plot(ts[pidx], os_[pidx], "o", color="C3", label="plateau")
        axes[0].axhline(r["omega"], color="C3", ls="--", lw=0.8)
    axes[0].axhline(THEORY_W, color="k", ls=":", lw=0.8, label="theory")
    axes[0].set_xlabel("start time t0"); axes[0].set_ylabel(r"$\omega M$")
    axes[0].legend(loc="best", fontsize=8)
    axes[1].plot(ts, tas, "o-")
    if pidx:
        axes[1].plot(ts[pidx], tas[pidx], "o", color="C3")
        axes[1].axhline(r["tau"], color="C3", ls="--", lw=0.8)
    axes[1].axhline(THEORY_T, color="k", ls=":", lw=0.8)
    axes[1].set_xlabel("start time t0"); axes[1].set_ylabel(r"$\tau / M$")
    fig.suptitle(f"Method 4 stability scan (fno_{version}, xq=2)")
    fig.tight_layout()
    out = OUT_DIR / f"fno_{version}_plateau_m4_xq2.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"[fig] {out}")


# --------------------------------------------------------------------------
# Figure 7 (NEW): M5 2-D plateau heatmap at xq=2 (v4 FNO)
# Mirrors Figs.~fwd-plateau-m5 / inv-plateau-m5 / cur-plateau-m5.
# --------------------------------------------------------------------------
def fig_plateau_m5(version="v4"):
    c = load_cache(version)
    x, t, pp = c["x"], c["t"], c["phi_pred"]
    ix = int(np.argmin(np.abs(x - 2.0)))
    y_p = pp[:, ix].astype(np.float64)

    # The M4/M5 extraction window matches the PINN convention
    # (t in [10, 50] M): beyond ~50 M the QNM signal is at the field-error
    # floor for both modalities, so the FNO uses the same extraction window
    # as the PINN runs and the heatmaps are directly comparable.
    m5 = qnm_method_5_2d_scan(
        t, y_p,
        t_start_min=10.0, t_start_max=25.0,
        t_end_min=30.0, t_end_max=50.0,
        n_starts=10, n_ends=9,
        potential="zerilli", ell=2,
    )
    e5 = percentage_errors({"omega": m5["omega"], "tau": m5["tau"]},
                           potential="zerilli", ell=2, M=1.0)
    m5_full = {**m5, **e5}

    # Persist the scan for the LaTeX table row and for reproducibility.
    scan_path = OUT_DIR / f"fno_{version}_method5_2d_scan.json"
    with open(scan_path, "w") as f:
        # Convert numpy arrays to lists for JSON serialisation.
        s = {}
        for k, v in m5_full.items():
            if isinstance(v, np.ndarray):
                s[k] = v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                s[k] = float(v)
            else:
                s[k] = v
        json.dump(s, f, indent=2)
    print(f"[m5]  {scan_path}")
    print(f"      M5 omega = {m5['omega']:.6f}  ({e5['omega_pct_err']:.4f}% err)")
    print(f"      M5 tau   = {m5['tau']:.6f}  ({e5['tau_pct_err']:.4f}% err)")

    # Render in the EXACT visual style used for every PINN M5 figure in the
    # paper: 1x2 horizontal, raw omega/tau values on imshow, viridis with a
    # shared linear colour scale (omega in [0.30, 0.50], tau in [5.0, 20.0]),
    # red rectangle outlining the plateau block. This is the convention from
    # scripts/exploration/replot_consistent_scales.py that produced
    # outputs/qnm/zerilli_l2_*/pinn_method5_2d_heatmap.png.
    ts = np.asarray(m5["t_starts"])
    tes = np.asarray(m5["t_ends"])
    og = np.asarray(m5["omegas_grid"])
    tg = np.asarray(m5["taus_grid"])
    OMEGA_LIM = (0.30, 0.50)
    TAU_LIM = (5.0, 20.0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    extent = [ts[0], ts[-1], tes[0], tes[-1]]
    im0 = axes[0].imshow(og, origin="lower", aspect="auto", extent=extent,
                         cmap="viridis", vmin=OMEGA_LIM[0], vmax=OMEGA_LIM[1])
    axes[0].set_xlabel(r"$t_0$"); axes[0].set_ylabel(r"$t_{\rm end}$")
    axes[0].set_title(rf"$\omega M$  (theory={THEORY_W})  shared scale")
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(tg, origin="lower", aspect="auto", extent=extent,
                         cmap="viridis", vmin=TAU_LIM[0], vmax=TAU_LIM[1])
    axes[1].set_xlabel(r"$t_0$"); axes[1].set_ylabel(r"$t_{\rm end}$")
    axes[1].set_title(rf"$\tau / M$  (theory={THEORY_T})  shared scale")
    plt.colorbar(im1, ax=axes[1])
    t0_lo = m5.get("t0_plateau_min"); t0_hi = m5.get("t0_plateau_max")
    te_lo = m5.get("te_plateau_min"); te_hi = m5.get("te_plateau_max")
    if all(v is not None and np.isfinite(v) for v in (t0_lo, t0_hi, te_lo, te_hi)):
        for ax in axes:
            ax.add_patch(plt.Rectangle(
                (t0_lo, te_lo), t0_hi - t0_lo, te_hi - te_lo,
                fill=False, edgecolor="red", lw=1.5,
            ))
    fig.suptitle(f"Method 5 2-D stability scan (fno_{version}, xq=2)")
    fig.tight_layout()
    out = OUT_DIR / f"fno_{version}_plateau_m5_xq2.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"[fig] {out}")


# --------------------------------------------------------------------------
# Figure 8 (NEW): training loss curve
# Mirrors Fig.~fwd-loss for the PINN.
# --------------------------------------------------------------------------
def fig_loss_curve(version="v4"):
    cfg = load_config(CONFIGS[version])
    out_dir_run = cfg["training"]["out_dir"]
    hist_path = os.path.join(out_dir_run, "history.json")
    if not os.path.exists(hist_path):
        print(f"[loss] no history at {hist_path}, skipping")
        return
    with open(hist_path) as f:
        history = json.load(f)["history"]
    out = OUT_DIR / f"fno_{version}_loss.png"
    pinn_plot.plot_loss_fno(
        history=history,
        outpath=str(out),
        title=f"FNO {version} training history",
    )
    print(f"[fig] {out}")


# --------------------------------------------------------------------------
# Figure 6: M4 1-D plateau diagnostic (legacy combined version removed)
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", nargs="+", default=["v4"],
                    help="FNO versions to cache (and use for ablation if v2/v3/v4 all present)")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--skip-cache", action="store_true",
                    help="skip stage A and just regenerate plots from existing cache")
    ap.add_argument("--only", nargs="+", default=None,
                    help="subset of figures to draw: snapshots xq_sweep ringdown population ablation plateau")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_cache:
        for ver in args.versions:
            ensure_cache(ver, recompute=args.recompute)

    do = set(args.only) if args.only else {"snapshots", "abs_diff", "pointwise",
                                            "xq_sweep", "ringdown",
                                            "population", "ablation",
                                            "plateau_m4", "plateau_m5",
                                            "loss"}
    if "snapshots" in do:
        fig_snapshots("v4")
    if "abs_diff" in do:
        fig_abs_diff_snapshots("v4")
    if "pointwise" in do:
        fig_pointwise_error("v4")
    if "xq_sweep" in do:
        fig_xq_sweep()
    if "ringdown" in do:
        fig_ringdown_xq2("v4")
    if "population" in do:
        fig_pop_scatter("v4")
    if "ablation" in do:
        fig_ablation()
    if "plateau_m4" in do or "plateau" in do:
        fig_plateau_m4("v4")
    if "plateau_m5" in do:
        fig_plateau_m5("v4")
    if "loss" in do:
        fig_loss_curve("v4")


if __name__ == "__main__":
    main()

"""Re-plot all error_heatmap, M4-stability, M5-2D-heatmap PNGs across the
forward, inverse and three curriculum runs using SHARED color/axis scales
so they can be compared visually. Overwrites existing PNGs in place.

The data themselves are unchanged; only the rendering is regenerated."""
from __future__ import annotations
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RUNS = [
    "zerilli_l2_greedy_f03_lbfgs30k",
    "zerilli_l2_inverse_qnm_combo",
    "zerilli_l2_curriculum",
    "zerilli_l2_curriculum_3w",
    "zerilli_l2_soft_overlap_curriculum",
]

# ---------------- 1. global scales ----------------------------------------

def load_pinn_fd(run: str):
    d = os.path.join(ROOT, "outputs", "pinn", run)
    pinn = glob.glob(os.path.join(d, "*pinn.npz"))
    fd   = glob.glob(os.path.join(d, "*fd.npz"))
    if not pinn or not fd:
        return None
    p = np.load(pinn[0]); f = np.load(fd[0])
    return p["x"], p["t"], f["phi"], p["phi"]

abs_max = 0.0
err_data = {}
for r in RUNS:
    out = load_pinn_fd(r)
    if out is None:
        print(f"[skip] {r} no data"); continue
    x, t, fd, pi = out
    err = np.abs(fd - pi)
    err_data[r] = (x, t, err)
    abs_max = max(abs_max, float(err.max()))
print(f"global abs-error max = {abs_max:.4e}")

# ---------------- 2. M4 / M5 ranges ---------------------------------------

OMEGA_LIM = (0.30, 0.50)
TAU_LIM   = (5.0, 20.0)

# ---------------- 3. error heatmaps ---------------------------------------

for r, (x, t, err) in err_data.items():
    out = os.path.join(ROOT, "outputs", "pinn", r, "error_heatmap.png")
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.pcolormesh(x, t, err, shading="auto", cmap="hot_r",
                       vmin=0.0, vmax=abs_max)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r"$|\Phi_{\mathrm{FD}} - \Phi_{\mathrm{PINN}}|$"
                   f"  (shared scale, max={abs_max:.2e})")
    ax.set_xlabel(r"$x_* / M$"); ax.set_ylabel("t / M")
    ax.set_title(f"Pointwise error  ({r})")
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)
    print(f"  wrote {out}")

# ---------------- 4. M4 stability plots -----------------------------------

def find_qnm(run: str):
    return os.path.join(ROOT, "outputs", "qnm", run)

for r in RUNS:
    qd = find_qnm(r)
    files = glob.glob(os.path.join(qd, "*method4_two_mode.json"))
    if not files: print(f"[skip M4] {r}"); continue
    j = json.load(open(files[0]))
    ts = np.asarray(j["t_starts"])
    om = np.asarray(j["omegas"], dtype=float)
    ta = np.asarray(j["taus"], dtype=float)
    pidx = j.get("plateau_idx") or []
    om_th = j.get("omega_theory", 0.3737)
    ta_th = j.get("tau_theory", 11.241)
    # mask out-of-range values to NaN so they do not draw spurious vertical
    # lines through the plateau when forced to a shared y-axis
    om_p = np.where((om >= OMEGA_LIM[0]) & (om <= OMEGA_LIM[1]), om, np.nan)
    ta_p = np.where((ta >= TAU_LIM[0])   & (ta <= TAU_LIM[1]),   ta, np.nan)
    n_clip_om = int(np.sum(np.isnan(om_p) & ~np.isnan(om)))
    n_clip_ta = int(np.sum(np.isnan(ta_p) & ~np.isnan(ta)))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(ts, om_p, "o-", label="per-window fit")
    if pidx:
        axes[0].plot(ts[pidx], om_p[pidx], "o", color="C3", label="plateau")
        axes[0].axhline(j["omega"], color="C3", ls="--", lw=0.8)
    axes[0].axhline(om_th, color="k", ls=":", lw=0.8, label="theory")
    axes[0].set_xlabel("start time t0"); axes[0].set_ylabel(r"$\omega M$")
    axes[0].set_ylim(OMEGA_LIM); axes[0].legend(loc="best", fontsize=8)
    if n_clip_om: axes[0].text(0.98, 0.02, f"{n_clip_om} pts off-scale", transform=axes[0].transAxes, ha="right", va="bottom", fontsize=7, color="gray")
    axes[1].plot(ts, ta_p, "o-")
    if pidx:
        axes[1].plot(ts[pidx], ta_p[pidx], "o", color="C3")
        axes[1].axhline(j["tau"], color="C3", ls="--", lw=0.8)
    axes[1].axhline(ta_th, color="k", ls=":", lw=0.8)
    axes[1].set_xlabel("start time t0"); axes[1].set_ylabel(r"$\tau / M$")
    axes[1].set_ylim(TAU_LIM)
    if n_clip_ta: axes[1].text(0.98, 0.02, f"{n_clip_ta} pts off-scale", transform=axes[1].transAxes, ha="right", va="bottom", fontsize=7, color="gray")
    fig.suptitle(f"Method 4 stability scan ({r})  shared y: omega in {OMEGA_LIM}, tau in {TAU_LIM}")
    fig.tight_layout()
    tag = os.path.basename(files[0]).replace("_method4_two_mode.json", "")
    out = os.path.join(qd, f"{tag}_method4_stability.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"  wrote {out}")

# ---------------- 5. M5 2D heatmaps ---------------------------------------

for r in RUNS:
    qd = find_qnm(r)
    files = glob.glob(os.path.join(qd, "*method5_2d_scan.json"))
    if not files: print(f"[skip M5] {r}"); continue
    j = json.load(open(files[0]))
    ts = np.asarray(j["t_starts"]); tes = np.asarray(j["t_ends"])
    og = np.asarray(j["omegas_grid"]); tg = np.asarray(j["taus_grid"])
    om_th = j.get("omega_theory", 0.3737); ta_th = j.get("tau_theory", 11.241)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    extent = [ts[0], ts[-1], tes[0], tes[-1]]
    im0 = axes[0].imshow(og, origin="lower", aspect="auto", extent=extent,
                         cmap="viridis", vmin=OMEGA_LIM[0], vmax=OMEGA_LIM[1])
    axes[0].set_xlabel(r"$t_0$"); axes[0].set_ylabel(r"$t_{\rm end}$")
    axes[0].set_title(rf"$\omega M$  (theory={om_th})  shared scale")
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(tg, origin="lower", aspect="auto", extent=extent,
                         cmap="viridis", vmin=TAU_LIM[0], vmax=TAU_LIM[1])
    axes[1].set_xlabel(r"$t_0$"); axes[1].set_ylabel(r"$t_{\rm end}$")
    axes[1].set_title(rf"$\tau / M$  (theory={ta_th})  shared scale")
    plt.colorbar(im1, ax=axes[1])
    t0_lo = j.get("t0_plateau_min"); t0_hi = j.get("t0_plateau_max")
    te_lo = j.get("te_plateau_min"); te_hi = j.get("te_plateau_max")
    if all(v is not None and np.isfinite(v) for v in (t0_lo, t0_hi, te_lo, te_hi)):
        for ax in axes:
            ax.add_patch(plt.Rectangle((t0_lo, te_lo), t0_hi - t0_lo, te_hi - te_lo,
                                       fill=False, edgecolor="red", lw=1.5))
    fig.suptitle(f"Method 5 2-D stability scan ({r})")
    fig.tight_layout()
    tag = os.path.basename(files[0]).replace("_method5_2d_scan.json", "")
    out = os.path.join(qd, f"{tag}_method5_2d_heatmap.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"  wrote {out}")

print("done")

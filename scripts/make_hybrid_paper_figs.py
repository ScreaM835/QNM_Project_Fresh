"""Generate the hybrid (coarse-FD + FNO residual) figures for the paper.

The hybrid trains and evaluates on the same t in [0, 50] M domain used by
the PINN sections, so every figure here mirrors the PINN visual convention
exactly (snapshot times, colour map, axis ranges, scan windows).

Outputs go to ``outputs/hybrid/fno_sw_k2_h64/figs/``.

Stages:
    A) canonical-BH field cache  -- run fine FD, coarse FD, upsample, then
       hybrid model on the canonical (M, x0, sigma) = (1, 4, 5) draw and
       store (psi_fine, psi_coarse_up, psi_hybrid) for plotting.
       Skipped if cache exists.
    B) plots                      -- read cache + summary.json/per_sample.json
       and write the figures.

Usage:
    python scripts/make_hybrid_paper_figs.py
    python scripts/make_hybrid_paper_figs.py --recompute
    python scripts/make_hybrid_paper_figs.py --skip-cache --only snapshots
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
from src.fd_solver import solve_fd
from src.hybrid_fno import build_hybrid_fno
from src.hybrid_data_pipe import upsample_to_fine
from src.qnm import (
    qnm_method_4_window_scan,
    qnm_method_5_2d_scan,
    percentage_errors,
)
from src import plotting as pinn_plot

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/hybrid_sw_train_k2_h64.yaml"
DATASET_CFG = "configs/hybrid_sw_dataset.yaml"     # for fine-grid/domain
# Immutable defaults (used as argparse defaults; CONFIG/DATASET_CFG above are
# reassigned at runtime to the run selected on the command line).
DEFAULT_CONFIG = CONFIG
DEFAULT_DATASET_CFG = DATASET_CFG
OUT_DIR = ROOT / "outputs" / "hybrid" / "fno_sw_k2_h64" / "figs"
CACHE_DIR = ROOT / "outputs" / "hybrid" / "fno_sw_k2_h64" / "field_cache"

M_CANON, X0_CANON, SIGMA_CANON = 1.0, 4.0, 5.0
THEORY_W = 0.3737
THEORY_T = 11.241

# Snapshot times in M -- match the PINN forward convention
# (times = [10, 20, 30, 40] M in scripts/run_pinn.py) for cross-model
# consistency of the field-snapshot figures.
SNAP_TIMES = [10.0, 20.0, 30.0, 40.0]


# --------------------------------------------------------------------------
# Stage A: canonical-BH field cache
# --------------------------------------------------------------------------
def _run_fd(base_cfg, dx, dt, space_scheme=None, bc_order=None):
    """Solve the FD problem at the given (dx, dt).

    ``space_scheme``/``bc_order`` upgrade the spatial stencil; they are passed
    only for the cheap coarse prior so the cached field matches the dataset the
    model was trained on (the fine reference is left on the default 2nd-order
    scheme). When both are ``None`` the default 2nd-order solver is used.
    """
    fd = {"dx": float(dx), "dt": float(dt),
          "scheme": str(base_cfg["fd"].get("scheme", "rk4_mol"))}
    if space_scheme is not None:
        fd["space_scheme"] = str(space_scheme)
    if bc_order is not None:
        fd["bc_order"] = int(bc_order)
    fd_cfg = {
        "physics": {"M": M_CANON, "potential": "zerilli", "l": 2,
                    "pde_sign": "standard"},
        "initial_data": {"A": 1.0, "x0": X0_CANON, "sigma": SIGMA_CANON,
                         "velocity_profile": "outgoing"},
        "domain": {"xmin": float(base_cfg["domain"]["xmin"]),
                   "xmax": float(base_cfg["domain"]["xmax"]),
                   "tmin": 0.0, "tmax": float(base_cfg["domain"]["tmax"])},
        "fd": fd,
    }
    sol = solve_fd(fd_cfg)
    return (sol["x"].astype(np.float64), sol["t"].astype(np.float64),
            sol["phi"].astype(np.float64), sol["V"].astype(np.float64))


def _assemble_input_one(phi_c_up, V_fine, M_val, dt_c, x_fine, phi_c, x_coarse):
    """Build the 5-channel input (1, 5, Nt_f, Nx_f) for one BH.

    Mirrors src.hybrid_data_pipe.assemble_split for a single sample.
    """
    Nt_f, Nx_f = phi_c_up.shape
    Phi0 = phi_c_up[0, :]
    # IC velocity computed from coarse field, then resampled to fine x grid.
    Pi0_c = (phi_c[1, :] - phi_c[0, :]) / dt_c
    Pi0 = np.interp(x_fine, x_coarse, Pi0_c).astype(np.float32)
    X = np.empty((1, 5, Nt_f, Nx_f), dtype=np.float32)
    X[0, 0] = phi_c_up.astype(np.float32)
    X[0, 1] = np.broadcast_to(Phi0.astype(np.float32), (Nt_f, Nx_f))
    X[0, 2] = np.broadcast_to(V_fine.astype(np.float32), (Nt_f, Nx_f))
    X[0, 3] = np.full((Nt_f, Nx_f), float(M_val), dtype=np.float32)
    X[0, 4] = np.broadcast_to(Pi0, (Nt_f, Nx_f))
    return X


def _solve_canonical():
    """Run fine FD + coarse FD + hybrid model on the canonical BH.

    Returns dict with x, t (fine), psi_fine, psi_coarse_up, psi_hybrid,
    plus the per-field rL2 and RMSD numbers.
    """
    base_cfg = load_config(DATASET_CFG)
    train_cfg = load_config(CONFIG)
    base_dx = float(base_cfg["fd"]["dx"])
    base_dt = float(base_cfg["fd"]["dt"])
    # k inferred from dataset path naming
    ds_path = train_cfg["dataset"]["path"]
    k = 2 if "k2" in os.path.basename(ds_path) else 4
    coarse_dx = k * base_dx
    coarse_dt = k * base_dt
    # The cheap coarse prior may use a higher-order / dispersion-relation-
    # preserving stencil (fd.coarse_space_scheme + fd.coarse_bc_order in the
    # dataset config); the fine reference always uses the default 2nd-order
    # scheme. Mirror exactly what src.hybrid_dataset did when building the
    # training set, otherwise the cached prior would not match the model.
    coarse_ss = base_cfg["fd"].get("coarse_space_scheme")
    coarse_bo = base_cfg["fd"].get("coarse_bc_order")
    print(f"[CANON] fine dx={base_dx} dt={base_dt}; coarse k={k} "
          f"dx={coarse_dx} dt={coarse_dt}  "
          f"coarse_scheme={coarse_ss or 'central2'} bc={coarse_bo or 2}")

    x_f, t_f, psi_fine, V_fine = _run_fd(base_cfg, base_dx, base_dt)
    x_c, t_c, psi_coarse, _    = _run_fd(base_cfg, coarse_dx, coarse_dt,
                                         space_scheme=coarse_ss,
                                         bc_order=coarse_bo)
    print(f"[CANON] fine grid Nt={t_f.size} Nx={x_f.size};   "
          f"coarse grid Nt={t_c.size} Nx={x_c.size}")

    # Cubic-spline upsample coarse to fine grid (matches src.hybrid_data_pipe).
    psi_coarse_up = upsample_to_fine(psi_coarse, x_c, t_c, x_f, t_f)

    # Build 5-channel input and run model
    X = _assemble_input_one(psi_coarse_up, V_fine, M_CANON,
                            float(t_c[1] - t_c[0]), x_f, psi_coarse, x_c)
    out_dir_run = train_cfg["logging"]["out_dir"]
    ckpt = os.path.join(out_dir_run, "model_best.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(ckpt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[CANON] loading model on {device}: {ckpt}")
    model = build_hybrid_fno(train_cfg).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    # A raw state_dict saved via torch.save(model.state_dict(), ...) carries an
    # OrderedDict `_metadata` attribute that round-trips as an unexpected key on
    # a strict load (e.g. for the _PriorFeatureFNO gate/grad wrapper); drop it.
    if isinstance(state, dict):
        state.pop("_metadata", None)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        delta = model(torch.from_numpy(X).to(device)).cpu().numpy()[0, 0]
    psi_hybrid = psi_coarse_up + delta.astype(np.float64)

    # Diagnostics
    def _rmsd(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
    def _rl2(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)) /
                     (np.sqrt(np.mean(a ** 2)) + 1e-30))

    out = dict(
        x=x_f, t=t_f,
        psi_fine=psi_fine,
        psi_coarse_up=psi_coarse_up,
        psi_hybrid=psi_hybrid,
        rmsd_hybrid=_rmsd(psi_fine, psi_hybrid),
        rmsd_baseline=_rmsd(psi_fine, psi_coarse_up),
        rl2_hybrid=_rl2(psi_fine, psi_hybrid),
        rl2_baseline=_rl2(psi_fine, psi_coarse_up),
        k=k,
        fine_dx=base_dx, fine_dt=base_dt,
        coarse_dx=coarse_dx, coarse_dt=coarse_dt,
    )
    print(f"[CANON] field RMSD hybrid={out['rmsd_hybrid']:.3e}   "
          f"baseline={out['rmsd_baseline']:.3e}   "
          f"ratio={out['rmsd_hybrid']/out['rmsd_baseline']:.3e}")
    print(f"[CANON] field rL2  hybrid={out['rl2_hybrid']:.3e}   "
          f"baseline={out['rl2_baseline']:.3e}   "
          f"ratio={out['rl2_hybrid']/out['rl2_baseline']:.3e}")
    return out


def ensure_cache(recompute: bool = False) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = CACHE_DIR / "canonical.npz"
    if p.exists() and not recompute:
        print(f"[cache] {p} exists, skipping recompute")
        return p
    c = _solve_canonical()
    np.savez(p, **c)
    print(f"[cache] wrote {p}")
    return p


def load_cache():
    z = np.load(CACHE_DIR / "canonical.npz")
    return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}


# --------------------------------------------------------------------------
# Figure 1a: snapshots overlay  (FD-fine vs Hybrid at four times)
# --------------------------------------------------------------------------
def fig_snapshots():
    c = load_cache()
    x = c["x"]; t = c["t"]; pf = c["psi_fine"]; ph = c["psi_hybrid"]
    times = [tj for tj in SNAP_TIMES if tj <= float(t.max()) + 1e-6]
    out = OUT_DIR / "hybrid_snapshots.png"
    pinn_plot.plot_snapshots(
        x=x, t=t, phi_fd=pf, phi_pinn=ph, times=times,
        outpath=str(out), model_label="Hybrid",
        title=("Hybrid vs FD-fine reference, canonical "
               r"$M=1,\,x_0=4,\,\sigma=5$"),
    )
    print(f"[fig] {out}")


# --------------------------------------------------------------------------
# Figure 1b: abs-diff snapshots
# --------------------------------------------------------------------------
def fig_abs_diff_snapshots():
    c = load_cache()
    x = c["x"]; t = c["t"]; pf = c["psi_fine"]; ph = c["psi_hybrid"]
    times = [tj for tj in SNAP_TIMES if tj <= float(t.max()) + 1e-6]
    tstr = r",\,".join(f"{tj:.0f}" for tj in times)
    out = OUT_DIR / "hybrid_abs_diff_snapshots.png"
    pinn_plot.plot_abs_diff_snapshots(
        x=x, t=t, phi_fd=pf, phi_pinn=ph, times=times,
        outpath=str(out),
        title=(r"$|\Phi_{\mathrm{FD\,fine}} - \Phi_{\mathrm{Hybrid}}|$  "
               r"at $t/M \in \{" + tstr + r"\}$ (canonical BH)"),
    )
    print(f"[fig] {out}")


# --------------------------------------------------------------------------
# Figure 1c: pointwise error heatmap on (x, t).
# Same colour-scale convention as the PINN and FNO figures.
# --------------------------------------------------------------------------
def fig_pointwise_error():
    from matplotlib.colors import LogNorm
    c = load_cache()
    x = c["x"]; t = c["t"]; pf = c["psi_fine"]; ph = c["psi_hybrid"]
    err = np.abs(pf - ph)
    vmax = 1.11e-1
    vmin = vmax * 1e-4
    err_clip = np.maximum(err, vmin)
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.pcolormesh(x, t, err_clip, shading="auto", cmap="magma_r",
                       norm=LogNorm(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r"$|\Phi_{\mathrm{FD\,fine}} - \Phi_{\mathrm{Hybrid}}|$"
                   f"  (shared log scale, max={vmax:.2e})")
    ax.set_xlabel(r"$x_* / M$"); ax.set_ylabel("t / M")
    ax.set_title("Hybrid pointwise error (canonical BH)")
    fig.tight_layout()
    out = OUT_DIR / "hybrid_pointwise_error.png"
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"[fig] {out}")


# --------------------------------------------------------------------------
# Figure 1d: same heatmap for the baseline coarse-upsampled field, so the
# reader can see the 300x field-RMSD reduction the FNO residual delivers.
# --------------------------------------------------------------------------
def fig_pointwise_error_baseline():
    from matplotlib.colors import LogNorm
    c = load_cache()
    x = c["x"]; t = c["t"]; pf = c["psi_fine"]; pb = c["psi_coarse_up"]
    err = np.abs(pf - pb)
    vmax = 1.11e-1
    vmin = vmax * 1e-4
    err_clip = np.maximum(err, vmin)
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.pcolormesh(x, t, err_clip, shading="auto", cmap="magma_r",
                       norm=LogNorm(vmin=vmin, vmax=vmax))
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(r"$|\Phi_{\mathrm{FD\,fine}} - \Phi_{\mathrm{coarse\!-\!up}}|$"
                   f"  (shared log scale, max={vmax:.2e})")
    ax.set_xlabel(r"$x_* / M$"); ax.set_ylabel("t / M")
    ax.set_title(r"Baseline (coarse FD + quintic upsample) pointwise error")
    fig.tight_layout()
    out = OUT_DIR / "hybrid_pointwise_error_baseline.png"
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"[fig] {out}")


# --------------------------------------------------------------------------
# Figure 1e: grad-channel diagnostic -- three (x, t) heatmaps on the shared
# paper scale (baseline error | |grad prior| | hybrid error) PLUS the
# quantitative speckle / grad-error metrics, written to <run>/eval/report.json.
# This is what makes every run emit the SAME diagnostic plot and one
# machine-readable metrics file (merging the population eval summary if present).
# --------------------------------------------------------------------------
def _grad_mag_unit(prior):
    """|grad(prior)| with UNIT spacing -- identical to the _PriorFeatureFNO
    wrapper feature (torch.gradient over dims t, x with default spacing=1)."""
    g_t = np.gradient(prior, axis=0)
    g_x = np.gradient(prior, axis=1)
    return np.sqrt(g_t * g_t + g_x * g_x + 1e-24)


def fig_grad_vs_error():
    from matplotlib.colors import LogNorm
    c = load_cache()
    x = c["x"]; t = c["t"]
    fine = c["psi_fine"]; prior = c["psi_coarse_up"]; hyb = c["psi_hybrid"]
    err_base = np.abs(fine - prior)
    err_hyb = np.abs(fine - hyb)
    g = _grad_mag_unit(prior)

    vmax = 1.11e-1
    vmin = vmax * 1e-4

    def _panel(ax, field, title):
        fc = np.maximum(field, vmin)
        im = ax.pcolormesh(x, t, fc, shading="auto", cmap="magma_r",
                           norm=LogNorm(vmin=vmin, vmax=vmax))
        ax.set_xlabel(r"$x_* / M$"); ax.set_title(title, fontsize=10)
        return im

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    _panel(axes[0], err_base,
           r"(1) baseline err  $|\Phi_{\rm fine}-\Phi_{\rm coarse\text{-}up}|$")
    im = _panel(axes[1], g,
                r"(2) FNO feature  $|\nabla\Phi_{\rm prior}|$ (unit spacing)")
    _panel(axes[2], err_hyb,
           r"(3) hybrid err  $|\Phi_{\rm fine}-\Phi_{\rm hybrid}|$")
    axes[0].set_ylabel("t / M")
    cbar = fig.colorbar(im, ax=axes, pad=0.015, fraction=0.025)
    cbar.set_label(f"shared log scale  [{vmin:.2e}, {vmax:.2e}]")
    fig.suptitle("grad-channel diagnostic: pointwise error vs |grad prior| "
                 "(canonical BH)", fontsize=12)
    out = OUT_DIR / "grad_vs_error.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] {out}")

    # ---- quantitative metrics -> <run>/eval/report.json ------------------
    lg = np.log10(np.maximum(g, 1e-30)).ravel()
    le = np.log10(np.maximum(err_base, 1e-30)).ravel()
    pearson_r = float(np.corrcoef(lg, le)[0, 1])

    gflat = g.ravel(); ebflat = err_base.ravel(); ehflat = err_hyb.ravel()
    order = np.argsort(gflat); nq = gflat.size
    qcut = np.linspace(0, nq, 11).astype(int)
    deciles = []
    for d in range(10):
        idx = order[qcut[d]:qcut[d + 1]]
        deciles.append({
            "decile": f"{10*d}-{10*(d+1)}%",
            "grad_mean": float(gflat[idx].mean()),
            "base_err_mean": float(ebflat[idx].mean()),
            "hybrid_err_mean": float(ehflat[idx].mean()),
        })

    speckle = {}
    for thr in (1e-5, 1e-4):
        clean = err_base < thr
        any_clean = bool(clean.any())
        pe = float(err_base[clean].mean()) if any_clean else float("nan")
        fe = float(err_hyb[clean].mean()) if any_clean else float("nan")
        speckle[f"prior_err_lt_{thr:.0e}"] = {
            "domain_frac": float(clean.mean()),
            "prior_err_mean": pe,
            "hybrid_err_mean": fe,
            "hybrid_over_prior": float(fe / (pe + 1e-30)) if any_clean else float("nan"),
        }
    ext = x > 80.0
    pe_e = float(err_base[:, ext].mean()); fe_e = float(err_hyb[:, ext].mean())
    speckle["causal_exterior_x_gt_80"] = {
        "field_amp_mean": float(np.abs(fine[:, ext]).mean()),
        "prior_err_mean": pe_e,
        "hybrid_err_mean": fe_e,
        "hybrid_over_prior": float(fe_e / (pe_e + 1e-30)),
    }

    report = {
        "field_canonical": {
            "rl2_prior": float(c["rl2_baseline"]),
            "rl2_hybrid": float(c["rl2_hybrid"]),
            "rmsd_prior": float(c["rmsd_baseline"]),
            "rmsd_hybrid": float(c["rmsd_hybrid"]),
        },
        "grad_error": {"pearson_r_loglog": pearson_r, "deciles": deciles},
        "speckle": speckle,
    }
    # Merge the population eval summary (test-set field rL2 + QNM medians) so
    # report.json is the single all-metrics file per run.
    eval_dir = OUT_DIR.parent / "eval"
    summ_path = eval_dir / "summary.json"
    if summ_path.exists():
        with open(summ_path) as f:
            summ = json.load(f)
        report["field_testset"] = summ.get("field", {})
        report["qnm_testset_median"] = summ.get("qnm_pct_err_median", {})
        report["qnm_window"] = summ.get("qnm_window", {})
    eval_dir.mkdir(parents=True, exist_ok=True)
    rep_path = eval_dir / "report.json"
    with open(rep_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[metrics] {rep_path}")
    print(f"          grad-error Pearson r(loglog) = {pearson_r:+.3f}")
    sp5 = speckle["prior_err_lt_1e-05"]
    print(f"          clean(prior<1e-5, {100*sp5['domain_frac']:.0f}% dom): "
          f"prior {sp5['prior_err_mean']:.2e} -> hybrid {sp5['hybrid_err_mean']:.2e} "
          f"({sp5['hybrid_over_prior']:.0f}x)")


# --------------------------------------------------------------------------
# Figure 2: ringdown overlay at xq=2.
# Two-panel linear + semilogy, FD-fine vs Hybrid (and baseline overlay
# on the log panel for contrast).
# --------------------------------------------------------------------------
def fig_ringdown_xq2():
    c = load_cache()
    x = c["x"]; t = c["t"]
    ix = int(np.argmin(np.abs(x - 2.0)))
    y_fine = c["psi_fine"][:, ix]
    y_hyb  = c["psi_hybrid"][:, ix]
    y_base = c["psi_coarse_up"][:, ix]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(t, y_fine, label="FD fine", linewidth=1.2)
    ax1.plot(t, y_hyb,  label="Hybrid",  linewidth=1.2, linestyle="--")
    ax1.set_ylabel(r"$\Phi(x_q, t)$")
    ax1.set_title(r"Ringdown at $x_q = 2\,M$, canonical BH")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.semilogy(t, np.abs(y_fine) + 1e-30, label="FD fine", linewidth=1.2)
    ax2.semilogy(t, np.abs(y_hyb)  + 1e-30, label="Hybrid",  linewidth=1.2,
                 linestyle="--")
    ax2.semilogy(t, np.abs(y_base) + 1e-30, label="coarse + upsample",
                 linewidth=1.0, linestyle=":", alpha=0.8, color="grey")
    ax2.set_xlabel("t / M"); ax2.set_ylabel(r"$|\Phi(x_q, t)|$")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = OUT_DIR / "hybrid_ringdown_xq2.png"
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"[fig] {out}")


# --------------------------------------------------------------------------
# Figure 3: M4 1-D plateau diagnostic at xq=2 for the hybrid prediction.
# Mirrors the PINN/FNO M4 convention (window t in [10, 50] M, t_start scan
# [10, 20] M, theory in dotted black, plateau in C3, dashed mean).
# --------------------------------------------------------------------------
def fig_plateau_m4():
    c = load_cache()
    x = c["x"]; t = c["t"]; ph = c["psi_hybrid"]
    ix = int(np.argmin(np.abs(x - 2.0)))
    y_p = ph[:, ix].astype(np.float64)
    # Scan parameters identical to scripts/eval_hybrid_sw.py (_all_methods),
    # so this single-BH diagnostic matches the population eval and the canonical
    # QNM table exactly (t_start in [10, 18] M, t_end = 50 M).
    r = qnm_method_4_window_scan(
        t, y_p, t_start_min=10.0, t_start_max=18.0,
        t_end=50.0, n_starts=12, potential="zerilli", ell=2,
    )
    e4 = percentage_errors({"omega": r["omega"], "tau": r["tau"]},
                           potential="zerilli", ell=2, M=1.0)
    scan_path = OUT_DIR / "hybrid_method4_two_mode.json"
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
    print(f"[m4]  {scan_path}")
    print(f"      M4 omega = {r['omega']:.6f}  ({e4['omega_pct_err']:.4f}% err)")
    print(f"      M4 tau   = {r['tau']:.6f}  ({e4['tau_pct_err']:.4f}% err)")

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
    fig.suptitle("Method 4 stability scan (Hybrid, canonical BH, xq=2)")
    fig.tight_layout()
    out = OUT_DIR / "hybrid_plateau_m4_xq2.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"[fig] {out}")


# --------------------------------------------------------------------------
# Figure 4: M5 2-D plateau heatmap at xq=2 for the hybrid prediction.
# Same visual convention as the PINN/FNO M5 plots.
# --------------------------------------------------------------------------
def fig_plateau_m5():
    c = load_cache()
    x = c["x"]; t = c["t"]; ph = c["psi_hybrid"]
    ix = int(np.argmin(np.abs(x - 2.0)))
    y_p = ph[:, ix].astype(np.float64)
    # Scan parameters identical to scripts/eval_hybrid_sw.py (_all_methods),
    # so this single-BH diagnostic matches the population eval and the canonical
    # QNM table exactly (t_0 in [10, 18] M, t_end in [40, 50] M).
    m5 = qnm_method_5_2d_scan(
        t, y_p,
        t_start_min=10.0, t_start_max=18.0,
        t_end_min=40.0, t_end_max=50.0,
        n_starts=8, n_ends=5,
        potential="zerilli", ell=2,
    )
    e5 = percentage_errors({"omega": m5["omega"], "tau": m5["tau"]},
                           potential="zerilli", ell=2, M=1.0)
    m5_full = {**m5, **e5}
    scan_path = OUT_DIR / "hybrid_method5_2d_scan.json"
    with open(scan_path, "w") as f:
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
    fig.suptitle("Method 5 2-D stability scan (Hybrid, canonical BH, xq=2)")
    fig.tight_layout()
    out = OUT_DIR / "hybrid_plateau_m5_xq2.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"[fig] {out}")


# --------------------------------------------------------------------------
# Figure 5: training loss curve (Adam + L-BFGS).
# Custom because the hybrid history records (train_mse, val_mse) per phase
# rather than the weighted components of the pure-FNO loss.
# --------------------------------------------------------------------------
def fig_loss_curve():
    train_cfg = load_config(CONFIG)
    hist_path = os.path.join(train_cfg["logging"]["out_dir"], "history.json")
    with open(hist_path) as f:
        h = json.load(f)["history"]
    adam = [r for r in h if r["phase"] == "adam"]
    lbfgs = [r for r in h if r["phase"] == "lbfgs"]
    n_adam = len(adam)

    fig, ax = plt.subplots(figsize=(9, 5))
    if adam:
        ep_a = np.array([r["epoch"] for r in adam])
        ax.semilogy(ep_a, [r["train_mse"] for r in adam], "-",
                    label="Adam train MSE", linewidth=1.5)
        ax.semilogy(ep_a, [r["val_mse"]   for r in adam], "-",
                    label="Adam val MSE", linewidth=1.5, alpha=0.8)
    if lbfgs:
        ep_l = np.array([n_adam + r["epoch"] for r in lbfgs])
        ax.semilogy(ep_l, [r["val_mse"] for r in lbfgs], "-",
                    label="L-BFGS val MSE", linewidth=1.5, color="C3")
        ax.axvline(n_adam, color="grey", linestyle="--", alpha=0.6, linewidth=1.0)
        ax.text(n_adam, ax.get_ylim()[1], " L-BFGS \N{RIGHTWARDS ARROW}",
                fontsize=8, va="top", ha="left", color="grey")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"MSE on residual $\delta\Phi$")
    ax.set_title("Hybrid training history")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out = OUT_DIR / "hybrid_loss.png"
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"[fig] {out}")


# --------------------------------------------------------------------------
# Figure 6: population scatter of M4 (hybrid vs fine-FD truth) on
# the 100-BH test set. Mirrors the FNO population scatter.
# --------------------------------------------------------------------------
def fig_pop_scatter():
    train_cfg = load_config(CONFIG)
    ps_path = os.path.join(train_cfg["logging"]["out_dir"], "eval", "per_sample.json")
    with open(ps_path) as f:
        recs = json.load(f)
    M_arr = np.array([r["M"] for r in recs])

    def collect(src, metric):
        return np.array([
            (r[f"xq2_{src}"]["M4"].get(metric)
             if r[f"xq2_{src}"]["M4"].get(metric) is not None else np.nan)
            for r in recs])
    hyb_w = collect("hyb",  "omega_pct_err")
    hyb_t = collect("hyb",  "tau_pct_err")
    bas_w = collect("base", "omega_pct_err")
    bas_t = collect("base", "tau_pct_err")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax, vals_h, vals_b, label in [
        (axes[0], hyb_w, bas_w, r"$\omega$ % err"),
        (axes[1], hyb_t, bas_t, r"$\tau$ % err"),
    ]:
        valid = np.isfinite(vals_h) & np.isfinite(vals_b)
        sc = ax.scatter(vals_b[valid], vals_h[valid],
                        c=M_arr[valid], cmap="viridis", s=22,
                        edgecolor="k", linewidth=0.3)
        finite = np.r_[vals_h[valid], vals_b[valid]]
        lim_hi = np.nanmax(finite) * 1.2
        lim_lo = max(min(np.nanmin(finite) * 0.5, 1e-3), 1e-4)
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=0.8,
                label="y = x (Hybrid ties baseline)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lim_lo, lim_hi); ax.set_ylim(lim_lo, lim_hi)
        ax.set_xlabel(f"Baseline (coarse+upsample) M4 {label}")
        ax.set_ylabel(f"Hybrid M4 {label}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        n_win = int(np.sum(vals_h[valid] < vals_b[valid]))
        ax.set_title(
            f"valid: {int(valid.sum())}/{len(recs)},  "
            f"Hyb<Base on {n_win}/{int(valid.sum())},  "
            f"med Hyb={np.nanmedian(vals_h):.3g}%  med Base={np.nanmedian(vals_b):.3g}%",
            fontsize=9)
        plt.colorbar(sc, ax=ax, label=r"BH mass $M$")
    fig.suptitle(r"Hybrid vs baseline (coarse FD + quintic upsample), M4 on "
                 r"100-BH test set at $x_q=2\,M$, $t\in[10,50]\,M$",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUT_DIR / "hybrid_population_scatter.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[fig] {out}")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="hybrid training config; figures are written under "
                         "its logging.out_dir/figs (default: flagship k2_h64).")
    ap.add_argument("--dataset-cfg", default=DEFAULT_DATASET_CFG,
                    help="dataset config providing the fine grid/domain and the "
                         "coarse-prior stencil (fd.coarse_space_scheme/bc_order).")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--skip-cache", action="store_true")
    ap.add_argument("--only", nargs="+", default=None,
                    help="subset of figures: snapshots abs_diff pointwise "
                         "pointwise_base grad ringdown plateau_m4 plateau_m5 "
                         "loss population")
    args = ap.parse_args()

    # Route every output through the selected run's out_dir so a new experiment
    # never overwrites the flagship figures.
    global CONFIG, DATASET_CFG, OUT_DIR, CACHE_DIR
    CONFIG = args.config
    DATASET_CFG = args.dataset_cfg
    run_out = Path(load_config(CONFIG)["logging"]["out_dir"])
    if not run_out.is_absolute():
        run_out = ROOT / run_out
    OUT_DIR = run_out / "figs"
    CACHE_DIR = run_out / "field_cache"
    print(f"[main] config={CONFIG}\n[main] dataset_cfg={DATASET_CFG}\n"
          f"[main] figures -> {OUT_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.skip_cache:
        ensure_cache(recompute=args.recompute)
    do = set(args.only) if args.only else {
        "snapshots", "abs_diff", "pointwise", "pointwise_base", "grad",
        "ringdown", "plateau_m4", "plateau_m5", "loss", "population",
    }
    if "snapshots" in do:        fig_snapshots()
    if "abs_diff" in do:         fig_abs_diff_snapshots()
    if "pointwise" in do:        fig_pointwise_error()
    if "pointwise_base" in do:   fig_pointwise_error_baseline()
    if "grad" in do:             fig_grad_vs_error()
    if "ringdown" in do:         fig_ringdown_xq2()
    if "plateau_m4" in do:       fig_plateau_m4()
    if "plateau_m5" in do:       fig_plateau_m5()
    if "loss" in do:             fig_loss_curve()
    if "population" in do:       fig_pop_scatter()


if __name__ == "__main__":
    main()

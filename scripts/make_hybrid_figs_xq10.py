"""xq=10 M versions of the hybrid paper figures (additive; core code untouched).

Reads the canonical field cache written by scripts/make_hybrid_paper_figs.py
and the population eval (eval/per_sample.json) and produces:
  - hybrid_ringdown_xq10.png          (two-panel linear + semilogy)
  - hybrid_plateau_m4_xq10.png        (M4 stability scan + JSON)
  - hybrid_population_scatter_xq10.png (hybrid vs baseline, M4, 100-BH test set)
Same visual conventions as the xq=2 originals; population-scatter label
corrected to "quintic upsample" (the pipeline uses kx=ky=5).

Usage:
    python scripts/make_hybrid_figs_xq10.py --config configs/hybrid_sw_gate_s1em3.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config                      # noqa: E402
from src.qnm import (                                    # noqa: E402
    qnm_method_4_window_scan, qnm_method_5_2d_scan, percentage_errors,
)

THEORY_W = 0.3737
THEORY_T = 11.241
XQ = 10.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hybrid_sw_gate_s1em3.yaml")
    args = ap.parse_args()

    run_out = Path(load_config(args.config)["logging"]["out_dir"])
    if not run_out.is_absolute():
        run_out = ROOT / run_out
    out_dir = run_out / "figs"
    cache = np.load(run_out / "field_cache" / "canonical.npz")
    x = cache["x"]; t = cache["t"]
    ix = int(np.argmin(np.abs(x - XQ)))
    y_fine = cache["psi_fine"][:, ix]
    y_hyb = cache["psi_hybrid"][:, ix]
    y_base = cache["psi_coarse_up"][:, ix]

    # ---- ringdown overlay -------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(t, y_fine, label="Fine FD reference", linewidth=1.2)
    ax1.plot(t, y_base, label="Numerical prior", linewidth=1.0,
             linestyle=":", alpha=0.8, color="grey")
    ax1.plot(t, y_hyb, label="hFNO", linewidth=1.2, linestyle="--")
    ax1.set_ylabel(r"$\Phi(x_q, t)$")
    ax1.set_title(r"Ringdown at $x_q = 10\,M$, canonical configuration")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.semilogy(t, np.abs(y_fine) + 1e-30, label="Fine FD reference", linewidth=1.2)
    ax2.semilogy(t, np.abs(y_base) + 1e-30, label="Numerical prior",
                 linewidth=1.0, linestyle=":", alpha=0.8, color="grey")
    ax2.semilogy(t, np.abs(y_hyb) + 1e-30, label="hFNO", linewidth=1.2,
                 linestyle="--")
    ax2.set_xlabel("t / M"); ax2.set_ylabel(r"$|\Phi(x_q, t)|$")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    out = out_dir / "hybrid_ringdown_xq10.png"
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"[fig] {out}")

    # ---- M4 stability scan (canonical Algorithm-1 grid; M4 uses the fixed
    #      usable-signal cutoff t_end=80 M matching the canonical extraction
    #      window, M5 scans to the domain end) ---------------------------------
    r = qnm_method_4_window_scan(
        t, y_hyb.astype(np.float64), t_start_min=10.0, t_start_max=25.0,
        t_end=80.0, n_starts=16, potential="zerilli", ell=2,
    )
    e4 = percentage_errors({"omega": r["omega"], "tau": r["tau"]},
                           potential="zerilli", ell=2, M=1.0)
    scan_path = out_dir / "hybrid_method4_xq10.json"
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

    ts = np.asarray(r["t_starts"]); os_ = np.asarray(r["omegas"])
    tas = np.asarray(r["taus"]); pidx = r.get("plateau_idx") or []
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
    fig.suptitle(r"Method 4 stability scan (Hybrid, canonical BH, $x_q=10\,M$)")
    fig.tight_layout()
    out = out_dir / "hybrid_plateau_m4_xq10.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"[fig] {out}")

    # ---- M5 2-D stability scan (canonical Algorithm-1 grid, [10,100]) ------
    m5 = qnm_method_5_2d_scan(
        t, y_hyb.astype(np.float64),
        t_start_min=10.0, t_start_max=25.0,
        t_end_min=30.0, t_end_max=100.0,
        n_starts=10, n_ends=6,
        potential="zerilli", ell=2,
    )
    e5 = percentage_errors({"omega": m5["omega"], "tau": m5["tau"]},
                           potential="zerilli", ell=2, M=1.0)
    print(f"      M5 omega = {m5['omega']:.6f}  ({e5['omega_pct_err']:.4f}% err)")
    print(f"      M5 tau   = {m5['tau']:.6f}  ({e5['tau_pct_err']:.4f}% err)")
    ts5 = np.asarray(m5["t_starts"]); tes = np.asarray(m5["t_ends"])
    og = np.asarray(m5["omegas_grid"]); tg = np.asarray(m5["taus_grid"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    extent = [ts5[0], ts5[-1], tes[0], tes[-1]]
    im0 = axes[0].imshow(og, origin="lower", aspect="auto", extent=extent,
                         cmap="viridis", vmin=0.30, vmax=0.50)
    axes[0].set_xlabel(r"$t_0$"); axes[0].set_ylabel(r"$t_{\rm end}$")
    axes[0].set_title(rf"$\omega M$  (theory={THEORY_W})  shared scale")
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(tg, origin="lower", aspect="auto", extent=extent,
                         cmap="viridis", vmin=5.0, vmax=20.0)
    axes[1].set_xlabel(r"$t_0$"); axes[1].set_ylabel(r"$\tau / M$")
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
    fig.suptitle(r"Method 5 2-D stability scan (Hybrid, canonical BH, $x_q=10\,M$)")
    fig.tight_layout()
    out = out_dir / "hybrid_plateau_m5_xq10.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"[fig] {out}")

    # ---- population scatter -----------------------------------------------
    with open(run_out / "eval" / "per_sample.json") as f:
        recs = json.load(f)
    M_arr = np.array([r_["M"] for r_ in recs])

    def collect(src, metric):
        return np.array([
            (r_[f"xq10_{src}"]["M4"].get(metric)
             if r_[f"xq10_{src}"]["M4"].get(metric) is not None else np.nan)
            for r_ in recs])

    hyb_w = collect("hyb", "omega_pct_err")
    hyb_t = collect("hyb", "tau_pct_err")
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
                 r"100-BH test set at $x_q=10\,M$, $t\in[10,50]\,M$",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = out_dir / "hybrid_population_scatter_xq10.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[fig] {out}")


if __name__ == "__main__":
    main()

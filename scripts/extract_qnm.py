from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import os
import numpy as np

from src.config import load_config
from src.utils import ensure_dir, save_json
from src.qnm import (
    qnm_method_1, qnm_method_2, qnm_method_3_esprit,
    qnm_method_4_two_mode, qnm_method_4_window_scan,
    qnm_method_5_2d_scan,
    percentage_errors,
)
from src.plotting import plot_ringdown, plot_ringdown_overlay


def _fmt_pct(val: float) -> str:
    """Format percentage error, handling NaN."""
    if val != val:  # NaN check
        return "N/A"
    return f"{val:.2f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--source", choices=["fd", "pinn"], required=True)
    ap.add_argument("--esprit", action="store_true",
                    help="Additionally run Method 3 (ESPRIT). Off by default.")
    ap.add_argument("--esprit-K", type=int, default=4,
                    help="Model order for ESPRIT (default 4).")
    ap.add_argument("--two-mode", action="store_true",
                    help="Additionally run Method 4 (two-mode NLS + start-time "
                         "stability scan). Off by default.")
    ap.add_argument("--two-mode-t0-min", type=float, default=None,
                    help="Min start time for Method 4 scan (default: qnm.t_start).")
    ap.add_argument("--two-mode-t0-max", type=float, default=None,
                    help="Max start time for Method 4 scan (default: qnm.t_start + 15).")
    ap.add_argument("--two-mode-n", type=int, default=16,
                    help="Number of start times in the Method 4 scan (default 16).")
    ap.add_argument("--two-mode-2d", action="store_true",
                    help="Additionally run Method 5 (two-mode NLS + 2-D (t0, t_end) "
                         "stability scan). Off by default.")
    ap.add_argument("--two-mode-2d-te-min", type=float, default=None,
                    help="Min end time for Method 5 2-D scan (default: qnm.t_end - 20).")
    ap.add_argument("--two-mode-2d-te-max", type=float, default=None,
                    help="Max end time for Method 5 2-D scan (default: qnm.t_end).")
    ap.add_argument("--two-mode-2d-n-t0", type=int, default=10,
                    help="Number of start times in the Method 5 2-D scan (default 10).")
    ap.add_argument("--two-mode-2d-n-te", type=int, default=6,
                    help="Number of end times in the Method 5 2-D scan (default 6).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    name = cfg["experiment"]["name"]
    xq = float(cfg["evaluation"]["xq"])
    potential = cfg["physics"]["potential"]       # "zerilli" or "regge_wheeler"
    ell = int(cfg["physics"]["l"])                # angular mode number

    if args.source == "fd":
        npz = np.load(os.path.join("outputs", "fd", f"{name}_fd.npz"))
        x, t, phi = npz["x"], npz["t"], npz["phi"]
        tag = "fd"
    else:
        # expects you ran scripts/run_pinn.py already
        npz = np.load(os.path.join("outputs", "pinn", name, f"{name}_pinn.npz"))
        x, t, phi = npz["x"], npz["t"], npz["phi"]
        tag = "pinn"

    # pick nearest x index
    ix = int(np.argmin(np.abs(x - xq)))
    y = phi[:, ix]

    t_start = float(cfg["qnm"]["t_start"])
    t_end = float(cfg["qnm"]["t_end"])

    m1 = qnm_method_1(t, y, t_start=t_start, t_end=t_end)
    m2 = qnm_method_2(t, y, t_start=t_start, t_end=t_end)

    # Compute percentage errors vs theoretical values (Patel et al. Table 3 style)
    e1 = percentage_errors(m1, potential=potential, ell=ell)
    e2 = percentage_errors(m2, potential=potential, ell=ell)

    # Merge percentage errors into result dicts for saving
    m1_full = {**m1, **e1}
    m2_full = {**m2, **e2}

    outdir = os.path.join("outputs", "qnm", name)
    ensure_dir(outdir)

    save_json(os.path.join(outdir, f"{tag}_method1.json"), m1_full)
    save_json(os.path.join(outdir, f"{tag}_method2.json"), m2_full)

    plot_ringdown(t, y, os.path.join(outdir, f"{tag}_ringdown.png"), title=f"Ringdown at xq={xq} ({tag})")

    # Print formatted comparison table (matching Patel et al. Table 3 format)
    print(f"\n{'='*65}")
    print(f"  QNM Extraction: {tag.upper()} | {potential} l={ell} | xq={xq}")
    print(f"  Theory:  omega*M = {e1['omega_theory']},  tau/M = {e1['tau_theory']}")
    print(f"{'='*65}")
    print(f"  {'Method':<10} {'tau/M':>10} {'(% err)':>10} {'omega*M':>10} {'(% err)':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Method 1':<10} {m1['tau']:>10.4f} {_fmt_pct(e1['tau_pct_err']):>10} {m1['omega']:>10.4f} {_fmt_pct(e1['omega_pct_err']):>10}")
    print(f"  {'Method 2':<10} {m2['tau']:>10.4f} {_fmt_pct(e2['tau_pct_err']):>10} {m2['omega']:>10.4f} {_fmt_pct(e2['omega_pct_err']):>10}")

    if args.esprit:
        m3 = qnm_method_3_esprit(t, y, t_start=t_start, t_end=t_end, K=args.esprit_K)
        e3 = percentage_errors(m3, potential=potential, ell=ell)
        m3_full = {**m3, **e3}
        save_json(os.path.join(outdir, f"{tag}_method3_esprit.json"), m3_full)
        print(f"  {'Method 3':<10} {m3['tau']:>10.4f} {_fmt_pct(e3['tau_pct_err']):>10} {m3['omega']:>10.4f} {_fmt_pct(e3['omega_pct_err']):>10}")
        # Show secondary modes (potential overtones / tail)
        amps = m3.get("all_amps", [])
        omegas = m3.get("all_omegas", [])
        taus = m3.get("all_taus", [])
        if amps:
            order = sorted(range(len(amps)), key=lambda i: -amps[i])
            print(f"  ESPRIT (K={m3['K']}) modes sorted by |amp|:")
            for rank, i in enumerate(order):
                print(f"    #{rank}: |amp|={amps[i]:.3e}  omega={omegas[i]:+.4f}  tau={taus[i]:+.4f}")

    if args.two_mode:
        t0_min = args.two_mode_t0_min if args.two_mode_t0_min is not None else t_start
        t0_max = args.two_mode_t0_max if args.two_mode_t0_max is not None else t_start + 15.0
        m4 = qnm_method_4_window_scan(
            t, y, t_start_min=t0_min, t_start_max=t0_max, t_end=t_end,
            n_starts=args.two_mode_n, potential=potential, ell=ell,
        )
        e4 = percentage_errors({"omega": m4["omega"], "tau": m4["tau"]},
                               potential=potential, ell=ell)
        m4_full = {**m4, **e4}
        save_json(os.path.join(outdir, f"{tag}_method4_two_mode.json"), m4_full)
        print(f"  {'Method 4':<10} {m4['tau']:>10.4f} {_fmt_pct(e4['tau_pct_err']):>10} {m4['omega']:>10.4f} {_fmt_pct(e4['omega_pct_err']):>10}")
        print(f"    plateau t0 in [{m4['t0_plateau_min']:.2f}, {m4['t0_plateau_max']:.2f}]"
              f"   omega = {m4['omega']:.6f} +/- {m4['omega_std']:.6f}"
              f"   tau = {m4['tau']:.6f} +/- {m4['tau_std']:.6f}")
        # Save the plateau scan plot
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            ts = np.asarray(m4["t_starts"])
            os_ = np.asarray(m4["omegas"])
            tas = np.asarray(m4["taus"])
            pidx = m4["plateau_idx"]
            axes[0].plot(ts, os_, "o-", label="per-window fit")
            if pidx:
                axes[0].plot(ts[pidx], os_[pidx], "o", color="C3", label="plateau")
                axes[0].axhline(m4["omega"], color="C3", ls="--", lw=0.8)
            axes[0].axhline(e4["omega_theory"], color="k", ls=":", lw=0.8, label="theory")
            axes[0].set_xlabel("start time t0"); axes[0].set_ylabel(r"$\omega M$"); axes[0].legend()
            axes[1].plot(ts, tas, "o-")
            if pidx:
                axes[1].plot(ts[pidx], tas[pidx], "o", color="C3")
                axes[1].axhline(m4["tau"], color="C3", ls="--", lw=0.8)
            axes[1].axhline(e4["tau_theory"], color="k", ls=":", lw=0.8)
            axes[1].set_xlabel("start time t0"); axes[1].set_ylabel(r"$\tau / M$")
            fig.suptitle(f"Method 4 stability scan ({tag}, xq={xq})")
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f"{tag}_method4_stability.png"), dpi=120)
            plt.close(fig)
        except Exception as exc:
            print(f"  [warn] could not save stability plot: {exc}")

        # ---- M2 (single-mode envelope fit) re-evaluated on the M4 plateau ----
        # Reporting convention: omega from M4 plateau (best-variance estimator
        # for the oscillation rate of the fundamental in a two-mode signal),
        # tau from M2 over the same plateau window (best-variance estimator
        # for the decay rate once the overtone has decayed below noise).
        try:
            t0_pl = float(m4["t0_plateau_min"])
            t1_pl = float(m4["t0_plateau_max"])
            if np.isfinite(t0_pl) and np.isfinite(t1_pl):
                # Use plateau LOWER edge as M2 fit start: M4 certifies that
                # t >= t0_pl is in the QNM-clean regime (no overtone, no prompt
                # burst), and starting as early as M4 allows maximises the
                # exponential-decay range available to constrain tau.
                t_start_m2 = t0_pl
                m2_pl = qnm_method_2(t, y, t_start=t_start_m2, t_end=t_end)
                e2_pl = percentage_errors(m2_pl, potential=potential, ell=ell)
                m2_pl_full = {
                    **m2_pl, **e2_pl,
                    "t_start_used": t_start_m2,
                    "t_end_used": t_end,
                    "m4_plateau_min": t0_pl, "m4_plateau_max": t1_pl,
                }
                save_json(os.path.join(outdir, f"{tag}_method2_on_m4_plateau.json"), m2_pl_full)
                print(f"  {'M2@M4pl':<10} {m2_pl['tau']:>10.4f} {_fmt_pct(e2_pl['tau_pct_err']):>10} {m2_pl['omega']:>10.4f} {_fmt_pct(e2_pl['omega_pct_err']):>10}")
                print(f"    fit window t in [{t_start_m2:.2f}, {t_end:.2f}]"
                      f"  (M4 plateau midpoint)")
        except Exception as exc:
            print(f"  [warn] M2-on-M4-plateau failed: {exc}")

    if args.two_mode_2d:
        t0_min = args.two_mode_t0_min if args.two_mode_t0_min is not None else t_start
        t0_max = args.two_mode_t0_max if args.two_mode_t0_max is not None else t_start + 15.0
        te_min = args.two_mode_2d_te_min if args.two_mode_2d_te_min is not None else max(t_start + 10.0, t_end - 20.0)
        te_max = args.two_mode_2d_te_max if args.two_mode_2d_te_max is not None else t_end
        m5 = qnm_method_5_2d_scan(
            t, y,
            t_start_min=t0_min, t_start_max=t0_max,
            t_end_min=te_min, t_end_max=te_max,
            n_starts=args.two_mode_2d_n_t0, n_ends=args.two_mode_2d_n_te,
            potential=potential, ell=ell,
        )
        e5 = percentage_errors({"omega": m5["omega"], "tau": m5["tau"]},
                               potential=potential, ell=ell)
        m5_full = {**m5, **e5}
        save_json(os.path.join(outdir, f"{tag}_method5_2d_scan.json"), m5_full)
        print(f"  {'Method 5':<10} {m5['tau']:>10.4f} {_fmt_pct(e5['tau_pct_err']):>10} {m5['omega']:>10.4f} {_fmt_pct(e5['omega_pct_err']):>10}")
        print(f"    plateau t0 in [{m5['t0_plateau_min']:.2f}, {m5['t0_plateau_max']:.2f}]"
              f"   te in [{m5['te_plateau_min']:.2f}, {m5['te_plateau_max']:.2f}]")
        print(f"    omega = {m5['omega']:.6f} +/- {m5['omega_std']:.6f}"
              f"   tau = {m5['tau']:.6f} +/- {m5['tau_std']:.6f}")
        # Heat-map of (t0, t_end) -> omega and tau
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            ts = np.asarray(m5["t_starts"])
            tes = np.asarray(m5["t_ends"])
            og = np.asarray(m5["omegas_grid"])
            tg = np.asarray(m5["taus_grid"])
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
            extent = [ts[0], ts[-1], tes[0], tes[-1]]
            im0 = axes[0].imshow(og, origin="lower", aspect="auto", extent=extent,
                                 cmap="viridis")
            axes[0].set_xlabel(r"$t_0$"); axes[0].set_ylabel(r"$t_{\rm end}$")
            axes[0].set_title(rf"$\omega M$  (theory={e5['omega_theory']})")
            plt.colorbar(im0, ax=axes[0])
            im1 = axes[1].imshow(tg, origin="lower", aspect="auto", extent=extent,
                                 cmap="viridis")
            axes[1].set_xlabel(r"$t_0$"); axes[1].set_ylabel(r"$t_{\rm end}$")
            axes[1].set_title(rf"$\tau / M$  (theory={e5['tau_theory']})")
            plt.colorbar(im1, ax=axes[1])
            # overlay plateau rectangle
            t0_lo, t0_hi = m5["t0_plateau_min"], m5["t0_plateau_max"]
            te_lo, te_hi = m5["te_plateau_min"], m5["te_plateau_max"]
            for ax in axes:
                ax.add_patch(plt.Rectangle(
                    (t0_lo, te_lo), t0_hi - t0_lo, te_hi - te_lo,
                    fill=False, edgecolor="red", lw=1.5,
                ))
            fig.suptitle(f"Method 5 2-D stability scan ({tag}, xq={xq})")
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f"{tag}_method5_2d_heatmap.png"), dpi=120)
            plt.close(fig)
        except Exception as exc:
            print(f"  [warn] could not save 2-D heat-map: {exc}")

    print(f"{'='*65}\n")

    print(f"[QNM:{tag}] Outputs in: {outdir}")


if __name__ == "__main__":
    main()

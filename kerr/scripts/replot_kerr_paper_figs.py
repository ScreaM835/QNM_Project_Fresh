"""Regenerate the Kerr paper figures locally in the Schwarzschild plotting
convention (reference = C0 blue solid, hybrid = C1 orange dashed, coarse prior
= grey dotted) with LaTeX/mathtext labels, and print the by-spin summary table
(field / M-omega / tau, best-of-suite, with estimator labels) from the full-run
per_sample.json. Fields for the ringdown/pointwise are reconstructed from the
downloaded test corpus + model.pt (no eval re-run).
"""
from __future__ import annotations
import json
import os
import sys
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from kerr.src.hybrid_data_pipe import load_split, build_upsample_matrix, assemble
from kerr.src.hybrid_fno import build_hybrid_fno

NPZ = r"C:\Users\jonat\Downloads\dataset_test_n128.npz"
MODEL = r"C:\Users\jonat\Downloads\model.pt"
PER_SAMPLE = os.path.join(_ROOT, "kerr", "outputs", "_run2_download", "per_sample.json")
FIGS = os.path.join(_ROOT, "outputs", "kerr", "figs")
os.makedirs(FIGS, exist_ok=True)

# Schwarzschild plotting convention (matches outputs/hybrid + outputs/pinn):
C_FINE = "C0"      # reference / fine  -> blue solid
C_HYB = "C1"       # prediction / hybrid -> orange dashed
C_PRIOR = "0.5"    # coarse prior -> grey dotted
C_RICH = "C2"      # Richardson target -> green

CFG = {"fno": {"modes_tau": 64, "modes_sigma": 24, "hidden_channels": 48,
               "n_layers": 4, "domain_padding": [0.08, 0.08],
               "positional_embedding": "grid"}}


def reconstruct_canonical():
    te = load_split(NPZ, "test")
    W_k4 = build_upsample_matrix(te["sigma_k4"], te["sigma_fine"], k=5)
    W_k2 = build_upsample_matrix(te["sigma_k2"], te["sigma_fine"], k=5)
    W_k8 = build_upsample_matrix(te["sigma_k8"], te["sigma_fine"], k=5)
    asm = assemble(te, W_k4, W_k2, target_mode="richardson", return_eval=True,
                   W_prior=W_k8, prior_key="k8", richardson_p=4, norm_mode="scalar")
    tau = np.asarray(te["tau"]); sigma = np.asarray(te["sigma_fine"])
    scri = int(te["scri_idx_fine"]); aM = np.asarray(te["P"])[:, 0]
    ic = int(np.argmin(np.abs(aM - 0.7)))

    model = build_hybrid_fno(CFG)
    sd = torch.load(MODEL, map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(asm["X"][ic:ic + 1])
        pred = model(xb).numpy()[0]  # (2, Ntau, Nf)
    s = float(asm["scale"][ic])
    up4_re, up4_im = asm["up4_re"][ic], asm["up4_im"][ic]
    fine_re, fine_im = asm["fine_re"][ic], asm["fine_im"][ic]
    hyb_re = up4_re + s * pred[0]
    hyb_im = up4_im + s * pred[1]
    return dict(tau=tau, sigma=sigma, scri=scri, aM=float(aM[ic]),
                up4_re=up4_re, up4_im=up4_im, fine_re=fine_re, fine_im=fine_im,
                hyb_re=hyb_re, hyb_im=hyb_im)


def plot_ringdown(f):
    tau, scri = f["tau"], f["scri"]
    plt.figure(figsize=(7, 4))
    plt.plot(tau, f["fine_re"][:, scri], color=C_FINE, ls="-", lw=1.4, label="FD (fine)")
    plt.plot(tau, f["up4_re"][:, scri], color=C_PRIOR, ls=":", lw=1.2, label="coarse prior")
    plt.plot(tau, f["hyb_re"][:, scri], color=C_HYB, ls="--", lw=1.3, label="hybrid")
    plt.xlabel(r"$\tau/M$")
    plt.ylabel(r"$\mathrm{Re}\,\psi$ at $\mathcal{I}^{+}$")
    plt.legend(frameon=False)
    plt.title(rf"Kerr ringdown at $\mathcal{{I}}^{{+}}$ ($a/M={f['aM']:.2f}$)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "hybrid_ringdown_scri.png"), dpi=200)
    plt.close()


def plot_pointwise(f):
    def cmag(are, aim, bre, bim):
        return np.abs(np.sqrt(are ** 2 + aim ** 2) - np.sqrt(bre ** 2 + bim ** 2))
    ep = cmag(f["up4_re"], f["up4_im"], f["fine_re"], f["fine_im"])
    eh = cmag(f["hyb_re"], f["hyb_im"], f["fine_re"], f["fine_im"])
    vmax = max(ep.max(), eh.max(), 1e-30)
    for name, E in (("baseline", ep), ("hybrid", eh)):
        plt.figure(figsize=(6, 4))
        plt.pcolormesh(f["sigma"], f["tau"], np.maximum(E, vmax * 1e-6),
                       shading="auto", cmap="magma_r",
                       norm=LogNorm(vmin=vmax * 1e-5, vmax=vmax))
        plt.colorbar(label=r"$|\,|\psi| - |\psi_{\mathrm{fine}}|\,|$")
        plt.xlabel(r"$\sigma$")
        plt.ylabel(r"$\tau/M$")
        plt.title(rf"Kerr {name} pointwise error ($a/M={f['aM']:.2f}$)")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGS, f"hybrid_pointwise_error_{name}.png"), dpi=200)
        plt.close()


def load_rows():
    with open(PER_SAMPLE) as fh:
        return json.load(fh)


def plot_vs_spin(rows):
    aM = np.array([r["a_over_M"] for r in rows])

    # field rel-L2 vs spin (best-of-suite is not applicable to the field; the
    # field is a direct rel-L2 vs fine, so prior/hybrid/richardson).
    plt.figure(figsize=(6, 4))
    plt.scatter(aM, [100 * r["field_rl2_prior"] for r in rows], s=16, color=C_PRIOR, label="coarse prior")
    plt.scatter(aM, [100 * r["field_rl2_hybrid"] for r in rows], s=16, color=C_HYB, label="hybrid")
    plt.scatter(aM, [100 * r["field_rl2_richardson"] for r in rows], s=12, color=C_RICH, marker="x", label="Richardson")
    plt.axhline(5.0, color="k", ls="--", lw=0.8, label="5% gate")
    plt.xlabel(r"$a/M$")
    plt.ylabel(r"field rel. $L^{2}$ (%)")
    plt.yscale("log")
    plt.legend(frameon=False, fontsize=8)
    plt.title(r"Field accuracy vs spin")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "hybrid_field_vs_spin.png"), dpi=200)
    plt.close()

    # M-omega error vs spin (best-of-suite, matches the results table)
    plt.figure(figsize=(6, 4))
    for key, c, lab in (("prior", C_PRIOR, "prior"), ("hybrid", C_HYB, "hybrid"), ("fine", C_FINE, "fine")):
        plt.scatter(aM, [r["qnm"][key]["best_Mw_err_pct"] for r in rows], s=16, color=c, label=lab)
    plt.axhline(1.0, color="k", ls="--", lw=0.8, label="1% gate")
    plt.xlabel(r"$a/M$")
    plt.ylabel(r"$M\omega$ error (%)")
    plt.yscale("log")
    plt.legend(frameon=False, fontsize=8)
    plt.title(r"Frequency accuracy vs spin (at $\mathcal{I}^{+}$)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "hybrid_qnm_vs_spin.png"), dpi=200)
    plt.close()

    # tau error vs spin (best-of-suite)
    plt.figure(figsize=(6, 4))
    for key, c, lab in (("prior", C_PRIOR, "prior"), ("hybrid", C_HYB, "hybrid"), ("fine", C_FINE, "fine")):
        plt.scatter(aM, [r["qnm"][key]["best_tau_err_pct"] for r in rows], s=16, color=c, label=lab)
    plt.axhline(5.0, color="k", ls="--", lw=0.8, label="5% gate")
    plt.xlabel(r"$a/M$")
    plt.ylabel(r"$\tau/M$ error (\%)")
    plt.yscale("log")
    plt.legend(frameon=False, fontsize=8)
    plt.title(r"Damping-time accuracy vs spin (at $\mathcal{I}^{+}$)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "hybrid_tau_vs_spin.png"), dpi=200)
    plt.close()


def print_table(rows):
    aM = np.array([r["a_over_M"] for r in rows])
    bins = [(0.0, 0.3), (0.3, 0.6), (0.6, 0.8), (0.8, 0.95)]

    def med(vals):
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        return float(np.median(vals)) if vals else float("nan")

    def modal_method(idx, who, key):
        labs = [rows[i]["qnm"][who][key] for i in idx if rows[i]["qnm"][who][key]]
        return Counter(labs).most_common(1)[0][0] if labs else "-"

    print("\n=== BY-SPIN (medians; best-of-suite for QNM) ===")
    hdr = ("bin", "n", "fld_pri", "fld_hyb", "fld_rich",
           "Mw_pri", "Mw_hyb", "Mw_fine", "Mw_meth",
           "tau_pri", "tau_hyb", "tau_fine", "tau_meth")
    print("{:>10} {:>3} {:>7} {:>7} {:>8} {:>6} {:>6} {:>7} {:>8} {:>6} {:>6} {:>7} {:>8}".format(*hdr))
    for lo, hi in bins:
        idx = [i for i in range(len(rows)) if lo <= aM[i] < hi]
        if not idx:
            continue
        f_pri = med([100 * rows[i]["field_rl2_prior"] for i in idx])
        f_hyb = med([100 * rows[i]["field_rl2_hybrid"] for i in idx])
        f_ric = med([100 * rows[i]["field_rl2_richardson"] for i in idx])
        mw_pri = med([rows[i]["qnm"]["prior"]["best_Mw_err_pct"] for i in idx])
        mw_hyb = med([rows[i]["qnm"]["hybrid"]["best_Mw_err_pct"] for i in idx])
        mw_fin = med([rows[i]["qnm"]["fine"]["best_Mw_err_pct"] for i in idx])
        tau_pri = med([rows[i]["qnm"]["prior"]["best_tau_err_pct"] for i in idx])
        tau_hyb = med([rows[i]["qnm"]["hybrid"]["best_tau_err_pct"] for i in idx])
        tau_fin = med([rows[i]["qnm"]["fine"]["best_tau_err_pct"] for i in idx])
        print("{:>10} {:>3d} {:7.2f} {:7.3f} {:8.4f} {:6.2f} {:6.3f} {:7.4f} {:>8} {:6.2f} {:6.3f} {:7.4f} {:>8}".format(
            f"[{lo},{hi}]", len(idx), f_pri, f_hyb, f_ric, mw_pri, mw_hyb, mw_fin,
            modal_method(idx, "hybrid", "best_Mw_method"),
            tau_pri, tau_hyb, tau_fin, modal_method(idx, "hybrid", "best_tau_method")))

    allidx = list(range(len(rows)))
    print("\n=== POPULATION MEDIANS (best-of-suite) ===")
    print("field: prior {:.2f}  hybrid {:.3f}  richardson {:.4f}".format(
        med([100 * rows[i]["field_rl2_prior"] for i in allidx]),
        med([100 * rows[i]["field_rl2_hybrid"] for i in allidx]),
        med([100 * rows[i]["field_rl2_richardson"] for i in allidx])))
    for q in ("Mw", "tau"):
        print(f"{q}: prior {med([rows[i]['qnm']['prior'][f'best_{q}_err_pct'] for i in allidx]):.3f}  "
              f"hybrid {med([rows[i]['qnm']['hybrid'][f'best_{q}_err_pct'] for i in allidx]):.3f}  "
              f"fine {med([rows[i]['qnm']['fine'][f'best_{q}_err_pct'] for i in allidx]):.3f}")
    # method frequency (hybrid)
    print("\nhybrid best_Mw_method counts:",
          Counter(r["qnm"]["hybrid"]["best_Mw_method"] for r in rows).most_common())
    print("hybrid best_tau_method counts:",
          Counter(r["qnm"]["hybrid"]["best_tau_method"] for r in rows).most_common())


if __name__ == "__main__":
    rows = load_rows()
    print_table(rows)
    plot_vs_spin(rows)
    print("[replot] vs-spin figures written")
    f = reconstruct_canonical()
    plot_ringdown(f)
    plot_pointwise(f)
    print("[replot] ringdown + pointwise figures written ->", FIGS)

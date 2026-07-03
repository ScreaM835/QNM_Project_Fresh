"""Comprehensive Kerr hybrid-FNO job: assemble -> train -> evaluate -> plot.

A single entry point that runs the full Schwarzschild-style workflow adapted for
the Kerr Teukolsky surrogate, in one job:

  1. assemble normalised Richardson-target tensors from the Phase C corpus
     (fast precomputed-matrix quintic upsample);
  2. train the no-gate additive FNO residual operator (Adam + cosine, val-best
     checkpoint, history);
  3. evaluate on the held-out test split -- field rel-L2 (prior / hybrid /
     Richardson vs fine) and QNM at scri (prior / hybrid / Richardson / fine vs
     Leaver, via the VALIDATED ``extract_fundamental``), stratified by spin;
  4. save metrics (report.json, per_sample.json, history.json) + model.pt;
  5. write the paper-style figure set.

Design rationale lives in ``kerr/src/hybrid_fno.py`` and
``kerr/src/hybrid_data_pipe.py``. Headline acceptance is the Phase C gate:
held-out field rel-L2 <= 5% AND QNM M*omega <= 1% / tau <= 5% vs Leaver.

Usage:
    python kerr/scripts/train_eval_hybrid_kerr.py --config kerr/configs/hybrid_kerr.yaml
    python kerr/scripts/train_eval_hybrid_kerr.py --config ... --smoke   # tiny dry run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from typing import Dict, List, Tuple

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import yaml

from kerr.src.hybrid_data_pipe import (
    HYBRID_IN_CHANNELS, HYBRID_OUT_CHANNELS,
    load_split, build_upsample_matrix, assemble,
)
from kerr.src.hybrid_fno import build_hybrid_fno, count_parameters
from kerr.src.qnm_kerr_reference import kerr_qnm
from kerr.src.qnm_ensemble_kerr import extract_qnm_kerr_ensemble


# ---------------------------------------------------------------------------
# Config / setup
# ---------------------------------------------------------------------------
def load_config(path: str) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------
def rel_l2_complex(pred_re, pred_im, tgt_re, tgt_im, axis=(1, 2)) -> np.ndarray:
    """Per-sample complex relative L2: ||pred - tgt|| / ||tgt||."""
    num = np.sqrt(np.sum((pred_re - tgt_re) ** 2 + (pred_im - tgt_im) ** 2, axis=axis))
    den = np.sqrt(np.sum(tgt_re ** 2 + tgt_im ** 2, axis=axis))
    return num / np.maximum(den, 1e-30)


def extract_qnm_scri(y_re, y_im, a, tau, qnm_ref) -> Dict[str, float]:
    """Multi-method consensus (M*omega, tau/M) + honest error bar at scri.

    Uses the stored per-sample Leaver reference ``qnm_ref`` = (Mw_R, Mw_I, tau/M)
    to set the spin-scaled fit band AND, at a>0, as the mode-ID target for the
    mode-selective ensemble path. ELL-AGNOSTIC: the target is read from the
    corpus, never a hardcoded ell, so this is correct for ell=2 and ell=4 alike.
    The mode-selective path fixes the high-l/high-spin tail-lock that the old
    single-method extractor silently suffered (it locked onto the slow late-time
    tail and reported a wrong mode with no flag). Returns the consensus omega/tau
    plus the across-method spread (the error bar), the selection distance
    (confidence guard), and the surviving-method count. nan on failure.
    """
    try:
        y = y_re.astype(np.float64) + 1j * y_im.astype(np.float64)
        tau_ref = float(qnm_ref[2])
        omega_target = complex(float(qnm_ref[0]), float(qnm_ref[1]))
        out = extract_qnm_kerr_ensemble(
            tau, y, a_over_M=float(a), tau_ref=tau_ref,
            tau_final=float(tau[-1]), omega_target=omega_target,
        )
        om = float(out.get("omega", np.nan))
        spr = float(out.get("omega_std", np.nan))
        return {
            "Mw": om, "tau": float(out.get("tau", np.nan)),
            "spread_pct": (spr / abs(om) * 100.0) if (np.isfinite(om) and om) else float("nan"),
            "sel_dist": float(out.get("sel_dist", np.nan)),
            "n_methods": int(out.get("n_omega", 0)),
        }
    except Exception:
        return {"Mw": float("nan"), "tau": float("nan"),
                "spread_pct": float("nan"), "sel_dist": float("nan"), "n_methods": 0}


def pct_err(val, ref) -> float:
    if not np.isfinite(val) or abs(ref) < 1e-30:
        return float("nan")
    return abs(val - ref) / abs(ref) * 100.0


# ---------------------------------------------------------------------------
# Parallel QNM extraction workers
#
# The mode-selective ensemble runs a window x order scan per field, so scoring
# four fields x N test samples serially is ~2 h. The work is embarrassingly
# parallel and pure-numpy (no torch / CUDA), so we fork a process pool AFTER the
# GPU forward pass (mirrors reextract_qnm_ensemble_l2). Workers inherit the big
# field arrays via fork (no IPC copy) through ``_eval_qnm_init``.
# ---------------------------------------------------------------------------
_EVALG: Dict[str, object] = {}


def _eval_qnm_init(g: Dict[str, object]) -> None:
    # Single-thread the BLAS/OpenMP pools inside each worker so that N workers x
    # M BLAS threads do not oversubscribe the node (the SLURM job exports
    # OMP_NUM_THREADS=cpus for the one-time assembly einsum). The QNM work is
    # curve_fit / small-SVD dominated, so 1 thread/worker is optimal here.
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS"):
        os.environ[_v] = "1"
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(1)
    except Exception:
        pass
    _EVALG.update(g)


def _eval_qnm_work(i: int) -> dict:
    g = _EVALG
    tau = g["tau"]; sc = g["scri"]; P = g["P"]; qnm = g["qnm"]
    a = float(P[i, 0]); qref = qnm[i]
    wL = float(qref[0]); tL = float(qref[2])
    row = {
        "i": i, "a_over_M": a, "r0": float(P[i, 1]), "w": float(P[i, 2]),
        "leaver_Mw": wL, "leaver_tau": tL,
        "field_rl2_prior": float(g["fl_prior"][i]),
        "field_rl2_hybrid": float(g["fl_hyb"][i]),
        "field_rl2_richardson": float(g["fl_rich"][i]),
        "qnm": {},
    }
    for name, RE, IM in (("prior", "up4_re", "up4_im"),
                         ("hybrid", "hyb_re", "hyb_im"),
                         ("richardson", "rich_re", "rich_im"),
                         ("fine", "fine_re", "fine_im")):
        q = extract_qnm_scri(g[RE][i, :, sc], g[IM][i, :, sc], a, tau, qref)
        row["qnm"][name] = {
            "Mw": q["Mw"], "tau": q["tau"],
            "Mw_err_pct": pct_err(q["Mw"], wL),
            "tau_err_pct": pct_err(q["tau"], tL),
            "spread_pct": q["spread_pct"], "sel_dist": q["sel_dist"],
            "n_methods": q["n_methods"],
        }
    return row


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def run_epoch(model, loader, device, optimizer=None) -> float:
    train = optimizer is not None
    model.train(train)
    total, count = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            pred = model(xb)
            loss = torch.mean((pred - yb) ** 2)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        bs = xb.shape[0]
        total += float(loss.item()) * bs
        count += bs
    return total / max(count, 1)


def train_model(model, Xtr, Ytr, Xva, Yva, cfg, device, out_dir) -> dict:
    tcfg = cfg["train"]
    epochs = int(tcfg.get("epochs", 150))
    lr = float(tcfg.get("lr", 1.0e-3))
    wd = float(tcfg.get("weight_decay", 1.0e-6))
    batch = int(tcfg.get("batch_size", 8))
    patience = int(tcfg.get("patience", 30))

    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)),
        batch_size=batch, shuffle=True, num_workers=0, pin_memory=(device == "cuda"),
        drop_last=False,
    )
    va_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xva), torch.from_numpy(Yva)),
        batch_size=batch, shuffle=False, num_workers=0, pin_memory=(device == "cuda"),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    hist = {"train_mse": [], "val_mse": [], "lr": []}
    best_val = float("inf")
    best_state = None
    best_epoch = -1
    since = 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        tr = run_epoch(model, tr_loader, device, optimizer)
        va = run_epoch(model, va_loader, device, None)
        scheduler.step()
        hist["train_mse"].append(tr)
        hist["val_mse"].append(va)
        hist["lr"].append(float(optimizer.param_groups[0]["lr"]))
        improved = va < best_val - 1e-12
        if improved:
            best_val = va
            best_state = {
                k: (v.detach().cpu().clone() if torch.is_tensor(v) else v)
                for k, v in model.state_dict().items()
            }
            best_epoch = ep
            since = 0
        else:
            since += 1
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(f"  epoch {ep:3d}/{epochs}  train {tr:.4e}  val {va:.4e}  "
                  f"best {best_val:.4e}@{best_epoch}  lr {hist['lr'][-1]:.2e}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if since >= patience:
            print(f"  early stop at epoch {ep} (no val improvement for {patience})", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    hist["best_val_mse"] = best_val
    hist["best_epoch"] = best_epoch
    hist["model_params"] = count_parameters(model)
    with open(os.path.join(out_dir, "history.json"), "w") as fh:
        json.dump(hist, fh, indent=2)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    return hist


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(model, X, device, batch=8) -> np.ndarray:
    model.eval()
    outs = []
    for lo in range(0, X.shape[0], batch):
        xb = torch.from_numpy(X[lo:lo + batch]).to(device)
        outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs, axis=0)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model, test, asm_test, split_test, cfg, device, out_dir) -> dict:
    tau = split_test["tau"]
    scri = int(split_test["scri_idx_fine"])
    P = np.asarray(split_test["P"])           # (N,3) a/M, r0, w
    qnm = np.asarray(split_test["qnm"])       # (N,3) Mw_R, Mw_I, tau/M
    N = P.shape[0]

    scale = asm_test["scale"]                 # (N,) scalar-norm or (N,Ntau) envelope-norm
    Yn = asm_test["Y"]                         # (N,2,Ntau,Nf) normalised Richardson target
    up4_re = asm_test["up4_re"]; up4_im = asm_test["up4_im"]
    fine_re = asm_test["fine_re"]; fine_im = asm_test["fine_im"]

    pred = predict(model, asm_test["X"], device, batch=int(cfg["train"].get("batch_size", 8)))
    # scalar norm -> scale is (N,); envelope norm -> (N, Ntau), a per-time scale.
    sB = scale[:, :, None] if scale.ndim == 2 else scale[:, None, None]
    # reconstruct physical fields
    hyb_re = up4_re + sB * pred[:, 0]
    hyb_im = up4_im + sB * pred[:, 1]
    rich_re = up4_re + sB * Yn[:, 0]
    rich_im = up4_im + sB * Yn[:, 1]

    # --- field rel-L2 (full 2-D) ---
    fl_prior = rel_l2_complex(up4_re, up4_im, fine_re, fine_im)
    fl_hyb = rel_l2_complex(hyb_re, hyb_im, fine_re, fine_im)
    fl_rich = rel_l2_complex(rich_re, rich_im, fine_re, fine_im)

    # --- QNM at scri (multi-method consensus ensemble; parallel) ---
    # The targeted ensemble is ~1-5 s/field, so 4 fields x N samples is ~2 h
    # serial -> fork a CPU pool over the pure-numpy extraction (the GPU forward
    # is already done, fields are plain arrays). ell-agnostic: the per-sample
    # Leaver target comes from the corpus ``qnm`` array.
    gvars = dict(
        tau=tau, scri=scri, P=P, qnm=qnm,
        up4_re=up4_re, up4_im=up4_im, hyb_re=hyb_re, hyb_im=hyb_im,
        rich_re=rich_re, rich_im=rich_im, fine_re=fine_re, fine_im=fine_im,
        fl_prior=fl_prior, fl_hyb=fl_hyb, fl_rich=fl_rich,
    )
    n_workers = int(cfg.get("eval", {}).get("qnm_workers", min(16, os.cpu_count() or 1)))
    t_qnm = time.time()
    if n_workers > 1:
        with Pool(processes=n_workers, initializer=_eval_qnm_init,
                  initargs=(gvars,)) as pool:
            rows = list(pool.imap_unordered(_eval_qnm_work, range(N)))
        rows.sort(key=lambda r: r["i"])
    else:
        _eval_qnm_init(gvars)
        rows = [_eval_qnm_work(i) for i in range(N)]
    print(f"  QNM consensus extracted {N}/{N} (ensemble, {n_workers} workers) "
          f"in {time.time()-t_qnm:.0f}s", flush=True)

    with open(os.path.join(out_dir, "per_sample.json"), "w") as fh:
        json.dump(rows, fh, indent=2)

    # --- aggregate ---
    def agg(key_path):
        vals = []
        for r in rows:
            v = r
            for k in key_path:
                v = v[k]
            if isinstance(v, float) and np.isfinite(v):
                vals.append(v)
        vals = np.array(vals) if vals else np.array([np.nan])
        return {"mean": float(np.nanmean(vals)), "median": float(np.nanmedian(vals)),
                "max": float(np.nanmax(vals)), "p90": float(np.nanpercentile(vals, 90))}

    aM = P[:, 0]
    spin_bins = [(0.0, 0.3), (0.3, 0.6), (0.6, 0.8), (0.8, 0.95)]
    by_spin = []
    for lo, hi in spin_bins:
        m = (aM >= lo) & (aM < hi)
        if not m.any():
            continue
        by_spin.append({
            "bin": [lo, hi], "n": int(m.sum()),
            "field_rl2_prior": float(np.mean(fl_prior[m])),
            "field_rl2_hybrid": float(np.mean(fl_hyb[m])),
            "field_rl2_richardson": float(np.mean(fl_rich[m])),
            "qnm_Mw_err_prior": float(np.nanmean([rows[i]["qnm"]["prior"]["Mw_err_pct"] for i in np.where(m)[0]])),
            "qnm_Mw_err_hybrid": float(np.nanmean([rows[i]["qnm"]["hybrid"]["Mw_err_pct"] for i in np.where(m)[0]])),
            "qnm_Mw_err_fine": float(np.nanmean([rows[i]["qnm"]["fine"]["Mw_err_pct"] for i in np.where(m)[0]])),
        })

    gate = cfg.get("acceptance", {"field_rl2": 0.05, "qnm_Mw": 0.01, "qnm_tau": 0.05})
    field_pass = float(np.median(fl_hyb)) <= gate["field_rl2"]
    qnm_pass = (agg(["qnm", "hybrid", "Mw_err_pct"])["median"] <= gate["qnm_Mw"] * 100
                and agg(["qnm", "hybrid", "tau_err_pct"])["median"] <= gate["qnm_tau"] * 100)

    report = {
        "n_test": N,
        "field_rl2": {
            "prior": {"mean": float(np.mean(fl_prior)), "median": float(np.median(fl_prior))},
            "hybrid": {"mean": float(np.mean(fl_hyb)), "median": float(np.median(fl_hyb))},
            "richardson": {"mean": float(np.mean(fl_rich)), "median": float(np.median(fl_rich))},
            "hybrid_over_prior_factor": float(np.mean(fl_prior) / max(np.mean(fl_hyb), 1e-30)),
        },
        "qnm_Mw_err_pct": {
            "prior": agg(["qnm", "prior", "Mw_err_pct"]),
            "hybrid": agg(["qnm", "hybrid", "Mw_err_pct"]),
            "richardson": agg(["qnm", "richardson", "Mw_err_pct"]),
            "fine": agg(["qnm", "fine", "Mw_err_pct"]),
        },
        "qnm_tau_err_pct": {
            "prior": agg(["qnm", "prior", "tau_err_pct"]),
            "hybrid": agg(["qnm", "hybrid", "tau_err_pct"]),
            "fine": agg(["qnm", "fine", "tau_err_pct"]),
        },
        "qnm_spread_pct": {
            "prior": agg(["qnm", "prior", "spread_pct"]),
            "hybrid": agg(["qnm", "hybrid", "spread_pct"]),
            "richardson": agg(["qnm", "richardson", "spread_pct"]),
            "fine": agg(["qnm", "fine", "spread_pct"]),
        },
        "qnm_sel_dist": {
            "prior": agg(["qnm", "prior", "sel_dist"]),
            "hybrid": agg(["qnm", "hybrid", "sel_dist"]),
            "richardson": agg(["qnm", "richardson", "sel_dist"]),
            "fine": agg(["qnm", "fine", "sel_dist"]),
        },
        "by_spin": by_spin,
        "acceptance_gate": {
            "field_rl2_threshold": gate["field_rl2"],
            "field_rl2_hybrid_median": float(np.median(fl_hyb)),
            "field_pass": bool(field_pass),
            "qnm_pass": bool(qnm_pass),
            "overall_pass": bool(field_pass and qnm_pass),
        },
    }
    with open(os.path.join(out_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    # bundle arrays needed by the plotting stage (canonical sample)
    fields = dict(
        tau=tau, scri=scri, P=P, qnm=qnm, scale=scale,
        up4_re=up4_re, up4_im=up4_im, hyb_re=hyb_re, hyb_im=hyb_im,
        fine_re=fine_re, fine_im=fine_im,
        fl_prior=fl_prior, fl_hyb=fl_hyb, fl_rich=fl_rich, rows=rows,
        sigma=split_test["sigma_fine"],
    )
    return report, fields


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def make_plots(hist, fields, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = os.path.join(out_dir, "figs")
    os.makedirs(figs, exist_ok=True)

    tau = fields["tau"]; sigma = fields["sigma"]; scri = fields["scri"]
    P = fields["P"]; aM = P[:, 0]
    ic = int(np.argmin(np.abs(aM - 0.7)))      # canonical mid-spin sample

    # 1) loss history
    plt.figure(figsize=(6, 4))
    plt.semilogy(hist["train_mse"], label="train")
    plt.semilogy(hist["val_mse"], label="val")
    plt.axvline(hist["best_epoch"] - 1, color="k", ls="--", lw=0.8, label=f"best@{hist['best_epoch']}")
    plt.xlabel("epoch"); plt.ylabel("normalised MSE"); plt.legend()
    plt.title("Kerr hybrid training"); plt.tight_layout()
    plt.savefig(os.path.join(figs, "hybrid_loss.png"), dpi=130); plt.close()

    # 2) pointwise |error| heatmaps prior vs hybrid (canonical), |psi| error
    def cmag_err(are, aim, bre, bim):
        return np.abs(np.sqrt(are ** 2 + aim ** 2) - np.sqrt(bre ** 2 + bim ** 2))
    ep = cmag_err(fields["up4_re"][ic], fields["up4_im"][ic], fields["fine_re"][ic], fields["fine_im"][ic])
    eh = cmag_err(fields["hyb_re"][ic], fields["hyb_im"][ic], fields["fine_re"][ic], fields["fine_im"][ic])
    vmax = max(ep.max(), eh.max(), 1e-30)
    from matplotlib.colors import LogNorm
    for name, E in (("baseline", ep), ("hybrid", eh)):
        plt.figure(figsize=(6, 4))
        plt.pcolormesh(sigma, tau, np.maximum(E, vmax * 1e-6), shading="auto",
                       cmap="magma_r", norm=LogNorm(vmin=vmax * 1e-5, vmax=vmax))
        plt.colorbar(label="|psi| abs error")
        plt.xlabel("sigma"); plt.ylabel("tau/M")
        plt.title(f"Kerr {name} pointwise error (a/M={aM[ic]:.3f})")
        plt.tight_layout()
        plt.savefig(os.path.join(figs, f"hybrid_pointwise_error_{name}.png"), dpi=130); plt.close()

    # 3) ringdown at scri: Re(psi) prior/hybrid/fine overlay (canonical)
    plt.figure(figsize=(7, 4))
    plt.plot(tau, fields["fine_re"][ic, :, scri], "k-", lw=1.4, label="fine FD")
    plt.plot(tau, fields["up4_re"][ic, :, scri], "C1--", lw=1.0, label="coarse prior")
    plt.plot(tau, fields["hyb_re"][ic, :, scri], "C0:", lw=1.4, label="hybrid")
    plt.xlabel("tau/M"); plt.ylabel("Re psi (scri)"); plt.legend()
    plt.title(f"Ringdown at scri (a/M={aM[ic]:.3f})"); plt.tight_layout()
    plt.savefig(os.path.join(figs, "hybrid_ringdown_scri.png"), dpi=130); plt.close()

    # 4) field rel-L2 vs spin (population)
    plt.figure(figsize=(6, 4))
    plt.scatter(aM, 100 * fields["fl_prior"], s=14, c="C1", label="coarse prior")
    plt.scatter(aM, 100 * fields["fl_hyb"], s=14, c="C0", label="hybrid")
    plt.scatter(aM, 100 * fields["fl_rich"], s=10, c="C2", marker="x", label="Richardson target")
    plt.axhline(5.0, color="k", ls="--", lw=0.8, label="5% gate")
    plt.xlabel("a/M"); plt.ylabel("field rel-L2 (%)"); plt.yscale("log"); plt.legend()
    plt.title("Field accuracy vs spin"); plt.tight_layout()
    plt.savefig(os.path.join(figs, "hybrid_field_vs_spin.png"), dpi=130); plt.close()

    # 5) QNM Mw error vs spin
    rows = fields["rows"]
    plt.figure(figsize=(6, 4))
    for key, c, lab in (("prior", "C1", "prior"), ("hybrid", "C0", "hybrid"), ("fine", "k", "fine")):
        ys = [r["qnm"][key]["Mw_err_pct"] for r in rows]
        plt.scatter(aM, ys, s=14, c=c, label=lab)
    plt.axhline(1.0, color="r", ls="--", lw=0.8, label="1% gate")
    plt.xlabel("a/M"); plt.ylabel("M*omega error (%)"); plt.yscale("log"); plt.legend()
    plt.title("QNM accuracy vs spin (at scri)"); plt.tight_layout()
    plt.savefig(os.path.join(figs, "hybrid_qnm_vs_spin.png"), dpi=130); plt.close()

    print(f"  wrote figures to {figs}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true", help="tiny dry run on CPU")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 0)))

    data_dir = cfg["data"]["dir"]
    target_mode = cfg["data"].get("target_mode", "richardson")
    prior_grid = cfg["data"].get("prior_grid")          # e.g. "k8" (decoupled) or None
    richardson_p = int(cfg["data"].get("richardson_p", 2))
    norm_mode = cfg["data"].get("norm_mode", "scalar")  # "envelope" = per-time tau fix
    envelope_floor = float(cfg["data"].get("envelope_floor", 1.0e-5))
    out_dir = cfg["logging"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() and not args.smoke else "cpu"
    print(f"[KERR-HYBRID] device={device} target_mode={target_mode} "
          f"prior_grid={prior_grid or 'k4(coupled)'} richardson_p={richardson_p} "
          f"norm_mode={norm_mode} out_dir={out_dir}", flush=True)

    # ---- load splits ----
    t0 = time.time()
    tr = load_split(os.path.join(data_dir, "dataset_train.npz"), "train")
    va = load_split(os.path.join(data_dir, "dataset_val.npz"), "val")
    te = load_split(os.path.join(data_dir, "dataset_test.npz"), "test")
    print(f"[KERR-HYBRID] loaded splits in {time.time()-t0:.0f}s "
          f"(train {tr['P'].shape[0]}, val {va['P'].shape[0]}, test {te['P'].shape[0]})", flush=True)

    if args.smoke:
        # tiny sample count + halve the tau axis so the full train/eval/plot
        # workflow completes on CPU; the production path is untouched.
        for d, n in ((tr, 8), (va, 4), (te, 4)):
            for k in list(d):
                if k.startswith("psi_") or k in ("P", "qnm"):
                    d[k] = d[k][:n]
            d["tau"] = d["tau"][::2]
            for k in list(d):
                if k.startswith("psi_"):
                    d[k] = d[k][:, ::2, :]
        cfg["fno"]["modes_tau"] = 16
        cfg["fno"]["modes_sigma"] = 12
        cfg["fno"]["hidden_channels"] = 16

    # ---- upsample matrices (once) ----
    W_k4 = build_upsample_matrix(tr["sigma_k4"], tr["sigma_fine"])
    W_k2 = build_upsample_matrix(tr["sigma_k2"], tr["sigma_fine"])
    # Decoupled prior: input grid (k8, N=101) is separate from the k2/k4
    # Richardson rungs so the cheap headroom-bearing prior is upsampled with its
    # own matrix while the order-p label is built on the finer pair.
    W_prior = (build_upsample_matrix(tr[f"sigma_{prior_grid}"], tr["sigma_fine"])
               if prior_grid else None)

    # ---- assemble ----
    t0 = time.time()
    _asm_kw = dict(W_prior=W_prior, prior_key=(prior_grid or "k8"),
                   richardson_p=richardson_p, norm_mode=norm_mode,
                   envelope_floor=envelope_floor)
    asm_tr = assemble(tr, W_k4, W_k2, target_mode=target_mode, return_eval=False, **_asm_kw)
    asm_va = assemble(va, W_k4, W_k2, target_mode=target_mode, return_eval=False, **_asm_kw)
    asm_te = assemble(te, W_k4, W_k2, target_mode=target_mode, return_eval=True, **_asm_kw)
    print(f"[KERR-HYBRID] assembled tensors in {time.time()-t0:.0f}s  "
          f"X_train {asm_tr['X'].shape}", flush=True)
    # free raw fields we no longer need (keep test meta)
    del tr, va

    # ---- model ----
    model = build_hybrid_fno(cfg).to(device)
    print(f"[KERR-HYBRID] model params = {count_parameters(model):,}  "
          f"modes=({cfg['fno'].get('modes_tau',64)},{cfg['fno'].get('modes_sigma',24)})  "
          f"hidden={cfg['fno'].get('hidden_channels',48)}", flush=True)

    if args.smoke:
        cfg["train"]["epochs"] = 2
        cfg["train"]["patience"] = 5

    # ---- train ----
    hist = train_model(model, asm_tr["X"], asm_tr["Y"], asm_va["X"], asm_va["Y"],
                       cfg, device, out_dir)
    del asm_tr, asm_va

    # ---- evaluate ----
    print("[KERR-HYBRID] evaluating test split ...", flush=True)
    report, fields = evaluate(model, te, asm_te, te, cfg, device, out_dir)

    # ---- plots ----
    make_plots(hist, fields, out_dir)

    # ---- summary ----
    fr = report["field_rl2"]; gq = report["acceptance_gate"]
    print("\n==================== KERR HYBRID SUMMARY ====================", flush=True)
    print(f" field rel-L2 (median): prior {fr['prior']['median']*100:.2f}%  "
          f"-> hybrid {fr['hybrid']['median']*100:.2f}%  "
          f"(Richardson target {fr['richardson']['median']*100:.2f}%)", flush=True)
    print(f" field factor (mean): {fr['hybrid_over_prior_factor']:.1f}x", flush=True)
    qh = report["qnm_Mw_err_pct"]
    print(f" QNM M*omega err (median %): prior {qh['prior']['median']:.3f}  "
          f"hybrid {qh['hybrid']['median']:.3f}  fine {qh['fine']['median']:.3f}", flush=True)
    print(f" acceptance gate: field_pass={gq['field_pass']} qnm_pass={gq['qnm_pass']} "
          f"OVERALL={gq['overall_pass']}", flush=True)
    print("============================================================", flush=True)


if __name__ == "__main__":
    main()

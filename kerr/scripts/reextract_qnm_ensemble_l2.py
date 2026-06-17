"""Re-extract the FINISHED ell=2 hybrid QNMs with the m1-m5 ENSEMBLE.

The production job scored the QNM with the SINGLE-method ``extract_fundamental``
(one complex-phase / one damped-cosine fit). That estimator cannot self-diagnose
a bad fit, and the per-spin means hinted it was injecting phase error. This
script reloads the SAME trained model + test corpus, reconstructs the four scri
waveforms per sample (prior up4 / hybrid / Richardson / fine), and re-scores the
QNM with BOTH the single method (to reproduce the report) and the multi-method
consensus (``extract_qnm_kerr_ensemble``). It writes a side-by-side table so we
can see whether the hybrid QNM was genuinely worse or a single-method artifact.

No training, no overwrite of the production report: writes a new
``report_ensemble_qnm.json`` next to the model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from kerr.src.hybrid_data_pipe import load_split, build_upsample_matrix, assemble
from kerr.src.hybrid_fno import build_hybrid_fno, count_parameters
from kerr.scripts.kv3_qnm import extract_fundamental
from kerr.src.qnm_ensemble_kerr import extract_qnm_kerr_ensemble
import yaml


def load_config(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def pct_err(val, ref):
    if not np.isfinite(val) or abs(ref) < 1e-30:
        return float("nan")
    return abs(val - ref) / abs(ref) * 100.0


def _single(tau, yc, a, tau_ref, tau_final):
    out = extract_fundamental(tau, yc, is_real=(abs(a) < 1e-6),
                              tau_ref=tau_ref, tau_final=tau_final)
    return float(out.get("omega", np.nan))


def _ens(tau, yc, a, tau_ref, tau_final):
    out = extract_qnm_kerr_ensemble(tau, yc, a_over_M=a, tau_ref=tau_ref,
                                    tau_final=tau_final)
    om = float(out.get("omega", np.nan))
    spr = float(out.get("omega_std", np.nan))
    return om, spr, int(out.get("n_omega", 0))


# Worker globals (filled by _init so the big arrays are inherited via fork).
_G = {}


def _init(tau, scri, P, qnm, prior_re, prior_im, hyb_re, hyb_im,
          rich_re, rich_im, fine_re, fine_im):
    _G.update(dict(tau=tau, scri=scri, P=P, qnm=qnm,
                   prior_re=prior_re, prior_im=prior_im,
                   hyb_re=hyb_re, hyb_im=hyb_im,
                   rich_re=rich_re, rich_im=rich_im,
                   fine_re=fine_re, fine_im=fine_im))


def _work(i):
    tau = _G["tau"]; sc = _G["scri"]
    a = float(_G["P"][i, 0]); wL = float(_G["qnm"][i, 0])
    tau_ref = float(_G["qnm"][i, 2]); tau_final = float(tau[-1])
    res = {"i": i, "a": a, "leaver": wL}
    for name, RE, IM in (("prior", "prior_re", "prior_im"),
                         ("hybrid", "hyb_re", "hyb_im"),
                         ("richardson", "rich_re", "rich_im"),
                         ("fine", "fine_re", "fine_im")):
        yc = (_G[RE][i, :, sc].astype(np.float64)
              + 1j * _G[IM][i, :, sc].astype(np.float64))
        sm = _single(tau, yc, a, tau_ref, tau_final)
        em, es, en = _ens(tau, yc, a, tau_ref, tau_final)
        res[name] = {
            "single_Mw": sm, "single_err": pct_err(sm, wL),
            "ens_Mw": em, "ens_err": pct_err(em, wL),
            "ens_spread_pct": (es / abs(em) * 100.0) if (np.isfinite(em) and em) else float("nan"),
            "ens_n": en,
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="kerr/configs/hybrid_kerr.yaml")
    ap.add_argument("--out_dir", default="kerr/outputs/phase_c/fno_hybrid_kerr")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = cfg["data"]["dir"]
    target_mode = cfg["data"].get("target_mode", "richardson")

    t0 = time.time()
    te = load_split(os.path.join(data_dir, "dataset_test.npz"), "test")
    # The sigma axes are shared across splits (nested grids), so the upsample
    # matrices are built from the TEST split -- no need to load the large train
    # fields just for the axes.
    W_k4 = build_upsample_matrix(te["sigma_k4"], te["sigma_fine"])
    W_k2 = build_upsample_matrix(te["sigma_k2"], te["sigma_fine"])
    asm = assemble(te, W_k4, W_k2, target_mode=target_mode, return_eval=True)
    print(f"[reextract] loaded+assembled test in {time.time()-t0:.0f}s", flush=True)

    tau = te["tau"]; scri = int(te["scri_idx_fine"])
    P = np.asarray(te["P"]); qnm = np.asarray(te["qnm"])
    N = P.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_hybrid_fno(cfg)
    sd = torch.load(os.path.join(args.out_dir, "model.pt"), map_location="cpu",
                    weights_only=False)
    model.load_state_dict(sd)
    model.eval()
    model.to(device)
    print(f"[reextract] model loaded ({count_parameters(model):,} params) on {device}", flush=True)

    scale = asm["scale"]; Yn = asm["Y"]
    up4_re = asm["up4_re"]; up4_im = asm["up4_im"]
    fine_re = asm["fine_re"]; fine_im = asm["fine_im"]

    torch.set_num_threads(max(1, args.workers))   # speed the CPU forward pass
    t_fwd = time.time()
    with torch.no_grad():
        preds = []
        for lo in range(0, N, 8):
            xb = torch.from_numpy(asm["X"][lo:lo + 8]).to(device)
            preds.append(model(xb).cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    print(f"[reextract] forward pass in {time.time()-t_fwd:.0f}s", flush=True)
    sB = scale[:, None, None]
    hyb_re = up4_re + sB * pred[:, 0]; hyb_im = up4_im + sB * pred[:, 1]
    rich_re = up4_re + sB * Yn[:, 0]; rich_im = up4_im + sB * Yn[:, 1]
    print(f"[reextract] reconstructed fields; extracting QNM "
          f"(single + ensemble) for {N} samples on {args.workers} workers ...",
          flush=True)

    t0 = time.time()
    with Pool(processes=args.workers, initializer=_init,
              initargs=(tau, scri, P, qnm, up4_re, up4_im, hyb_re, hyb_im,
                        rich_re, rich_im, fine_re, fine_im)) as pool:
        rows = []
        for k, r in enumerate(pool.imap_unordered(_work, range(N)), 1):
            rows.append(r)
            if k % 25 == 0:
                print(f"  {k}/{N}", flush=True)
    rows.sort(key=lambda r: r["i"])
    print(f"[reextract] extraction done in {time.time()-t0:.0f}s", flush=True)

    # --- aggregate: median err per field, single vs ensemble ---
    def med(field, key):
        v = [r[field][key] for r in rows if np.isfinite(r[field][key])]
        return float(np.median(v)) if v else float("nan")

    fields = ["prior", "hybrid", "richardson", "fine"]
    summary = {}
    for f in fields:
        summary[f] = {
            "single_err_median": med(f, "single_err"),
            "ens_err_median": med(f, "ens_err"),
            "ens_spread_median": med(f, "ens_spread_pct"),
        }

    aM = P[:, 0]
    bins = [(0.0, 0.3), (0.3, 0.6), (0.6, 0.8), (0.8, 0.95)]
    by_spin = []
    for lo, hi in bins:
        idx = [r["i"] for r in rows if lo <= r["a"] < hi]
        if not idx:
            continue
        row = {"bin": [lo, hi], "n": len(idx)}
        for f in fields:
            se = [rows[i][f]["single_err"] for i in idx if np.isfinite(rows[i][f]["single_err"])]
            ee = [rows[i][f]["ens_err"] for i in idx if np.isfinite(rows[i][f]["ens_err"])]
            row[f] = {
                "single_mean": float(np.mean(se)) if se else float("nan"),
                "ens_mean": float(np.mean(ee)) if ee else float("nan"),
            }
        by_spin.append(row)

    report = {"n_test": N, "summary": summary, "by_spin": by_spin}
    out = os.path.join(args.out_dir, "report_ensemble_qnm.json")
    with open(out, "w") as fh:
        json.dump({"report": report, "per_sample": rows}, fh, indent=2)

    # --- print ---
    print("\n================ ENSEMBLE vs SINGLE QNM (ell=2 hybrid) ================")
    print(f"{'field':>11} | {'single med':>11} {'ens med':>9} {'ens spread':>10}")
    print("-" * 52)
    for f in fields:
        s = summary[f]
        print(f"{f:>11} | {s['single_err_median']:10.3f}% {s['ens_err_median']:8.3f}% "
              f"{s['ens_spread_median']:9.3f}%")
    print("\n--- by spin (MEAN % err, single -> ensemble) ---")
    print(f"{'bin':>12} {'n':>3} | "
          + " ".join(f"{f[:5]:>16}" for f in fields))
    for r in by_spin:
        cells = " ".join(
            f"{r[f]['single_mean']:6.2f}->{r[f]['ens_mean']:6.2f}%" for f in fields)
        print(f"  [{r['bin'][0]:.1f},{r['bin'][1]:.2f}) {r['n']:3d} | {cells}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

"""Population QNM eval under the printed protocol (Algorithm 1 / Sec. methods).

Identical data flow to scripts/eval_hybrid_sw.py (same dataset, model,
reconstruction), but the plateau scans use the report's printed convention
-- the central-package defaults -- instead of the narrower grids the older
eval hardcoded:

    M4: t0 in [10, 25], 16 starts, plateau width 8, t_end = 50
    M5: t0 in [10, 25] x t_end in [30, 50], 10 x 6 grid, plateau 5 x 3

M1--M3 are single-window methods on [10, 50], unchanged. Observer xq = 10 M.
Reports per-method medians over the test population for hybrid / coarse-up
baseline / fine FD, plus an explicitly-labelled oracle best-of-suite bound
(selected using the true answer; diagnostic only, never a primary value).

Additive script; core code untouched.

Usage:
    python scripts/eval_hybrid_protocol1.py --config configs/hybrid_sw_gate_s1em3.yaml
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import warnings
from typing import Any, Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.hybrid_data_pipe import assemble_split
from src.hybrid_dataset import load_dataset
from src.hybrid_fno import build_hybrid_fno
from src.qnm import (
    qnm_method_1,
    qnm_method_2,
    qnm_method_3_esprit,
    qnm_method_4_window_scan,
    qnm_method_5_2d_scan,
    percentage_errors,
    theory_ref,
)

warnings.filterwarnings("ignore")

XQ = 10.0
T_START, T_END = 10.0, 50.0
METHODS = ("M1", "M2", "M3", "M4", "M5")


def _safe(fn, *args, **kwargs) -> Dict[str, float]:
    try:
        r = fn(*args, **kwargs)
        return {"omega": float(r.get("omega", float("nan"))),
                "tau": float(r.get("tau", float("nan")))}
    except Exception:
        return {"omega": float("nan"), "tau": float("nan")}


def _protocol1(t: np.ndarray, y: np.ndarray, potential: str, ell: int,
               M: float, t_start: float, t_end: float
               ) -> Dict[str, Dict[str, float]]:
    """M1-M5 with the printed (Algorithm-1) scan grids.

    For the canonical window (t_start=10, t_end=50) the M4/M5 grids are exactly
    Algorithm 1. A larger t_end (e.g. 80 for the t=100 corpus) keeps the same
    t0 scan [10,25] but lets the fit reach further into the ringdown tail --
    an explicitly-labelled extended-window variant, never the primary number.
    """
    raw = {
        "M1": _safe(qnm_method_1, t, y, t_start, t_end),
        "M2": _safe(qnm_method_2, t, y, t_start, t_end),
        "M3": _safe(qnm_method_3_esprit, t, y, t_start, t_end, K=4),
        "M4": _safe(qnm_method_4_window_scan, t, y,
                    t_start_min=10.0, t_start_max=25.0, t_end=t_end,
                    potential=potential, ell=ell),        # defaults: 16, w=8
        "M5": _safe(qnm_method_5_2d_scan, t, y,
                    t_start_min=10.0, t_start_max=25.0,
                    t_end_min=30.0, t_end_max=t_end,
                    potential=potential, ell=ell),        # defaults: 10x6, 5x3
    }
    out: Dict[str, Dict[str, float]] = {}
    for name, res in raw.items():
        err = percentage_errors(res, potential=potential, ell=ell, M=M)
        out[name] = {"omega_pct_err": err.get("omega_pct_err", float("nan")),
                     "tau_pct_err": err.get("tau_pct_err", float("nan"))}
    return out


# --- multiprocessing worker: one BH, three sources -------------------------
# Pure numpy/scipy (no torch) so it pickles cleanly under Windows spawn.
_WT: Dict[str, Any] = {}


def _winit(t_fine, potential, ell, t_start, t_end) -> None:
    _WT["t"] = t_fine
    _WT["pot"] = potential
    _WT["ell"] = ell
    _WT["t_start"] = t_start
    _WT["t_end"] = t_end


def _worker(task):
    i, M_i, y_hyb, y_base, y_fine = task
    t, pot, ell = _WT["t"], _WT["pot"], _WT["ell"]
    ts, te = _WT["t_start"], _WT["t_end"]
    rec: Dict[str, Any] = {"i": i, "M": M_i}
    for tag, y in (("hyb", y_hyb), ("base", y_base), ("fine", y_fine)):
        rec[tag] = _protocol1(t, y, pot, ell, M_i, ts, te)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hybrid_sw_gate_s1em3.yaml")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--ell", type=int, default=2)
    ap.add_argument("--potential", default="zerilli")
    ap.add_argument("--t_start", type=float, default=10.0)
    ap.add_argument("--t_end", type=float, default=50.0,
                    help="fit-window end (default 50 = canonical Algorithm 1;"
                         " use 80 for the extended t=100 corpus variant).")
    ap.add_argument("--procs", type=int, default=12,
                    help="worker processes for the extraction loop (default 12,"
                         " capped below cpu_count; thermal headroom on 24-core).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg["logging"]["out_dir"]
    ckpt_path = os.path.join(out_dir, "model_best.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[P1] device={device}  ckpt={ckpt_path}")

    splits, grid, meta = load_dataset(cfg["dataset"]["path"])
    test = splits["test"]
    n_total = test["Phi_fine"].shape[0]
    n_eval = n_total if args.n <= 0 else min(args.n, n_total)
    print(f"[P1] test: {n_total} samples, evaluating {n_eval}")

    t0 = time.time()
    X_te, Y_te, up_te = assemble_split(
        test, grid.x_coarse, grid.t_coarse, grid.x_fine, grid.t_fine,
    )
    print(f"[P1] assembled in {time.time()-t0:.1f}s")

    model = build_hybrid_fno(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    if isinstance(state, dict):
        state.pop("_metadata", None)
    model.load_state_dict(state)
    model.eval()

    delta_pred = np.empty_like(Y_te)
    with torch.no_grad():
        for i0 in range(0, n_eval, 4):
            i1 = min(i0 + 4, n_eval)
            xb = torch.from_numpy(X_te[i0:i1]).to(device)
            delta_pred[i0:i1] = model(xb).cpu().numpy()

    psi_fine = test["Phi_fine"][:n_eval].astype(np.float64)
    psi_up = up_te[:n_eval].astype(np.float64)
    psi_hyb = psi_up + delta_pred[:n_eval, 0].astype(np.float64)

    t_fine = grid.t_fine.astype(np.float64)
    x_fine = grid.x_fine.astype(np.float64)
    ix = int(np.argmin(np.abs(x_fine - XQ)))
    ref = theory_ref(args.potential, args.ell)
    print(f"[P1] xq={XQ:g}  window [{args.t_start:g},{args.t_end:g}]  "
          f"theory M*omega={ref['omega']:.4f} tau/M={ref['tau']:.4f}")

    P = test["P"][:n_eval]
    tasks = [(
        i, float(P[i, 0]),
        np.ascontiguousarray(psi_hyb[i, :, ix]),
        np.ascontiguousarray(psi_up[i, :, ix]),
        np.ascontiguousarray(psi_fine[i, :, ix]),
    ) for i in range(n_eval)]

    nproc = max(1, min(args.procs, mp.cpu_count()))
    print(f"[P1] extracting {n_eval} samples x 3 sources on {nproc} processes ...")
    t0 = time.time()
    with mp.Pool(nproc, initializer=_winit,
                 initargs=(t_fine, args.potential, args.ell,
                           args.t_start, args.t_end)) as pool:
        per_sample = pool.map(_worker, tasks, chunksize=1)
    per_sample.sort(key=lambda r: r["i"])
    print(f"[P1] extraction done in {time.time()-t0:.0f}s")

    accum: Dict[str, List[float]] = {}
    for rec in per_sample:
        for tag in ("hyb", "base", "fine"):
            res = rec[tag]
            for m in METHODS:
                for k in ("omega_pct_err", "tau_pct_err"):
                    accum.setdefault(f"{tag}_{m}_{k}", []).append(res[m][k])
            # oracle best-of-suite (uses the truth; diagnostic bound only)
            for k in ("omega_pct_err", "tau_pct_err"):
                vals = [res[m][k] for m in METHODS if np.isfinite(res[m][k])]
                accum.setdefault(f"{tag}_ORACLE_{k}", []).append(
                    min(vals) if vals else float("nan"))

    med = {k: float(np.nanmedian(np.asarray(v, dtype=float)))
           for k, v in accum.items()}

    print(f"\n[P1] === population medians, xq={XQ:g}, "
          f"window [{args.t_start:g},{args.t_end:g}] ===")
    print("method  | hybrid w%/tau%      | baseline w%/tau%    | fine w%/tau%")
    for m in list(METHODS) + ["ORACLE"]:
        row = [f"{m:7s}"]
        for tag in ("hyb", "base", "fine"):
            w = med[f"{tag}_{m}_omega_pct_err"]
            ta = med[f"{tag}_{m}_tau_pct_err"]
            row.append(f"{w:7.3f}/{ta:8.3f}")
        print("  ".join(row))
    print("(ORACLE = best across M1-M5 selected using the true answer; "
          "diagnostic bound, not a primary value.)")

    te_tag = f"te{args.t_end:g}"
    out = {"n_eval": n_eval, "xq": XQ, "window": [args.t_start, args.t_end],
           "protocol": f"Algorithm-1 grids (M4 16 starts w=8 t0[10,25]; "
                       f"M5 10x6 t0[10,25] te[30,{args.t_end:g}])",
           "medians": med}
    out_path = os.path.join(out_dir, "eval", f"protocol1_xq10_{te_tag}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    ps_path = os.path.join(out_dir, "eval",
                           f"protocol1_xq10_{te_tag}_per_sample.json")
    with open(ps_path, "w") as fh:
        json.dump(per_sample, fh, indent=2)
    print(f"\n[P1] wrote {out_path}")


if __name__ == "__main__":
    main()

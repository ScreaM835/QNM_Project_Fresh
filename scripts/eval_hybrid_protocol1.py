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
T_START, T_END = 10.0, 50.0   # legacy defaults; actual window comes from argv
METHODS = ("M1", "M2", "M3", "M4", "M5")


def _safe(fn, *args, **kwargs) -> Dict[str, float]:
    try:
        r = fn(*args, **kwargs)
        return {"omega": float(r.get("omega", float("nan"))),
                "tau": float(r.get("tau", float("nan")))}
    except Exception:
        return {"omega": float("nan"), "tau": float("nan")}


def _protocol1(t: np.ndarray, y: np.ndarray, potential: str, ell: int,
               M: float, t_start: float, t_end_m4: float
               ) -> Dict[str, Dict[str, float]]:
    """M1-M5 with the printed (Algorithm-1) scan grids.

    M4 (1-D scan) uses a FIXED end time ``t_end_m4`` (the usable-signal cutoff:
    50 for the t=50 corpus, 90 for t=100). M5 (2-D scan) always explores the
    FULL domain -- its end-time axis runs to the last available time t[-1] --
    and self-selects the plateau rectangle, so it is never hand-windowed.
    M1-M3 (single-window) use [t_start, t_end_m4].
    """
    t_end_m5 = float(t[-1])   # M5 explores the full time domain
    raw = {
        "M1": _safe(qnm_method_1, t, y, t_start, t_end_m4),
        "M2": _safe(qnm_method_2, t, y, t_start, t_end_m4),
        "M3": _safe(qnm_method_3_esprit, t, y, t_start, t_end_m4, K=4),
        "M4": _safe(qnm_method_4_window_scan, t, y,
                    t_start_min=10.0, t_start_max=25.0, t_end=t_end_m4,
                    potential=potential, ell=ell),        # defaults: 16, w=8
        "M5": _safe(qnm_method_5_2d_scan, t, y,
                    t_start_min=10.0, t_start_max=25.0,
                    t_end_min=30.0, t_end_max=t_end_m5,
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


def _winit(t_fine, potential, ell, t_start, t_end_m4) -> None:
    _WT["t"] = t_fine
    _WT["pot"] = potential
    _WT["ell"] = ell
    _WT["t_start"] = t_start
    _WT["t_end_m4"] = t_end_m4


def _worker(task):
    i, M_i, y_hyb, y_base, y_fine = task
    t, pot, ell = _WT["t"], _WT["pot"], _WT["ell"]
    ts, te = _WT["t_start"], _WT["t_end_m4"]
    rec: Dict[str, Any] = {"i": i, "M": M_i}
    for tag, y in (("hyb", y_hyb), ("base", y_base), ("fine", y_fine)):
        rec[tag] = _protocol1(t, y, pot, ell, M_i, ts, te)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hybrid_sw_gate_s1em3.yaml")
    ap.add_argument("--dataset-config", default="configs/hybrid_sw_dataset.yaml",
                    help="dataset config giving the fine grid/domain for the "
                         "canonical-BH solve (use *_t100.yaml for t=100).")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--ell", type=int, default=2)
    ap.add_argument("--potential", default="zerilli")
    ap.add_argument("--t_start", type=float, default=10.0)
    ap.add_argument("--t_end_m4", type=float, default=50.0,
                    help="M4 fixed fit-window end (50 = canonical Algorithm 1;"
                         " use 90 for the t=100 corpus). M5 always scans the"
                         " full domain regardless.")
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
    print(f"[P1] xq={XQ:g}  M4 end={args.t_end_m4:g}  M5 full-domain to "
          f"{t_fine[-1]:g}  theory M*omega={ref['omega']:.4f} "
          f"tau/M={ref['tau']:.4f}")

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
                           args.t_start, args.t_end_m4)) as pool:
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

    med = {k: float(np.nanmedian(np.asarray(v, dtype=float)))
           for k, v in accum.items()}

    # ---- canonical BH (M,x0,sigma)=(1,4,5), same as the PINNs ---------------
    os.environ.setdefault("MPLBACKEND", "Agg")
    import make_hybrid_paper_figs as mh
    mh.CONFIG = args.config
    mh.DATASET_CFG = args.dataset_config
    print("[P1] solving canonical BH (1,4,5) ...")
    c = mh._solve_canonical()
    xc, tc = c["x"], c["t"]
    ixc = int(np.argmin(np.abs(xc - XQ)))
    canon = {}
    for tag, key in (("hyb", "psi_hybrid"), ("base", "psi_coarse_up"),
                     ("fine", "psi_fine")):
        canon[tag] = _protocol1(tc, c[key][:, ixc].astype(np.float64),
                                args.potential, args.ell, 1.0,
                                args.t_start, args.t_end_m4)

    canon_field = {k: float(c[k]) for k in
                   ("rmsd_hybrid", "rmsd_baseline", "rl2_hybrid", "rl2_baseline")}
    print(f"[P1] canonical field RMSD: hybrid={canon_field['rmsd_hybrid']:.3e}  "
          f"baseline={canon_field['rmsd_baseline']:.3e}  "
          f"rL2 hybrid={canon_field['rl2_hybrid']*100:.3f}%")

    def _print_table(title, getter):
        print(f"\n[P1] === {title} ===")
        print("method  | hybrid w%/tau%      | baseline w%/tau%    | fine w%/tau%")
        for m in METHODS:
            row = [f"{m:7s}"]
            for tag in ("hyb", "base", "fine"):
                w, ta = getter(tag, m)
                row.append(f"{w:7.3f}/{ta:8.3f}")
            print("  ".join(row))

    _print_table(
        f"population medians, xq={XQ:g} (M4 end {args.t_end_m4:g}, M5 full-domain)",
        lambda tag, m: (med[f"{tag}_{m}_omega_pct_err"],
                        med[f"{tag}_{m}_tau_pct_err"]))
    _print_table(
        f"canonical BH (1,4,5), xq={XQ:g} (M4 end {args.t_end_m4:g}, M5 full-domain)",
        lambda tag, m: (canon[tag][m]["omega_pct_err"],
                        canon[tag][m]["tau_pct_err"]))

    te_tag = f"m4end{args.t_end_m4:g}"
    out = {"n_eval": n_eval, "xq": XQ,
           "t_start": args.t_start, "t_end_m4": args.t_end_m4,
           "t_end_m5": float(t_fine[-1]),
           "protocol": "M4 1-D scan t0[10,25] fixed end; "
                       "M5 2-D scan t0[10,25] x te[30,domain-end] self-select",
           "population_medians": med,
           "canonical": canon,
           "canonical_field": canon_field}
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

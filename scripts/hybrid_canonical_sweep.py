"""Canonical-BH observer-position sweep for the hybrid surrogate.

Additive analysis script: reuses the canonical-field solver of
scripts/make_hybrid_paper_figs.py and the M1-M5 extraction stack of
scripts/eval_hybrid_sw.py, at multiple observer positions. No core code
is modified.

Usage:
    python scripts/hybrid_canonical_sweep.py \
        --config configs/hybrid_sw_gate_s1em3.yaml --xq 2 5 10 15 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import make_hybrid_paper_figs as mh          # noqa: E402
from eval_hybrid_sw import _all_methods      # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hybrid_sw_gate_s1em3.yaml")
    ap.add_argument("--dataset-config", default="configs/hybrid_sw_dataset.yaml")
    ap.add_argument("--xq", type=float, nargs="+", default=[2.0, 5.0, 10.0, 15.0, 20.0])
    ap.add_argument("--t_start", type=float, default=10.0)
    ap.add_argument("--t_end", type=float, default=50.0)
    args = ap.parse_args()

    mh.CONFIG = args.config
    mh.DATASET_CFG = args.dataset_config

    c = mh._solve_canonical()
    x, t = c["x"], c["t"]
    sources = {"hybrid": c["psi_hybrid"],
               "baseline": c["psi_coarse_up"],
               "fine": c["psi_fine"]}

    results = {"canonical": {"M": mh.M_CANON, "x0": mh.X0_CANON, "sigma": mh.SIGMA_CANON},
               "window": [args.t_start, args.t_end],
               "field": {k: c[k] for k in
                         ("rmsd_hybrid", "rmsd_baseline", "rl2_hybrid", "rl2_baseline")}}

    for xq in args.xq:
        ix = int(np.argmin(np.abs(x - xq)))
        key = f"xq{xq:g}"
        results[key] = {"x_actual": float(x[ix])}
        for label, field in sources.items():
            res = _all_methods(t, field[:, ix], args.t_start, args.t_end,
                               potential="zerilli", ell=2, M=1.0)
            results[key][label] = res

    # ---- print table -------------------------------------------------------
    methods = ["M1", "M2", "M3", "M4", "M5"]
    for xq in args.xq:
        key = f"xq{xq:g}"
        print(f"\n=== canonical BH, xq = {xq:g} M  (window [{args.t_start:g},{args.t_end:g}] M) ===")
        hdr = "method | " + " | ".join(f"{s:>8s} w%  {s:>8s} tau%" for s in sources)
        print(hdr)
        for m in methods:
            row = [m]
            for s in sources:
                d = results[key][s][m]
                row.append(f"{d['omega_pct_err']:9.3f}  {d['tau_pct_err']:12.3f}")
            print("  ".join(row))

    out_dir = None
    try:
        from src.config import load_config
        out_dir = load_config(args.config)["logging"]["out_dir"]
    except Exception:
        pass
    if out_dir:
        os.makedirs(os.path.join(out_dir, "eval"), exist_ok=True)
        out_path = os.path.join(out_dir, "eval", "canonical_sweep.json")
        with open(out_path, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\n[SWEEP] wrote {out_path}")


if __name__ == "__main__":
    main()

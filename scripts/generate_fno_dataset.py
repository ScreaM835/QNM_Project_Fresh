"""Generate the multi-parameter Zerilli dataset for FNO training.

Sweeps (M, x0, sigma) and runs the existing FD solver for each draw.

Usage:
    python scripts/generate_fno_dataset.py --config configs/fno_zerilli_l2.yaml
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import time

import numpy as np

from src.config import load_config
from src.utils import ensure_dir
from src.fno_dataset import (
    generate_split,
    sample_params,
    save_dataset,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--quick", action="store_true",
                    help="Override n_train/val/test to (8/2/2) for a smoke test.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    name = cfg["experiment"]["name"]
    dcfg = cfg["dataset"]

    n_train = 8 if args.quick else int(dcfg["n_train"])
    n_val = 2 if args.quick else int(dcfg["n_val"])
    n_test = 2 if args.quick else int(dcfg["n_test"])

    rng = np.random.default_rng(int(dcfg["seed"]))
    sweep = dcfg["sweep"]
    train_p = sample_params(rng, sweep, n_train)
    val_p   = sample_params(rng, sweep, n_val)
    test_p  = sample_params(rng, sweep, n_test)

    stride_t = int(dcfg.get("stride_t", 1))
    stride_x = int(dcfg.get("stride_x", 1))

    out_dir = os.path.join("outputs", "fno", name)
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "dataset.npz")

    splits = {}
    grid = None
    t0 = time.time()
    for split_name, params in [("train", train_p), ("val", val_p), ("test", test_p)]:
        print(f"[fno-data] generating {split_name} ({len(params)} samples)")
        X, Y, P, g, V = generate_split(
            cfg, params,
            stride_t=stride_t, stride_x=stride_x,
            progress_prefix=f"[{split_name}]",
        )
        splits[split_name] = {"X": X, "Y": Y, "P": P, "V": V}
        grid = g
        print(f"[fno-data]   X{X.shape}  Y{Y.shape}  P{P.shape}  V{V.shape}")

    meta = {
        "name": name,
        "stride_t": stride_t,
        "stride_x": stride_x,
        "fd_dx": float(cfg["fd"]["dx"]),
        "fd_dt": float(cfg["fd"]["dt"]),
        "potential": cfg["physics"]["potential"],
        "l": int(cfg["physics"]["l"]),
        "xq": float(cfg["evaluation"]["xq"]),
        "qnm_t_start": float(cfg["qnm"]["t_start"]),
        "qnm_t_end": float(cfg["qnm"]["t_end"]),
        "sweep": sweep,
    }
    save_dataset(out_path, splits, grid, meta)
    print(f"[fno-data] wrote {out_path}  (elapsed {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()

"""Build the coarse/fine FD dataset for hybrid training.

Usage:
    python scripts/build_hybrid_dataset.py \\
        --config configs/hybrid_sw_dataset.yaml --k 2 --out outputs/hybrid/dataset_sw_k2.npz
    python scripts/build_hybrid_dataset.py \\
        --config configs/hybrid_sw_dataset.yaml --k 4 --out outputs/hybrid/dataset_sw_k4.npz

The `--smoke` flag shrinks each split to a few samples for pipeline validation
(Phase 1 acceptance check).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.hybrid_dataset import generate_split, save_dataset, sobol_params


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--k", type=int, required=True, choices=[2, 3, 4])
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="Use 4/2/2 samples instead of full split sizes "
                         "for pipeline validation.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sweep = cfg["hybrid_sweep"]
    if args.smoke:
        n_tr, n_va, n_te = 4, 2, 2
    else:
        n_tr = int(sweep["n_train"])
        n_va = int(sweep["n_val"])
        n_te = int(sweep["n_test"])
    seed = int(sweep.get("seed", 0))

    total = n_tr + n_va + n_te
    print(f"[HYBRID-DATA] k={args.k}, total samples={total} "
          f"(train={n_tr}, val={n_va}, test={n_te})")

    all_params = sobol_params(sweep, total, seed=seed)
    train_p = all_params[:n_tr]
    val_p   = all_params[n_tr:n_tr + n_va]
    test_p  = all_params[n_tr + n_va:]

    splits = {}
    grid = None
    for name, plist in [("train", train_p), ("val", val_p), ("test", test_p)]:
        t0 = time.time()
        Phi_f, Phi_c, V_f, P, g = generate_split(
            cfg, plist, k=args.k, progress_prefix=name,
        )
        splits[name] = {"Phi_fine": Phi_f, "Phi_coarse": Phi_c,
                        "V_fine": V_f, "P": P}
        grid = g
        print(f"[HYBRID-DATA] {name}: {len(plist)} samples in {time.time()-t0:.1f}s "
              f"(Phi_fine {Phi_f.shape}, Phi_coarse {Phi_c.shape})")

    meta = {
        "k": args.k,
        "base_dx": float(cfg["fd"]["dx"]),
        "base_dt": float(cfg["fd"]["dt"]),
        "coarse_dx": args.k * float(cfg["fd"]["dx"]),
        "coarse_dt": args.k * float(cfg["fd"]["dt"]),
        "sweep": sweep,
        "smoke": args.smoke,
    }
    save_dataset(args.out, splits, grid, meta)
    print(f"[HYBRID-DATA] wrote {args.out}")
    print(f"[HYBRID-DATA] file size: "
          f"{os.path.getsize(args.out) / 1024**2:.1f} MiB")


if __name__ == "__main__":
    main()

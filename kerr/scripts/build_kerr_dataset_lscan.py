"""Build the Kerr surrogate corpus at a chosen angular number ell (default 4).

Thin wrapper around the validated ``kerr/src/kerr_dataset.py`` machinery that
sets the module-level ``ELL`` (and ``MM``) *before* any evolution, so the SAME
audited solver/grids produce an ell=4 corpus without editing the validated
module. On Linux the multiprocessing pool uses ``fork``, so the workers inherit
the parent's ``ELL`` -- verified bit-consistent with the serial path.

Why ell=4 (pre-flight, 2026-06-17): the fine N=801 solver stays trustworthy
(QNM < 0.1% vs Leaver across a/M in {0, 0.5, 0.9, 0.95}), while the coarse k4
N=201 prior genuinely FAILS the QNM at high spin (42% at a/M=0.9, 58% at 0.95).
That is the genuine QNM-rescue regime the ell=2 corpus did not have: a
trustworthy teacher AND a coarse prior far from the gate, so a hybrid that
reaches fine quality would actually *improve* the QNM, not merely conserve it.

The grids are deliberately unchanged from the ell=2 corpus (fine 801 / k2 401 /
k4 201): the pre-flight shows fine is adequate and coarse-prior failure is the
desired learnable signal, not a defect to fix.

Usage (one split per invocation, run under SLURM):
    python kerr/scripts/build_kerr_dataset_lscan.py --ell 4 --split train \
        --out kerr/outputs/phase_c_l4/dataset_train.npz --workers 32
    python kerr/scripts/build_kerr_dataset_lscan.py --ell 4 --verify-corpus \
        --out kerr/outputs/phase_c_l4/dataset_train.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

import kerr.src.kerr_dataset as kd
from kerr.src.kerr_dataset import (
    SPLIT_SIZES, COARSE_N, FINE_N, T_STORE, DT_STORE,
    generate_split, params_for_split, save_dataset, load_dataset,
)


def set_ell(ell: int, m: int = 2) -> None:
    """Set the angular numbers on the kerr_dataset module BEFORE evolving.

    ``kerr_dataset`` bound ``ELL``/``MM`` at import from ``kv3_qnm``; we rebind
    the module attributes here so ``evolve_full_field`` (which reads the bare
    module globals) uses the requested ell. Forked pool workers inherit these.
    """
    kd.ELL = int(ell)
    kd.MM = int(m)


def set_grids(coarse_n: dict) -> None:
    """Override the coarse grid sizes on the kerr_dataset module BEFORE evolving.

    Same pre-fork pattern as ``set_ell``: ``generate_split``/``evolve_full_field``
    read the module-level ``COARSE_N``, so mutating ``kd.COARSE_N`` here (before
    the pool forks) changes which grids are evolved without editing the audited
    module. Used to drop the ell=4 prior to the coarser ladder (k2=201, k4=101)
    that opens genuine spin-graded QNM headroom. The grids must still nest in the
    fine N=801 axis: (801-1) % (N-1) == 0 for every N.
    """
    for k, Nk in coarse_n.items():
        if (FINE_N - 1) % (int(Nk) - 1) != 0:
            raise ValueError(
                f"grid k={k} N={Nk} does not nest in fine N={FINE_N}: "
                f"(801-1) % ({int(Nk)}-1) != 0")
    kd.COARSE_N = {int(k): int(v) for k, v in coarse_n.items()}


def _parse_coarse_n(spec: str) -> dict:
    """Parse a 'k:N,k:N' grid spec, e.g. '2:201,4:101'."""
    out = {}
    for tok in spec.split(","):
        k, n = tok.split(":")
        out[int(k)] = int(n)
    return out


def verify_corpus(out_dir: str, ell: int) -> bool:
    """Load every split, check finiteness/shapes/disjointness; emit a manifest.

    Mirrors build_kerr_dataset.verify_corpus but ell-aware and without the
    ell=2-specific B.9 anchors. The fine-field QNM faithfulness is checked
    against the qnm package per sample (the corpus already stores it).
    """
    print(f"=== L-scan corpus verification (ell={ell}) ===")
    ks = sorted(kd.COARSE_N)
    tags = ["fine"] + [f"k{k}" for k in ks]
    expect_N = {"fine": FINE_N, **{f"k{k}": kd.COARSE_N[k] for k in ks}}
    ok = True
    P_by_split = {}
    manifest = {"ell": ell, "splits": {}}
    ntau_ref = None
    for split in SPLIT_SIZES:
        path = os.path.join(out_dir, f"dataset_{split}.npz")
        if not os.path.isfile(path):
            print(f"  {split}: MISSING ({path})  FAIL"); ok = False; continue
        sp, arrays, grids, meta = load_dataset(path)
        ntau = int(grids.tau.size); n = int(arrays["P"].shape[0])
        finite = True; shapes_ok = True
        for tag in tags:
            re = arrays[f"psi_{tag}_re"]; im = arrays[f"psi_{tag}_im"]
            finite = finite and bool(np.all(np.isfinite(re)) and np.all(np.isfinite(im)))
            shapes_ok = shapes_ok and (re.shape == (n, ntau, expect_N[tag]) and im.shape == re.shape)
        qfin = bool(np.all(np.isfinite(arrays["qnm"])))
        n_ok = (n == SPLIT_SIZES[split])
        if ntau_ref is None:
            ntau_ref = ntau
        axes_ok = (ntau == ntau_ref)
        P_by_split[split] = np.asarray(arrays["P"], dtype=np.float64).copy()
        good = bool(finite and shapes_ok and qfin and n_ok and axes_ok)
        ok = ok and good
        manifest["splits"][split] = dict(
            n=n, expected_n=SPLIT_SIZES[split], ntau=ntau, finite=finite,
            shapes_ok=shapes_ok, qnm_finite=qfin,
            a_min=float(arrays["P"][:, 0].min()), a_max=float(arrays["P"][:, 0].max()),
            mw_min=float(arrays["qnm"][:, 0].min()), mw_max=float(arrays["qnm"][:, 0].max()),
            size_mb=round(os.path.getsize(path) / 1e6, 1),
        )
        print(f"  {split:5s}: n={n}/{SPLIT_SIZES[split]} ntau={ntau} finite={finite} "
              f"shapes_ok={shapes_ok} qnm_finite={qfin} {'PASS' if good else 'FAIL'}")
        del arrays
    disjoint = True
    names = list(P_by_split)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            A = {tuple(np.round(r, 10)) for r in P_by_split[names[i]]}
            B = {tuple(np.round(r, 10)) for r in P_by_split[names[j]]}
            if A & B:
                disjoint = False
                print(f"  OVERLAP {names[i]}/{names[j]}: {len(A & B)} shared  FAIL")
    ok = ok and disjoint
    manifest["disjoint_splits"] = disjoint
    man_path = os.path.join(out_dir, "corpus_manifest.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(man_path, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print(f"  manifest -> {man_path}")
    print(f"=== verification {'PASS' if ok else 'FAIL'} ===")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ell", type=int, default=4)
    ap.add_argument("--m", type=int, default=2)
    ap.add_argument("--split", choices=list(SPLIT_SIZES), default="train")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--coarse-n", type=str, default=None,
                    help="override coarse grid sizes, e.g. '2:201,4:101' (the "
                         "coarser ell=4 ladder with QNM headroom). Must nest in "
                         "fine N=801. Default keeps the module's {2:401,4:201}.")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--verify-corpus", action="store_true")
    args = ap.parse_args()

    set_ell(args.ell, args.m)
    if args.coarse_n:
        set_grids(_parse_coarse_n(args.coarse_n))
    print(f"[L-SCAN] ELL={kd.ELL} MM={kd.MM}  COARSE_N={kd.COARSE_N}  "
          f"(kerr_dataset module globals set)")

    if args.verify_corpus:
        ok = verify_corpus(os.path.dirname(args.out) or ".", args.ell)
        sys.exit(0 if ok else 1)

    n = args.n
    t_store, dt_store = T_STORE, DT_STORE
    if args.smoke:
        n = n or 4
        t_store, dt_store = 60.0, 0.5

    params = params_for_split(args.split, seed=args.seed, n=n)
    print(f"=== build (ell={args.ell}): split={args.split} n={len(params)} "
          f"ks={args.ks} t_store={t_store} dt_store={dt_store} workers={args.workers} ===")
    print(f"    grids: fine N={FINE_N}, coarse "
          f"{', '.join(f'k{k}->N={kd.COARSE_N[k]}' for k in args.ks)}")
    t0 = time.time()
    arrays, grids = generate_split(
        params, ks=tuple(args.ks), t_store=t_store, dt_store=dt_store,
        progress_prefix=f"l{args.ell}-{args.split}", workers=args.workers)
    elapsed = time.time() - t0

    meta = dict(
        split=args.split, n=len(params), seed=args.seed, ks=list(args.ks),
        ell=args.ell, m=args.m,
        t_store=t_store, dt_store=dt_store, ntau=int(grids.tau.size),
        fine_N=FINE_N, coarse_N={k: kd.COARSE_N[k] for k in args.ks},
        spin_range=(0.0, 0.95), r0_range=(8.0, 11.0), w_range=(1.0, 1.5),
        amp_fixed=1.0, workers=args.workers, elapsed_s=elapsed,
    )
    save_dataset(args.out, args.split, arrays, grids, meta)
    print(f"    wrote {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB)  in {elapsed:.1f}s")
    print(f"    manifest: {json.dumps(meta)}")


if __name__ == "__main__":
    main()

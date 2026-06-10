"""Build the Phase C Kerr coarse/fine COMPLEX-field dataset (C.1).

Generates one split (``train``/``val``/``test``) of the Kerr surrogate corpus by
evolving the verified Phase B operator on the fine (``N=801``) and coarse
(``N=401`` / ``N=201``) grids for each Sobol ``(a/M, r0, w)`` sample, storing the
full complex field ``psi(tau, sigma)`` on a shared canonical tau-axis. See
``kerr/src/kerr_dataset.py`` and ``kerr/notes/phase_c_plan.md`` (C.1).

Usage
-----
    # quick login-node smoke + acceptance self-test (a few samples, short window)
    python kerr/scripts/build_kerr_dataset.py --smoke --selftest

    # one production split (run under SLURM, C.2)
    python kerr/scripts/build_kerr_dataset.py --split train \
        --out kerr/outputs/phase_c/dataset_train.npz

The ``--selftest`` acceptance (C.1 gate) regenerates the three B.9 anchor spins
at the B.9 pulse (``r0=10, w=1``) and checks that the stored fine field
reproduces the B.9 multi-observer gate estimate of ``M*omega_220`` -- which
simultaneously validates the record cadence ``DT_STORE``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from kerr.src.kerr_dataset import (          # noqa: E402
    DT_STORE,
    T_STORE,
    SPLIT_SIZES,
    COARSE_N,
    FINE_N,
    evolve_full_field,
    generate_split,
    params_for_split,
    save_dataset,
    load_dataset,
)
from kerr.scripts.kv3_qnm import (           # noqa: E402
    OBSERVERS_RM,
    scri_index,
    observer_index,
    extract_fundamental,
    gate_estimate,
)

# B.9 recorded M*omega_220 gate estimates at the canonical pulse (r0=10, w=1),
# from kerr/outputs/phase_b/kerr_sweep.npz (see phase_b_plan.md B.9 table).
B9_ANCHOR_MW = {0.0: 0.373665, 0.5: 0.464047, 0.9: 0.671593}


def _extract_mw(tau, rec, op, is_real, tau_ref):
    """Multi-observer B.9 gate estimate of M*omega_220 from a recorded field."""
    tau_final = float(tau[-1])
    per = []
    for lab, rM in OBSERVERS_RM.items():
        idx = scri_index(op) if rM is None else observer_index(op, rM)
        sm = extract_fundamental(tau, rec[:, idx], is_real=is_real,
                                 tau_ref=tau_ref, tau_final=tau_final)
        per.append((lab, sm.get("omega", np.nan), sm.get("tau", np.nan),
                    sm.get("omega_std", np.nan)))
    mw_ext, _, _ = gate_estimate(per)
    return mw_ext


def run_selftest() -> bool:
    """C.1 acceptance. Returns ``ok``.

    Two checks per anchor spin a/M in {0, 0.5, 0.9} at the B.9 pulse (r0=10, w=1):

    1. Fine-field faithfulness (decisive): the stored fine field (N=801) must
       reproduce both the qnm-package M*omega_220 AND the B.9 multi-observer gate
       estimate to B.9's own 5e-3 tolerance -- this validates the record cadence.

    2. Coarse VALIDITY: each coarse field (N=401/N=201) must be finite AND itself
       extract the QNM to the SAME 5e-3 gate -- proving it is a valid (if
       under-resolved) ringdown, not numerical garbage. The field L2 vs fine at
       the shared nested nodes is REPORTED as the learnable-residual magnitude
       (the very signal the FNO must correct): a large residual at k=4/high-spin
       is expected and desirable, so it is informational, NOT a pass/fail gate.
    """
    from kerr.src.qnm_kerr_reference import kerr_qnm

    print("=== C.1 self-test: fine B.9 faithfulness + coarse validity ===")
    ok = True
    for a in (0.0, 0.5, 0.9):
        ref = kerr_qnm(a_over_M=a, ell=2, m=2, n=0)
        tau_ref = ref.tau_over_M
        mw_ref = float(ref.M_omega_R)
        is_real = (a == 0.0)
        # fine field at the B.9 pulse
        tau, rec, op, info = evolve_full_field(a, FINE_N, r0=10.0, w=1.0)
        mw_ext = _extract_mw(tau, rec, op, is_real, tau_ref)
        b9 = B9_ANCHOR_MW[a]
        e_ref = abs(mw_ext - mw_ref) / mw_ref
        e_b9 = abs(mw_ext - b9) / b9
        good = bool(np.isfinite(mw_ext) and e_ref < 5e-3 and e_b9 < 5e-3)
        ok = ok and good
        print(f"  a/M={a:.2f}: Mw_ext={mw_ext:.6f}  qnm={mw_ref:.6f} "
              f"(e={e_ref:.2e})  B.9={b9:.6f} (e={e_b9:.2e})  "
              f"finite_field={info['finite']}  {'PASS' if good else 'FAIL'}")

        # coarse validity: finite AND extracts the QNM to B.9's gate (5e-3).
        # L2 vs fine@nodes is the learnable residual -> reported, not gated.
        for k, Nk in COARSE_N.items():
            tc, recc, opc, infoc = evolve_full_field(a, Nk, r0=10.0, w=1.0)
            mw_c = _extract_mw(tc, recc, opc, is_real, tau_ref)
            e_c = abs(mw_c - mw_ref) / mw_ref
            fine_at_nodes = rec[:, ::k]
            rel = float(np.linalg.norm(fine_at_nodes - recc)
                        / np.linalg.norm(fine_at_nodes))
            cgood = bool(infoc["finite"] and np.isfinite(mw_c) and e_c < 5e-3)
            ok = ok and cgood
            print(f"      k={k} (N={Nk}): finite={infoc['finite']}  "
                  f"Mw_c={mw_c:.6f} (e={e_c:.2e})  "
                  f"residual relL2={rel:.3e} [signal, not gated]  "
                  f"{'PASS' if cgood else 'FAIL'}")
    print(f"=== self-test {'PASS' if ok else 'FAIL'} ===")
    return ok


def verify_corpus(out_dir: str, allow_partial: bool = False) -> bool:
    """C.2 acceptance: load every split, check finiteness/shapes/disjointness.

    Writes a small machine-readable manifest (``corpus_manifest.json``) next to
    the data and prints it. Splits are checked one at a time so peak memory stays
    at the size of the largest single split (``train``), never all three at once.
    With ``allow_partial`` the per-split count is reported but not gated (used by
    the reduced-size plumbing dry-run); the production run leaves it strict.
    Returns ``ok``.
    """
    print("=== C.2 corpus verification ===")
    ks = sorted(COARSE_N)
    tags = ["fine"] + [f"k{k}" for k in ks]
    expect_N = {"fine": FINE_N, **{f"k{k}": COARSE_N[k] for k in ks}}

    ok = True
    ntau_ref = None
    sig_sizes_ref = None
    scri_ref = None
    P_by_split: Dict[str, np.ndarray] = {}
    manifest: Dict[str, object] = {"splits": {}}

    for split in SPLIT_SIZES:
        path = os.path.join(out_dir, f"dataset_{split}.npz")
        if not os.path.isfile(path):
            print(f"  {split}: MISSING ({path})  FAIL")
            ok = False
            continue
        sp, arrays, grids, meta = load_dataset(path)
        ntau = int(grids.tau.size)
        n = int(arrays["P"].shape[0])
        finite = True
        shapes_ok = True
        mw = arrays["qnm"][:, 0]
        for tag in tags:
            re = arrays[f"psi_{tag}_re"]
            im = arrays[f"psi_{tag}_im"]
            finite = finite and bool(np.all(np.isfinite(re))
                                     and np.all(np.isfinite(im)))
            shapes_ok = shapes_ok and (re.shape == (n, ntau, expect_N[tag])
                                       and im.shape == re.shape)
        qfin = bool(np.all(np.isfinite(arrays["qnm"])))
        exp_n = SPLIT_SIZES[split]
        n_ok = (n == exp_n)
        # consistency of shared axes across splits
        if ntau_ref is None:
            ntau_ref = ntau
            sig_sizes_ref = {t: int(grids.sigma[1 if t == "fine" else int(t[1:])].size)
                             for t in tags}
            scri_ref = dict(grids.scri_idx)
        axes_ok = (ntau == ntau_ref)
        P_by_split[split] = np.asarray(arrays["P"], dtype=np.float64).copy()
        count_ok = n_ok or allow_partial
        good = bool(finite and shapes_ok and qfin and count_ok and axes_ok)
        ok = ok and good
        manifest["splits"][split] = dict(
            n=n, expected_n=exp_n, n_ok=n_ok, ntau=ntau, finite=finite,
            shapes_ok=shapes_ok, qnm_finite=qfin,
            mw_min=float(mw.min()), mw_max=float(mw.max()),
            a_min=float(arrays["P"][:, 0].min()),
            a_max=float(arrays["P"][:, 0].max()),
            size_mb=round(os.path.getsize(path) / 1e6, 1),
        )
        print(f"  {split:5s}: n={n}/{exp_n}  ntau={ntau}  finite={finite}  "
              f"shapes_ok={shapes_ok}  qnm_finite={qfin}  "
              f"{'PASS' if good else 'FAIL'}")
        del arrays  # free the big fields before the next split

    # disjointness: no parameter row shared across splits (Sobol slices)
    disjoint = True
    names = list(P_by_split)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            A = {tuple(np.round(r, 10)) for r in P_by_split[names[i]]}
            B = {tuple(np.round(r, 10)) for r in P_by_split[names[j]]}
            if A & B:
                disjoint = False
                print(f"  OVERLAP between {names[i]} and {names[j]}: "
                      f"{len(A & B)} shared params  FAIL")
    ok = ok and disjoint
    manifest["disjoint_splits"] = disjoint
    manifest["ntau"] = ntau_ref
    manifest["sigma_sizes"] = sig_sizes_ref
    manifest["scri_idx"] = scri_ref
    manifest["t_store"] = T_STORE
    manifest["dt_store"] = DT_STORE
    manifest["ks"] = ks
    print(f"  disjoint_splits={disjoint}")

    man_path = os.path.join(out_dir, "corpus_manifest.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(man_path, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print(f"  manifest -> {man_path}")
    print(json.dumps(manifest, sort_keys=True))
    print(f"=== corpus verification {'PASS' if ok else 'FAIL'} ===")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=list(SPLIT_SIZES), default="train")
    ap.add_argument("--n", type=int, default=None, help="override split count")
    ap.add_argument("--seed", type=int, default=0, help="master Sobol seed")
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--t-store", type=float, default=T_STORE)
    ap.add_argument("--dt-store", type=float, default=DT_STORE)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--workers", type=int, default=1,
                    help="process-pool size for the per-sample evolutions "
                         "(bit-identical to 1; >1 fans out over cores)")
    ap.add_argument("--smoke", action="store_true",
                    help="quick: 4 samples, short window (login-node spot-check)")
    ap.add_argument("--selftest", action="store_true",
                    help="run the C.1 B.9-faithfulness acceptance and exit")
    ap.add_argument("--verify-corpus", action="store_true",
                    help="C.2 acceptance: check all written splits and emit a "
                         "manifest (uses --out's directory)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="with --verify-corpus: report but do not gate on the "
                         "production per-split counts (plumbing dry-run)")
    args = ap.parse_args()

    if args.selftest:
        ok = run_selftest()
        sys.exit(0 if ok else 1)

    if args.verify_corpus:
        out_dir = (os.path.dirname(args.out) if args.out
                   else os.path.join(_ROOT, "kerr", "outputs", "phase_c"))
        ok = verify_corpus(out_dir, allow_partial=args.allow_partial)
        sys.exit(0 if ok else 1)

    n = args.n
    t_store, dt_store = args.t_store, args.dt_store
    if args.smoke:
        n = n or 4
        if args.t_store == T_STORE:
            t_store = 60.0
        if args.dt_store == DT_STORE:
            dt_store = 0.5

    params = params_for_split(args.split, seed=args.seed, n=n)
    out = args.out or os.path.join(
        _ROOT, "kerr", "outputs", "phase_c", f"dataset_{args.split}.npz")

    print(f"=== build_kerr_dataset: split={args.split} n={len(params)} "
          f"ks={args.ks} t_store={t_store} dt_store={dt_store} "
          f"workers={args.workers} ===")
    print(f"    grids: fine N={FINE_N}, coarse "
          f"{', '.join(f'k{k}->N={COARSE_N[k]}' for k in args.ks)}")
    t0 = time.time()
    arrays, grids = generate_split(
        params, ks=tuple(args.ks), t_store=t_store, dt_store=dt_store,
        progress_prefix=args.split, workers=args.workers)
    elapsed = time.time() - t0

    meta = dict(
        split=args.split, n=len(params), seed=args.seed, ks=list(args.ks),
        t_store=t_store, dt_store=dt_store, ntau=int(grids.tau.size),
        fine_N=FINE_N, coarse_N={k: COARSE_N[k] for k in args.ks},
        spin_range=(0.0, 0.95), r0_range=(8.0, 11.0), w_range=(1.0, 1.5),
        amp_fixed=1.0, workers=args.workers, elapsed_s=elapsed,
    )
    save_dataset(out, args.split, arrays, grids, meta)
    size_mb = os.path.getsize(out) / 1e6
    print(f"    wrote {out}  ({size_mb:.1f} MB)  in {elapsed:.1f}s")
    print(f"    manifest: {json.dumps(meta)}")


if __name__ == "__main__":
    main()

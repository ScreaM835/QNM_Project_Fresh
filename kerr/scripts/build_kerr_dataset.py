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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=list(SPLIT_SIZES), default="train")
    ap.add_argument("--n", type=int, default=None, help="override split count")
    ap.add_argument("--seed", type=int, default=0, help="master Sobol seed")
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--t-store", type=float, default=T_STORE)
    ap.add_argument("--dt-store", type=float, default=DT_STORE)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="quick: 4 samples, short window (login-node spot-check)")
    ap.add_argument("--selftest", action="store_true",
                    help="run the C.1 B.9-faithfulness acceptance and exit")
    args = ap.parse_args()

    if args.selftest:
        ok = run_selftest()
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
          f"ks={args.ks} t_store={t_store} dt_store={dt_store} ===")
    print(f"    grids: fine N={FINE_N}, coarse "
          f"{', '.join(f'k{k}->N={COARSE_N[k]}' for k in args.ks)}")
    t0 = time.time()
    arrays, grids = generate_split(
        params, ks=tuple(args.ks), t_store=t_store, dt_store=dt_store,
        progress_prefix=args.split)
    elapsed = time.time() - t0

    meta = dict(
        split=args.split, n=len(params), seed=args.seed, ks=list(args.ks),
        t_store=t_store, dt_store=dt_store, ntau=int(grids.tau.size),
        fine_N=FINE_N, coarse_N={k: COARSE_N[k] for k in args.ks},
        spin_range=(0.0, 0.95), r0_range=(8.0, 11.0), w_range=(1.0, 1.5),
        amp_fixed=1.0, elapsed_s=elapsed,
    )
    save_dataset(out, args.split, arrays, grids, meta)
    size_mb = os.path.getsize(out) / 1e6
    print(f"    wrote {out}  ({size_mb:.1f} MB)  in {elapsed:.1f}s")
    print(f"    manifest: {json.dumps(meta)}")


if __name__ == "__main__":
    main()

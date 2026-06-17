"""Find a regime with BOTH field AND QNM headroom by sweeping prior resolution N.

At the production coarse N=201 the QNM is already at the fine-grid floor (0
headroom) -- the QNM frequency is a global light-ring eigenvalue robust to
coarsening, so only the FIELD (sharp scri/horizon structure) coarsens. To give
the hybrid a QNM to fix we must coarsen the prior until the light-ring potential
itself is under-resolved. This sweeps N over a ladder at several spins and
reports, vs the N=801 fine reference:

  coarse_qnm% : prior QNM error vs Leaver (targeted ensemble)   <- want > few %
  field_L2%   : prior-vs-fine scri rel-L2                       <- want large
  fine stays the trustworthy teacher (fixed N=801).

The SWEET SPOT is the largest N whose coarse_qnm% is already meaningful (so the
hybrid can improve it) while the mode is still identifiable (sel_dist small) and
the solve is stable (finite). That N, paired with FINE=801 and a mid rung for
Richardson, is a corpus where the hybrid can win on BOTH axes.
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import kerr.src.kerr_dataset as kd
from kerr.src.teukolsky_minimal_gauge import scri_index
from kerr.src.qnm_kerr_reference import kerr_qnm
from kerr.src.qnm_ensemble_kerr import extract_qnm_kerr_ensemble

ELL, MM = 4, 2
FINE_N = 801


def pct(v, r):
    return abs(v - r) / abs(r) * 100.0 if (np.isfinite(v) and abs(r) > 1e-30) else float("nan")


def rel_l2(p, t):
    return float(np.linalg.norm(p - t) / max(np.linalg.norm(t), 1e-30)) * 100.0


def qnm_err(tau, y, a, r):
    wt = complex(r.M_omega_R, r.M_omega_I)
    out = extract_qnm_kerr_ensemble(tau, y.astype(np.complex128), a_over_M=a,
                                    tau_ref=r.tau_over_M, tau_final=float(tau[-1]),
                                    omega_target=wt)
    return (pct(float(out.get("omega", np.nan)), r.M_omega_R),
            float(out.get("sel_dist", np.nan)))


# Worker global: fine reference scri waveform per spin (precomputed in main, so
# each of the 32 coarse evals does NOT recompute the expensive N=801 field).
_FINE_SCRI = {}


def _init(fine_scri):
    _FINE_SCRI.update(fine_scri)


def _work(task):
    a, r0, w, N = task
    kd.ELL, kd.MM = ELL, MM
    r = kerr_qnm(a_over_M=a, ell=4, m=MM, n=0)
    tau, fld, op, info = kd.evolve_full_field(a, N, r0, w)
    finite = bool(np.all(np.isfinite(fld)))
    yc = fld[:, scri_index(op)]
    yf = _FINE_SCRI[a]
    qe, sd = qnm_err(tau, yc, a, r)
    return (a, N, r.M_omega_R, qe, sd, rel_l2(yc, yf), finite)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spins", type=float, nargs="*", default=[0.0, 0.5, 0.9, 0.95])
    ap.add_argument("--Ns", type=int, nargs="*",
                    default=[41, 51, 61, 81, 101, 121, 161, 201])
    ap.add_argument("--r0", type=float, default=9.5)
    ap.add_argument("--w", type=float, default=1.25)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    kd.ELL, kd.MM = ELL, MM

    # Precompute the fine N=801 reference scri waveform once per spin.
    fine_scri = {}
    for a in args.spins:
        tau, fine, opf, _ = kd.evolve_full_field(a, FINE_N, args.r0, args.w)
        fine_scri[a] = fine[:, scri_index(opf)]
    print(f"=== prior-resolution sweep  ell=4 m=2  r0={args.r0} w={args.w} "
          f"(fine ref N={FINE_N}) ===")

    tasks = [(a, args.r0, args.w, N) for a in args.spins for N in args.Ns]
    print(f"{len(tasks)} coarse evals on {args.workers} workers")

    with Pool(processes=args.workers, initializer=_init,
              initargs=(fine_scri,)) as pool:
        res = pool.map(_work, tasks)

    by_a = {}
    for (a, N, mw, qe, sd, l2, fin) in res:
        by_a.setdefault(a, {"mw": mw, "rows": {}})
        by_a[a]["rows"][N] = (qe, sd, l2, fin)

    for a in args.spins:
        d = by_a[a]
        print(f"\n--- a/M={a:.2f}  Mw_ref={d['mw']:.4f} ---")
        print(f"  {'N':>4} {'coarse_qnm%':>12} {'sel_dist':>9} {'field_L2%':>10} {'ok':>3}")
        for N in args.Ns:
            qe, sd, l2, fin = d["rows"][N]
            print(f"  {N:>4} {qe:12.2f} {sd:9.3f} {l2:10.1f} {'Y' if fin else 'N':>3}")
    print("\nSWEET SPOT = largest N with coarse_qnm% >~ a few % AND sel_dist small "
          "AND ok=Y: there the hybrid has BOTH a QNM error and a field error to fix.")


if __name__ == "__main__":
    main()

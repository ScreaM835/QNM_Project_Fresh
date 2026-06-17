"""Teacher-quality check for a COARSER ell=4 ladder (fine / mid / prior).

The prior-resolution sweep showed N=101 opens genuine spin-graded QNM error
(0.3->42%) AND field error (40-72%), unlike the saturated N=201 prior. To train
the hybrid LABEL-FREE we need the Richardson teacher built from the two coarse
grids to be a GOOD teacher: cleaner QNM AND smaller field error than the prior it
corrects. Richardson cancels the leading O(h^2) error, but ONLY in the asymptotic
regime -- at high spin the prior (101) may be too under-resolved for that, so we
must MEASURE it, not assume it.

For the proposed ladder fine=801 / mid=201 / prior=101 (all nest: 800=4*200=8*100)
this evolves each grid and reports, vs the Leaver (4,2,0) and the fine field:
  prior : up(101)                         -- what the hybrid corrects
  mid   : up(201)                         -- the finer coarse grid (context)
  rich  : (4*up(201) - up(101))/3         -- the LABEL-FREE training teacher
  fine  : N=801                           -- reference / ceiling
QNM via the targeted ensemble (scri); field via full-(tau,sigma) rel-L2 vs fine.

GO criterion: rich_qnm << prior_qnm (clean, carries QNM gain) and rich_field <
prior_field (carries field gain). If rich QNM degrades at high spin (Richardson
amplifying an under-resolved prior), fall back to target_mode=supervised (fine).
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
from kerr.src.hybrid_data_pipe import build_upsample_matrix, _apply_W

ELL, MM = 4, 2


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


_CFG = {}


def _init(fine_n, mid_n, prior_n):
    _CFG.update(fine=fine_n, mid=mid_n, prior=prior_n)


def _work(task):
    a, r0, w = task
    kd.ELL, kd.MM = ELL, MM
    fine_n, mid_n, prior_n = _CFG["fine"], _CFG["mid"], _CFG["prior"]
    r = kerr_qnm(a_over_M=a, ell=4, m=MM, n=0)

    tau, fine, opf, _ = kd.evolve_full_field(a, fine_n, r0, w)
    _, mid, opm, _ = kd.evolve_full_field(a, mid_n, r0, w)
    _, pri, opp, _ = kd.evolve_full_field(a, prior_n, r0, w)

    # full-field upsample to the fine sigma grid (matches the corpus pipeline)
    sigf = opf.sigma; sigm = opm.sigma; sigp = opp.sigma
    Wm = build_upsample_matrix(sigm, sigf)
    Wp = build_upsample_matrix(sigp, sigf)
    up_mid = (_apply_W(Wm, mid.real[None]).astype(np.float64)[0]
              + 1j * _apply_W(Wm, mid.imag[None]).astype(np.float64)[0])
    up_pri = (_apply_W(Wp, pri.real[None]).astype(np.float64)[0]
              + 1j * _apply_W(Wp, pri.imag[None]).astype(np.float64)[0])
    rich_f = (4.0 * up_mid - up_pri) / 3.0

    f_pri = rel_l2(up_pri, fine)
    f_mid = rel_l2(up_mid, fine)
    f_rich = rel_l2(rich_f, fine)

    # scri QNM (all grids share scri at index 0; Richardson combines pointwise)
    sf, sm, sp = scri_index(opf), scri_index(opm), scri_index(opp)
    y_fine = fine[:, sf]; y_mid = mid[:, sm]; y_pri = pri[:, sp]
    y_rich = (4.0 * y_mid - y_pri) / 3.0
    qp, sdp = qnm_err(tau, y_pri, a, r)
    qm, _ = qnm_err(tau, y_mid, a, r)
    qr, sdr = qnm_err(tau, y_rich, a, r)
    qf, _ = qnm_err(tau, y_fine, a, r)
    return (a, r.M_omega_R, qp, qm, qr, qf, sdp, sdr, f_pri, f_mid, f_rich)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fine", type=int, default=801)
    ap.add_argument("--mid", type=int, default=201)
    ap.add_argument("--prior", type=int, default=101)
    ap.add_argument("--spins", type=float, nargs="*", default=[0.0, 0.5, 0.7, 0.9, 0.95])
    ap.add_argument("--r0s", type=float, nargs="*", default=[8.5, 9.5, 10.5])
    ap.add_argument("--ws", type=float, nargs="*", default=[1.25])
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    for n in (args.fine, args.mid, args.prior):
        if (args.fine - 1) % (n - 1) != 0:
            print(f"WARNING: grids do not nest: (fine-1)={args.fine-1} not "
                  f"divisible by (N-1)={n-1} for N={n}")

    tasks = [(a, r0, w) for a in args.spins for r0 in args.r0s for w in args.ws]
    print(f"=== coarse-ladder teacher check  ell=4 m=2 "
          f"(fine={args.fine} mid={args.mid} prior={args.prior}) ===")
    print(f"{len(tasks)} evals on {args.workers} workers; targeted ensemble; "
          f"rich=(4*up_mid-up_prior)/3")

    with Pool(processes=args.workers, initializer=_init,
              initargs=(args.fine, args.mid, args.prior)) as pool:
        res = pool.map(_work, tasks)

    by_a = {}
    for row in res:
        a = row[0]
        by_a.setdefault(a, []).append(row)

    print(f"\n{'a/M':>5} {'Mw_ref':>7} | {'prior_q%':>8} {'mid_q%':>7} "
          f"{'RICH_q%':>8} {'fine_q%':>7} {'sd_pri':>6} {'sd_rich':>7} | "
          f"{'prior_F%':>8} {'mid_F%':>7} {'RICH_F%':>8}")
    print("-" * 104)
    for a in args.spins:
        rows = by_a[a]
        mw = rows[0][1]
        col = lambda i: np.array([row[i] for row in rows], float)
        qp, qm, qr, qf = col(2), col(3), col(4), col(5)
        sdp, sdr = col(6), col(7)
        fp, fm, fr = col(8), col(9), col(10)
        print(f"{a:5.2f} {mw:7.4f} | "
              f"{np.nanmedian(qp):8.2f} {np.nanmedian(qm):7.2f} "
              f"{np.nanmedian(qr):8.2f} {np.nanmedian(qf):7.2f} "
              f"{np.nanmax(sdp):6.3f} {np.nanmax(sdr):7.3f} | "
              f"{np.nanmedian(fp):8.1f} {np.nanmedian(fm):7.1f} {np.nanmedian(fr):8.1f}")
    print("-" * 104)
    print("GO if RICH_q% << prior_q% (clean teacher, carries QNM gain) AND "
          "RICH_F% < prior_F% (carries field gain).")
    print("sd_* = max ensemble sel_dist (small => mode genuinely identified; "
          "large => that QNM% is noisy).")


if __name__ == "__main__":
    main()

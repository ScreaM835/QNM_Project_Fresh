"""Worst-corner stability / resolution check for the N=101 prior (ell=4).

The corpus docstring guaranteed ">=5 points across the pulse" only for the OLD
coarsest grid N=201. Dropping the prior to N=101 doubles the sigma spacing, so
the narrowest pulse (w=1.0) at the corner of the Sobol box must be re-checked
before committing a 6 h corpus build. Resolution (points across the Gaussian
pulse, std=w in r) is worst where (r_plus/r0^2)*w is smallest: max r0 (=11),
min w (=1.0), and smallest r_plus (highest spin). This evolves those corners on
N=101 (prior), N=201 (mid) and N=801 (fine) and reports:
  npts_pulse : grid points within +-2 std of the pulse on N=101 (resolution)
  finite     : N=101 field has no NaN/Inf (the build's hard gate)
  prior_q%   : targeted-ensemble QNM error of the N=101 prior  (large = real)
  RICH_q%    : QNM of the (4*up201-up101)/3 teacher           (must stay clean)
  rich_F%    : Richardson field rel-L2 vs fine                (teacher quality)
  sd_rich    : ensemble sel_dist on the teacher (small => mode genuinely ID'd)
GO if finite=Y everywhere AND npts_pulse not absurdly low (>~3) AND the
Richardson teacher stays clean (RICH_q% < ~1%, sd_rich small) at the corner.
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
FINE_N, MID_N, PRIOR_N = 801, 201, 101


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


def npts_across_pulse(op, r0, w):
    """Grid points within +-2 std (=2w in r) of the pulse center, on op.sigma."""
    r = op.r
    return int(np.sum(np.abs(r - r0) <= 2.0 * w))


def _work(task):
    a, r0, w = task
    kd.ELL, kd.MM = ELL, MM
    r = kerr_qnm(a_over_M=a, ell=4, m=MM, n=0)
    tau, fine, opf, info_f = kd.evolve_full_field(a, FINE_N, r0, w)
    _, mid, opm, info_m = kd.evolve_full_field(a, MID_N, r0, w)
    _, pri, opp, info_p = kd.evolve_full_field(a, PRIOR_N, r0, w)

    fin = bool(np.all(np.isfinite(pri)) and np.all(np.isfinite(mid))
               and np.all(np.isfinite(fine)))
    npts = npts_across_pulse(opp, r0, w)

    sigf = opf.sigma
    Wm = build_upsample_matrix(opm.sigma, sigf)
    Wp = build_upsample_matrix(opp.sigma, sigf)
    up_mid = (_apply_W(Wm, mid.real[None]).astype(np.float64)[0]
              + 1j * _apply_W(Wm, mid.imag[None]).astype(np.float64)[0])
    up_pri = (_apply_W(Wp, pri.real[None]).astype(np.float64)[0]
              + 1j * _apply_W(Wp, pri.imag[None]).astype(np.float64)[0])
    rich_f = (4.0 * up_mid - up_pri) / 3.0
    rich_F = rel_l2(rich_f, fine)
    prior_F = rel_l2(up_pri, fine)

    sp = scri_index(opp); sm = scri_index(opm); sf = scri_index(opf)
    y_pri = pri[:, sp]; y_mid = mid[:, sm]
    y_rich = (4.0 * y_mid - y_pri) / 3.0
    pq, _ = qnm_err(tau, y_pri, a, r)
    rq, sdr = qnm_err(tau, y_rich, a, r)
    return (a, r0, w, npts, fin, pq, rq, sdr, prior_F, rich_F)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    kd.ELL, kd.MM = ELL, MM

    # Corner of the Sobol box (worst resolution) + a couple of neighbours.
    corners = []
    for a in (0.90, 0.95):
        for r0 in (10.5, 11.0):
            for w in (1.0, 1.1):
                corners.append((a, r0, w))
    # also the extreme low-w at mid spin and the literal box corner a=0.95,r0=11,w=1
    corners += [(0.70, 11.0, 1.0), (0.50, 11.0, 1.0), (0.0, 11.0, 1.0)]

    print(f"=== worst-corner N=101 prior check  ell=4  (fine={FINE_N} mid={MID_N} "
          f"prior={PRIOR_N}) ===")
    print(f"{len(corners)} corners on {args.workers} workers; "
          f"resolution worst at max r0 / min w / high spin")
    with Pool(processes=args.workers) as pool:
        res = pool.map(_work, corners)
    res.sort(key=lambda x: (x[0], x[1], x[2]))

    print(f"\n{'a/M':>5} {'r0':>5} {'w':>4} | {'npts':>4} {'finite':>6} | "
          f"{'prior_q%':>8} {'RICH_q%':>8} {'sd_rich':>7} | {'prior_F%':>8} {'RICH_F%':>8}")
    print("-" * 86)
    all_fin = True
    min_npts = 1e9
    worst_richq = 0.0
    for (a, r0, w, npts, fin, pq, rq, sdr, pf, rf) in res:
        all_fin = all_fin and fin
        min_npts = min(min_npts, npts)
        if np.isfinite(rq):
            worst_richq = max(worst_richq, rq)
        print(f"{a:5.2f} {r0:5.1f} {w:4.1f} | {npts:4d} {'Y' if fin else 'N':>6} | "
              f"{pq:8.2f} {rq:8.2f} {sdr:7.3f} | {pf:8.1f} {rf:8.1f}")
    print("-" * 86)
    print(f"all finite = {all_fin};  min npts across pulse (N=101) = {min_npts};  "
          f"worst RICH_q% = {worst_richq:.2f}")
    verdict = ("GO" if (all_fin and min_npts >= 3 and worst_richq < 2.0)
               else "CHECK")
    print(f"VERDICT: {verdict}  (need finite=Y, npts>=3, RICH_q%<2 at every corner)")


if __name__ == "__main__":
    main()

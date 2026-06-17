#!/usr/bin/env python
"""
DECISIVE TEST: is the QNM degradation in the Richardson TARGET or in the FNO?

The label-free run trains the FNO toward Phi_R = (4*up2 - up4)/3 (Richardson
extrapolation of two coarse FD solves), NOT toward the fine field. This script
computes Phi_R directly from the dataset's coarse solves -- no neural network
involved at all -- and extracts the M4 QNM at x_q=2 for every test BH:

    up4   : k4 coarse prior, quintic-upsampled            (the FNO input prior)
    up2   : k2 coarse, quintic-upsampled                  (the finer coarse solve)
    Phi_R : (4*up2 - up4)/3                                (the FNO TRAINING TARGET)
    fine  : true fine FD field                            (ground truth)

If Phi_R already has degraded M4 QNM relative to up4 -- matching the FNO
hybrid's eval medians -- then the network is faithfully learning a target whose
QNM is worse: the degradation is a property of RICHARDSON EXTRAPOLATION, not the
neural network. If Phi_R is clean, the degradation is a network artifact.

Login-safe: pure numpy/scipy on cached datasets, no FD solve, no torch.
  venv_csd3/bin/python scripts/diag_richardson_target_qnm.py
"""
from __future__ import annotations

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.hybrid_data_pipe import upsample_to_fine  # noqa: E402
from src.qnm import qnm_method_4_window_scan        # noqa: E402

OMEGA_TRUE = 0.3737
TAU_TRUE = 11.241
XQ = 2.0
K2 = "outputs/hybrid/dataset_sw_k2.npz"
K4 = "outputs/hybrid/dataset_sw_k4.npz"
N_MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 100


def pct(v, ref):
    return abs(v - ref) / ref * 100.0


def main():
    d2 = np.load(K2)
    d4 = np.load(K4)
    xf = d2["x_fine"].astype(float)
    tf = d2["t_fine"].astype(float)
    x2 = d2["x_coarse"].astype(float)
    t2 = d2["t_coarse"].astype(float)
    x4 = d4["x_coarse"].astype(float)
    t4 = d4["t_coarse"].astype(float)
    Phi_fine = d4["test_Phi_fine"]
    Phi_c4 = d4["test_Phi_coarse"]
    Phi_c2 = d2["test_Phi_coarse"]
    P = d4["test_P"]                      # (N,3) = (M, x0, sigma)
    # sample-aligned check (same as the pipeline asserts)
    assert np.allclose(Phi_fine, d2["test_Phi_fine"], atol=1e-5), "k2/k4 not aligned"
    assert np.allclose(P, d2["test_P"], atol=1e-5), "k2/k4 P not aligned"

    ix = int(np.argmin(np.abs(xf - XQ)))
    N = min(N_MAX, Phi_fine.shape[0])
    print(f"k2 grid {Phi_c2.shape[1:]} , k4 grid {Phi_c4.shape[1:]} , fine {Phi_fine.shape[1:]}")
    print(f"observer x_q={XQ} -> ix={ix} (x={xf[ix]:.3f}); N_test={N}")
    print("Phi_R = (4*up2 - up4)/3 ; both quintic-upsampled (matches training).\n")

    def m4(y, M):
        # rescale to dimensionless (M*omega, tau/M) before comparing to theory
        r = qnm_method_4_window_scan(tf, y, 10.0, 18.0, 50.0, n_starts=12,
                                     potential="zerilli", ell=2)
        o = r["omega"] * M
        ta = r["tau"] / M
        return o, ta

    # Load the FNO eval per-sample QNM (already M-rescaled: *_pct_err keys).
    import json
    per = json.load(open("outputs/hybrid/fno_sw_richardson/eval/per_sample.json"))

    rows = {k: {"o": [], "t": []} for k in ("up4", "up2", "PhiR", "fine")}
    persample = []  # aligned records for the gap table / JSON
    for i in range(N):
        M_i = float(P[i, 0])
        up4 = upsample_to_fine(Phi_c4[i], x4, t4, xf, tf, method="quintic")
        up2 = upsample_to_fine(Phi_c2[i], x2, t2, xf, tf, method="quintic")
        PhiR = (4.0 * up2 - up4) / 3.0
        vals = {}
        for key, fld in (("up4", up4), ("up2", up2),
                         ("PhiR", PhiR), ("fine", Phi_fine[i])):
            o, ta = m4(fld[:, ix], M_i)
            oe, te = pct(o, OMEGA_TRUE), pct(ta, TAU_TRUE)
            rows[key]["o"].append(oe)
            rows[key]["t"].append(te)
            vals[key] = (oe, te)
        # pull the FNO eval record for the SAME index; verify alignment by M
        rec = per[i]
        assert abs(rec["M"] - M_i) < 1e-3, (
            f"index {i} misaligned: per_sample M={rec['M']} vs dataset M={M_i}")
        hyb = rec["xq2_hyb"]["M4"]
        base = rec["xq2_base"]["M4"]
        persample.append({
            "i": i, "M": M_i,
            "phiR_omega": vals["PhiR"][0], "phiR_tau": vals["PhiR"][1],
            "up4_omega": vals["up4"][0], "up4_tau": vals["up4"][1],
            "fine_omega": vals["fine"][0], "fine_tau": vals["fine"][1],
            "fno_hyb_omega": hyb["omega_pct_err"], "fno_hyb_tau": hyb["tau_pct_err"],
            "fno_base_omega": base["omega_pct_err"], "fno_base_tau": base["tau_pct_err"],
        })
        if (i + 1) % 20 == 0:
            print(f"  ...{i+1}/{N}")

    print("\n=== M4 omega %err  (xq=2, median over test set) ===")
    print(f"  {'field':6s}  median     mean      max")
    for key in ("up4", "up2", "PhiR", "fine"):
        a = np.array(rows[key]["o"])
        a = a[np.isfinite(a)]
        print(f"  {key:6s}  {np.median(a):7.4f}%  {np.mean(a):7.4f}%  {np.max(a):8.4f}%")

    print("\n=== M4 tau %err  (xq=2, median over test set) ===")
    print(f"  {'field':6s}  median     mean      max")
    for key in ("up4", "up2", "PhiR", "fine"):
        a = np.array(rows[key]["t"])
        a = a[np.isfinite(a)]
        print(f"  {key:6s}  {np.median(a):7.4f}%  {np.mean(a):7.4f}%  {np.max(a):8.4f}%")

    # ---------------- PER-SAMPLE alignment: FNO hybrid vs its target Phi_R ----
    print("\n" + "=" * 72)
    print("PER-SAMPLE: FNO hybrid M4 vs its OWN training target Phi_R (xq=2)")
    print("=" * 72)
    print("  i   M      omega%: PhiR  FNOhyb   gap | tau%: PhiR  FNOhyb    gap")
    do_w = np.array([r["fno_hyb_omega"] - r["phiR_omega"] for r in persample])
    dt_w = np.array([r["fno_hyb_tau"] - r["phiR_tau"] for r in persample])
    do_b = np.array([r["fno_hyb_omega"] - r["fno_base_omega"] for r in persample])
    dt_b = np.array([r["fno_hyb_tau"] - r["fno_base_tau"] for r in persample])
    show = min(N, 25)
    for r in persample[:show]:
        print(f"  {r['i']:2d}  {r['M']:.3f}   "
              f"{r['phiR_omega']:6.3f} {r['fno_hyb_omega']:6.3f}  "
              f"{r['fno_hyb_omega']-r['phiR_omega']:+6.3f} | "
              f"{r['phiR_tau']:6.3f} {r['fno_hyb_tau']:6.3f}  "
              f"{r['fno_hyb_tau']-r['phiR_tau']:+7.3f}")
    if N > show:
        print(f"  ... ({N-show} more rows omitted; stats below use all {N})")

    print(f"\n  --- paired summary over all {N} test BHs ---")
    print(f"  FNO hybrid worse than its target Phi_R:  "
          f"omega {np.mean(do_w > 0):.0%} of BHs (median gap {np.median(do_w):+.3f}%),  "
          f"tau {np.mean(dt_w > 0):.0%} (median gap {np.median(dt_w):+.3f}%)")
    print(f"  FNO hybrid worse than coarse prior up4:  "
          f"omega {np.mean(do_b > 0):.0%} of BHs (median gap {np.median(do_b):+.3f}%),  "
          f"tau {np.mean(dt_b > 0):.0%} (median gap {np.median(dt_b):+.3f}%)")

    out = "outputs/hybrid/fno_sw_richardson/eval/richardson_target_qnm_persample.json"
    json.dump({
        "note": ("Per-BH M4 QNM @xq2: FNO hybrid (from eval per_sample.json) vs "
                 "its own label-free training target Phi_R=(4*up2-up4)/3 computed "
                 "directly from coarse FD solves (no network). M-rescaled."),
        "n": N, "per_sample": persample,
        "summary": {
            "hyb_worse_than_phiR_frac_omega": float(np.mean(do_w > 0)),
            "hyb_worse_than_phiR_frac_tau": float(np.mean(dt_w > 0)),
            "hyb_minus_phiR_median_omega": float(np.median(do_w)),
            "hyb_minus_phiR_median_tau": float(np.median(dt_w)),
        },
    }, open(out, "w"), indent=2)
    print(f"\n  saved per-sample JSON -> {out}")
    print("\n  VERDICT: if FNO hybrid is worse than Phi_R on a MAJORITY of BHs,")
    print("  the QNM degradation is a per-sample NETWORK artifact, not a median")
    print("  fluke and not a defect of the Richardson target.")


if __name__ == "__main__":
    main()

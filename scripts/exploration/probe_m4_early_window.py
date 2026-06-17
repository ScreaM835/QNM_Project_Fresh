"""
Quick probe: re-run M4 two-mode plateau on existing waveforms with an EARLY
start-time window, the Giesler+2019 recipe. No retraining: just refits the
two-mode NLS on data already on disk.

Searches the strain peak per waveform, scans t0 in [t_peak, t_peak + 12 M],
and prints the resulting (Mw_0, Mw_1, tau_0/M, tau_1/M) plateau averages.
"""
from __future__ import annotations
import os, sys, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.qnm import qnm_method_4_window_scan  # noqa: E402

THEORY = {"Mw0": 0.373672, "tauOverM0": 11.2407,
          "Mw1": 0.346711, "tauOverM1": 3.65114}

CASES = [
    # (label, npz_path, xq, M)  -- xq matches each run's config
    ("FD reference (M=1.009, xq=2)",
     "outputs/fd/fno_zerilli_l2_v2_GT_001_M1.009_fd.npz", 2.0, 1.009),
    ("FNO prediction (M=1.009, xq=2)",
     "outputs/fd/fno_zerilli_l2_v2_pred_001_M1.009_fd.npz", 2.0, 1.009),
    ("Forward PINN (M=1)",
     "outputs/pinn/zerilli_l2_greedy_f03_lbfgs30k/"
     "zerilli_l2_greedy_f03_lbfgs30k_pinn.npz", 10.0, 1.0),
    ("Inverse PINN combo (M=1)",
     "outputs/pinn/zerilli_l2_inverse_qnm_combo/"
     "zerilli_l2_inverse_qnm_combo_pinn.npz", 10.0, 1.0),
    ("Curriculum 3w PINN (M=1)",
     "outputs/pinn/zerilli_l2_curriculum_3w/"
     "zerilli_l2_curriculum_3w_pinn.npz", 10.0, 1.0),
]


def _try_load(path):
    full = os.path.join(ROOT, path)
    if not os.path.isfile(full):
        return None
    return np.load(full)


def probe(label, path, xq, M):
    z = _try_load(path)
    if z is None:
        print(f"[skip] {label}: {path} missing"); return
    x, t, phi = z["x"], z["t"], z["phi"]
    ix = int(np.argmin(np.abs(x - xq)))
    y = phi[:, ix]
    # Locate strain peak (max |y|) within t in [0, 30 M]
    win = (t >= 0.0) & (t <= 30.0)
    if win.sum() < 8:
        print(f"[skip] {label}: too few samples in [0,30]M"); return
    tpk = float(t[win][np.argmax(np.abs(y[win]))])
    print(f"\n[{label}] peak at t={tpk:.2f} M, |peak|={np.max(np.abs(y[win])):.3e}")
    for (t0_lo, t0_hi, n_starts, tag) in [
        (max(0.0, tpk),         tpk + 12.0, 13, "early (peak->peak+12)"),
        (max(0.0, tpk + 2.0),   tpk + 14.0, 13, "early+2  (peak+2->peak+14)"),
        (10.0, 25.0, 16, "standard [10,25] M"),
        (20.0, 35.0, 16, "late [20,35] M"),
        (25.0, 40.0, 16, "later [25,40] M"),
    ]:
        r = qnm_method_4_window_scan(
            t, y, t_start_min=t0_lo, t_start_max=t0_hi, t_end=50.0,
            n_starts=n_starts, potential="zerilli", ell=2,
        )
        Mw0 = M * r["omega"]; tau0 = r["tau"] / M
        Mw1 = M * r["omega1"]; tau1 = r["tau1"] / M
        e0 = 100*abs(Mw0 - THEORY["Mw0"])/THEORY["Mw0"]
        e1 = 100*abs(Mw1 - THEORY["Mw1"])/THEORY["Mw1"]
        et0 = 100*abs(tau0 - THEORY["tauOverM0"])/THEORY["tauOverM0"]
        et1 = 100*abs(tau1 - THEORY["tauOverM1"])/THEORY["tauOverM1"]
        print(f"  {tag:30s}  Mw0={Mw0:.4f} ({e0:5.2f}%)  tau0/M={tau0:6.3f} ({et0:5.1f}%)"
              f"  | Mw1={Mw1:.4f} ({e1:5.1f}%)  tau1/M={tau1:6.3f} ({et1:5.1f}%)")


def main():
    for label, path, xq, M in CASES:
        probe(label, path, xq, M)


if __name__ == "__main__":
    main()

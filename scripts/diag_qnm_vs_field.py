"""Diagnostic: why does the hybrid improve global field L2 but degrade QNM?

Resolves the apparent paradox and tests fixability, with NO retraining:
  1. GLOBAL field rL2 (hyb vs fine, prior vs fine) -- reproduces the eval.
  2. WINDOWED rL2 at the observer x_q over the QNM window [t_start,t_end]
     -- the prediction is hyb << prior globally but hyb >= prior in-window.
  3. FFT of the correction (hyb-prior) in-window -> texture (high-freq, fixable
     by low-pass) vs coherent bias (low-freq, needs tail-weighted loss).
  4. Low-pass the hyb observer series, re-extract omega/tau -> does the QNM
     snap back to the prior level? (direct fixability test).

Inference-only; intended for a handful of samples on the login node.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.hybrid_data_pipe import assemble_split
from src.hybrid_dataset import load_dataset
from src.hybrid_fno import build_hybrid_fno
from src.qnm import qnm_method_2, qnm_method_4_window_scan, percentage_errors

warnings.filterwarnings("ignore")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--xq", type=float, default=2.0)
    ap.add_argument("--t_start", type=float, default=10.0)
    ap.add_argument("--t_end", type=float, default=50.0)
    ap.add_argument("--ell", type=int, default=2)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg["logging"]["out_dir"]
    ckpt_path = args.ckpt or os.path.join(out_dir, "model_best.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DIAG] device={device}  ckpt={ckpt_path}")

    splits, grid, meta = load_dataset(cfg["dataset"]["path"])
    test = splits["test"]
    n = min(args.n, test["Phi_fine"].shape[0])
    sub = {k: test[k][:n] for k in test}
    print(f"[DIAG] using {n} test samples")

    X, Y, up = assemble_split(
        sub, grid.x_coarse, grid.t_coarse, grid.x_fine, grid.t_fine,
    )

    model = build_hybrid_fno(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"] if isinstance(state, dict)
                          and "model_state" in state else state)
    model.eval()

    delta = np.empty_like(Y)
    with torch.no_grad():
        for i0 in range(0, n, 4):
            i1 = min(i0 + 4, n)
            xb = torch.from_numpy(X[i0:i1]).to(device)
            delta[i0:i1] = model(xb).cpu().numpy()

    psi_fine = sub["Phi_fine"][:n].astype(np.float64)
    psi_prior = up[:n].astype(np.float64)
    psi_hyb = psi_prior + delta[:n, 0].astype(np.float64)

    t = grid.t_fine.astype(np.float64)
    x = grid.x_fine.astype(np.float64)
    ix = int(np.argmin(np.abs(x - args.xq)))
    wmask = (t >= args.t_start) & (t <= args.t_end)
    dt = float(np.median(np.diff(t)))

    def rl2(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean(b ** 2)) + 1e-30))

    # ---- 1+2: global vs windowed-at-observer field error -------------------
    print("\n=== FIELD ERROR vs fine (per sample): GLOBAL field  vs  observer-window ===")
    print(f"  observer x_q={args.xq} (ix={ix}), window t in [{args.t_start},{args.t_end}]")
    print(f"  {'i':>2} | {'glob_hyb':>9}{'glob_pri':>9} | {'win_hyb':>9}{'win_pri':>9} | {'hyb worse in-window?':>20}")
    gw = []
    for i in range(n):
        g_h = rl2(psi_hyb[i], psi_fine[i])
        g_p = rl2(psi_prior[i], psi_fine[i])
        w_h = rl2(psi_hyb[i, wmask, ix], psi_fine[i, wmask, ix])
        w_p = rl2(psi_prior[i, wmask, ix], psi_fine[i, wmask, ix])
        gw.append((g_h, g_p, w_h, w_p))
        flag = "YES <--" if w_h > w_p else "no"
        print(f"  {i:>2} | {g_h:9.4f}{g_p:9.4f} | {w_h:9.4f}{w_p:9.4f} | {flag:>20}")
    gw = np.array(gw)
    print(f"  MEDIAN | {np.median(gw[:,0]):9.4f}{np.median(gw[:,1]):9.4f} | "
          f"{np.median(gw[:,2]):9.4f}{np.median(gw[:,3]):9.4f} |")
    print(f"  -> GLOBAL: hybrid better in {(gw[:,0]<gw[:,1]).sum()}/{n}.   "
          f"IN-WINDOW: hybrid WORSE in {(gw[:,2]>gw[:,3]).sum()}/{n}.")

    # ---- 3: texture vs bias in the correction (FFT in window) --------------
    print("\n=== CORRECTION (hyb-prior) AT OBSERVER, IN WINDOW: texture vs bias ===")
    # QNM angular freq ~ omega_R/M ~ 0.37 -> period ~17M; texture is higher freq.
    f_qnm = 0.3737  # rad/M units approx; cut at 3x for 'high freq' bucket
    fcut = 3.0 * f_qnm / (2 * np.pi)  # cycles per M
    for i in range(min(n, 4)):
        corr = psi_hyb[i, wmask, ix] - psi_prior[i, wmask, ix]
        sig = psi_fine[i, wmask, ix]
        sp = np.fft.rfft(corr * np.hanning(len(corr)))
        fr = np.fft.rfftfreq(len(corr), d=dt)
        pw = np.abs(sp) ** 2
        hi = pw[fr > fcut].sum() / (pw.sum() + 1e-30)
        print(f"  i={i}: |corr|_rms={np.sqrt(np.mean(corr**2)):.2e}  "
              f"|sig|_rms={np.sqrt(np.mean(sig**2)):.2e}  "
              f"frac corr power above {fcut:.3f} cyc/M = {hi*100:5.1f}%")

    # ---- 4: low-pass the hybrid observer series, re-extract QNM ------------
    try:
        from scipy.signal import savgol_filter
        have_sg = True
    except Exception:
        have_sg = False
    win = int(max(5, round(2.0 / dt)))          # ~2M smoothing window
    if win % 2 == 0:
        win += 1
    print(f"\n=== QNM re-extraction (method 2), low-pass = Savgol win={win} (~2M) ===")
    print(f"  {'i':>2}{'M':>6} | {'w%_pri':>8}{'w%_hyb':>8}{'w%_hLP':>8} | "
          f"{'t%_pri':>8}{'t%_hyb':>8}{'t%_hLP':>8}")
    agg = {k: [] for k in ("w_pri", "w_hyb", "w_hlp", "t_pri", "t_hyb", "t_hlp")}
    for i in range(n):
        M = float(sub["P"][i, 0])
        y_pri = psi_prior[i, :, ix]
        y_hyb = psi_hyb[i, :, ix]
        y_hlp = savgol_filter(y_hyb, win, 3) if have_sg else y_hyb
        try:
            e_pri = percentage_errors(qnm_method_2(t, y_pri, args.t_start, args.t_end), M=M)
            e_hyb = percentage_errors(qnm_method_2(t, y_hyb, args.t_start, args.t_end), M=M)
            e_hlp = percentage_errors(qnm_method_2(t, y_hlp, args.t_start, args.t_end), M=M)
            agg["w_pri"].append(e_pri["omega_pct_err"]); agg["t_pri"].append(e_pri["tau_pct_err"])
            agg["w_hyb"].append(e_hyb["omega_pct_err"]); agg["t_hyb"].append(e_hyb["tau_pct_err"])
            agg["w_hlp"].append(e_hlp["omega_pct_err"]); agg["t_hlp"].append(e_hlp["tau_pct_err"])
            print(f"  {i:>2}{M:6.2f} | {e_pri['omega_pct_err']:8.4f}{e_hyb['omega_pct_err']:8.4f}"
                  f"{e_hlp['omega_pct_err']:8.4f} | {e_pri['tau_pct_err']:8.4f}"
                  f"{e_hyb['tau_pct_err']:8.4f}{e_hlp['tau_pct_err']:8.4f}")
        except Exception as e:
            print(f"  {i:>2}{M:6.2f} | fit fail: {type(e).__name__}")
    med = lambda a: float(np.median(np.array(a))) if a else float("nan")
    print(f"  MED    |        {med(agg['w_pri']):8.4f}{med(agg['w_hyb']):8.4f}{med(agg['w_hlp']):8.4f} |"
          f"        {med(agg['t_pri']):8.4f}{med(agg['t_hyb']):8.4f}{med(agg['t_hlp']):8.4f}")
    print("\n  Reading: if w%_hLP/t%_hLP drop back toward w%_pri/t%_pri, the QNM")
    print("  degradation is HIGH-FREQ TEXTURE (fixable by smoothness/low-pass).")
    print("  If low-pass does NOT recover it, it is a coherent late-time BIAS.")

    # ---- 5: M4 (sensitive plateau) per-sample prior vs hyb -----------------
    print("\n=== M4 (two-mode plateau scan) per-sample: prior vs hybrid ===")
    print("  (M4 is the hypersensitive extractor; is the summary.json gap real")
    print("   per-sample or an artefact of medianing tiny noisy numbers?)")
    print(f"  {'i':>2}{'M':>6} | {'w%_pri':>8}{'w%_hyb':>8}{'dw':>9} | "
          f"{'t%_pri':>8}{'t%_hyb':>8}{'dt':>9}")
    m4 = {k: [] for k in ("w_pri", "w_hyb", "t_pri", "t_hyb")}
    for i in range(n):
        M = float(sub["P"][i, 0])
        try:
            r_pri = qnm_method_4_window_scan(t, psi_prior[i, :, ix],
                       t_start_min=args.t_start, t_start_max=args.t_start + 8.0,
                       t_end=args.t_end, n_starts=12, ell=args.ell)
            r_hyb = qnm_method_4_window_scan(t, psi_hyb[i, :, ix],
                       t_start_min=args.t_start, t_start_max=args.t_start + 8.0,
                       t_end=args.t_end, n_starts=12, ell=args.ell)
            e_pri = percentage_errors(r_pri, M=M)
            e_hyb = percentage_errors(r_hyb, M=M)
            dw = e_hyb["omega_pct_err"] - e_pri["omega_pct_err"]
            dtt = e_hyb["tau_pct_err"] - e_pri["tau_pct_err"]
            m4["w_pri"].append(e_pri["omega_pct_err"]); m4["w_hyb"].append(e_hyb["omega_pct_err"])
            m4["t_pri"].append(e_pri["tau_pct_err"]); m4["t_hyb"].append(e_hyb["tau_pct_err"])
            print(f"  {i:>2}{M:6.2f} | {e_pri['omega_pct_err']:8.4f}{e_hyb['omega_pct_err']:8.4f}"
                  f"{dw:+9.4f} | {e_pri['tau_pct_err']:8.4f}{e_hyb['tau_pct_err']:8.4f}{dtt:+9.4f}")
        except Exception as e:
            print(f"  {i:>2}{M:6.2f} | M4 fail: {type(e).__name__}")
    if m4["w_pri"]:
        print(f"  MED    |        {med(m4['w_pri']):8.4f}{med(m4['w_hyb']):8.4f}"
              f"{med(m4['w_hyb'])-med(m4['w_pri']):+9.4f} |        "
              f"{med(m4['t_pri']):8.4f}{med(m4['t_hyb']):8.4f}{med(m4['t_hyb'])-med(m4['t_pri']):+9.4f}")
        wins = sum(1 for a, b in zip(m4["w_hyb"], m4["w_pri"]) if a < b)
        print(f"  -> M4 omega: hybrid better in {wins}/{len(m4['w_pri'])} samples "
              f"(sign of dw is mixed => extractor noise; consistent => real bias)")


if __name__ == "__main__":
    main()

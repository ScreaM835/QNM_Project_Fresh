#!/usr/bin/env python
"""
Mechanism diagnostic: WHY does the hybrid FNO improve global field L2 yet
DEGRADE the plateau QNM extractors (M4/M5) relative to BOTH the coarse prior
AND the fine FD floor?

This script asserts NOTHING. It loads the already-computed canonical fields
(fine FD, coarse-upsampled prior, hybrid = prior + FNO correction) for one
black hole and measures, in order:

  TEST A  Observer time series at x_q = 2: signal envelope vs absolute error of
          prior and hybrid, in 5M time bins. Does the hybrid error DECAY with
          the exponential ringdown tail (like a wave-equation solution), or
          stay on a flat absolute floor?  ->  relative error per bin.

  TEST B  Loss accounting. The training loss is global field MSE. What FRACTION
          of that MSE lives in the high-amplitude early field vs the late tail
          where the QNM is read?  Does the FNO have any incentive to fix the
          tail?

  TEST C  Frequency content of the hybrid correction at the observer (FFT).
          Is the tail contamination high-frequency "texture" or a smooth
          low-frequency bias?  (Sanity check against an earlier claim.)

  TEST D  DECISIVE: extract M4 (plateau two-mode) and M2 (robust single-mode)
          omega/tau for fine, prior, hybrid as a function of the fit end-time
          t_end in {25,30,...,50}M. If the tail is the culprit, the
          (hybrid - prior) error GAP must grow as the window reaches deeper
          into the contaminated tail, and shrink when t_end is pulled back.

Run (login-safe, pure numpy/scipy):
  venv_csd3/bin/python scripts/diag_qnm_mechanism.py
"""
from __future__ import annotations

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.qnm import (  # noqa: E402
    qnm_method_2,
    qnm_method_4_window_scan,
)

OMEGA_TRUE = 0.3737
TAU_TRUE = 11.241
XQ = 2.0
CACHE = "outputs/hybrid/fno_sw_richardson/field_cache/canonical.npz"


def envelope_peaks(t, y):
    """Robust envelope: interpolate through local maxima of |y|."""
    from scipy.signal import find_peaks
    ay = np.abs(y)
    pk, _ = find_peaks(ay)
    if pk.size < 2:
        return ay
    return np.interp(t, t[pk], ay[pk])


def banner(s):
    print("\n" + "=" * 72)
    print(s)
    print("=" * 72)


def main():
    d = np.load(CACHE)
    x = d["x"]
    t = d["t"]
    pf = d["psi_fine"]          # (Nt, Nx) truth
    pp = d["psi_coarse_up"]     # (Nt, Nx) coarse prior (upsampled)
    ph = d["psi_hybrid"]        # (Nt, Nx) prior + FNO correction
    ix = int(np.argmin(np.abs(x - XQ)))
    print(f"cache={CACHE}")
    print(f"grid: Nt={t.size} (t in [{t[0]:.1f},{t[-1]:.1f}]), "
          f"Nx={x.size} (x in [{x[0]:.1f},{x[-1]:.1f}])")
    print(f"observer x_q={XQ} -> ix={ix} (x={x[ix]:.4f})")
    print(f"global rl2: prior={float(d['rl2_baseline']):.4%}  "
          f"hybrid={float(d['rl2_hybrid']):.4%}")

    yf = pf[:, ix]
    yp = pp[:, ix]
    yh = ph[:, ix]
    ep = yp - yf   # prior error at observer
    eh = yh - yf   # hybrid error at observer

    # ------------------------------------------------------------------ TEST A
    banner("TEST A  observer signal vs error, in 5M time bins (x_q=2)")
    print("  bin[M]      rms|fine|     rms|prior-fine|  rms|hyb-fine|   "
          "relErr_prior  relErr_hyb   hyb/prior")
    edges = np.arange(10, 51, 5)
    rowsA = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (t >= a) & (t < b)
        sf = np.sqrt(np.mean(yf[m] ** 2))
        sp = np.sqrt(np.mean(ep[m] ** 2))
        sh = np.sqrt(np.mean(eh[m] ** 2))
        rp = sp / sf
        rh = sh / sf
        rowsA.append((a, b, sf, sp, sh, rp, rh, rh / rp))
        print(f"  [{a:2d},{b:2d})   {sf:.4e}    {sp:.4e}      {sh:.4e}    "
              f"{rp:8.2%}     {rh:8.2%}    {rh/rp:6.2f}x")
    print("\n  read: rms|fine| should fall ~e-fold per ~11M (tail). For each "
          "bin,\n  relErr = error_rms / signal_rms. If hybrid leaves a flat "
          "absolute\n  floor, rms|hyb-fine| stops falling while rms|fine| keeps "
          "falling, so\n  relErr_hyb climbs into the tail and the last column "
          "exceeds 1x.")

    # absolute-floor check: ratio of error rms (tail / early) vs signal ratio
    m_early = (t >= 10) & (t < 15)
    m_tail = (t >= 45) & (t <= 50)
    sig_drop = np.sqrt(np.mean(yf[m_tail] ** 2)) / np.sqrt(np.mean(yf[m_early] ** 2))
    perr_drop = np.sqrt(np.mean(ep[m_tail] ** 2)) / np.sqrt(np.mean(ep[m_early] ** 2))
    herr_drop = np.sqrt(np.mean(eh[m_tail] ** 2)) / np.sqrt(np.mean(eh[m_early] ** 2))
    print(f"\n  tail/early rms ratio  signal={sig_drop:.3e}  "
          f"prior_err={perr_drop:.3e}  hyb_err={herr_drop:.3e}")
    print("  (if hyb_err ratio >> signal ratio, the hybrid error does NOT "
          "decay\n   with the ringdown -> flat absolute floor in the tail.)")

    # ------------------------------------------------------------------ TEST B
    banner("TEST B  loss accounting: where does the global field MSE live?")
    # global MSE over the FULL 2D field, split by time region
    se_h = (ph - pf) ** 2  # (Nt,Nx) hybrid sq error
    se_p = (pp - pf) ** 2
    tot_h = se_h.mean()
    tot_p = se_p.mean()
    print(f"  global field MSE: prior={tot_p:.4e}  hybrid={tot_h:.4e}")
    print("\n  time region   %of fine signal power   %of hybrid MSE   "
          "%of prior MSE")
    sig_pow = (pf ** 2)
    regions = [(0, 20), (20, 30), (30, 40), (40, 50)]
    for a, b in regions:
        m = (t >= a) & (t < b) if b < 50 else (t >= a) & (t <= b)
        fp = sig_pow[m].sum() / sig_pow.sum()
        fh = se_h[m].sum() / se_h.sum()
        fpr = se_p[m].sum() / se_p.sum()
        print(f"  [{a:2d},{b:2d})M     {fp:8.2%}              {fh:8.2%}         "
              f"{fpr:8.2%}")
    print("\n  read: if the late tail (where M4/M5 read the QNM) carries a "
          "tiny\n  fraction of the MSE, the FNO has almost no loss incentive to "
          "get it\n  right -> it optimizes the high-amplitude early field and "
          "leaves the\n  tail on a floor.")

    # ------------------------------------------------------------------ TEST C
    banner("TEST C  is the hybrid correction at x_q=2 high-freq texture or "
           "low-freq bias?")
    corr = yh - yp  # the FNO correction at the observer
    from scipy.fft import rfft, rfftfreq
    dt = float(t[1] - t[0])
    # restrict to ringdown window to avoid prompt transient
    m = (t >= 10) & (t <= 50)
    cw = corr[m] - corr[m].mean()
    C = np.abs(rfft(cw * np.hanning(cw.size)))
    fr = rfftfreq(cw.size, d=dt)  # cycles/M
    w = 2 * np.pi * fr            # rad/M
    P = C ** 2
    P /= P.sum()
    # fraction of correction power BELOW the qnm frequency vs ABOVE
    below = P[w < OMEGA_TRUE].sum()
    near = P[(w >= OMEGA_TRUE) & (w < 2 * OMEGA_TRUE)].sum()
    above = P[w >= 2 * OMEGA_TRUE].sum()
    kpk = int(np.argmax(P[1:])) + 1
    print(f"  correction power: omega<{OMEGA_TRUE:.3f}: {below:.1%}   "
          f"[{OMEGA_TRUE:.3f},{2*OMEGA_TRUE:.3f}): {near:.1%}   "
          f">{2*OMEGA_TRUE:.3f}: {above:.1%}")
    print(f"  spectral peak of correction at omega={w[kpk]:.4f} rad/M "
          f"(qnm={OMEGA_TRUE})")
    print("  read: if power is overwhelmingly at/below the QNM frequency, the "
          "tail\n  contamination is a smooth LOW-freq bias, NOT high-freq "
          "texture, so a\n  low-pass filter cannot remove it (matches earlier "
          "finding).")

    # ------------------------------------------------------------------ TEST D
    banner("TEST D  DECISIVE: QNM vs fit end-time t_end (t0 scan 10->18M)")
    t_ends = [25, 30, 35, 40, 45, 50]

    def m4(y, te):
        r = qnm_method_4_window_scan(t, y, 10.0, 18.0, float(te),
                                     n_starts=8, potential="zerilli", ell=2)
        return r["omega"], r["tau"]

    def m2(y, te):
        try:
            r = qnm_method_2(t, y, 10.0, float(te))
            return r["omega"], r["tau"]
        except Exception:
            return float("nan"), float("nan")

    def pct(v, ref):
        return abs(v - ref) / ref * 100

    print("\n  -- Method 4 (plateau two-mode) omega %err --")
    print("  t_end   fine     prior    hybrid    (hyb-prior gap)")
    gapsD = []
    for te in t_ends:
        of, _ = m4(yf, te)
        op, _ = m4(yp, te)
        oh, _ = m4(yh, te)
        pf_ = pct(of, OMEGA_TRUE)
        pp_ = pct(op, OMEGA_TRUE)
        ph_ = pct(oh, OMEGA_TRUE)
        gap = ph_ - pp_
        gapsD.append((te, pf_, pp_, ph_, gap))
        print(f"  {te:3d}    {pf_:6.3f}%  {pp_:6.3f}%  {ph_:6.3f}%   "
              f"{gap:+6.3f}%")

    print("\n  -- Method 4 (plateau two-mode) tau %err --")
    print("  t_end   fine     prior    hybrid    (hyb-prior gap)")
    for te in t_ends:
        _, tf_ = m4(yf, te)
        _, tp_ = m4(yp, te)
        _, th_ = m4(yh, te)
        a = pct(tf_, TAU_TRUE)
        b = pct(tp_, TAU_TRUE)
        c = pct(th_, TAU_TRUE)
        print(f"  {te:3d}    {a:6.3f}%  {b:6.3f}%  {c:6.3f}%   {c-b:+6.3f}%")

    print("\n  -- Method 2 (robust single-mode, full window) omega %err --")
    print("  t_end   fine     prior    hybrid    (hyb-prior gap)")
    for te in t_ends:
        of, _ = m2(yf, te)
        op, _ = m2(yp, te)
        oh, _ = m2(yh, te)
        a = pct(of, OMEGA_TRUE)
        b = pct(op, OMEGA_TRUE)
        c = pct(oh, OMEGA_TRUE)
        print(f"  {te:3d}    {a:6.3f}%  {b:6.3f}%  {c:6.3f}%   {c-b:+6.3f}%")

    print("\n  read: if the (hyb-prior) gap in M4 GROWS with t_end (window "
          "reaches\n  deeper into the contaminated tail) but M2 (amplitude-"
          "weighted over the\n  whole window) barely moves, the tail "
          "contamination is the mechanism.")

    banner("DONE")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
TEST THE MODE-TRUNCATION HYPOTHESIS (training-free).

Claim under test: the FNO's spectral layers keep only modes_t=16 temporal and
modes_x=32 spatial Fourier modes, and that band-limit is too small to represent
the ringdown's spectral structure -> QNM (esp. tau) is distorted no matter what
the loss does.

Decisive probe: take the CLEAN target field Phi_R=(4*up2-up4)/3 (verified good
QNM, ~fine) and spectrally TRUNCATE it exactly as an FNO spectral conv would
(keep lowest modes_t along t from both freq ends, lowest modes_x along x via
rfft), inverse-transform, and extract the M4 QNM at x_q=2. Sweep modes_t.

  - If truncation to modes_t=16 ALONE destroys the QNM -> the representation is
    the bottleneck (your hypothesis is right; fix = raise modes_t).
  - If the truncated field still has good QNM at modes_t=16 -> mode count is NOT
    the bottleneck (the FNO COULD represent it; the failure is elsewhere).

NOTE on faithfulness: a real FNO is NOT a pure low-pass — each block has a W
bypass (1x1 conv) that can carry high frequencies, plus nonlinearities. So pure
truncation is an UPPER BOUND on the harm from limited modes: if even pure
truncation does not hurt the QNM, limited modes are definitely not the cause.

Login-safe-ish (FFT cheap; QNM fits are the cost). Default 3 BHs; pass N.
  venv_csd3/bin/python scripts/diag_mode_truncation.py [N]
"""
from __future__ import annotations

import os
import sys
import numpy as np
from scipy.fft import dct, idct

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.hybrid_data_pipe import upsample_to_fine        # noqa: E402
from src.qnm import qnm_method_4_window_scan             # noqa: E402

OMEGA_TRUE = 0.3737
TAU_TRUE = 11.241
XQ = 2.0
K2 = "outputs/hybrid/dataset_sw_k2.npz"
K4 = "outputs/hybrid/dataset_sw_k4.npz"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3


def pct(v, ref):
    return abs(v - ref) / ref * 100.0


def truncate_t_only(field, modes_t):
    """Keep the lowest `modes_t` temporal Fourier modes (both freq ends), all x.

    field: (Nt, Nx). FFT along t (axis 0), zero |freq-mode| >= modes_t, inverse.
    This isolates the TEMPORAL band-limit (the QNM is a temporal-frequency
    phenomenon), which is the cleanest test of 'modes_t too small'.

    WARNING: raw DFT truncation of a NON-periodic transient injects Gibbs ringing
    (wraparound discontinuity), which corrupts the tail independent of modes_t.
    Use the *_pad / *_dct variants below for a faithful FNO-style probe.
    """
    Nt = field.shape[0]
    F = np.fft.fft(field, axis=0)
    mask = np.zeros(Nt, dtype=bool)
    mask[:modes_t] = True
    mask[-(modes_t - 1):] = True if modes_t > 1 else False
    F[~mask, :] = 0.0
    return np.fft.ifft(F, axis=0).real


def truncate_t_pad(field, modes_t, pad_frac=0.10):
    """FNO-faithful: zero-pad t by pad_frac (mimics domain_padding=0.10), keep
    lowest modes_t on the PADDED grid, inverse, crop back. Reduces the wraparound
    Gibbs artifact the way the real FNO does."""
    Nt, Nx = field.shape
    npad = int(round(pad_frac * Nt))
    fp = np.concatenate([field, np.zeros((npad, Nx))], axis=0)
    Np = fp.shape[0]
    F = np.fft.fft(fp, axis=0)
    mask = np.zeros(Np, dtype=bool)
    mask[:modes_t] = True
    if modes_t > 1:
        mask[-(modes_t - 1):] = True
    F[~mask, :] = 0.0
    out = np.fft.ifft(F, axis=0).real
    return out[:Nt, :]


def truncate_t_dct(field, modes_t):
    """Cleanest capacity probe: DCT-II along t (even reflection => NO wraparound
    discontinuity), keep lowest modes_t cosine modes, inverse. Answers purely
    'can modes_t basis functions represent this signal?' with no Gibbs confound."""
    C = dct(field, type=2, axis=0, norm="ortho")
    C[modes_t:, :] = 0.0
    return idct(C, type=2, axis=0, norm="ortho")


def truncate_2d_fno(field, modes_t, modes_x):
    """FNO-style: keep lowest modes_t along t (both ends) and lowest modes_x
    along x (rfft, low end). field (Nt, Nx)."""
    Nt, Nx = field.shape
    F = np.fft.rfft(field, axis=1)            # (Nt, Nx//2+1)
    F = np.fft.fft(F, axis=0)                 # full FFT along t
    mt = np.zeros(Nt, dtype=bool)
    mt[:modes_t] = True
    if modes_t > 1:
        mt[-(modes_t - 1):] = True
    F[~mt, :] = 0.0
    F[:, modes_x:] = 0.0                      # drop high spatial modes
    F = np.fft.ifft(F, axis=0)
    return np.fft.irfft(F, n=Nx, axis=1).real


def main():
    d2 = np.load(K2)
    d4 = np.load(K4)
    xf = d2["x_fine"].astype(float)
    tf = d2["t_fine"].astype(float)
    x2 = d2["x_coarse"].astype(float); t2 = d2["t_coarse"].astype(float)
    x4 = d4["x_coarse"].astype(float); t4 = d4["t_coarse"].astype(float)
    P = d4["test_P"]
    ix = int(np.argmin(np.abs(xf - XQ)))
    Nt = tf.size

    # frequency reference: temporal mode k <-> omega = k * 2*pi / T
    T = tf[-1] - tf[0]
    dw = 2 * np.pi / T
    print(f"grid Nt={Nt}, T={T:.0f}M, temporal mode spacing dω={dw:.4f} rad/M")
    print(f"QNM ω_true={OMEGA_TRUE} rad/M  -> temporal mode k≈{OMEGA_TRUE/dw:.2f}")
    print(f"envelope decay 1/τ={1/TAU_TRUE:.4f} /M -> mode k≈{(1/TAU_TRUE)/dw:.2f}")
    print(f"=> a damped sinusoid's spectral peak sits near mode 3 with Lorentzian "
          f"width ~1 mode.\nobserver x_q={XQ} ix={ix}; N={N} BHs\n")

    def m4(y, M):
        r = qnm_method_4_window_scan(tf, y, 10.0, 18.0, 50.0, n_starts=12,
                                     potential="zerilli", ell=2)
        return r["omega"] * M, r["tau"] / M

    # build the clean Phi_R fields once
    fields = []
    for i in range(N):
        up4 = upsample_to_fine(d4["test_Phi_coarse"][i], x4, t4, xf, tf, "quintic")
        up2 = upsample_to_fine(d2["test_Phi_coarse"][i], x2, t2, xf, tf, "quintic")
        fields.append(((4 * up2 - up4) / 3.0, float(P[i, 0])))

    # ---- TEST 1: temporal-only truncation of the observer signal ------------
    print("=" * 72)
    print("TEST 1  truncate Phi_R to K temporal modes (all x kept), QNM @xq2")
    print("  3 methods: raw DFT (Gibbs-confounded) | PAD (FNO domain_padding) | "
          "DCT (no wraparound)")
    print("=" * 72)
    print(f"  {'modes_t':>7} | {'DFT ω%':>8} {'DFT τ%':>9} | {'PAD ω%':>8} "
          f"{'PAD τ%':>9} | {'DCT ω%':>8} {'DCT τ%':>9}")
    Ks = [4, 6, 8, 12, 16, 24, 32, 64, Nt]
    for K in Ks:
        res = {}
        for name, fn in (("dft", truncate_t_only), ("pad", truncate_t_pad),
                         ("dct", truncate_t_dct)):
            oe, te = [], []
            for fld, M in fields:
                ytr = fn(fld, K)[:, ix]
                o, t_ = m4(ytr, M)
                oe.append(pct(o, OMEGA_TRUE)); te.append(pct(t_, TAU_TRUE))
            res[name] = (np.median(oe), np.median(te))
        tag = " <== FNO modes_t" if K == 16 else (" (none)" if K == Nt else "")
        print(f"  {K:>7} | {res['dft'][0]:7.3f}% {res['dft'][1]:8.3f}% | "
              f"{res['pad'][0]:7.3f}% {res['pad'][1]:8.3f}% | "
              f"{res['dct'][0]:7.3f}% {res['dct'][1]:8.3f}%{tag}")

    # ---- TEST 2: full FNO-style 2D truncation -------------------------------
    print("\n" + "=" * 72)
    print("TEST 2  FNO-style 2D truncation (modes_t along t, modes_x=32 along x)")
    print("=" * 72)
    print(f"  {'modes_t':>7}  modes_x   {'ω%err(med)':>10}  {'τ%err(med)':>10}")
    for mt, mx in [(16, 32), (16, 64), (32, 32), (64, 64), (Nt, 501)]:
        oe, te = [], []
        for fld, M in fields:
            ytr = truncate_2d_fno(fld, mt, mx)[:, ix]
            o, t_ = m4(ytr, M)
            oe.append(pct(o, OMEGA_TRUE)); te.append(pct(t_, TAU_TRUE))
        tag = " <== FNO config" if (mt, mx) == (16, 32) else ""
        print(f"  {mt:>7}  {mx:>5}    {np.median(oe):9.4f}%  "
              f"{np.median(te):9.4f}%{tag}")

    print("\nVERDICT:")
    print("  - If ω/τ stay small (~0.02-0.1%) down to modes_t=16, the 16-mode")
    print("    representation CAN hold the QNM -> mode count is NOT the bottleneck.")
    print("  - If τ blows up as modes_t drops to 16, the band-limit IS the cause")
    print("    -> fix = raise modes_t (architectural), not loss weighting.")


if __name__ == "__main__":
    main()

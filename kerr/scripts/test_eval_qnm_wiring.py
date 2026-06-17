"""Cheap wiring test for the rewritten eval QNM path (no model/corpus needed).

Builds a synthetic single-mode Kerr scri waveform psi(tau)=A exp(-i omega tau)
with a KNOWN Leaver (4,2,0) frequency, then checks:
  1. extract_qnm_scri recovers M*omega to <1% and returns spread/sel_dist/n;
  2. the parallel _eval_qnm_work worker + Pool produce a correctly-shaped row
     with the preserved schema (Mw_err_pct) plus the new error-bar fields.
This validates the ensemble+omega_target wiring and the fork-pool plumbing
before launching the real (hours-long) train+eval job.
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from kerr.src.qnm_kerr_reference import kerr_qnm
import kerr.scripts.train_eval_hybrid_kerr as ev


def synth_scri(tau, mw_R, mw_I, phi=0.3, amp=1.0):
    """psi(tau) = A e^{-i omega tau}, omega = mw_R - i|mw_I| (decaying)."""
    omega = mw_R - 1j * abs(mw_I)
    return amp * np.exp(-1j * omega * tau + 1j * phi)


def main():
    tau = np.arange(0.0, 220.0 + 1e-9, 0.25)
    a = 0.9
    q = kerr_qnm(a_over_M=a, ell=4, m=2, n=0)
    qref = np.array([q.M_omega_R, q.M_omega_I, q.tau_over_M], dtype=np.float64)
    psi = synth_scri(tau, q.M_omega_R, q.M_omega_I)

    print(f"synthetic (4,2,0) a={a}: Leaver Mw={q.M_omega_R:.4f}  tau={q.tau_over_M:.3f}")

    # 1) direct helper
    out = ev.extract_qnm_scri(psi.real, psi.imag, a, tau, qref)
    err = abs(out["Mw"] - q.M_omega_R) / q.M_omega_R * 100.0
    print(f"  [1] extract_qnm_scri: Mw={out['Mw']:.4f}  err={err:.2f}%  "
          f"spread={out['spread_pct']:.2f}%  sel_dist={out['sel_dist']:.3f}  "
          f"n={out['n_methods']}")
    assert np.isfinite(out["Mw"]) and err < 1.0, "helper failed to recover omega"

    # 2) worker + pool with a synthetic (N=3, Ntau, Nsigma=1) field stack
    N, Nt = 3, tau.size
    def stk(p):
        a3 = np.empty((N, Nt, 1), dtype=np.float32)
        a3[:, :, 0] = p[None, :]
        return a3
    re = stk(psi.real); im = stk(psi.imag)
    P = np.tile([a, 9.5, 1.25], (N, 1)).astype(np.float32)
    qnm = np.tile(qref, (N, 1))
    fl = np.full(N, 0.1, dtype=np.float64)
    gvars = dict(tau=tau, scri=0, P=P, qnm=qnm,
                 up4_re=re, up4_im=im, hyb_re=re, hyb_im=im,
                 rich_re=re, rich_im=im, fine_re=re, fine_im=im,
                 fl_prior=fl, fl_hyb=fl, fl_rich=fl)
    with Pool(processes=2, initializer=ev._eval_qnm_init, initargs=(gvars,)) as pool:
        rows = list(pool.imap_unordered(ev._eval_qnm_work, range(N)))
    rows.sort(key=lambda r: r["i"])
    r0 = rows[0]
    keys_ok = all(k in r0["qnm"]["prior"] for k in
                  ("Mw", "tau", "Mw_err_pct", "tau_err_pct", "spread_pct",
                   "sel_dist", "n_methods"))
    print(f"  [2] pool worker: rows={len(rows)} schema_ok={keys_ok}  "
          f"prior.Mw_err={r0['qnm']['prior']['Mw_err_pct']:.2f}%  "
          f"field_rl2_prior={r0['field_rl2_prior']}")
    assert len(rows) == N and keys_ok, "worker/pool schema broken"
    assert r0["qnm"]["prior"]["Mw_err_pct"] < 1.0, "worker omega wrong"

    print("\nALL WIRING TESTS PASSED")


if __name__ == "__main__":
    main()

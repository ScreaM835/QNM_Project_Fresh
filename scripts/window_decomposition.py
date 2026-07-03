"""Decompose the aligned-window improvement: start-shift vs end-extension.

Three window schemes on the T=100 fine FD solve (canonical BH):
  fixed    [10, 50]              -- current protocol
  start    [10+d, 50]            -- aligned start, end clamped at the corpus
                                     horizon t=50 (usable by ALL current models)
  full     [10+d, 50+d]          -- fully retarded-aligned (needs t>50 data)
with d = xq - 2. Also applies the start scheme to the existing HYBRID
canonical field (t<=50 cache) at xq=10 to see what the trained model gains
without any retraining. Additive script; no core code touched.
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from src.fd_solver import solve_fd                 # noqa: E402
from eval_hybrid_sw import _all_methods            # noqa: E402


def _fd(tmax):
    cfg = {
        "physics": {"M": 1.0, "potential": "zerilli", "l": 2,
                    "pde_sign": "standard"},
        "initial_data": {"A": 1.0, "x0": 4.0, "sigma": 5.0,
                         "velocity_profile": "outgoing"},
        "domain": {"xmin": -50.0, "xmax": 150.0, "tmin": 0.0, "tmax": tmax},
        "fd": {"dx": 0.2, "dt": 0.1, "scheme": "rk4_mol"},
    }
    sol = solve_fd(cfg)
    return (sol["x"].astype(np.float64), sol["t"].astype(np.float64),
            sol["phi"].astype(np.float64))


def main() -> None:
    print("[DECOMP] solving fine FD tmax=100 ...")
    x, t, psi = _fd(100.0)

    for xq in (10.0, 15.0, 20.0):
        ix = int(np.argmin(np.abs(x - xq)))
        d = xq - 2.0
        schemes = {
            "fixed [10,50]":               (10.0, 50.0),
            f"start [{10+d:g},50]":        (10.0 + d, 50.0),
            f"full  [{10+d:g},{50+d:g}]":  (10.0 + d, 50.0 + d),
        }
        print(f"\n=== fine FD, xq = {xq:g} M ===")
        print("scheme                |    M4 w%    M4 tau% |    M5 w%    M5 tau%")
        for name, (t0, t1) in schemes.items():
            r = _all_methods(t, psi[:, ix], t0, t1,
                             potential="zerilli", ell=2, M=1.0)
            m4, m5 = r["M4"], r["M5"]
            print(f"{name:21s} | {m4['omega_pct_err']:8.3f} {m4['tau_pct_err']:9.3f}"
                  f" | {m5['omega_pct_err']:8.3f} {m5['tau_pct_err']:9.3f}")

    # ---- hybrid field, within its own corpus horizon --------------------
    cache = np.load(os.path.join(ROOT, "outputs", "hybrid", "fno_sw_gate_s1em3",
                                 "field_cache", "canonical.npz"))
    xh, th = cache["x"], cache["t"]
    ix = int(np.argmin(np.abs(xh - 10.0)))
    print("\n=== hybrid corpus fields (t<=50), xq = 10 M ===")
    print("source   scheme       |    M4 w%    M4 tau% |    M5 w%    M5 tau%")
    for label, key in (("hybrid", "psi_hybrid"), ("baseline", "psi_coarse_up"),
                       ("fine", "psi_fine")):
        y = cache[key][:, ix].astype(np.float64)
        for name, (t0, t1) in (("fixed [10,50]", (10.0, 50.0)),
                               ("start [18,50]", (18.0, 50.0))):
            r = _all_methods(th, y, t0, t1, potential="zerilli", ell=2, M=1.0)
            m4, m5 = r["M4"], r["M5"]
            print(f"{label:8s} {name:12s} | {m4['omega_pct_err']:8.3f} {m4['tau_pct_err']:9.3f}"
                  f" | {m5['omega_pct_err']:8.3f} {m5['tau_pct_err']:9.3f}")


if __name__ == "__main__":
    main()

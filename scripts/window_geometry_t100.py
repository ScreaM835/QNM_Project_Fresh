"""Test: is the large-xq extraction degradation just window geometry?

Doubles the evolution time (tmax=100) for the canonical BH using ONLY the
FD solver (fine reference + coarse k=4 quintic-upsampled baseline -- no
trained model needed) and extracts QNMs at each observer with:
  (a) the fixed window [10, 50] M      (what the existing sweep used)
  (b) a retarded-time-aligned window [10 + (xq-2), 50 + (xq-2)] M
so every observer gets the same 40 M of usable ringdown measured from its
own signal-arrival time. If hypothesis is right, (b) restores xq=2-level
accuracy at xq = 10-20 while (a) stays broken.

Additive analysis script; no core code touched.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from src.fd_solver import solve_fd                # noqa: E402
from src.hybrid_data_pipe import upsample_to_fine  # noqa: E402
from eval_hybrid_sw import _all_methods            # noqa: E402

TMAX = 100.0
XQS = [2.0, 5.0, 10.0, 15.0, 20.0]


def _fd(dx, dt):
    cfg = {
        "physics": {"M": 1.0, "potential": "zerilli", "l": 2,
                    "pde_sign": "standard"},
        "initial_data": {"A": 1.0, "x0": 4.0, "sigma": 5.0,
                         "velocity_profile": "outgoing"},
        "domain": {"xmin": -50.0, "xmax": 150.0, "tmin": 0.0, "tmax": TMAX},
        "fd": {"dx": float(dx), "dt": float(dt), "scheme": "rk4_mol"},
    }
    sol = solve_fd(cfg)
    return (sol["x"].astype(np.float64), sol["t"].astype(np.float64),
            sol["phi"].astype(np.float64))


def main() -> None:
    print(f"[T100] solving fine FD (dx=0.2, dt=0.1, tmax={TMAX:g}) ...")
    x_f, t_f, psi_f = _fd(0.2, 0.1)
    print(f"[T100] fine grid Nt={len(t_f)} Nx={len(x_f)}")
    print("[T100] solving coarse k=4 FD (dx=0.8, dt=0.4) + quintic upsample ...")
    x_c, t_c, psi_c = _fd(0.8, 0.4)
    psi_cu = upsample_to_fine(psi_c, x_c, t_c, x_f, t_f)

    sources = {"fine": psi_f, "baseline": psi_cu}
    results = {"tmax": TMAX}
    methods = ["M1", "M2", "M3", "M4", "M5"]

    for xq in XQS:
        ix = int(np.argmin(np.abs(x_f - xq)))
        shift = xq - 2.0
        wins = {"fixed_10_50": (10.0, 50.0),
                "aligned": (10.0 + shift, 50.0 + shift)}
        key = f"xq{xq:g}"
        results[key] = {}
        for wname, (t0, t1) in wins.items():
            results[key][wname] = {"window": [t0, t1]}
            for label, field in sources.items():
                res = _all_methods(t_f, field[:, ix], t0, t1,
                                   potential="zerilli", ell=2, M=1.0)
                results[key][wname][label] = res

    for xq in XQS:
        key = f"xq{xq:g}"
        for wname in ("fixed_10_50", "aligned"):
            t0, t1 = results[key][wname]["window"]
            print(f"\n=== xq = {xq:g} M, window [{t0:g},{t1:g}] M ({wname}) ===")
            print("method |     fine w%      fine tau% | baseline w%  baseline tau%")
            for m in methods:
                row = [m]
                for s in ("fine", "baseline"):
                    d = results[key][wname][s][m]
                    row.append(f"{d['omega_pct_err']:9.3f}  {d['tau_pct_err']:12.3f}")
                print("  ".join(row))

    out = os.path.join(ROOT, "outputs", "hybrid", "fno_sw_gate_s1em3",
                       "eval", "window_geometry_t100.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n[T100] wrote {out}")


if __name__ == "__main__":
    main()

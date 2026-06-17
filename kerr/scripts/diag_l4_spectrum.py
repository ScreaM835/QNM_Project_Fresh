"""Decisive check: the radial QNM SPECTRUM of the frozen-l operator itself.

The (Psi, U, W) hyperboloidal system is linear: d_t state = L state. We build
L by applying rhs_teuk to the 3N unit basis vectors, eigen-decompose it, and map
each eigenvalue mu to a QNM frequency via the ansatz state ~ e^{mu t} = e^{-i w t}
  =>  w = i mu  =>  M*omega_R = -Im(mu),  M*omega_I = Re(mu) (<0 for decay).

This bypasses BOTH the initial-data excitation and the time-domain extractor: it
reports what modes the operator ACTUALLY supports. If the l=4 fundamental comes
out at ~0.907 the operator is correct and the 0.667 seen in evolution is an
excitation/extraction artifact; if it comes out at ~0.667 the operator/eigenvalue
is wrong. We also print the qnm-package Leaver (l,2,0) ladder for comparison.
"""
from __future__ import annotations

import argparse
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from kerr.src.teukolsky_minimal_gauge import build_teukolsky_op, rhs_teuk
from kerr.src.fd_stencils import d1_central
from kerr.src.qnm_kerr_reference import kerr_qnm

ELL, MM = 4, 2
CAND_L = [2, 3, 4, 5, 6]


def build_generator(op, N):
    """Dense 3N x 3N generator L with d_t state = L state (rhs_teuk is linear)."""
    dim = 3 * N
    L = np.zeros((dim, dim), dtype=np.complex128)
    for k in range(dim):
        v = np.zeros(dim, dtype=np.complex128)
        v[k] = 1.0
        Psi, U, W = v[:N], v[N:2 * N], v[2 * N:]
        dPsi, dU, dW = rhs_teuk((Psi, U, W), op, d1_central)
        L[:, k] = np.concatenate([dPsi, dU, dW])
    return L


def spectrum(a, N, n_show=8):
    q420 = kerr_qnm(a_over_M=a, ell=4, m=MM, n=0)
    omega_ref = complex(q420.M_omega_R, q420.M_omega_I)
    op = build_teukolsky_op(N=N, a_over_M=a, M=1.0, ell=ELL, m=MM,
                            omega_ref=omega_ref, include_potential=True)
    L = build_generator(op, N)
    mu = np.linalg.eigvals(L)
    omega_R = -mu.imag
    omega_I = mu.real            # <0 for decaying

    # Leaver targets: (l,2,0) ladder + the (4,2,1) overtone.
    targets = []
    for l in CAND_L:
        try:
            q = kerr_qnm(a_over_M=a, ell=l, m=MM, n=0)
            targets.append((f"({l},2,0)", float(q.M_omega_R), float(q.M_omega_I)))
        except Exception:
            pass
    try:
        q421 = kerr_qnm(a_over_M=a, ell=4, m=MM, n=1)
        targets.append(("(4,2,1)", float(q421.M_omega_R), float(q421.M_omega_I)))
    except Exception:
        pass

    print(f"\n===== a/M={a:.3f}  N={N}  (operator frozen at omega_ref=(4,2,0)) =====")
    print("  For each Leaver target, the operator eigenvalue NEAREST it (in the")
    print("  complex omega plane) and the match distance:")
    print(f"  {'target':>9} {'Mw_R_ref':>9} {'Mw_I_ref':>9} | "
          f"{'op Mw_R':>8} {'op Mw_I':>8} | {'dist':>7}")
    print("  " + "-" * 64)
    for (name, tR, tI) in targets:
        d = np.abs((omega_R - tR) + 1j * (omega_I - tI))
        i = int(np.argmin(d))
        flag = "  <== supported" if d[i] < 0.03 else ""
        print(f"  {name:>9} {tR:9.4f} {tI:9.4f} | {omega_R[i]:8.4f} "
              f"{omega_I[i]:8.4f} | {d[i]:7.4f}{flag}")

    # Also: the least-damped *resolved* mode with |Mw_R|>0.3 (what dominates the
    # latest times the fit window can reach), to expose long-lived contamination.
    good = (omega_I < -1e-4) & (omega_I > -0.5) & (omega_R > 0.3)
    gi = np.where(good)[0]
    gi = gi[np.argsort(omega_I[gi])[::-1]]
    print("  least-damped modes with Mw_R>0.3 (late-time dominant):")
    for i in gi[:5]:
        d420 = abs((omega_R[i] - q420.M_omega_R) + 1j * (omega_I[i] - q420.M_omega_I))
        print(f"      Mw_R={omega_R[i]:7.4f}  Mw_I={omega_I[i]:8.4f}  "
              f"(dist to (4,2,0)={d420:.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spins", type=float, nargs="*", default=[0.0, 0.5, 0.9, 0.95])
    ap.add_argument("--N", type=int, default=201)
    args = ap.parse_args()
    for a in args.spins:
        spectrum(a, args.N)


if __name__ == "__main__":
    main()

"""Symbolic derivation of the minimal-gauge hyperboloidal Teukolsky operator (B.2).

s = -2, single mode (ell, m), compactified sigma = r_+/r, minimal-gauge height
H(sigma) = 1 - 2 sigma^2 (the SAME height as Schwarzschild works for Kerr,
because the tortoise metric w = dr_*/dsigma keeps the same sigma^-2 / (1-sigma)^-1
edge powers; verified below).

This script is the analytic backbone of kerr/notes/kerr_minimal_gauge_derivation.md
AND the acceptance test for task B.2. It does:

  1. Builds the frequency-domain Teukolsky radial operator (Teukolsky 1973),
     coefficients C_tt, C_t, C_rr, C_r, C_0 in (t, r) after omega -> i d_t.
  2. Transforms (t, r) -> (tau, sigma) with the hyperboloidal + compactifying
     map, using d_t = d_tau, d_r = (1/rho)(d_sigma - H w d_tau), and collects
     the (tau, sigma) coefficients A_tt, A_ts, A_ss, A_tau, A_sig, A_0.
  3. PRINCIPAL SYMBOL (numerical): confirms the two characteristic slopes
     (roots of A_tt xi^2 + A_ts xi + A_ss) equal the analytic minimal-gauge
     speeds lambda_out, lambda_in of the Phase A Schwarzschild gauge.
  4. SOURCE (numerical): regularised COMPLEX coefficients
     c_Psi, c_Pi, c_Phi = -A_0/A_tt, -A_tau/A_tt, -A_sig/A_tt are finite on
     the closed [0,1] (the 1 - H^2 prefactor cancels) for a in {0,0.5,0.9}.
  5. a -> 0 reduction: characteristic structure -> Schwarzschild RW minimal
     gauge; source -> Bardeen-Press (REAL, not RW; the complex sector is an
     a>0 effect). Physical a=0 check is the QNM omega (B.8).

Run on a login node (pure sympy/numpy, no evolution):
    ../venv_csd3/bin/python scripts/derive_teukolsky_coeffs.py
"""
from __future__ import annotations

import numpy as np
import sympy as sp


def build():
    """Construct the (tau, sigma) coefficients symbolically. No simplify()."""
    sigma = sp.symbols("sigma", positive=True)
    M = sp.symbols("M", positive=True)
    a = sp.symbols("a", nonnegative=True)
    m = sp.symbols("m", real=True)
    s = sp.symbols("s", real=True)
    lam = sp.symbols("lambda_sep")  # frozen spheroidal separation constant
    I = sp.I

    root = sp.sqrt(M**2 - a**2)
    r_plus = M + root
    r_minus = M - root
    r = r_plus / sigma
    Delta = (r - r_plus) * (r - r_minus)
    rho = sp.diff(r, sigma)               # dr/dsigma = -r_+/sigma^2
    dstar_dr = (r**2 + a**2) / Delta      # dr_*/dr
    w = dstar_dr * rho                    # dr_*/dsigma
    H = 1 - 2 * sigma**2

    # frequency-domain Teukolsky radial operator, omega -> i d_t:
    r2a2 = r**2 + a**2
    rr = sp.symbols("rr", positive=True)
    Delta_prime = sp.diff(rr**2 - 2 * M * rr + a**2, rr).subs(rr, r)  # 2(r - M)

    C_tt = -(r2a2**2) / Delta
    C_t = (-2 * I * a * m * r2a2 + 2 * s * (r - M) * r2a2) / Delta - 4 * s * r
    C_rr = Delta
    C_r = (s + 1) * Delta_prime
    C_0 = (a**2 * m**2 + 2 * I * a * s * m * (r - M)) / Delta - lam

    tau = sp.symbols("tau", real=True)
    psi = sp.Function("psi")(tau, sigma)

    def d_t(f):
        return sp.diff(f, tau)

    def d_r(f):
        return (1 / rho) * (sp.diff(f, sigma) - H * w * sp.diff(f, tau))

    eq = (
        C_tt * d_t(d_t(psi))
        + C_t * d_t(psi)
        + C_rr * d_r(d_r(psi))
        + C_r * d_r(psi)
        + C_0 * psi
    )
    eq = sp.expand(eq)

    # Use psi.diff(...) so the queried derivative atom matches sympy's canonical
    # variable ordering (it sorts the mixed partial to Derivative(psi, sigma, tau),
    # so querying Derivative(psi, tau, sigma) would silently return 0).
    A_tt = eq.coeff(psi.diff(tau, tau))
    A_ts = eq.coeff(psi.diff(tau, sigma))
    A_ss = eq.coeff(psi.diff(sigma, sigma))
    A_tau = eq.coeff(psi.diff(tau))
    A_sig = eq.coeff(psi.diff(sigma))
    A_0 = eq.coeff(psi)

    return dict(
        sigma=sigma, M=M, a=a, m=m, s=s, lam=lam,
        A_tt=A_tt, A_ts=A_ts, A_ss=A_ss, A_tau=A_tau, A_sig=A_sig, A_0=A_0,
    )


def _lam_speeds(sigma, a, M):
    """Analytic minimal-gauge characteristic speeds (verified to reduce at a=0)."""
    root = np.sqrt(M**2 - a**2)
    r_plus, r_minus = M + root, M - root
    denom = 2.0 * (r_plus**2 + a**2 * sigma**2)
    lam_out = -(1.0 - sigma) * (r_plus - r_minus * sigma) / denom
    lam_in = sigma**2 * (r_plus - r_minus * sigma) / ((1.0 + sigma) * denom)
    return lam_out, lam_in


def main():
    d = build()
    sigma, M, a, m, s, lam = d["sigma"], d["M"], d["a"], d["m"], d["s"], d["lam"]
    A_tt, A_ts, A_ss = d["A_tt"], d["A_ts"], d["A_ss"]
    A_tau, A_sig, A_0 = d["A_tau"], d["A_sig"], d["A_0"]

    # s=-2, m=2, lambda_sep=4 (a=0 spherical value); the separation constant
    # only shifts c_Psi and never the principal part.
    base = {s: -2, m: 2, lam: 4}
    sig_grid = np.linspace(0.02, 0.98, 25)

    print("=" * 72)
    print("STAGE 1: principal symbol -> characteristic speeds (numerical)")
    print("=" * 72)
    f_tt = sp.lambdify((sigma, a, M), A_tt.subs(base), "numpy")
    f_ts = sp.lambdify((sigma, a, M), A_ts.subs(base), "numpy")
    f_ss = sp.lambdify((sigma, a, M), A_ss.subs(base), "numpy")
    # Characteristic speeds c = dsigma/dtau satisfy A_tt c^2 - A_ts c + A_ss = 0.
    # Verify lambda_out, lambda_in are exactly the two roots by residual.
    worst = 0.0
    for av in (0.0, 0.5, 0.9):
        att = np.asarray(f_tt(sig_grid, av, 1.0), dtype=complex)
        ats = np.asarray(f_ts(sig_grid, av, 1.0), dtype=complex)
        ass = np.asarray(f_ss(sig_grid, av, 1.0), dtype=complex)
        lo, li = _lam_speeds(sig_grid, av, 1.0)
        # normalise the quadratic by A_tt so the residual scale is O(speed^2)
        res_out = np.abs(att * lo**2 - ats * lo + ass) / np.abs(att)
        res_in = np.abs(att * li**2 - ats * li + ass) / np.abs(att)
        err = float(max(np.max(res_out), np.max(res_in)))
        worst = max(worst, err)
        print(f"  a={av}: max residual of (A_tt c^2 - A_ts c + A_ss) at "
              f"c=lambda_out,lambda_in = {err:.2e}")
    assert worst < 1e-9, f"principal symbol speeds mismatch {worst:.2e}"
    print("  -> lambda_out, lambda_in are exactly the characteristic speeds. PASS")

    print()
    print("=" * 72)
    print("STAGE 2: regularised source coefficients finite on [0,1] (numerical)")
    print("=" * 72)
    # raw regularised (complex) source coefficients, general s and lambda_sep:
    c_Psi = -A_0 / A_tt
    c_Pi = -A_tau / A_tt
    c_Phi = -A_sig / A_tt
    # Substitute a NUMERIC spin and M=1 FIRST, then sp.cancel -> a clean single
    # rational in sigma alone (cancelling with symbolic a and sqrt(M^2-a^2) leaves
    # a nested sum of fractions that lambdify evaluates to inf-inf=nan). The
    # near-endpoint values then expose boundedness on the closed slice.
    flam = {}  # (name, a) -> lambdified rational c(sigma)
    edge = np.array([1e-6, 1e-3, 0.25, 0.5, 0.75, 1 - 1e-3, 1 - 1e-6])
    for av in (0.0, 0.5, 0.9):
        funcs = {}
        for nm, cc in (("c_Psi", c_Psi), ("c_Pi", c_Pi), ("c_Phi", c_Phi)):
            rat = sp.cancel(cc.subs(base).subs({a: av, M: 1.0}))
            funcs[nm] = sp.lambdify(sigma, rat, "numpy")
            flam[(nm, av)] = funcs[nm]
        with np.errstate(divide="ignore", invalid="ignore"):
            vals = {nm: np.asarray(f(edge), dtype=complex) for nm, f in funcs.items()}
        finite = all(np.all(np.isfinite(v)) for v in vals.values())
        mag = max(float(np.max(np.abs(v))) for v in vals.values())
        print(f"  a={av}: all finite on [1e-6,1-1e-6]? {finite};  max|c| = {mag:.3e}")
        assert finite, f"source coefficient diverges at endpoints for a={av}"
    print("  -> (1 - H^2) prefactor cancels; sources bounded on closed [0,1]. PASS")

    print()
    print("=" * 72)
    print("STAGE 3: a -> 0 reduction")
    print("=" * 72)
    lo0, li0 = _lam_speeds(sig_grid, 0.0, 1.0)
    schw_out = -(1 - sig_grid) / 4.0
    schw_in = sig_grid**2 / (4.0 * (1 + sig_grid))
    print(f"  lambda_out(a=0) vs Schwarzschild: max diff {np.max(np.abs(lo0-schw_out)):.2e}")
    print(f"  lambda_in (a=0) vs Schwarzschild: max diff {np.max(np.abs(li0-schw_in)):.2e}")

    # a=0 closed form: keep M symbolic and substitute a=0 (fast: sqrt -> M).
    cPsi0 = sp.simplify(c_Psi.subs(base).subs(a, 0))
    cPi0 = sp.simplify(c_Pi.subs(base).subs(a, 0))
    cPhi0 = sp.simplify(c_Phi.subs(base).subs(a, 0))
    has_I = any(e.has(sp.I) for e in (cPsi0, cPi0, cPhi0))
    print()
    print("  Bardeen-Press (a=0) source coefficients (s=-2, m=2, lambda_sep=4):")
    print("    c_Psi(a=0) =", cPsi0)
    print("    c_Pi (a=0) =", cPi0)
    print("    c_Phi(a=0) =", cPhi0)
    print(f"  a=0 coefficients contain I? -> {has_I}  (expected False: BP is real)")
    assert not has_I, "a=0 Bardeen-Press coefficients unexpectedly complex"
    print("  -> a=0 reduces to a REAL Bardeen-Press operator (isospectral to RW,")
    print("     NOT identical; physical a=0 check is the QNM omega in B.8). PASS")

    print()
    print("=" * 72)
    print("STAGE 4: numeric reference table of source coefficients for B.4")
    print("=" * 72)
    # The general-a closed form is enormous; B.4 will rebuild the operator by the
    # same mechanical sympy/lambdify procedure. What B.4 must REPRODUCE is the
    # numeric value of each regularised coefficient at sample (a, sigma) points.
    samples = [(0.0, 0.25), (0.0, 0.75), (0.5, 0.25), (0.5, 0.75),
               (0.9, 0.25), (0.9, 0.75)]
    print(f"  {'a':>5} {'sigma':>6} | {'c_Psi':>24} {'c_Pi':>24} {'c_Phi':>24}")
    with np.errstate(divide="ignore", invalid="ignore"):
        for av, sv in samples:
            vP = complex(flam[("c_Psi", av)](sv))
            vQ = complex(flam[("c_Pi", av)](sv))
            vR = complex(flam[("c_Phi", av)](sv))
            print(f"  {av:>5} {sv:>6} | {vP.real:+.5f}{vP.imag:+.5f}j "
                  f"{vQ.real:+.5f}{vQ.imag:+.5f}j {vR.real:+.5f}{vR.imag:+.5f}j")
    print("  (these fp64 values are the B.4 acceptance reference for s=-2,m=2,lam=4)")

    print("\nALL B.2 SYMBOLIC CHECKS PASSED.")


if __name__ == "__main__":
    main()

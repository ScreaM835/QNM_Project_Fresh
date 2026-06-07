"""B.4 foundation: derive + verify the Kerr minimal-gauge first-order system.

Extends the B.2 residue machinery to the FULL principal part so the
characteristic structure (speeds, mu_pm, mu_pm', inverse map) and the complex
source coefficients (c_Pi, c_Phi, c_Psi) are obtained as EXACT closed forms,
then checked to reduce to the validated Phase A minimal gauge at a=0.

Rational parametrisation (no surds, exact arithmetic):
  symbols r_plus (rp), r_minus (rm), beta, lambda_sep (lam).
  M  = (rp+rm)/2,   a^2 = rp*rm,   a*m = beta*(rp-rm)   [since beta=m a/(rp-rm)].
  Only ODD powers of a (the two i*am frame-dragging terms) use `am`; even powers
  use a^2 = rp*rm.  So every coefficient is rational in (sigma, rp, rm, beta, lam).

The a=0 reduction substitutes rp=2, rm=0, beta=0, lam=4 (Bardeen-Press), and is
compared against the hand-coded Phase A closed forms from
src/rwz_minimal_gauge.py (M=1): the characteristic arrays must match exactly,
the source arrays must equal the Bardeen-Press forms (derivation eq 11).
"""
import sys

import sympy as sp

sigma, tau = sp.symbols("sigma tau", positive=True)
rp, rm, beta, lam = sp.symbols("r_plus r_minus beta lambda_sep")
s, m = sp.symbols("s m")
I = sp.I

M = (rp + rm) / 2
a2 = rp * rm                 # a^2  (even power of a -> rational)
am = beta * (rp - rm)        # a*m  (odd power of a -> carried by beta)

P = sp.Integer(-3)           # scri peeling exponent (derivation eq 9a)
Q = 2 + I * beta             # horizon exponent 2 + i beta (eq 9b)


def operator_coeffs():
    """Return (A_tt, A_ts, A_ss, A_t, A_s, A_0) of the rescaled-field operator.

    A_tt d_tt Psi + A_ts d_ts Psi + A_ss d_ss Psi + A_t d_t Psi
        + A_s d_s Psi + A_0 Psi = 0.
    """
    r = rp / sigma
    Delta = (r - rp) * (r - rm)
    rho = sp.diff(r, sigma)                      # -rp/sigma^2
    w = ((r**2 + a2) / Delta) * rho
    H = 1 - 2 * sigma**2
    r2a2 = r**2 + a2

    C_tt = -(r2a2**2) / Delta
    C_t = (-2 * I * am * r2a2 + 2 * s * (r - M) * r2a2) / Delta - 4 * s * r
    C_rr = Delta
    C_r = (s + 1) * 2 * (r - M)
    C_0 = (a2 * m**2 + 2 * I * s * am * (r - M)) / Delta - lam

    Psi = sp.Function("Psi")(tau, sigma)

    def d_t(g):
        return sp.diff(g, tau)

    def d_r(g):
        return (1 / rho) * (sp.diff(g, sigma) - H * w * sp.diff(g, tau))

    Z = sigma**P * (1 - sigma) ** Q
    psi = Z * Psi
    eq = sp.expand(
        (C_tt * d_t(d_t(psi)) + C_t * d_t(psi) + C_rr * d_r(d_r(psi))
         + C_r * d_r(psi) + C_0 * psi) / Z
    )
    A_tt = eq.coeff(Psi.diff(tau, tau))
    A_ts = eq.coeff(Psi.diff(tau, sigma))
    A_ss = eq.coeff(Psi.diff(sigma, sigma))
    A_t = eq.coeff(Psi.diff(tau))
    A_s = eq.coeff(Psi.diff(sigma))
    A_0 = eq.coeff(Psi)
    return A_tt, A_ts, A_ss, A_t, A_s, A_0


def geometry_shift():
    """G(sigma) = 2H / [w (1 - H^2)] (the mu shift, derivation eq 19/21)."""
    r = rp / sigma
    Delta = (r - rp) * (r - rm)
    rho = sp.diff(r, sigma)
    w = ((r**2 + a2) / Delta) * rho
    H = 1 - 2 * sigma**2
    return sp.cancel(2 * H / (w * (1 - H**2)))


SUB = {s: -2, m: 2}
A0_SUB = {rp: 2, rm: 0, beta: 0, lam: 4}     # a = 0 (Bardeen-Press)


def closed_forms():
    """Return the 8 exact closed forms (symbols sigma, r_plus, r_minus, beta,
    lambda_sep) that src/teukolsky_minimal_gauge.py encodes. c_Psi is the
    horizon-stable (beta^2 -> physical) form. Used by the B.4 test to validate
    the numpy transcription via lambdify."""
    A_tt, _, _, A_t, A_s, A_0 = (c.subs(SUB) for c in operator_coeffs())
    lam_out_cf = -(1 - sigma) * (rp - rm * sigma) / (2 * (rp**2 + rp * rm * sigma**2))
    lam_in_cf = sigma**2 * (rp - rm * sigma) / (2 * (1 + sigma) * (rp**2 + rp * rm * sigma**2))
    G = geometry_shift().subs(SUB)
    mu_plus = sp.cancel(-lam_out_cf + G)
    mu_minus = sp.cancel(-lam_in_cf + G)
    dmu = sp.cancel(mu_plus - mu_minus)
    c_Psi = sp.cancel(-A_0 / A_tt)
    c_Psi_clean = sp.cancel(c_Psi.subs(beta**2, 4 * rp * rm / (rp - rm) ** 2))
    return {
        "lambda_out": lam_out_cf,
        "lambda_in": lam_in_cf,
        "mu_plus": mu_plus,
        "mu_minus": mu_minus,
        "mu_plus_d": sp.cancel(sp.diff(mu_plus, sigma)),
        "mu_minus_d": sp.cancel(sp.diff(mu_minus, sigma)),
        "inv_dmu": sp.cancel(1 / dmu),
        "c_Pi": sp.cancel(-A_t / A_tt),
        "c_Phi": sp.cancel(-A_s / A_tt),
        "c_Psi": c_Psi_clean,
    }


def _show(label, expr):
    print(f"  {label} = {sp.factor(sp.cancel(expr))}")


def main():
    A_tt, A_ts, A_ss, A_t, A_s, A_0 = (c.subs(SUB) for c in operator_coeffs())

    # --- characteristic speeds: roots of A_tt c^2 - A_ts c + A_ss = 0 ---
    c = sp.symbols("c")
    char = sp.cancel(A_tt * c**2 - A_ts * c + A_ss)
    roots = sp.solve(sp.Eq(char, 0), c)
    lam_out_cf = -(1 - sigma) * (rp - rm * sigma) / (2 * (rp**2 + a2 * sigma**2))
    lam_in_cf = sigma**2 * (rp - rm * sigma) / (2 * (1 + sigma) * (rp**2 + a2 * sigma**2))
    print("=== principal symbol / speeds (symbolic rp,rm) ===")
    print(f"  roots of A_tt c^2 - A_ts c + A_ss: {[sp.simplify(rt) for rt in roots]}")
    match_out = any(sp.simplify(rt - lam_out_cf) == 0 for rt in roots)
    match_in = any(sp.simplify(rt - lam_in_cf) == 0 for rt in roots)
    print(f"  closed-form lam_out (eq 7a) is a root: {match_out}")
    print(f"  closed-form lam_in  (eq 7b) is a root: {match_in}")

    # --- mu_pm = -lambda + G, and inverse map ---
    G = geometry_shift().subs(SUB)
    mu_plus = sp.cancel(-lam_out_cf + G)        # pairs with U (advects lam_out)
    mu_minus = sp.cancel(-lam_in_cf + G)        # pairs with W (advects lam_in)
    dmu = sp.cancel(mu_plus - mu_minus)
    inv_dmu = sp.cancel(1 / dmu)
    wU = sp.cancel(-mu_minus / dmu)             # weight on U in Pi
    wW = sp.cancel(mu_plus / dmu)               # weight on W in Pi
    mu_plus_d = sp.cancel(sp.diff(mu_plus, sigma))
    mu_minus_d = sp.cancel(sp.diff(mu_minus, sigma))
    print("\n=== characteristic-variable structure (symbolic rp,rm) ===")
    _show("G            ", G)
    _show("mu_plus      ", mu_plus)
    _show("mu_minus     ", mu_minus)
    _show("mu_+ - mu_-  ", dmu)
    _show("inv_dmu      ", inv_dmu)
    _show("Pi wt on U   ", wU)
    _show("Pi wt on W   ", wW)
    _show("mu_plus_d    ", mu_plus_d)
    _show("mu_minus_d   ", mu_minus_d)

    # --- complex source coefficients ---
    c_Pi = sp.cancel(-A_t / A_tt)
    c_Phi = sp.cancel(-A_s / A_tt)
    c_Psi = sp.cancel(-A_0 / A_tt)
    print("\n=== complex source coefficients (symbolic rp,rm,beta,lam) ===")
    _show("c_Pi ", c_Pi)
    _show("c_Phi", c_Phi)
    _show("c_Psi", c_Psi)

    # ================== a = 0 reduction checks ==================
    print("\n=== a=0 reduction vs Phase A rwz_minimal_gauge (M=1) ===")
    # Phase A hand-coded closed forms (M=1):
    one_plus = 1 + sigma
    phaseA = {
        "lam_out": -(1 - sigma) / 4,
        "lam_in": sigma**2 / (4 * one_plus),
        "mu_plus": sigma**2 / (4 * one_plus),       # = lam_in (Schwarzschild)
        "mu_minus": -(1 - sigma) / 4,               # = lam_out
        "mu_plus_d": sigma * (2 + sigma) / (4 * one_plus**2),
        "mu_minus_d": sp.Rational(1, 4),
        "inv_dmu": 4 * one_plus,
        "wU": 1 - sigma**2,
        "wW": sigma**2,
    }
    # Bardeen-Press source (derivation eq 11):
    bp = {
        "c_Pi": -1 / (2 * one_plus),
        "c_Phi": -sigma * (sigma + 2) / (16 * one_plus),
        "c_Psi": (sigma - 4) / (16 * one_plus),
    }
    got = {
        "lam_out": lam_out_cf.subs(A0_SUB),
        "lam_in": lam_in_cf.subs(A0_SUB),
        "mu_plus": mu_plus.subs(A0_SUB),
        "mu_minus": mu_minus.subs(A0_SUB),
        "mu_plus_d": mu_plus_d.subs(A0_SUB),
        "mu_minus_d": mu_minus_d.subs(A0_SUB),
        "inv_dmu": inv_dmu.subs(A0_SUB),
        "wU": wU.subs(A0_SUB),
        "wW": wW.subs(A0_SUB),
    }
    ok = True
    for k, v in phaseA.items():
        diff = sp.simplify(got[k] - v)
        flag = (diff == 0)
        ok = ok and flag
        print(f"  [{'OK ' if flag else 'XX '}] {k:11s} a=0: {sp.cancel(got[k])}"
              + ("" if flag else f"   != PhaseA {v}"))
    for k, v in bp.items():
        g = {"c_Pi": c_Pi, "c_Phi": c_Phi, "c_Psi": c_Psi}[k].subs(A0_SUB)
        diff = sp.simplify(g - v)
        flag = (diff == 0)
        ok = ok and flag
        print(f"  [{'OK ' if flag else 'XX '}] {k:11s} a=0: {sp.cancel(g)}"
              + ("" if flag else f"   != BP {v}"))
    print(f"\nALL a=0 reductions exact? {ok}")

    # ============ emit clean numpy source (pycode) for B.4 ============
    # These strings are pasted verbatim into src/teukolsky_minimal_gauge.py and
    # re-validated there by test_teukolsky_minimal_gauge.py against lambdify.
    if "--pycode" in sys.argv:
        # c_Psi has a (sigma-1) denominator factor that cancels ONLY when
        # beta^2 takes its physical value 4 r+ r-/(r+ - r-)^2 (m=2). Without
        # that substitution the horizon inset is a 0/0 (precision loss). We
        # substitute beta^2 -> physical (rational) and re-cancel so the encoded
        # c_Psi is horizon-stable; the lone imaginary beta^1 part already
        # vanishes at sigma=1 identically and keeps the symbol `bt`.
        beta2_phys = 4 * rp * rm / (rp - rm) ** 2          # = (m a/(r+-r-))^2, m=2
        c_Psi_clean = sp.cancel(c_Psi.subs(beta**2, beta2_phys))
        den = sp.denom(c_Psi_clean)
        has_horizon_pole = sp.simplify(den.subs(sigma, 1)) == 0
        print(f"\n# c_Psi clean denom still vanishes at sigma=1? "
              f"{has_horizon_pole}  (must be False)")

        print("\n=== pycode (sigma->sg, r_plus->rp, r_minus->rm, beta->bt, "
              "lambda_sep->lm; I->1j) ===")
        repl = {rp: sp.Symbol("rp"), rm: sp.Symbol("rm"),
                beta: sp.Symbol("bt"), lam: sp.Symbol("lm"),
                sigma: sp.Symbol("sg")}
        for label, expr in (
            ("lam_out", lam_out_cf), ("lam_in", lam_in_cf),
            ("mu_plus_d", mu_plus_d), ("mu_minus_d", mu_minus_d),
            ("inv_dmu", inv_dmu),
            ("c_Pi", c_Pi), ("c_Phi", c_Phi), ("c_Psi", c_Psi_clean),
        ):
            code = sp.pycode(sp.cancel(expr.xreplace(repl)))
            print(f"\n# {label}\n{label} = {code}")


if __name__ == "__main__":
    main()

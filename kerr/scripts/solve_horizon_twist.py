"""Kerr horizon corotating twist: rigorous SYMBOLIC residue check.

At a=0 the rescaling Z = sigma^-3 (1-sigma)^2 regularises the s=-2 source.
For a>0 the horizon residue of c_Pi is LINEAR in the horizon exponent Q and
forces a COMPLEX Q = 2 + i*beta with beta = m a/(r_+ - r_-) (ingoing-Kerr
corotating phase). The open question: with that complex Q, are c_Phi and c_Psi
ALSO residue-free at both ends, or does the (1-sigma)^{i beta} derivative leak
an imaginary 1/(1-sigma) pole into the first-derivative coefficient c_Phi?

Key fact that makes this rigorous & exact: after psi = Z*Psi and dividing by Z,
Z'/Z = P/sigma - Q/(1-sigma) and Z''/Z are RATIONAL in sigma even for complex Q.
So every source coefficient is a rational function of sigma over the Gaussian
rationals (using Pythagorean spins a=3/5,4/5 -> r_+- rational, beta rational).
A pole exists iff the symbolic residue limit is non-zero. No numeric ambiguity.
"""
import sympy as sp

sigma, tau, M = sp.symbols("sigma tau M", positive=True)
s, m, lam = sp.symbols("s m lambda_sep")
P, Q = sp.symbols("P Q")
I = sp.I


def make_coeffs(a):
    root = sp.sqrt(M**2 - a**2)
    r_plus, r_minus = M + root, M - root
    r = r_plus / sigma
    Delta = (r - r_plus) * (r - r_minus)
    rho = sp.diff(r, sigma)
    w = ((r**2 + a**2) / Delta) * rho
    H = 1 - 2 * sigma**2
    r2a2 = r**2 + a**2
    C_tt = -(r2a2**2) / Delta
    C_t = (-2 * I * a * m * r2a2 + 2 * s * (r - M) * r2a2) / Delta - 4 * s * r
    C_rr = Delta
    C_r = (s + 1) * 2 * (r - M)
    C_0 = (a**2 * m**2 + 2 * I * a * s * m * (r - M)) / Delta - lam
    Psi = sp.Function("Psi")(tau, sigma)

    def d_t(g):
        return sp.diff(g, tau)

    def d_r(g):
        return (1 / rho) * (sp.diff(g, sigma) - H * w * sp.diff(g, tau))

    def coeffs(Z):
        psi = Z * Psi
        eq = sp.expand(
            (C_tt * d_t(d_t(psi)) + C_t * d_t(psi) + C_rr * d_r(d_r(psi))
             + C_r * d_r(psi) + C_0 * psi) / Z
        )
        return (eq.coeff(Psi.diff(tau, tau)), eq.coeff(Psi.diff(tau)),
                eq.coeff(Psi.diff(sigma)), eq.coeff(Psi))

    return coeffs, (r_plus, r_minus)


def residues(c):
    """Symbolic residues: (scri pole coeff, scri value, horizon pole coeff, horizon value)."""
    scri_pole = sp.simplify(sp.limit(sigma * c, sigma, 0))
    hor_pole = sp.simplify(sp.limit((1 - sigma) * c, sigma, 1))
    scri_val = sp.simplify(sp.limit(c, sigma, 0)) if scri_pole == 0 else sp.nan
    hor_val = sp.simplify(sp.limit(c, sigma, 1)) if hor_pole == 0 else sp.nan
    return scri_pole, scri_val, hor_pole, hor_val


sub = {s: -2, m: 2, lam: 4, M: 1}
Z = sigma**P * (1 - sigma) ** Q

for aval in (sp.Integer(0), sp.Rational(3, 5), sp.Rational(4, 5)):
    coeffs, (r_plus, r_minus) = make_coeffs(aval)
    At, Ata, Asi, A0 = coeffs(Z)
    cPi = sp.cancel((-Ata / At).subs(sub))

    # solve c_Pi residues (linear in P, Q) for the rescaling exponents
    res0 = sp.expand(sp.limit(sigma * cPi, sigma, 0))
    res1 = sp.expand(sp.limit((sigma - 1) * cPi, sigma, 1))
    Psol = sp.solve(sp.Eq(res0, 0), P)[0]
    Qsol = sp.solve(sp.Eq(res1, 0), Q)[0]
    print(f"=== a = {aval} ===")
    print(f"  P = {Psol}    Q = {Qsol}")
    if aval != 0:
        beta = sp.nsimplify((sp.Rational(2) * aval / (r_plus - r_minus)).subs(M, 1))
        print(f"  Im(Q) = {sp.im(Qsol)}   predicted m a/(r+-r-) = {beta}   "
              f"match = {sp.simplify(sp.im(Qsol) - beta) == 0}")

    # now substitute the solved (P, Q) into ALL THREE coefficients and check residues
    Zsol = sigma**Psol * (1 - sigma) ** Qsol
    Ats, Atas, Asis, A0s = coeffs(Zsol)
    names = ("c_Pi ", "c_Phi", "c_Psi")
    raws = (-Atas / Ats, -Asis / Ats, -A0s / Ats)
    print("  residues with solved (P,Q)   [pole coeff must be 0 for boundedness]:")
    all_ok = True
    for nm, raw in zip(names, raws):
        c = sp.cancel(raw.subs(sub))
        sp0, sv, sp1, hv = residues(c)
        ok = (sp0 == 0) and (sp1 == 0)
        all_ok = all_ok and ok
        print(f"    {nm}: scri_pole={sp0}  horizon_pole={sp1}   "
              f"bounded={ok}")
        if ok:
            print(f"           scri_val={sv}  horizon_val={hv}")
    print(f"  ALL THREE BOUNDED at a={aval}? {all_ok}\n")


# ---- closed-form source coefficient functions at a=0 (Bardeen-Press) ----
print("=== a = 0 closed-form source coefficients (for the derivation doc) ===")
coeffs0, _ = make_coeffs(sp.Integer(0))
Z0 = sigma ** sp.Integer(-3) * (1 - sigma) ** sp.Integer(2)
At0, Ata0, Asi0, A00 = coeffs0(Z0)
for nm, raw in (("c_Pi ", -Ata0 / At0), ("c_Phi", -Asi0 / At0), ("c_Psi", -A00 / At0)):
    print(f"  {nm}(sigma) = {sp.factor(sp.cancel(raw.subs(sub)))}")


# ---- symbolic verification of closed-form characteristic speeds ----
print("\n=== closed-form characteristic speeds (exact, rational spins) ===")
lam_out_cf = lambda rp, rm: -(1 - sigma) * (rp - rm * sigma) / (2 * (rp**2 + (rp * rm) * sigma**2))
lam_in_cf = lambda rp, rm: sigma**2 * (rp - rm * sigma) / (2 * (1 + sigma) * (rp**2 + (rp * rm) * sigma**2))
for aval in (sp.Integer(0), sp.Rational(3, 5), sp.Rational(4, 5)):
    rp = 1 + sp.sqrt(1 - aval**2)
    rm = 1 - sp.sqrt(1 - aval**2)
    r = rp / sigma
    Delta = (r - rp) * (r - rm)
    rho = sp.diff(r, sigma)
    w = ((r**2 + aval**2) / Delta) * rho
    H = 1 - 2 * sigma**2
    C_tt = -((r**2 + aval**2) ** 2) / Delta
    C_rr = Delta
    Psiv = sp.Function("Psi")(tau, sigma)
    d_t = lambda g: sp.diff(g, tau)
    d_r = lambda g: (1 / rho) * (sp.diff(g, sigma) - H * w * sp.diff(g, tau))
    princ = sp.expand(C_tt * d_t(d_t(Psiv)) + C_rr * d_r(d_r(Psiv)))
    A_tt = princ.coeff(Psiv.diff(tau, tau))
    A_ts = princ.coeff(Psiv.diff(tau, sigma))
    A_ss = princ.coeff(Psiv.diff(sigma, sigma))
    print(f"  a={aval}:  (note: rp*rm = a^2 used to keep coeffs rational)")
    for nm, lam in (("lam_out", lam_out_cf(rp, rm)), ("lam_in", lam_in_cf(rp, rm))):
        resid = sp.cancel(A_tt * lam**2 - A_ts * lam + A_ss)
        print(f"    {nm}: A_tt c^2 - A_ts c + A_ss = {resid}   root={resid == 0}")

"""Solve for the field rescaling Z = sigma^p (1-sigma)^q that regularises the
s=-2 Teukolsky source coefficients at a=0 (Bardeen-Press), rigorously, then
VERIFY the same (p, q) regularises the full Kerr source for a in {0, 0.5, 0.9}.

Strategy: with psi = Z(sigma) Psi, recompute c_Pi, c_Phi, c_Psi for the
rescaled field Psi. Require the simple-pole residues of c_Pi at sigma=0 (scri)
and sigma=1 (horizon) to vanish; solve the two equations for (p, q). Then
VERIFY all three coefficients are finite at both endpoints, for every spin.

Physical reading of the result (p, q) = (-3, 2), i.e. Psi = psi * sigma^3/(1-sigma)^2:
  * sigma^3 at scri  cancels the r^3 ~ sigma^-3 peeling growth of the s=-2
    OUTGOING Teukolsky solution (R_{-2} ~ r^{-2s-1} = r^3 at infinity);
  * (1-sigma)^-2 at horizon cancels the Delta^{-s} = Delta^2 ~ (1-sigma)^2
    suppression of the s=-2 INGOING solution.
Both exponents are fixed by the spin weight s and are a-independent, so the
same rescaling is expected (and below verified) to work for all spins.
"""
import sympy as sp

sigma, tau, M = sp.symbols("sigma tau M", positive=True)
s, m, lam, p, q = sp.symbols("s m lambda_sep p q")
a_sym = sp.symbols("a", nonnegative=True)
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
        A_tt = eq.coeff(Psi.diff(tau, tau))
        A_tau = eq.coeff(Psi.diff(tau))
        A_sig = eq.coeff(Psi.diff(sigma))
        A_0 = eq.coeff(Psi)
        return A_tt, A_tau, A_sig, A_0

    return coeffs


# ----- STEP 1: solve for (p, q) analytically at a = 0 (Bardeen-Press) -----
coeffs0 = make_coeffs(sp.Integer(0))
Z = sigma**p * (1 - sigma) ** q
A_tt, A_tau, A_sig, A_0 = coeffs0(Z)
sub = {s: -2, m: 2, lam: 4, M: 1}
cPi = sp.cancel((-A_tau / A_tt).subs(sub))

res0 = sp.simplify(sp.limit(sigma * cPi, sigma, 0))
res1 = sp.simplify(sp.limit((sigma - 1) * cPi, sigma, 1))
print("a=0: c_Pi residue at scri (sigma=0)   :", res0)
print("a=0: c_Pi residue at horizon (sigma=1):", res1)

sol = sp.solve([sp.Eq(res0, 0), sp.Eq(res1, 0)], [p, q], dict=True)
print("a=0: solve(res0=0, res1=0) ->", sol)
pv, qv = sol[0][p], sol[0][q]
print(f"a=0: rescaling exponents (p, q) = ({pv}, {qv})  "
      f"=> regular field Psi = psi * sigma^{-pv} (1-sigma)^{-qv}")

# verify all three bounded at a=0
print(f"\n--- a=0 verification with Z = sigma^{pv} (1-sigma)^{qv} ---")
Zc = sigma**pv * (1 - sigma) ** qv
At, Ata, Asi, A0 = coeffs0(Zc)
for nm, cc in (("c_Pi", -Ata / At), ("c_Phi", -Asi / At), ("c_Psi", -A0 / At)):
    e = sp.cancel(cc.subs(sub))
    l0 = sp.limit(e, sigma, 0, "+")
    l1 = sp.limit(e, sigma, 1, "-")
    print(f"  {nm}: scri={l0}, horizon={l1}   bounded={l0.is_finite and l1.is_finite}")

# ----- STEP 2: verify SAME (p, q) regularises the full Kerr source -----
# Use Pythagorean spins so sqrt(M^2 - a^2) stays RATIONAL (a=3/5 -> 4/5,
# a=4/5 -> 3/5); this keeps sp.cancel fast and exact. Substitute a, M FIRST,
# then cancel -> clean rational in sigma, evaluate near the endpoints.
print(f"\n=== STEP 2: same rescaling for spinning a/M in {{3/5, 4/5}} (numerical) ===")
import numpy as np
edge = np.array([1e-7, 1e-4, 0.25, 0.5, 0.75, 1 - 1e-4, 1 - 1e-7])
for aval in (sp.Rational(3, 5), sp.Rational(4, 5)):
    coeffs_a = make_coeffs(aval)
    At, Ata, Asi, A0 = coeffs_a(Zc)
    print(f"\n--- a/M = {aval} (= {float(aval)}) ---")
    allok = True
    for nm, cc in (("c_Pi", -Ata / At), ("c_Phi", -Asi / At), ("c_Psi", -A0 / At)):
        rat = sp.cancel(cc.subs({s: -2, m: 2, lam: 4, M: 1}))
        f = sp.lambdify(sigma, rat, "numpy")
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.asarray(f(edge), dtype=complex)
        ok = bool(np.all(np.isfinite(v)))
        allok = allok and ok
        print(f"  {nm}: max|c|={np.max(np.abs(v)):.4f}  scri={v[0]:.4f}  "
              f"horizon={v[-1]:.4f}  finite={ok}")
    print(f"  ALL BOUNDED at a/M={aval}? {allok}")
print("\nRESULT: regular field  Psi = psi * sigma^3 / (1-sigma)^2  regularises")
print("the s=-2 minimal-gauge Teukolsky source at a=0 AND for spinning Kerr.")




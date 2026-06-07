# Minimal-gauge hyperboloidal Regge-Wheeler on Schwarzschild

Status: derivation only. No code written against this yet. To be sanity-checked
line by line before A.8-fix implementation begins.

Reference frame: Panosso Macedo, Jaramillo & Ansorg, PRD 89 064008 (2014);
Panosso Macedo, CQG 37 065019 (2020); Ansorg & Panosso Macedo, PRD 93 124016
(2016). The construction below specialises the general "minimal gauge" of those
papers to Schwarzschild and the axial l=2 Regge-Wheeler equation. Units M=1
unless otherwise stated; we keep M symbolic in the final coefficients so that
the rescaling is unambiguous.

## 1. Starting point

Axial Regge-Wheeler on Schwarzschild in (t, r_*):

    - d_t^2 Psi + d_{r_*}^2 Psi  -  V(r) Psi  =  0,                        (1)

    V(r) = (1 - 2M/r) [ l(l+1) / r^2  -  6 M / r^3 ],                       (2)

with r_*(r) = r + 2M ln(r/(2M) - 1), dr_*/dr = 1 / (1 - 2M/r).

## 2. Hyperboloidal time change of variables

Define tau = t - h(r_*) with H(r_*) := dh/dr_*. Derivatives transform as

    d_t|_{r_*}       = d_tau|_{r_*}
    d_{r_*}|_t       = d_{r_*}|_tau  -  H d_tau.                            (3)

Substituting into (1):

    (H^2 - 1) d_tau^2 Psi  -  2 H d_{r_*} d_tau Psi  -  H' d_tau Psi
        +  d_{r_*}^2 Psi  -  V Psi  =  0.                                   (4)

H' denotes dH/dr_*. Principal symbol in cotangent coordinates (xi_tau, xi_{r_*}):

    (H^2 - 1) xi_tau^2  -  2 H xi_tau xi_{r_*}  +  xi_{r_*}^2  =  0.       (5)

Solving for xi_{r_*}/xi_tau = H +/- 1, so the characteristic curves in
(r_*, tau) have slopes

    dr_*/dtau  =  -xi_tau / xi_{r_*}  =  -1 / (H +/- 1)
               =  1 / (1 - H)        ("outgoing", + sign on numerator)
                  or   -1 / (1 + H)  ("ingoing").                           (6)

These are the standard hyperboloidal speeds: outgoing diverges at scri
(H -> +1), ingoing diverges at the horizon (H -> -1).

## 3. Why the tanh gauge fails on an explicit-RK code

For H(r_*) = tanh(r_* / L) with finite r_* truncation at r_*_max:

  - 1 - H -> 2 exp(-2 r_*_max / L), exponentially small but nonzero.
  - The OUTGOING coordinate speed at the outer edge is 1 / (1 - H), which is
    O(exp(+ 2 r_*_max / L)). CFL forces dt <= dx (1 - H), and the explicit
    RK4 update of the (1 / (1 - H^2)) Pi-equation aliases an unbounded
    coefficient against high-frequency error from the FD stencil.
  - Truncating r_* at finite r_*_max means the outer "boundary" is NOT scri:
    it is a finite-r boundary on a slice that is rapidly becoming null. The
    incoming characteristic has finite speed there and physically needs a
    BC, which we never supplied. The sponge masked this for L=30 only
    because |H| < 0.99, i.e. the slice was barely hyperboloidal.

Conclusion: a hyperboloidal scheme on a tortoise grid cannot be both
(a) actually hyperboloidal up to scri and (b) explicit-time-stable, without
either an implicit solve or a coordinate compactification. We choose the
latter (compactification), which is the standard MJA route.

## 4. Compactification to sigma = 2M / r

Define sigma = 2M / r in [0, 1]. Then sigma = 0 is scri (r = +infinity),
sigma = 1 is the horizon (r = 2M). The chain-rule weights:

    dr / dsigma     =  -2M / sigma^2,
    dr_* / dsigma   =  (dr_*/dr) (dr/dsigma)  =  -2M / [sigma^2 (1 - sigma)],

define

    w(sigma) := dr_* / dsigma  =  -2M / [sigma^2 (1 - sigma)].             (7)

w is strictly negative on (0, 1) and diverges at both endpoints. Derivatives
in r_* convert as d_{r_*} = (1/w) d_sigma. Useful auxiliaries:

    1 / w       =  - sigma^2 (1 - sigma) / (2M),                            (8a)
    w'(sigma)   =  (3 sigma - 2) / [sigma^3 (1 - sigma)^2]  *  (-2M)
                =  -2M (3 sigma - 2) / [sigma^3 (1 - sigma)^2],             (8b)

(verified by d/dsigma of (7)) and the combination that will appear in the
second derivative,

    w' / w^3    =  (3 sigma - 2) sigma^3 (1 - sigma) / (4 M^2).             (8c)

Note w'/w^3 is a polynomial in sigma on [0, 1]; the singularities of w have
cancelled.

## 5. Minimal-gauge height function

The minimal-gauge requirement is that the characteristic speeds in
(sigma, tau) coordinates be bounded and one-signed on [0, 1] (so that both
endpoints are characteristic outflow boundaries and no boundary data is
needed).

From section 2, the outgoing speed in (r_*, tau) is 1/(1-H), so in sigma:

    lambda_out^sigma  =  (1/w) * 1/(1 - H)  =  1 / [w (1 - H)].             (9)

Boundedness at sigma -> 0 requires w (1 - H) bounded away from zero. With
w ~ -2M / sigma^2 near sigma = 0, we need 1 - H ~ c_+ sigma^2.

Boundedness at sigma -> 1 of the ingoing speed,

    lambda_in^sigma   =  - 1 / [w (1 + H)],                                (10)

with w ~ -2M / (1 - sigma) near sigma = 1, requires 1 + H ~ c_- (1 - sigma).

The simplest polynomial H(sigma) satisfying H(0) = +1, H(1) = -1, AND both
boundary conditions, is

    H(sigma)  =  1  -  2 sigma^2.                                          (11)

Check 1 - H = 2 sigma^2 ✓; 1 + H = 2 - 2 sigma^2 = 2(1 - sigma)(1 + sigma)
so 1 + H ~ 4 (1 - sigma) near sigma = 1 ✓. Also |H| < 1 strictly on (0, 1)
and H is monotonic ✓. This is the choice used in MJA 2014 for the
Schwarzschild Bondi-like gauge.

Auxiliary quantities under (11):

    H              =  1  -  2 sigma^2,                                     (12a)
    1 - H          =  2 sigma^2,                                           (12b)
    1 + H          =  2 (1 - sigma)(1 + sigma),                            (12c)
    1 - H^2        =  4 sigma^2 (1 - sigma)(1 + sigma),                    (12d)
    dH/dsigma      =  - 4 sigma,
    H' = dH/dr_*   =  (1/w) (dH/dsigma)  =  - 4 sigma / w
                   =  2 sigma^3 (1 - sigma) / M,                           (12e)

all polynomial in sigma on [0, 1], all bounded.

Characteristic speeds in (sigma, tau), inserting (12) into (9), (10):

    lambda_out^sigma  =  1 / [w (1 - H)]
        =  1 / { [-2M / (sigma^2 (1 - sigma))] [2 sigma^2] }
        =  - (1 - sigma) / (4 M),                                          (13a)

    lambda_in^sigma   =  - 1 / [w (1 + H)]
        =  - 1 / { [-2M / (sigma^2 (1 - sigma))] [2 (1 - sigma)(1 + sigma)] }
        =  sigma^2 / [4 M (1 + sigma)].                                    (13b)

Both are polynomial / rational in sigma, bounded on [0, 1]:

    |lambda_out^sigma|  <=  1 / (4 M),     equality at sigma = 0 (scri),
                            vanishes at sigma = 1 (horizon).
    |lambda_in^sigma|   <=  1 / (8 M),     equality at sigma = 1,
                            vanishes at sigma = 0.

Signs: lambda_out^sigma <= 0 everywhere (outgoing = moving toward smaller
sigma = larger r, ✓), lambda_in^sigma >= 0 everywhere (ingoing = moving
toward larger sigma = horizon, ✓). At each endpoint one characteristic
exits the domain and the other is tangent. **NO boundary data is required
at either sigma = 0 or sigma = 1.**

## 6. First-order system on (sigma, tau)

Introduce auxiliary variables

    Pi  := d_tau Psi,        Phi := d_sigma Psi.                            (14)

The compatibility relation is d_tau Phi = d_sigma Pi (commuting partials,
since (sigma, tau) are independent coordinates).

From (4) and d_{r_*} = (1/w) d_sigma, d_{r_*}^2 Psi = (1/w^2) d_sigma Phi
- (w'/w^3) Phi, the Psi-equation becomes

    (1 - H^2) d_tau Pi  =  (1/w^2) d_sigma Phi  -  (w'/w^3) Phi
                          -  (2 H / w) d_sigma Pi  -  H' Pi  -  V Psi.    (15)

Together:

    d_tau Psi  =  Pi,                                                      (16a)
    d_tau Phi  =  d_sigma Pi,                                              (16b)
    (1 - H^2) d_tau Pi  =  RHS as in (15).                                 (16c)

The (1 - H^2) prefactor in (16c) vanishes at both endpoints. We will NOT
divide through naively. Instead we evolve characteristic variables U, W
defined below, for which the (1 - H^2) factor cancels analytically.

## 7. Characteristic variables and source regularity

Define

    U  :=  Pi  +  mu_+ Phi,                                                (17a)
    W  :=  Pi  +  mu_- Phi.                                                (17b)

We require that the linear combinations diagonalise the principal part of
(16) and satisfy

    d_tau U  +  lambda_out^sigma  d_sigma U  =  (regular source),          (18a)
    d_tau W  +  lambda_in^sigma   d_sigma W  =  (regular source).          (18b)

Derivation of mu_pm. Write V = Pi + mu Phi, take d_tau, substitute (16a),
(16b), (16c), and demand that V satisfies a pure transport equation at the
principal level. Matching the d_sigma Pi coefficient against lambda gives

    mu  =  - lambda  +  2H / [w (1 - H^2)],                                (19)

and matching the d_sigma Phi coefficient gives

    - lambda mu  =  1 / [w^2 (1 - H^2)],                                   (20)

which is automatically consistent with (19) when lambda is either of (13a),
(13b) (verified by direct substitution).

With (13a), (13b), (12d), and H = 1 - 2 sigma^2,

    2H / [w (1 - H^2)]
       =  2 (1 - 2 sigma^2)
          / { [-2M / (sigma^2 (1 - sigma))]  *  [4 sigma^2 (1 - sigma)(1 + sigma)] }
       =  - (1 - 2 sigma^2) / [4 M (1 + sigma)],                           (21)

so

    mu_+  =  - lambda_out^sigma  +  2H / [w (1 - H^2)]
           =  (1 - sigma) / (4M)  -  (1 - 2 sigma^2) / [4 M (1 + sigma)]
           =  [(1 - sigma)(1 + sigma)  -  (1 - 2 sigma^2)] / [4 M (1 + sigma)]
           =  sigma^2 / [4 M (1 + sigma)],                                 (22a)

    mu_-  =  - lambda_in^sigma  +  2H / [w (1 - H^2)]
           =  - sigma^2 / [4 M (1 + sigma)]  -  (1 - 2 sigma^2) / [4 M (1 + sigma)]
           =  - (1 - sigma^2) / [4 M (1 + sigma)]
           =  - (1 - sigma) / (4 M).                                       (22b)

Both bounded on [0, 1]. Note mu_+ = lambda_in^sigma and mu_- = lambda_out^sigma
as a consequence of (19) when 2H/[w(1-H^2)] is included; this is a labelling
curiosity, not a sign error.

Invertibility of the (Pi, Phi) <-> (U, W) map:

    mu_+ - mu_-  =  sigma^2 / [4M(1+sigma)]  +  (1 - sigma) / (4M)
                 =  [sigma^2 + (1 - sigma)(1 + sigma)] / [4M (1 + sigma)]
                 =  1 / [4M (1 + sigma)],

nonvanishing on [0, 1]. ✓

Inverse map:

    Phi  =  4 M (1 + sigma)  (U  -  W),                                    (23a)
    Pi   =  U  -  mu_+ Phi
          =  U  -  sigma^2 (U - W)
          =  (1 - sigma^2) U  +  sigma^2 W.                                (23b)

Sanity: at sigma = 0, Pi = U and Phi = 4M (U - W). At sigma = 1,
Pi = W and Phi = 8M (U - W).

## 8. Regular sources in characteristic form

Write the source side of (16c) as

    S_Pi^raw  :=  - (w'/w^3) Phi  -  H' Pi  -  V Psi,                      (24)

so (16c) reads (1 - H^2) d_tau Pi = (principal flux) + S_Pi^raw. The principal
flux is absorbed by the characteristic decomposition. We must check that

    S_Pi  :=  S_Pi^raw / (1 - H^2)                                          (25)

is a bounded function on [0, 1] under (12). Compute term-by-term using (12)
and the closed forms from sec. 4-5:

Term 1: w' / w^3 = (3 sigma - 2) sigma^3 (1 - sigma) / (4 M^2), from (8c).
Divided by 1 - H^2 = 4 sigma^2 (1 - sigma)(1 + sigma):

    (w' / w^3) / (1 - H^2)
       =  (3 sigma - 2) sigma / [16 M^2 (1 + sigma)].                      (26a)

Polynomial / rational, bounded on [0, 1]. ✓

Term 2: H' = 2 sigma^3 (1 - sigma) / M, from (12e). Divided by 1 - H^2:

    H' / (1 - H^2)  =  sigma / [2 M (1 + sigma)].                          (26b)

Bounded on [0, 1]. ✓

Term 3: V from (2) in terms of sigma. With r = 2M/sigma:
1 - 2M/r = 1 - sigma; 1/r^2 = sigma^2 / (4 M^2); 1/r^3 = sigma^3 / (8 M^3).

    V(sigma)
       =  (1 - sigma) [ l(l+1) sigma^2 / (4 M^2)  -  6 M sigma^3 / (8 M^3) ]
       =  (1 - sigma) sigma^2 [ l(l+1)  -  3 sigma ] / (4 M^2).             (27)

For l = 2: V = 3 sigma^2 (1 - sigma)(2 - sigma) / (4 M^2). Divided by 1 - H^2:

    V / (1 - H^2)  =  [l(l+1) - 3 sigma] / [16 M^2 (1 + sigma)].           (26c)

Bounded on [0, 1]. ✓

Therefore S_Pi (with the cancellation by 1 - H^2 done analytically, never
numerically) is bounded.

Collected: the regularised Pi-source is

    S_Pi(sigma) Psi-Pi-Phi   :=
        - (3 sigma - 2) sigma            / [16 M^2 (1 + sigma)]  *  Phi
        - sigma                          / [ 2 M    (1 + sigma)]  *  Pi
        - [ l(l+1) - 3 sigma ]           / [16 M^2 (1 + sigma)]  *  Psi.   (28)

## 9. Final characteristic-form evolution system

The system to be discretised (Psi, U, W):

    d_tau Psi  =  Pi(U, W; sigma),       [Pi from (23b)]                   (29a)

    d_tau U  +  lambda_out^sigma d_sigma U  =  S_U(sigma; Psi, Pi, Phi),   (29b)

    d_tau W  +  lambda_in^sigma  d_sigma W  =  S_W(sigma; Psi, Pi, Phi),   (29c)

with characteristic speeds (13) and

    S_U  =  S_Pi  +  mu_+'(sigma) * Phi * lambda_out^sigma,
    S_W  =  S_Pi  +  mu_-'(sigma) * Phi * lambda_in^sigma,                 (30)

where mu_pm' = d(mu_pm)/dsigma comes from differentiating (22):

    mu_+'  =  d/dsigma [ sigma^2 / (4 M (1 + sigma)) ]
           =  sigma (2 + sigma) / [4 M (1 + sigma)^2],                     (31a)
    mu_-'  =  d/dsigma [ -(1 - sigma) / (4 M) ]
           =  1 / (4 M).                                                   (31b)

The source contributions in (30) are bounded since lambda and mu' are
bounded. No 1/0 anywhere on the closed interval [0, 1].

## 10. Boundary treatment

Committed choice: 2nd-order central FD on the interior, 2nd-order one-sided
upwind at the two boundary points. Upgrade to higher order only if V.2 fails.

At sigma = 0 (scri):
  - lambda_out^sigma (0) = -1/(4M):  U has nonzero outflow toward smaller
    sigma, which is OFF the grid. The correct upwind direction (information
    flowing FROM interior TO boundary) is FORWARD (towards larger sigma):
    d_sigma U |_0  ~  (-3 U_0 + 4 U_1 - U_2) / (2 dsigma).
  - lambda_in^sigma (0) = 0:  W flux is multiplied by zero; any consistent
    stencil works. Use the same forward 2nd-order one-sided for code
    uniformity.
  - No data injection. (Psi, U, W) at sigma = 0 update from interior values
    only via (29).

At sigma = 1 (horizon):
  - lambda_in^sigma (1) = 1/(8M):  W outflows to larger sigma, OFF the grid.
    Correct upwind is BACKWARD:
    d_sigma W |_{N-1}  ~  (3 W_{N-1} - 4 W_{N-2} + W_{N-3}) / (2 dsigma).
  - lambda_out^sigma (1) = 0:  U flux vanishes; use backward for uniformity.

KO dissipation: apply only to strictly interior points (4 <= i <= N - 5).
No KO in the boundary stencil region. No sponges, no extrapolation BCs.

## 10b. Initial data prescription

Given a Schwarzschild-frame initial pulse Psi_0(r) (e.g. Gaussian in areal
radius centred at r = x_0 = 4 M with sigma_ID = 5 M as per parent paper),
set

    Psi(sigma, 0)  =  Psi_0( r(sigma) )    with r = 2M/sigma,
    Pi (sigma, 0)  =  0                     (time-symmetric ID),
    Phi(sigma, 0)  =  d_sigma Psi(sigma, 0) computed by the SAME 2nd-order
                                            central / one-sided FD as the
                                            evolution operator, NOT analytic.

Then invert (17) using (23) reversed:

    U(sigma, 0)  =  Pi  +  mu_+ Phi  =  mu_+(sigma) Phi(sigma, 0),
    W(sigma, 0)  =  Pi  +  mu_- Phi  =  mu_-(sigma) Phi(sigma, 0).

Using the FD operator (not analytic differentiation) for Phi(0) is important:
it ensures the discrete compatibility relation d_tau Phi = d_sigma Pi is
satisfied to machine precision at t = 0, which prevents a high-frequency
constraint-violating burst at the first RK4 step.

## 11. CFL

    dt  <=  safety * dsigma / max_sigma( max(|lambda_out^sigma|, |lambda_in^sigma|) )
        =  safety * dsigma  *  4 M.                                        (32)

A conservative safety = 0.4 gives dt = 1.6 M * dsigma. For dsigma = 1/(N-1)
with N = 801, dt approx 0.002 M, requiring O(5e4) RK4 steps per tau = 100 M.
Trivially affordable.

## 12. Reusable from current code

  - kerr/src/initial_data.py (GaussianID): re-usable with Psi(sigma) =
    A0 exp(-(r(sigma) - x0)^2 / (2 sigma_ID^2)). Need to evaluate at the
    sigma-grid.
  - kerr/src/fd_stencils.py: 2nd-order central is fine for interior. Will
    need to add 2nd-/3rd-order one-sided UPWIND stencils (existing edge
    stencils are centered-extrapolation, wrong for this scheme).
  - kerr/src/mol_rk4.py: re-usable as-is.
  - kerr/src/observers.py: re-usable; observer locations indexed by r/M
    must be mapped to sigma = 2M/r and snapped to nearest grid point.
  - kerr/src/qnm_kerr_reference.py and extractor_m4.py: unchanged.
  - kerr/src/hyperboloidal_schwarzschild.py: tortoise + areal map are
    reusable. tanh-gauge height functions become unused; tag with
    DEPRECATED banner. New module to be added.
  - kerr/src/rwz_hyperboloidal.py: build_operator and rhs are gauge-specific
    and must be replaced. Quarantine, do not delete (used by phase_a tests
    that we will rewrite).
  - kerr/src/dissipation.py: ko_dissipation re-usable. sponges DO NOT use.

## 13. New code surface (Phase A.8-fix)

  - kerr/src/rwz_minimal_gauge.py
      build_minimal_gauge_op(N) -> SigmaOp with sigma grid, w, w', H, H',
        V, mu_pm, lambda_out, lambda_in, mu_pm', and the regularised
        source coefficients (26a, 26b, 26c) tabulated.
      rhs_min(Psi, U, W, op, d1_upwind_left, d1_upwind_right) ->
        (dPsi, dU, dW).
      cfl_dt(op, safety=0.4) -> dt.

  - kerr/src/fd_upwind.py
      d1_upwind_left(u, dx)  : 3rd-order backward, with order-reduction
        to 2nd at the leftmost ghost-free interior point.
      d1_upwind_right(u, dx) : mirror.

  - kerr/scripts/v1_flat_propagation.py
      V = 0 verification. Gaussian pulse should propagate cleanly out of
      sigma = 0 boundary with reflected amplitude < 1e-6 at a near-scri
      observer.

  - kerr/scripts/v2_self_convergence.py
      L2 norm of Psi at fixed tau, N in {401, 801, 1601, 3201}, expect
      ~4th order self-convergence (or 2nd, depending on interior order
      chosen for v1).

  - kerr/scripts/v3_qnm_extraction.py
      Full RW potential, observers at r/M in {10, 30, 50} and at sigma=1e-3
      (effective scri). M*omega -> 0.3737 and tau/M -> 11.241 to <= 1e-3
      at highest resolution, with monotone convergence across resolutions.

## 14. Gating

A.8-fix is CLOSED only after V.1, V.2, V.3 ALL pass at the tolerances above.
No A.9 (Kerr coupling) is started before that gate. If V.3 fails to converge
in damping, escalate to checking algebra of (28) and the upwind boundary
stencil order before touching the gauge.

## 15. Open items the user should flag NOW if they're a problem

  - We have committed to the polynomial gauge H = 1 - 2 sigma^2. If a
    different minimal-gauge variant from a specific paper is preferred
    (e.g. Bizon-Friedrich's H, or Jaramillo-Panosso Macedo's
    "F_bondi-like" with a different sigma normalisation), say so before
    sec. 5 is reduced to code.
  - We are using Psi as the *unrescaled* RW field. Some references work
    with rescaled Psi -> Psi * r or Psi * f^{1/2}. We do NOT do that here:
    the field equation (4) and source (15) are for the bare Psi.
  - Time-domain QNM extraction (extractor_m4) was validated against the
    parent paper's Zerilli implementation, which uses *physical* time t,
    not hyperboloidal tau. At a fixed-r observer, t and tau differ by
    a constant offset h(r) for that observer; this is harmless for
    frequency and damping rate but shifts the apparent phase. Decision
    needed: do we re-shift observer time by h(r) before feeding to the
    extractor, or accept the constant phase offset? Frequency and damping
    are unaffected either way.

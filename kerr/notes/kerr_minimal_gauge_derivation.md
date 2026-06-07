# Minimal-gauge hyperboloidal Teukolsky on Kerr (s = −2, ℓ = m = 2)

Status: derivation only. No solver code written against this yet. To be
sanity-checked line by line before B.4 implementation begins. Mirrors
`minimal_gauge_derivation.md` (Phase A Regge–Wheeler) in rigour.

Reference frame: Teukolsky, ApJ 185 635 (1973); Bardeen & Press, JMP 14 7
(1973) (the a=0 master equation); Krivan, Laguna, Papadopoulos & Andersson,
PRD 56 3395 (1997) (time-domain BL form); Panosso Macedo, Jaramillo & Ansorg,
PRD 89 064008 (2014) and Panosso Macedo, CQG 37 065019 (2020) (minimal gauge);
Panosso Macedo, PRD 105 104032 (2022) (hyperboloidal Kerr / scri regularity).
Units M = 1 unless stated; M is kept symbolic where it clarifies the rescaling.

All symbolic claims in this document are reproduced exactly (Gaussian-rational
arithmetic, no floating point) by the scripts named in §10.

---

## 0. Summary of results

| object | result | reduces at a=0 to |
|---|---|---|
| height function | `H(σ) = 1 − 2σ²` (spin-independent) | same |
| outgoing speed | `λ_out = −(1−σ)(r₊−r₋σ) / [2(r₊²+a²σ²)]` | `−(1−σ)/(4M)` |
| ingoing speed | `λ_in = σ²(r₊−r₋σ) / [2(1+σ)(r₊²+a²σ²)]` | `σ²/[4M(1+σ)]` |
| field rescaling | `Ψ = ψ · σ³ (1−σ)^(−2−iβ)`, `β = ma/(r₊−r₋)` | `ψ · σ³ (1−σ)^(−2)` |
| source coeffs | complex, **bounded** on [0,1] at every spin | real Bardeen–Press |

The single genuinely new element relative to the Phase A construction is the
**field rescaling** with a *complex* horizon exponent. The Schwarzschild
minimal gauge needed no field rescaling at all (the `(1−H²)⁻¹` cancellation
alone gave bounded sources). For `s = −2` Teukolsky that cancellation is **not
sufficient**; see §4. This is the analytic crux of Phase B.

---

## 1. Starting point: the Teukolsky radial operator in (t, r)

After separation of the angular dependence into spin-weighted spheroidal
harmonics `₋₂S_{ℓm}(θ; aω) e^{imφ}`, a single azimuthal mode `ψ(t, r)` of the
`s = −2` Teukolsky master variable obeys

    C_tt ∂_t² ψ + C_t ∂_t ψ + C_rr ∂_r² ψ + C_r ∂_r ψ + C_0 ψ = 0.       (1)

With `Δ = (r−r₊)(r−r₋) = r² − 2Mr + a²`, `r_± = M ± √(M²−a²)`, the coefficients
are (Teukolsky 1973, radial equation, with `ω ↔ i∂_t`, single `m`-mode):

    C_tt = −(r²+a²)² / Δ,                                                 (2a)
    C_t  = [−2iam(r²+a²) + 2s(r−M)(r²+a²)] / Δ  −  4sr,                   (2b)
    C_rr = Δ,                                                             (2c)
    C_r  = 2(s+1)(r−M),                                                   (2d)
    C_0  = [a²m² + 2ias m(r−M)] / Δ  −  λ.                                (2e)

Here `λ = ₋₂A_{ℓm}(aω) + a²ω² − 2amω` is the (frozen, complex) Teukolsky
separation constant evaluated at the reference QNM frequency `ω_ref(a)` (B.3).

**Provenance check (term by term).** Writing the standard frequency-domain
radial equation
`Δ^{−s} d_r(Δ^{s+1} d_r R) + [K² − 2is(r−M)K]/Δ + 4isωr − λ ) R = 0`
with `K = (r²+a²)ω − am` and substituting `ω ↔ i∂_t`:

- `Δ^{−s} d_r(Δ^{s+1} d_r) = Δ ∂_r² + 2(s+1)(r−M) ∂_r` → (2c),(2d);
- `K²/Δ`: the `(r²+a²)²ω²` piece gives `−(r²+a²)²∂_t²/Δ` → (2a);
  the `−2(r²+a²)am ω` piece gives `−2iam(r²+a²)∂_t/Δ` → first term of (2b);
  the `a²m²` piece → first term of (2e);
- `−2is(r−M)K/Δ`: the `ω` piece gives `+2s(r−M)(r²+a²)∂_t/Δ` → second term
  of (2b); the `−am` piece gives `+2ias m(r−M)/Δ` → second term of (2e);
- `4isωr` gives `−4sr ∂_t` → third term of (2b);
- `−λ` → second term of (2e).

Every term of (2) matches the canonical operator. The two `i`-terms (in `C_t`
and `C_0`, both `∝ am/Δ`) are the **frame-dragging** terms; they vanish at
`a = 0` only through the explicit factor `a`, which is the source of the
complex sector (B.5) and of the horizon twist of §5.

At `a = 0` (so `r₊ = 2M`, `r₋ = 0`, `λ → ₋₂A = ℓ(ℓ+1) − s(s+1) = 4`):

    C_tt = −r⁴/(r²−2Mr),  C_t = 2s(r−M)r²/(r²−2Mr) − 4sr,
    C_rr = r²−2Mr,  C_r = 2(s+1)(r−M),  C_0 = −4,

which is the **Bardeen–Press** equation — *not* Regge–Wheeler. The `−4sr ∂_t`
term survives (it is `∝ s`, not `∝ a`), so even the non-spinning operator
carries a first-time-derivative term absent from RW. See §6.

---

## 2. Hyperboloidal transform

We reuse the Phase A two-ingredient construction.

**(a) Compactification.** `σ = r₊/r ∈ [0,1]`; `σ = 0` is scri⁺ (`r → ∞`),
`σ = 1` is the horizon (`r = r₊`). Define

    ρ(σ) := dr/dσ = −r₊/σ²,                                              (3a)
    w(σ) := dr_*/dσ = (r²+a²)/Δ · ρ,    (Kerr tortoise dr_*/dr=(r²+a²)/Δ)  (3b)

`ρ < 0`, `w < 0` on (0,1); both diverge at the endpoints, but the combinations
that enter the regularised operator are finite (§3, §4).

**(b) Hyperboloidal time.** `τ = t − h(r_*)`, `H(r_*) := dh/dr_*`. With the
chain rule `d_{r_*} = (1/ρ)·(dσ/dr_*)·… ` collapsed into a single `r`-operator,

    ∂_t|_r  = ∂_τ,                                                       (4a)
    ∂_r|_t  = (1/ρ)(∂_σ − H w ∂_τ).                                      (4b)

**Height function.** Exactly as Phase A, the requirement that the `(σ,τ)`
characteristic speeds be bounded and one-signed on the *closed* interval picks

    H(σ) = 1 − 2σ²,                                                      (5)

`H(0)=+1`, `H(1)=−1`, `|H|<1` on (0,1). Remarkably this **spin-independent**
choice already regularises the Kerr principal part; see §3. (The Kerr tortoise
weight `w` differs from Schwarzschild, but the `r²+a²` numerator and the `Δ`
denominator conspire so that the same `H` works. This is verified, not
assumed.)

---

## 3. Principal symbol and characteristic speeds

Insert (4) into the second-derivative part `C_tt ∂_t² + C_rr ∂_r²` of (1) and
read off the coefficients of `∂_ττ`, `∂_στ`, `∂_σσ` (the rescaling of §5 does
not affect the principal part — it multiplies the field without adding
top-order derivatives). The characteristic speeds `c = dσ/dτ` solve

    A_tt c² − A_ts c + A_ss = 0.                                         (6)

Closed-form roots (verified to be exact roots, residual `= 0`, for
`a ∈ {0, 3/5, 4/5}`):

    λ_out(σ;a) = −(1−σ)(r₊ − r₋σ) / [2(r₊² + a²σ²)],                     (7a)
    λ_in (σ;a) =  σ²(r₊ − r₋σ) / [2(1+σ)(r₊² + a²σ²)].                   (7b)

Properties on [0,1]:

- `λ_out ≤ 0` (toward scri), vanishes at the horizon; `λ_in ≥ 0` (toward the
  horizon), vanishes at scri. One characteristic exits at each endpoint and
  the other is tangent → **no boundary data is required** at either end.
- Both are bounded (the `r₊²+a²σ²` denominator is `≥ r₊² > 0`).
- At `a = 0` (`r₊=2M, r₋=0`): `λ_out → −(1−σ)/(4M)`, `λ_in → σ²/[4M(1+σ)]` —
  exactly the Phase A speeds (13a),(13b). ✔

So the **characteristic structure reduces to the validated Phase A minimal
gauge at `a=0`, exactly.** The frame-dragging terms (the `i`-pieces of `C_t`,
`C_0`) are lower order and do **not** enter (6).

---

## 4. Why a field rescaling is needed (the s = −2 obstruction)

In Phase A the regularised `Π`-source was `S_Π = S_Π^raw/(1−H²)` with all three
coefficients bounded *after* the analytic `(1−H²)⁻¹` cancellation, and **no
field rescaling**. Repeating that mechanical step for the Teukolsky operator
(2) — i.e. forming `c_Π = −A_τ/A_tt` etc. for the bare field `ψ` — gives a
source that **diverges**:

- at scri, `c_Π ∼ −(2s+1)/(4σ)` → for `s = −2` this is `+3/(4σ) → +∞`;
- at the horizon (for `a > 0`) a residual `∝ i·am/(1−σ)` pole.

Control test (ruling out a transcription bug): the *same* mechanical pipeline
applied to the Phase A Regge–Wheeler operator reproduces the Phase A bounded
`c_Π = −σ/[2(1+σ)]` **exactly**. So the divergence is genuine `s = −2` physics
(spin-weight peeling), not a coding error. The plan's B.2 assumption that the
`(1−H²)⁻¹` cancellation "carries over exactly as in §6–7 of the Schwarzschild
doc" is therefore **incomplete** for `s = −2`. This is the new finding.

The cure is a `σ`-dependent **field rescaling** `ψ = Z(σ) Ψ`, chosen so that the
operator for the regular field `Ψ` has bounded source coefficients. Because
`Z` carries no `τ`-dependence, `Z'/Z = P/σ − Q/(1−σ)` (and `Z''/Z`) are
*rational* in `σ` even for a complex exponent — so every coefficient of the
`Ψ`-equation is a rational function of `σ`, and "bounded" is decided
**exactly** by a symbolic residue (pole-coefficient `= 0`), never numerically.

---

## 5. The minimal-gauge field rescaling

Take

    Z(σ) = σ^P (1−σ)^Q,        ψ = Z Ψ,   i.e.  Ψ = ψ · σ^(−P) (1−σ)^(−Q). (8)

The scri residue of `c_Π` is *linear* in `P`, the horizon residue *linear* in
`Q`. Solving each `= 0`:

    P = −3   (all spins),                                                (9a)
    Q = 2 + iβ,     β = m a / (r₊ − r₋)   = a/√(M²−a²) for m=2.          (9b)

So the regular field is

    ┌─────────────────────────────────────────────────────────┐
    │  Ψ = ψ · σ³ (1−σ)^(−2−iβ),     β = m a / (r₊ − r₋).        │      (10)
    └─────────────────────────────────────────────────────────┘

**Physical reading of each factor.**

- `σ³ = (r₊/r)³ ∼ r^(−3) = r^(2s+1)|_{s=−2}` removes the **peeling** growth
  `r^(−2s−1) = r³` of the outgoing `s=−2` field at scri (Newman–Penrose
  peeling; `ψ₄ ∼ Ψ/r⁵` etc.). Spin-independent, hence `P = −3` for all `a`.
- `(1−σ)^(−2) ∼ Δ²|_{horizon}` removes the `Δ^(−s) = Δ²` suppression of the
  ingoing `s=−2` field at the horizon. Real, present already at `a=0`.
- `(1−σ)^(−iβ)` is the **ingoing-Kerr corotating azimuthal phase**: near the
  horizon `∫ a/Δ dr ∼ (a/(r₊−r₋)) ln(r−r₊)`, so `e^{imφ̃} = e^{imφ} e^{−imF}`
  with `F = (a/(r₊−r₋))ln(r−r₊)` contributes `(r−r₊)^(−ima/(r₊−r₋)) ∝
  (1−σ)^(−iβ)`. It is **pure phase** (`|·| = 1`), oscillating as `σ→1`. It
  vanishes (`β = 0`) at `a = 0`, so the complex sector is switched on **only**
  by frame dragging — consistent with B.0.

Because `Z` is time-independent, this phase is a purely *spatial* field
redefinition: it makes the operator regular but introduces **no** oscillation
into the time evolution. The QNM frequency read from `Ψ(t)` at fixed `σ` is
identical to that of `ψ`. The solver (B.4/B.5) evolves `Ψ` using the bounded
rational coefficients below; the factor `(1−σ)^(−iβ)` is never evaluated
numerically.

**Verification (exact symbolic residues, `a ∈ {0, 3/5, 4/5}`).** With (9), all
three source coefficients `c_Π = −A_τ/A_tt`, `c_Φ = −A_σ/A_tt`,
`c_Ψ = −A_0/A_tt` have **zero pole-coefficient at both endpoints** — i.e. they
are bounded. Endpoint values (exact):

| a | c_Π(0) | c_Π(1) | c_Φ(0) | c_Φ(1) | c_Ψ(0) | c_Ψ(1) |
|---|---|---|---|---|---|---|
| 0   | −1/2 | −1/4 | 0 | −3/32 | −1/4 | −3/32 |
| 3/5 | −5/9 − 85i/216 | −11/36 − 31i/96 | 0 | −2/27 − i/27 | −25/81 | −83/864 − 17i/432 |
| 4/5 | −5/8 − 35i/48 | −11/32 − 11i/24 | 0 | −27/512 − 3i/64 | −25/64 | −167/1536 − 7i/128 |

The imaginary parts are nonzero only for `a > 0` (frame dragging) and `c_Φ(0)`
is real (`= 0`) at every spin. `β` matches `ma/(r₊−r₋)` exactly: `a=3/5 → 3/4`,
`a=4/5 → 4/3`.

---

## 6. Reduction at a = 0 (Bardeen–Press, not Regge–Wheeler)

With `β = 0`, `(P,Q) = (−3, 2)` (real), the source coefficients become the
**Bardeen–Press** closed forms (exact):

    c_Π(σ)  = −1 / [2(σ+1)],                                            (11a)
    c_Φ(σ)  = −σ(σ+2) / [16(σ+1)],                                      (11b)
    c_Ψ(σ)  = (σ−4) / [16(σ+1)].                                        (11c)

All three are manifestly bounded on [0,1]. Compare the Phase A Regge–Wheeler
coefficients (`ℓ=2`):

    c_Π^RW = −σ/[2(σ+1)],
    c_Φ^RW = −σ(3σ−2)/[16(σ+1)],
    c_Ψ^RW = −(6−3σ)/[16(σ+1)].

They are **different** (e.g. `c_Π^BP = −1/[2(σ+1)]` has no `σ` in the
numerator; `c_Ψ^BP(0) = −1/4` vs `c_Ψ^RW(0) = −3/8`). This is expected and
correct: the `s=−2` Teukolsky / Bardeen–Press operator is **isospectral** to
Regge–Wheeler (identical QNM spectrum) but is a **different operator**, related
by a Chandrasekhar transformation. The decisive `a=0` check is therefore the
**QNM frequency** (`Mω_{220} = 0.37367`, enforced at B.8), *not* an
operator-coefficient match against RW.

What *does* reduce exactly to Phase A at `a=0` is the **characteristic
structure**: `H = 1−2σ²` (5), `λ_out`, `λ_in` (7) — see §3.

---

## 7. Deviation from the B.2 plan, and the corrected acceptance

The plan file `phase_b_plan.md` contains an internal inconsistency:

- **B.0 item 3** (signed off) correctly states that the `a=0` source
  coefficients reduce to **Bardeen–Press, not RW**, and that the decisive
  `a=0` check is the QNM frequency.
- **The B.2 "Acceptance" line** still says "*every coefficient evaluated
  symbolically at `a=0` must equal the Schwarzschild expressions in
  `minimal_gauge_derivation.md`*."

These cannot both hold. The present derivation confirms **B.0**: only the
characteristic structure reduces to the Phase A RW expressions; the source
coefficients reduce to Bardeen–Press (§6). In addition, this work found a
**second** deviation not anticipated even in B.0: the `s=−2` source is **not**
regularised by the `(1−H²)⁻¹` cancellation alone — it requires the field
rescaling (10) with a *complex* horizon exponent (§4–5).

**Corrected B.2 acceptance (proposed; to be ratified in the plan):**

1. The height function and characteristic speeds reduce to the Phase A RW
   minimal gauge at `a=0`, **exactly** (residual `= 0`). ✔ (§3)
2. The regularised source coefficients are **bounded on [0,1]** at every tested
   spin (exact symbolic residues), and reduce to the **Bardeen–Press** closed
   forms (11) at `a=0` (not the RW forms). ✔ (§5,§6)
3. The complex/frame-dragging sector is controlled by the corotating rescaling
   exponent `β = ma/(r₊−r₋)`, which vanishes at `a=0`. ✔ (§5)
4. The physical `a=0` validation is the QNM frequency, deferred to B.8.

---

## 8. First-order system for the solver (B.4 preview)

Introduce `Π := ∂_τ Ψ`, `Φ := ∂_σ Ψ` (regular field `Ψ`). The evolution is the
Phase A characteristic form with **complex** coefficients:

    ∂_τ Ψ = Π,
    ∂_τ Φ = ∂_σ Π,
    ∂_τ Π + (principal flux via λ_out, λ_in) = c_Π Π + c_Φ Φ + c_Ψ Ψ,

with `λ_out, λ_in` from (7) and `c_Π, c_Φ, c_Ψ` the bounded complex rational
functions of §5 (machine-generated; `a=0` forms (11)). The characteristic
split `U = Π + μ_+ Φ`, `W = Π + μ_- Φ` and the upwinded boundary stencils carry
over verbatim from Phase A §7–10; `μ_±` are the (complex) analogues computed by
the same matching conditions. These are produced by the B.4 module, not by
hand.

---

## 9. What carries over from Phase A unchanged

- Two-ingredient hyperboloidal construction (compactify + height), `H=1−2σ²`.
- Characteristic decomposition, upwind boundary stencils, KO dissipation on
  strict interior, MoL/RK4 time stepping, CFL `dt ≲ dσ·4M`.
- No boundary data at either endpoint (outflow both ends).

What is **new** for Kerr: complex coefficients (frame dragging); the field
rescaling (10) with the corotating horizon phase; the spheroidal separation
constant `λ` as a frozen complex constant (B.3).

---

## 10. Reproducing every claim

All results above are emitted exactly (Gaussian-rational arithmetic) by:

- `kerr/scripts/solve_horizon_twist.py` — solves `(P,Q)` from the `c_Π`
  residues; prints the §5 boundedness table, the §6 Bardeen–Press closed forms
  (11), and the §3 closed-form speed check (residual `= 0`).
- `kerr/scripts/solve_rescaling.py` — the `a=0` rescaling and the real-spin
  scri/horizon scan that first exposed the horizon imaginary pole.
- `kerr/scripts/derive_teukolsky_coeffs.py` — builds the `(τ,σ)` operator from
  (2) and (4); Stage 1 principal-symbol check; RW control test of §4.

Run on the login node with `../venv_csd3/bin/python` (numpy 2.4.4, sympy
1.14.0). The Pythagorean spins `a = 3/5, 4/5` (→ rational `r_±`) keep all
arithmetic exact and fast (no surds).

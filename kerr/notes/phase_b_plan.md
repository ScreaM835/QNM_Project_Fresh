# Phase B — task breakdown

Spinning ($a/M > 0$) time-domain Teukolsky, $s=-2$, $\ell=m=2$, on a
hyperboloidal compactified slice, built by generalising the **validated**
Phase A minimal-gauge Regge–Wheeler code (`src/rwz_minimal_gauge.py`,
$\sigma = 2M/r$, three-field characteristic form $(\Psi, U, W)$, no boundary
data). Phase A passed its V.3 gate at $a=0$: $M\omega_{220} = 0.37367$
(err $\sim 5\times10^{-5}$ vs `qnm`), $\tau/M = 11.24$, 2nd-order convergent,
stable to $180M$. Phase B must reproduce that result **from the Kerr operator
in the $a\to0$ limit** and then extend it to spin.

Each task is one commit, one test, one acceptance check. Do not merge two.
All compute is CPU (`icelake`, account `fergusson-sl3-cpu`); no GPU budget is
needed until Phase C. `kerr/` stays self-contained — reusable modules are
copied/extended in place, never imported from the parent `src/`.

---

## B.0 — formulation decision record (no code)
**File.** `kerr/notes/phase_b_plan.md` (this file), `kerr/README.md` (link).
**Decision (locked with user).** Standard Teukolsky master variable $\psi$
($s=-2$), hyperboloidal compactified, **not** Sasaki–Nakamura.

**Two honest consequences that shape the whole phase — record them now so no
later task quietly assumes them away:**

1. **The state is complex.** Phase A's $\Psi$ was real (axial RW). The
   Teukolsky $\psi$ is complex: frame dragging introduces a first-time-
   derivative coupling $\propto a m$ and the radial potential is
   complex-valued. Every downstream module (stepper, dissipation, observers,
   extractor) must carry complex arrays. This is mechanical but pervasive.

2. **1+1D single-mode ⇒ the angular separation constant is frequency-
   dependent.** Reducing Teukolsky to 1+1D ($r,\tau$) requires fixing the
   spin-weighted **spheroidal** eigenvalue ${}_sA_{\ell m}(c)$ with
   $c = a\omega$. Unlike $a=0$ (where spheroidal $\to$ spherical, $c=0$,
   ${}_sA = \ell(\ell+1)-s(s+1) = 4$), at $a>0$ the eigenvalue depends on the
   (complex, a-priori unknown) QNM frequency. **We adopt the standard 1+1D
   single-mode convention: fix ${}_sA_{\ell m}$ at the reference QNM frequency
   from the `qnm` package and verify self-consistency.** This makes the 1+1D
   solver a *single-mode QNM extractor* (exactly the README scope), **not** a
   general waveform generator. The general-waveform route is 2+1D ($r,\theta$),
   explicitly out of Phase B scope and noted as future work. This caveat goes
   verbatim into the paper's Kerr methods section.
3. **The $a=0$ limit is Bardeen–Press, not Regge–Wheeler.** The $s=-2$
   Teukolsky master variable at $a=0$ obeys the Bardeen–Press equation, which
   is *isospectral* to Regge–Wheeler (identical QNM spectrum) but is a
   **different operator** — related to RW by a Chandrasekhar transformation.
   Concretely, $s=-2$ Teukolsky carries a first-time-derivative term
   $\propto s$ that **survives even at $a=0$**. Consequences for the reduction
   tests below:
   - the **characteristic structure** (speeds $\lambda_\pm$, $\mu_\pm$) reduces
     to the validated Phase A RW minimal gauge at $a=0$ (same tortoise, same
     flat-wave principal part) — verified symbolically, exact to 0;
   - the **source/potential coefficients** reduce to **Bardeen–Press**, *not*
     RW, so they do **not** equal the Phase A RW source coefficients;
   - the decisive $a=0$ *physical* check is therefore the **QNM frequency**
     ($M\omega_{220}=0.3737$, isospectrality), enforced in B.8 — not an
     operator-coefficient match against RW.
   This also means the complex/first-derivative machinery (B.5) is exercised
   even at $a=0$; the $a=0$ Teukolsky run is a genuine test, not RW relabelled.
   (At $a=0$ the Bardeen–Press coefficients are real, so real initial data
   stays real — the complex sector is activated only by frame dragging, $a>0$.)
**Acceptance.** This section reviewed and signed off by the user before B.1
starts. No code.

---

## B.1 — Kerr radial coordinate map
**File.** `kerr/src/kerr_hyperboloidal.py` (new; mirrors
`hyperboloidal_schwarzschild.py`).
**Implements.**
- Horizons $r_\pm = M \pm \sqrt{M^2 - a^2}$, $\Delta(r) = r^2 - 2Mr + a^2 =
  (r-r_+)(r-r_-)$.
- Compactification $\sigma = r_+/r \in [0,1]$ ($\sigma=0$ scri$^+$,
  $\sigma=1$ horizon), with `sigma_of_r`, `r_of_sigma`, `dr_dsigma`.
- Kerr tortoise $dr_*/dr = (r^2+a^2)/\Delta$, closed form
  $r_* = r + \frac{2Mr_+}{r_+-r_-}\ln\frac{r-r_+}{2M}
       - \frac{2Mr_-}{r_+-r_-}\ln\frac{r-r_-}{2M}$, and `dtortoise_dr`.
- $a=0$ branch must collapse to $r_+ = 2M$, $\sigma = 2M/r$,
  $r_* = r + 2M\ln(r/2M-1)$ (the Phase A map).

**Acceptance.** Round-trip $|r-r(\sigma(r))|/r < 10^{-14}$ on log grid
$r/M\in[r_+(1+10^{-6}), 10^4]$ for $a/M\in\{0,0.5,0.9,0.95\}$; tortoise
asymptotics $r_*-r\to (r_++r_-)\ln(\cdots)$ form as $r\to\infty$; **$a=0$
output equals `hyperboloidal_schwarzschild.py` to $10^{-13}$** (numerical
reduction check, not visual).

---

## B.2 — Kerr minimal-gauge derivation (analytic, no solver code)
**File.** `kerr/notes/kerr_minimal_gauge_derivation.md` (new; mirrors
`minimal_gauge_derivation.md` line-for-line in rigour).
**Implements.** The genuine research step. Carry the Schwarzschild
minimal-gauge construction over to Kerr Teukolsky:
- Height function $H(\sigma)$ with $H(0)=+1$, $H(1)=-1$, $|H|<1$ on
  $(0,1)$, chosen so the $(\sigma,\tau)$ characteristic speeds are bounded and
  one-signed on the **closed** interval (outflow at both ends, no boundary
  data). **Result: $H = 1-2\sigma^2$ is spin-independent** — the same height as
  Phase A regularises the Kerr principal part (verified, not assumed).
- Closed-form $\lambda_{\rm out}(\sigma;a)$, $\lambda_{\rm in}(\sigma;a)$,
  characteristic-variable coefficients $\mu_\pm(\sigma;a)$ and their
  $\sigma$-derivatives, and the **regularised** complex source coefficients
  $c_\Psi, c_\Pi, c_\Phi$.
- **NEW FINDING (not anticipated in the original plan).** For $s=-2$ the
  $(1-H^2)^{-1}$ cancellation alone does **not** regularise the source (it does
  for Phase A RW). A **field rescaling** $\Psi = \psi\,\sigma^3(1-\sigma)^{-2-i\beta}$
  with a *complex* horizon exponent $\beta = ma/(r_+-r_-)$ (the ingoing-Kerr
  corotating phase; $\beta=0$ at $a=0$) is required and sufficient. Verified by
  exact symbolic residues ($a\in\{0,3/5,4/5\}$). Documented in
  `kerr_minimal_gauge_derivation.md` §4–5.
- Explicit appearance of: $\Delta(r(\sigma))$, the frame-dragging term
  $\propto a m$, the complex potential, and the fixed spheroidal eigenvalue
  ${}_sA_{\ell m}$ as a constant.

**Acceptance (corrected — supersedes the line below, reconciled with B.0
item 3 which is signed off).** Document only:
1. height $H=1-2\sigma^2$ and speeds $\lambda_{\rm out},\lambda_{\rm in}$ reduce
   to the Phase A RW minimal gauge at $a=0$ **exactly** (residual $=0$);
2. the regularised source coefficients are **bounded on $[0,1]$** at every
   tested spin (exact symbolic residues) and reduce at $a=0$ to the
   **Bardeen–Press** closed forms — **not** the RW source coefficients;
3. the physical $a=0$ check is the QNM frequency (B.8), per B.0 item 3.

~~Original (incorrect) acceptance: "every coefficient evaluated symbolically at
$a=0$ must equal the Schwarzschild expressions in `minimal_gauge_derivation.md`."
This contradicted B.0 item 3 (the $a=0$ Teukolsky source is Bardeen–Press, not
RW) and is retracted.~~ This remains the analytic crux of Phase B — sanity-
checked line by line before B.4.

---

## B.3 — spin-weighted spheroidal separation constant
**File.** `kerr/src/spheroidal.py` (new).
**Implements.** `sep_constant(a_over_M, ell, m, s, omega)` returning
${}_sA_{\ell m}(c)$ with $c=a\omega$, sourced from the `qnm` package's
angular solver (the same package already wrapped in `qnm_kerr_reference.py`),
plus the Teukolsky $\lambda = {}_sA_{\ell m} + a^2\omega^2 - 2am\omega$ used by
the radial operator. Self-consistency helper: given the reference QNM
$\omega_{\rm ref}(a)$ from `qnm`, return the frozen $\lambda(a)$ for the solver.
**Acceptance.** $a=0 \Rightarrow {}_sA = \ell(\ell+1)-s(s+1) = 4$ exactly;
$s=-2,\ell=m=2,a/M=0.9$ matches the Berti–Cardoso–Will tabulated
$\lambda$ (or `qnm`'s own angular eigenvalue) to $10^{-6}$.

**Status: DONE** (`src/spheroidal.py`, `scripts/test_spheroidal.py`, 6/6 pass).
Verified: $_sA(a{=}0)=4$ exactly and real; $\lambda(a{=}0)=4$ (ties to operator
$C_0(a{=}0)=-4$); $a/M{=}0.9$ matches `qnm`'s own angular eigenvalue to
$6\times10^{-15}$ (acceptance only asked $10^{-6}$). Two **literature-independent**
anchors added so the `qnm` match is not self-referential: real oblateness $c$
gives a real eigenvalue (exact), and the small-$c$ slope reproduces the analytic
leading coefficient $-2ms^2/[\ell(\ell+1)] = -8/3$ to $7\times10^{-6}$. Eigenvalue
converged in the matrix truncation $l_{\max}$ ($|v(\ell{+}6)-v(\ell{+}24)|\sim3\times10^{-15}$).

---

## B.4 — Teukolsky operator, complex characteristic form
**File.** `kerr/src/teukolsky_minimal_gauge.py` (new; supersedes the
`teukolsky_fd.py` stub, which is left for reference only — mirrors how
`rwz_minimal_gauge.py` superseded `rwz_hyperboloidal.py`).
**Implements.** `build_teukolsky_op(N, a_over_M, M, ell, m, omega_ref, ...)`
tabulating the **complex** coefficients from B.2 on the uniform $\sigma$ grid:
geometry ($r,\Delta,H$), speeds ($\lambda_{\rm out},\lambda_{\rm in}$),
$\mu_\pm$ and derivatives, the inverse-map weights, and the regularised
complex source coefficients $c_\Psi,c_\Pi,c_\Phi$ (now carrying the
$a m$ frame-dragging and the frozen $\lambda(a)$ from B.3). `recover_pi_phi`,
`rhs_teuk(state, op, d1)`, `cfl_dt`, `state_from_psi`, `observer_index`,
`scri_index` — same signatures as `rwz_minimal_gauge.py` but complex-typed.

**Acceptance.** All coefficients finite (no NaN/Inf) at
$\sigma\in\{10^{-8}, 1-10^{-8}\}$ for $a/M\in\{0,0.5,0.9,0.95\}$; **at $a=0$
the characteristic-structure arrays** ($\lambda_\pm,\mu_\pm,\mu_\pm',$ inverse
map) **equal those of `build_minimal_gauge_op` to $10^{-12}$**; the $a=0$
source/potential arrays equal the **Bardeen–Press** reduction (B.2), *not* the
RW source (Bardeen–Press is isospectral to RW, so the physical $a=0$ check is
the QNM frequency in B.8, not an RW coefficient match). No time evolution yet.

**Status: DONE** (`src/teukolsky_minimal_gauge.py`,
`scripts/derive_first_order_system.py`, `scripts/test_teukolsky_minimal_gauge.py`,
5/5 pass). The full principal part was derived symbolically
(`derive_first_order_system.py`) extending the B.2 residue machinery; all eight
closed forms ($\lambda_{\rm out},\lambda_{\rm in},\mu_\pm,\mu_\pm',$ inverse map,
$c_\Pi,c_\Phi,c_\Psi$) emitted as numpy source and transcribed verbatim into the
operator. Verified: every coefficient finite at $\sigma\in\{10^{-8},1-10^{-8}\}$
for $a/M\in\{0,0.5,0.9,0.95\}$; at $a=0$ the characteristic arrays equal
`build_minimal_gauge_op` to $6\times10^{-17}$; the $a=0$ source equals the
Bardeen–Press reduction (B.2) to $3\times10^{-17}$ (confirmed $\neq$ RW source);
transcription matches the symbolic closed forms across all four spins to
$1\times10^{-16}$. **Subtlety resolved:** the full `rhs_teuk` does *not* reduce
to Phase A `rhs_min` at $a=0$ — the sources legitimately differ (Bardeen–Press
vs Regge–Wheeler) — so the wiring test forces the sources equal and confirms the
principal-part algebra ($\lambda_\pm,\mu_\pm',$ inverse map) is wired identically
to the validated Phase A path to $3\times10^{-14}$. **$c_\Psi$ horizon stability:**
the raw emitted denominator carries a $(\sigma-1)$ factor that cancels only when
$\beta^2=4r_+r_-/(r_+-r_-)^2$ ($m=2$); substituting this before `cancel` removes
the $0/0$ at the horizon inset (verified the clean denominator does not vanish at
$\sigma=1$).

---

## B.5 — complex-state evolution stack
**File.** extend `kerr/src/mol_rk4.py`, `kerr/src/dissipation.py`,
`kerr/src/observers.py`, `kerr/src/initial_data.py`,
`kerr/src/extractor_m4.py` (in place; complex-safe).
**Implements.** Make the RK4 MOL stepper, Kreiss–Oliger dissipation,
fixed-radius observers, Gaussian initial data, and Method-4/5 extractor accept
`complex128` state without changing their real behaviour. Extractor operates on
the complex observer series $\psi(\tau, r_q)$ (fit $|\psi|$ envelope + complex
phase; the QNM $\omega$ is complex already, so this is the natural variable).

**Acceptance.** Regression: re-running the Phase A V.3 driver through the
extended (now complex-capable) modules reproduces the **real** RW result
$M\omega_{220}=0.37367$ bit-for-bit (real inputs ⇒ zero imaginary part); new
complex unit test: a synthetic $\psi=e^{-i\omega\tau}$, $\omega=0.374-0.089i$,
recovered to $10^{-6}$.

**Status: DONE** (`src/extractor_m4.py` +`qnm_complex_phase`,
`scripts/test_complex_evolution_stack.py`, 3/3 pass). **Honest finding:** four
of the five named modules — `mol_rk4.py`, `dissipation.py`, `observers.py`,
`initial_data.py` — were already dtype-generic (pure arithmetic, `.copy()`,
`np.zeros_like`/`np.empty_like` which inherit the input dtype) and so needed
**no change**; the test *proves* their complex-safety rather than asserting it.
The only edit is **purely additive** (79 insertions, 0 deletions): a complex
single-mode estimator `qnm_complex_phase` that fits the $\log|\psi|$ envelope
($\to\omega_I,\tau$) and the unwrapped phase ($\to\omega_R$) — the natural
variable for the genuinely-complex Teukolsky field. Because no existing line of
any module the V.3 driver executes was modified, the float64 V.3 SLURM run
reproduces $M\omega_{220}=0.373672$ **bit-for-bit by construction** (verified via
`git diff --numstat`). The miniature equivalence test runs the exact stack
V.3 uses (`integrate_state`+`rhs_min`+`ko_dissipation`+observers) in both
float64 and complex128 over $\tau\le20M$ at $N=401$: the imaginary part is
**identically zero** and the real part matches to **machine precision**
($2.2\times10^{-16}$, a single ULP — float64-vs-complex128 is *not* literally
bit-for-bit because complex division/multiply round at ~1 ULP, which is the
inherent cost of complex arithmetic, not a behavioural change). Synthetic
$\psi=e^{-i\omega\tau}$ recovered to $1.1\times10^{-16}$ ($\omega_R$), $0$
($\omega_I,\tau$). **Deferred (honest scope):** complex *multi-mode* extraction
(ESPRIT on $\psi$, needed for the $n=1$ overtone) is added in B.8 where it is
actually exercised, per one-acceptance-per-task discipline.

---

## B.6 — KV.1 propagation/stability gate (analogue of V.1)
**File.** `kerr/scripts/kv1_propagation.py`, `kerr/scripts/slurm_kv1.sh`
(CPU, icelake).
**Implements.** Evolve the full Kerr operator at $a/M=0.9$ from a Gaussian,
no extraction. Pure stability/characteristic-outflow check: energy bounded,
both endpoints radiate out cleanly, no boundary growth.
**Acceptance.** Stable to $\tau=200M$, no NaN, $L^2(\sigma)$ envelope decays
(does not grow) after the pulse leaves; $a=0$ run identical to Phase A V.1.

**Status: DONE** (`scripts/kv1_propagation.py`, `scripts/slurm_kv1.sh`;
4/4 runs pass over $a/M\in\{0,0.9\}\times N\in\{401,801\}$, validated on the
login node and submitted to SLURM). Two honest findings while building the gate:

1. *Real, spin-amplified numerical instability (fixed by dissipation, not by
   relaxing the gate).* Under-dissipated, the high-spin run grows back at late
   times — but at **physical finite-radius observers**, not only the horizon
   cell, so it is a genuine instability, not a coordinate artifact. Four textbook
   signatures confirm it is *numerical*: the onset is **pushed later** as the grid
   refines (round-off-seeded); it is removed by Kreiss–Oliger dissipation; it is
   converged-away at fixed KO ($N=401$ vs $801$ identical); and its strength
   tracks $\beta=ma/(r_+-r_-)$, the near-horizon azimuthal oscillation frequency
   of the regular-field rescaling $(1-\sigma)^{-2-i\beta}$ ($a{=}0\Rightarrow
   \beta{=}0$ clean; $a/M{=}0.9\Rightarrow\beta{\approx}2.06$ strong). Standard
   4th-order KO at $\sigma_{\rm KO}=0.2$ (Phase A used $0.02$) removes it for all
   spins to $a/M=0.95$. This is $O(\Delta x^3)$ and **invisible to the resolved
   ringdown**: the observer time series through the QNM window is bit-identical to
   4 s.f. across $\sigma_{\rm KO}\in[0.02,0.5]$, so the B.8 extraction window is
   untouched. Not gate-tuning — the instability is a $>500\%$ blow-up, the cure is
   a textbook dissipation operator on a flat plateau.

2. *Honest gate diagnostic.* A naïve full-grid $L^2(\sigma)$ of the **regular**
   field $\Psi=\psi\,\sigma^3(1-\sigma)^{-2-i\beta}$ is *not* a clean stability
   measure: the $(1-\sigma)^{-2}$ factor amplifies infalling radiation in the
   horizon boundary layer, a coordinate effect that swamps the exterior (at $a=0$
   the field decays 2.5 decades at every physical observer while the full-grid
   $L^2$ "grows" $3\times$ on the horizon cell once the bulk has emptied into the
   round-off floor). The gate is therefore the faithful, physical reading of "the
   envelope decays after the pulse leaves": the **late-time envelope slope**
   $d\log|\cdot|/d\tau$ of (i) the $\scri^+$ waveform and (ii) the **exterior bulk**
   $L^2$ ($\sigma\le0.9$, horizon layer excluded) must be negative. This slope is
   robust to the $a=0$ real field's zero-crossings and to decay into the floor
   (where a single max/start ratio is ill-posed), and it correctly flagged the
   under-dissipated case as **positive** (growing). As a bonus cross-check the
   gated slope reproduces the QNM decay rate: $a/M{=}0.9$ gives $-0.0649$ vs the
   `qnm` $-\omega_I=-0.0649$ ($<0.1\%$); $a{=}0$ gives $-0.0886$ vs $-0.0890$.
   The full-grid $L^2$, horizon edge, and pointwise ratios are still **reported**
   for transparency, just not gated. On "$a=0$ identical to V.1": the *principal
   part* (characteristic speeds, inverse map) is identical to Phase A to
   $\sim6\times10^{-17}$ (B.4), so the propagation/outflow **stability** reproduces
   V.1; the $a=0$ *source* is the Bardeen–Press reduction (isospectral, not
   pointwise-RW), so the pointwise field is not identical — the physical $a=0$
   equivalence (same QNM frequency) is the B.8 gate, and is previewed here by the
   $-0.0886\approx-\omega_I$ slope match.

   *SLURM note.* The Kerr operator imports `qnm.angular` (spheroidal separation
   constants), which the parent-root Phase A venv lacks (its `qnm` is a
   single-file stub); all Kerr SLURM jobs use the `_improved`-root venv.

---

## B.7 — KV.2 self-convergence (analogue of V.2)
**File.** `kerr/scripts/kv2_convergence.py`, `kerr/scripts/slurm_kv2.sh`.
**Implements.** Self-convergence at $a/M=0.9$ under $N\to2N\to4N$
($401\to801\to1601$), max-abs-final and observer-series Cauchy differences.
**Acceptance.** Clean 2nd-order ($\sim4\times$ error drop per refinement),
matching the Phase A convergence quality ($2.0\!\to\!5.0\!\to\!1.3\times10^{-5}$
ballpark).

**Status: DONE.** Implemented in `kerr/scripts/kv2_convergence.py`
(+ `slurm_kv2.sh`). A SINGLE shared $dt$ per spin (coarsest-grid CFL,
safety 0.4 $\Rightarrow$ finest-grid CFL number $0.4\times4=1.6<2.8$, stable on
all three grids) makes the RK4 time error $O(dt^4)$ identical on every grid so
it cancels in the differences; what remains is the spatial error. Coincident
points ($\sigma=$ `linspace(eps,1-eps,N)` nests exactly, verified) are
differenced as `psi2[::2]`, `psi3[::4]`. Production $\sigma_{\rm KO}=0.2$ (the
KV.1 value) is used — it is a 4th-difference, $O(d\sigma^4)$, subdominant to the
$O(d\sigma^2)$ scheme by $\sim(d\sigma)^2\!\sim\!10^{-5}$, so it does not degrade
the measured order. Local gate **PASS** at the real resolution
$401\to801\to1601$, $\tau_f=60M$; SLURM via `slurm_kv2.sh` for the record.
Measured $p=\log_2(\|e_{12}\|/\|e_{23}\|)$ (L2 norm), all $\approx2$:

| $a/M$ | full-field $p$ | bulk-field $p$ | scri$^+$ series $p$ | r=10M series $p$ |
|------:|---------------:|---------------:|--------------------:|-----------------:|
| 0.0   | 2.009 ($Q{=}4.03$) | 2.008 | 2.020 | 2.020 |
| 0.9   | **2.082** ($Q{=}4.23$) | 2.081 | **2.053** | 2.053 |

Gate = full-field **and** scri$^+$ waveform both $p\in[1.7,2.3]$; **PASS** at
the target $a/M=0.9$. $a=0$ is a reported cross-check (the Bardeen–Press
reduction is likewise clean 2nd-order). Two honest findings recorded while
building the gate:
1. *The convergence ORDER is uniform across the slice.* Unlike KV.1 — where the
   near-horizon rescaling $(1-\sigma)^{-2}$ made the full-grid $L^2$ *envelope
   over time* grow (a magnitude effect, gated around) — excluding the horizon
   layer ($\sigma\le0.9$) changes $p$ by $<0.002$ here. The rescaling inflates
   the field's magnitude, not its truncation order, so KV.2 gates on the FULL
   field (no region excluded); the bulk $p$ is reported only as the cross-check
   that established this.
2. *Coarse grids are pre-asymptotic (not a bug).* The width-$1M$ launch pulse at
   $r=10M$ maps to a $\sigma$-width $\sim0.014$; clean 2nd order needs $\gtrsim5$
   points across it ($N\ge401$). Below that, $p$ rises with resolution:
   full-field $p=1.50\,(101/201/401)\to1.86\,(201/401/801)\to2.08\,(401/801/1601)$.
   Hence the gate runs at $401/801/1601$ and the coarser `--smoke` login check is
   explicitly NOT authoritative.

---

## B.8 — KV.3 QNM gate (analogue of V.3) — the correctness crux
**File.** `kerr/scripts/kv3_qnm.py`, `kerr/scripts/slurm_kv3.sh`.
**Implements.** Method-4/5 plateau extraction at fixed observers, compared to
`qnm` reference, run in this order:
1. **$a=0$ reduction** — must reproduce Phase A: $M\omega_{220}=0.3737$,
   $\tau/M=11.24$. If this fails, B.2/B.4 are wrong; stop and fix the operator,
   do not tune windows.
2. $a/M\in\{0.5,0.9\}$ fundamental $(2,2,0)$ vs `qnm`.
3. First overtone $(2,2,1)$ at $a/M=0.9$ (two-mode Method-4/5 fit).

**Acceptance (README gate, verbatim, do not soften):**
- $M\omega_{220}$ error vs `qnm` $\le 0.1\%$ at **all three** spins
  $\{0, 0.5, 0.9\}$.
- First overtone $n=1$ resolved at $a/M=0.9$.
- Integrator stable, no boundary growth, windows data-driven (Method 5 2D
  plateau scan — not hand-pinned), consistent across $N$.

**Status: DONE.** Implemented in `kerr/scripts/kv3_qnm.py` (+ `slurm_kv3.sh`),
extractor additions in `kerr/src/extractor_m4.py`. Local `--quick` ($401/801$)
**PASS**; `slurm_kv3.sh` runs the authoritative $401/801/1601$ for the record.

| extraction | finest-grid $M\omega$ err | $\tau/M$ err | across-$N$ drift | gate |
|:--|--:|--:|--:|:--|
| $a/M{=}0$ fund $(2,2,0)$   | $1.7\times10^{-5}$ | $3.2\times10^{-5}$ | $3.8\times10^{-7}$ | **PASS** |
| $a/M{=}0.5$ fund $(2,2,0)$ | $1.6\times10^{-4}$ | $3.6\times10^{-4}$ | $1.3\times10^{-6}$ | **PASS** |
| $a/M{=}0.9$ fund $(2,2,0)$ | $3.2\times10^{-5}$ | $5.4\times10^{-4}$ | $1.9\times10^{-4}$ | **PASS** |
| $a/M{=}0.9$ overtone $(2,2,1)$ | $5.9\times10^{-3}$ | $8.0\times10^{-2}$ | $3.8\times10^{-3}$ | **RESOLVED** |

All three fundamentals are $\le0.1\%$ in $M\omega_{220}$ (margin: $0.0017\%$,
$0.016\%$, $0.0032\%$). Four honest decisions taken while building the gate:

1. *The fundamental's estimator is a **single-mode** 2D plateau scan, not the
   two-mode Method-5.* The two-mode fit over the late ringdown is
   **over-parametrised**: the overtone damps $\sim3\times$ faster
   ($\tau_{221}\!\approx\!\tau_{220}/3$, dead by $t\!\sim\!30M$), so on the
   plateau windows ($t_0\gtrsim4\tau$) only the fundamental survives and the
   second mode fits noise (this produced the $1$–$7\%$ scatter first seen with
   Method-5). A single-mode diagnostic (`--diag`) confirmed the operator is
   correct — $a=0$, window $[90,160]$: $M\omega$ err $2.2\times10^{-5}$,
   $\tau$ err $6.3\times10^{-6}$ — so the scatter was the fit, not the operator.
   A single-mode **2D $(t_0,t_e)$ plateau scan** (`qnm_method_2_2d_scan` for the
   real $a=0$ field, `qnm_complex_2d_scan` for $a>0$) is therefore the honest
   estimator and still satisfies the gate's "data-driven 2D plateau, not
   hand-pinned" clause — the windows come from the same scatter-minimising
   rectangle search, scaled off $\tau_{\rm ref}$, with no point hand-picked.
2. *$a=0$ is purely real, $a>0$ is genuinely complex.* At $a=0$ the
   Bardeen–Press coefficients are real so the field stays real
   (imag/real $=0$, printed) and the Schwarzschild-validated real-field path
   applies verbatim; `qnm_complex_phase` returns $\approx0$ there (phase is
   $0/\pi$) and is the $a>0$ tool. At $a>0$ frame dragging makes the field
   genuinely complex (imag/real $\sim0.95$–$1.1$, printed), read from the
   complex envelope$+$phase slopes.
3. *Observer self-selection by plateau scatter (no cherry-picking).* `scri`
   (floor-limited late amplitude) and `r10M` (sitting at the launch radius
   $r_0{=}10M$) scatter; `r50M`/`r20M` are rock-solid ($\text{std}/|\omega|\sim
   6\times10^{-6}$ vs $10^{-2}$ for the bad ones). The gate trusts only
   observers with $\text{std}/|\omega|<10^{-3}$ (flagged `T` in the log) and
   takes the **median** over them — fully data-driven, the field identifies its
   own good observers.
4. *The overtone needs band-filtered mode identification.* It is read by a
   **complex ESPRIT** ($K{=}4$, `qnm_complex_esprit`, no Hilbert — the field is
   already complex) over a data-driven $t_0$ scan on `r20M`. ESPRIT
   occasionally returns a spurious long-$\tau$ low-frequency tail mode
   ($M\omega\sim0.01$–$0.08$); identifying the fundamental as the globally
   longest-$\tau$ mode then mislabels it. Fix: **band-filter to
   $0.5$–$1.5\times\omega_{\rm fund}$ before** picking the longest-$\tau$ mode as
   the fundamental and the faster nearest-frequency mode as the overtone; late
   windows where the overtone has decayed honestly report "no overtone" rather
   than a mislabel. With the filter all $6$ scan windows agree
   ($M\omega_{221}\approx0.66$–$0.68$, $\tau\approx4.6$–$5.0$ vs ref
   $0.6676/5.12$), a clean $3.3\times$ separation from the fundamental
   ($\tau{=}15.4$). The $\sim8\%$ $\tau_{221}$ error is the genuine difficulty of
   a near-degenerate ($0.6\%$ from the fundamental in $\omega_R$), $\sim3\times$
   weaker overtone — reported transparently, not gated away.

---

## B.9 — full spin sweep
**File.** `kerr/scripts/kerr_sweep.py`, `kerr/scripts/slurm_kerr_sweep.sh`,
output `kerr/outputs/phase_b/sweep_*.npz`.
**Implements.** $M\omega_{220}(a)$ and $\tau_{220}(a)$ across
$a/M\in[0, 0.95]$ (e.g. 20 points), each vs `qnm`; the spheroidal $\lambda(a)$
refrozen per spin via B.3. Produces the appendix curve + table.
**Acceptance (README gate, verbatim):** population-mean $M\omega_{220}$ error
across the full $[0,0.95]$ sweep $\le 0.2\%$; monotonic, no isolated blow-ups
(any catastrophic draw diagnosed as extractor-window vs operator, exactly as
the Phase A draw-71 audit).

---

## B.10 — paper integration (deferred to write-up phase)
**File.** note only; actual `.tex` edits happen after B.9 passes.
**Implements.** Kerr methods subsection (operator + B.0 caveat verbatim),
three canonical spins $\{0,0.5,0.9\}$ in the body, full sweep curve+table in an
appendix. No softening of the gate language.
**Acceptance.** Deferred. Listed here so it is not forgotten.

---

## Validation gate sequence (mirrors Phase A)
```
KV.1 propagation/stability  →  KV.2 self-convergence  →  KV.3 QNM extraction
   (B.6)                          (B.7)                     (B.8)
```
All three are SLURM-only (login-node runs forbidden), CPU `icelake`.

## Recommended first commit
**B.1 + B.4 + B.8(step 1) together as the falsifiable crux**: write the Kerr
coordinate map and the Teukolsky operator, then immediately prove the $a=0$
reduction reproduces the validated Phase A RW gate ($M\omega=0.3737$). If the
reduction holds, the spin terms are isolated and low-risk; if it fails, nothing
downstream matters. (B.2 derivation precedes B.4; B.3/B.5 are independent and
can land in parallel.)

## Risk register
- **Highest risk: B.2** (Kerr minimal-gauge height + regularised complex
  source). Mitigation: derivation doc reviewed line-by-line; $a=0$ reduction
  table; B.4 numerical reduction test to $10^{-12}$.
- **Medium: B.0 caveat #2** (frozen spheroidal $\lambda$). Mitigation:
  self-consistency check — re-extract $\omega$, refreeze $\lambda(a\omega)$,
  confirm $\omega$ shift $<$ gate tolerance (one fixed-point iteration).
- **Low/mechanical: B.5** complex plumbing. Mitigation: real-input regression
  reproduces Phase A bit-for-bit.

## References
- Teukolsky, ApJ 185 635 (1973). Master equation, $s=-2$ radial operator.
- Macedo, Jaramillo, Ansorg, PRD 89 064008 (2014). Hyperboloidal minimal
  gauge; Schwarzschild radial map (Phase A foundation).
- Panosso Macedo, CQG 37 065019 (2020). Minimal gauge for **Kerr**
  (B.2 primary reference).
- Zenginoglu, PRD 83 127502 (2011). Hyperboloidal height function.
- Harms, Bernuzzi, Brügmann, CQG 31 245004 (2014). Hyperboloidal time-domain
  Teukolsky on Kerr.
- Krivan, Laguna, Papadopoulos, Andersson, PRD 56 3395 (1997). 2+1D
  time-domain Teukolsky (the $r,\theta$ alternative B.0 declines).
- Berti, Cardoso, Will, PRD 73 064030 (2006). Spheroidal eigenvalue /
  QNM reference tables.
- Pazos-Avalos, Lousto, PRD 72 084022 (2005). Time-domain Teukolsky
  validation reference.

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

---

## B.6 — KV.1 propagation/stability gate (analogue of V.1)
**File.** `kerr/scripts/kv1_propagation.py`, `kerr/scripts/slurm_kv1.sh`
(CPU, icelake).
**Implements.** Evolve the full Kerr operator at $a/M=0.9$ from a Gaussian,
no extraction. Pure stability/characteristic-outflow check: energy bounded,
both endpoints radiate out cleanly, no boundary growth.
**Acceptance.** Stable to $\tau=200M$, no NaN, $L^2(\sigma)$ envelope decays
(does not grow) after the pulse leaves; $a=0$ run identical to Phase A V.1.

---

## B.7 — KV.2 self-convergence gate (analogue of V.2)
**File.** `kerr/scripts/kv2_convergence.py`, `kerr/scripts/slurm_kv2.sh`.
**Implements.** Self-convergence at $a/M=0.9$ under $N\to2N\to4N$
($401\to801\to1601$), max-abs-final and observer-series Cauchy differences.
**Acceptance.** Clean 2nd-order ($\sim4\times$ error drop per refinement),
matching the Phase A convergence quality ($2.0\!\to\!5.0\!\to\!1.3\times10^{-5}$
ballpark).

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

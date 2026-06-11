# Phase C — Kerr neural waveform solver (task breakdown)

The machine-learning phase. Builds a **physics-informed neural solver** (PINN)
for the Kerr ringdown waveform on top of the **validated, audited** Phase B
finite-difference Teukolsky solver (`kerr/src/teukolsky_minimal_gauge.py`,
$s=-2$, $\ell=m=2$, hyperboloidal, $a/M\in[0,0.95]$). Phase B passed its B.9
gate (population-mean $M\omega_{220}$ error $4.7\times10^{-4}$) and was
independently verified non-circular and $2$nd-order convergent. The neural
solver produces the time-domain field $\psi(\sigma,\tau)$ **by minimising the
Teukolsky PDE residual + initial data — no FD field in the loss** — and QNMs are
then extracted from the predicted waveform by the already-validated Phase B
extractor, exactly as the Schwarzschild forward PINN did for the Zerilli
waveform (`scripts/run_pinn.py`, `src/pinn.py`: trained on PDE residual +
analytic IC + Sommerfeld BC, with FD used only to *score* RMSD after training).

Each task is one commit, one test, one acceptance check. Do not merge two.
`kerr/` stays self-contained — reusable parent modules are copied/extended in
`kerr/src/`, never imported from the parent `src/`. Dataset generation
(C.1/C.2, **done**) was CPU (`icelake`); PINN training (C.4) needs **GPU**
(CSD3 `ampere`/A100 — confirm before C.4).

---

## C.0′ — PIVOT (2026-06-11): hybrid FNO → parametric PINN

**This supersedes the original C.0 hybrid decision record** (committed `e24e1b5`,
amended `f678ef4`; preserved verbatim in git history and below under "C.0
(superseded)"). The pivot was made in a direct design review with the user.
Recorded honestly so the change of direction is explicit, not silent.

**Why the hybrid (coarse-FD prior → FNO residual → fine-FD target) was dropped.**
Three structural defects, none cosmetic:
1. **Not solver-free.** It still runs a coarse FD solve at inference, so it is a
   solver *accelerator*, not a solver. The project's goal is a model that
   produces the waveform from the equation + conditions alone.
2. **Capped by its teacher.** A supervised FNO regresses against the fine-FD
   field; its best case is "almost as good as the solver we already have." It
   can never *be* the solver.
3. **No transfer to the second goal.** The user's stated deliverable is also a
   reusable method for **higher-dimensional PDEs**. Learning to undo a specific
   1+1D stencil's truncation error teaches nothing that scales in dimension.
   In 1+1D the fine solve is cheap anyway, so the hybrid's only honest headline
   (speed) is weak here.

**What replaces it — a parametric, physics-informed PINN.** A single network
$\psi_\theta(\sigma,\tau;\,a/M,r_0,w)$ trained on the **Teukolsky PDE residual +
initial data**, with **no FD field in the loss**. Inference is **solver-free**:
feed $(a/M,r_0,w)$, read $\psi(\sigma,\tau)$, extract QNMs at scri. The FD corpus
(C.1/C.2) is **demoted from training data to an independent validation oracle**;
`qnm`/Leaver is the second oracle. The build effort on the corpus is not wasted —
it changes job (teacher → grader), not value.

**Why PINN, not PINO/FNO (decision, with the honest caveat).**
- Our input family is **three scalars** $(a/M,r_0,w)$ — the amplitude axis is
  dropped (linear PDE, C.0 addendum fact iv). FNO/PINO exist to map arbitrary
  input *functions* (IC fields, coefficient fields) to solutions; that
  function-space machinery is **wasted** on a 3-scalar family.
- Vanilla FNO/PINO differentiate via **FFT**, which assumes a **periodic,
  uniform** grid. Our domain is **bounded, non-periodic, hyperboloidal**
  $\sigma\in[0,1]$ with physically distinct endpoints (horizon $\sigma=1$,
  scri $\sigma=0$); spectral differentiation there produces **Gibbs oscillations
  at exactly the two boundaries the physics cares about**. A PINN differentiates
  by **autodiff** — pointwise, mesh-free, geometry-agnostic — so the endpoints
  are handled cleanly and collocation points may sit anywhere.
- A PINN is the only NN approach that scales past $3{+}1$D (no $N^d$ grid blow-up),
  which is the *higher-dimensional PDE package* goal.
- **Caveat (kept on the table):** PINO remains the right tool **iff** we later
  want to generalise over arbitrary initial-data *fields* (true function inputs).
  Documented as a future option; not built now.

**The model = a parametric conditioning of the proven SW forward PINN.** The SW
forward PINN (`src/pinn.py`) already trains with **no FD target** — verified:
`_make_ic_bcs` supplies an analytic Gaussian IC + Sommerfeld BCs + the
Regge–Wheeler/Zerilli PDE residual; `scripts/run_pinn.py` calls `solve_fd` only
to compute RMSD/RL2 *after* `train_pinn` returns. The Kerr solver adds the PDE
parameters as **extra input neurons** and enforces the residual over a
*distribution* of $(a/M,r_0,w)$, so one trained network gives the waveform across
the spin family (including held-out spins) — the operator behaviour, achieved
over a 3-scalar family without FNO.

### The exact PDE the PINN enforces (derived + verified, 2026-06-11)

The FD solver evolves a first-order **characteristic** system (state
$(\Psi,U,W)$ with $U=\Pi+\mu_+\Phi$, $W=\Pi+\mu_-\Phi$, $\Pi=\partial_\tau\Psi$,
$\Phi=\partial_\sigma\Psi$, $\mu_+=\lambda_{\rm in}$, $\mu_-=\lambda_{\rm out}$;
`teukolsky_minimal_gauge.rhs_teuk`). Eliminating $(U,W)$ (the $\mu'\lambda$ terms
cancel — verified from **both** the $U$ and $W$ equations) collapses it to a
single **second-order scalar** PDE in $(\sigma,\tau)$ that the PINN enforces:

$$
\mathcal{R}[\Psi]\;\equiv\;\partial_\tau^2\Psi
+(\lambda_{\rm in}+\lambda_{\rm out})\,\partial_\tau\partial_\sigma\Psi
+\lambda_{\rm in}\lambda_{\rm out}\,\partial_\sigma^2\Psi
-c_\Pi\,\partial_\tau\Psi-c_\Phi\,\partial_\sigma\Psi-c_\Psi\,\Psi\;=\;0 .
$$

- $\lambda_{\rm in}(\sigma),\lambda_{\rm out}(\sigma)$ are **real** closed-form
  rational functions of $\sigma$ (fixed by $a/M\!\to\!r_\pm$); $c_\Pi,c_\Phi,c_\Psi$
  are **complex** closed forms carrying frame dragging ($\propto i\beta$,
  $\beta=ma/(r_+-r_-)$). $c_\Psi$ carries the spheroidal constant
  $\lambda(a)=\texttt{teukolsky\_lambda}(a,\ell,m,-2,\omega_{\rm ref}(a))$, frozen
  at the reference QNM. All are **evaluated, not differentiated** → ported
  verbatim from `build_teukolsky_op` into a torch elementwise module of
  $(\sigma,a/M)$.
- $\Psi$ is **complex** → two output channels $p=\mathrm{Re}\,\Psi$,
  $q=\mathrm{Im}\,\Psi$ → two real residuals (split $\mathcal{R}$).
- **Initial data (purely real, time-symmetric, all spins):**
  $p(\sigma,0)=A\,e^{-(r-r_0)^2/2w^2}$ with $r=r_+/\sigma$;
  $q(\sigma,0)=0$; $\partial_\tau p(\sigma,0)=\partial_\tau q(\sigma,0)=0$. The
  imaginary part is *generated* dynamically by the complex coefficients (frame
  dragging), not imposed.
- **No boundary conditions.** The hyperboloidal characteristic speeds are
  **outflow at both ends** (horizon and scri), so PDE + IC fully determine the
  solution. This removes the Sommerfeld BC terms the SW PINN needed — a concrete
  simplification, not a hand-wave.
- **No KO dissipation.** $\sigma_{\rm KO}=0.2$ is an FD-only numerical
  regulariser; the PINN solves the *clean* PDE. (The tiny KO smoothing is part of
  why PINN-vs-FD agreement is gated at the QNM/field-RL2 level, not bit-level.)

### Acceptance gate — LOCKED (2026-06-11)

The original README field-RMSD-vs-coarse-up gate is **void** (no coarse-up in a
PINN). Replacement, anchored to the two oracles we trust, now written into
`kerr/README.md`:

> On **held-out spins** (not seen in training), the PINN's scri waveform matches
> the FD oracle to **field rel-$L^2 \le 5\%$**, **and** the QNM extracted from the
> PINN waveform matches **Leaver** ($M\omega$ to $\le 1\%$, $\tau$ to $\le 5\%$),
> i.e. within a small multiple of Phase B's own FD-vs-Leaver error. **Both** must
> hold.

Rationale: measures the thing that matters (solver-free generalisation +
waveform fidelity), and anchors truth to Leaver/FD rather than to a baseline the
model is unfairly strong against. Locked after the user chose to proceed rather
than keep the gate open. **Calibration rule:** these thresholds may be *tightened*
at C.3 once single-config numbers exist (e.g. if FD-vs-Leaver headroom proves
larger), but **not loosened** — consistent with the README's "do not soften".

**Abort policy (carried over).** If the PINN fails this gate after a genuine
training+tuning effort: ship Phase A+B (classical solver) only; the neural solver
becomes future work. No open-ended architecture grind.

### Prior art, positioning & framework decision (literature review 2026-06-11)

A focused review of the top-journal PINN/QNM and PINN-training literature was done
**before writing any C.3 code** to confirm the approach. Findings:

**1. Where the field is — and the gap we fill.** Essentially *all* prior PINN work
on black-hole QNMs solves the **frequency-domain eigenvalue** problem (the radial,
or 2-D radial+angular, Teukolsky ODE) and reads the complex frequency directly:
- Luna, Calderón Bustillo, Seoane, Torres-Forné & Font, *PRD* **107**, 064025
  (2023), "Solving the Teukolsky equation with PINNs" — Kerr QNMs, PyTorch, ≲1%.
  The closest prior work; **frequency domain**.
- Cornell, Ncube & Harmsen, *PRD* **106**, 124047 (2022) — eigenvalue solver,
  <0.01%, but explicitly *"not recommended when considering overall performance"*
  (slower than a continued fraction).
- Cornell, Herbst, Ncube & Noshad (2024, arXiv:2402.11343) — RW + Teukolsky,
  supervised vs unsupervised; **errors grow with spin $a$ and overtone $n$**.
- Pombo & Pizzuti (2025, arXiv:2511.15796) "Teukolsky by Design / SpectralPINN" —
  Chebyshev activations, separated + joint 2-D, hard Leaver normalisation, ~0.001%.
- Gu et al. (2026, arXiv:2604.23625) "DeepOPiraKAN" — physics-informed *operator*
  learning of the parameter-dependent Kerr spectrum: a **single** net spans the
  full spin range, overtones to $n=7$, $\mathcal{O}(10^{-6})$.
- Modified-gravity / higher-D variants (Luna et al. *PRD* **109**, 124064 (2024);
  Ncube 2021; dRGT 2023; Lobos 2024; Ahmad 2026) — all frequency-domain.

  **Our approach is different and, as far as this review found, novel: we solve the
  TIME-DOMAIN ringdown waveform on a hyperboloidal slice and extract the QNM from
  the waveform** (as the SW pipeline did). No prior work combines hyperboloidal
  slicing with a PINN (the `hyperboloidal`+`neural network` search returns only
  unrelated "hyperbolic-geometry" networks). This is a deliberate, harder choice:
  the frequency-domain shortcut is easier and more accurate, **but it does not
  produce a waveform and does not generalise** to the higher-dimensional,
  non-separable evolution solver that is the project's second goal. The
  time-domain route uniquely serves *both* stated goals — at the cost of landing
  in the known-hard PINN regime (oscillatory, long-time, transport).

**2. Framework: DeepXDE first (user-confirmed), pure-PyTorch deferred.** The
parent SW PINN (`src/pinn.py`) is a DeepXDE harness that *already* implements the
exact literature-standard mitigations for this hard regime; reusing it (swap in
the Kerr complex residual + IC, drop the BCs) is the lowest-risk way to get an
honest go/no-go. A clean-room PyTorch core is deferred to *if/when* DeepXDE works
and we push to higher-D (where DeepXDE's low-D geometry machinery becomes the
limiter). Proving the method in DeepXDE first does not block the package — we port
a *working* method later, not a guess.

**3. The literature-mandated ingredients (all but one already in `src/pinn.py`):**
- **Temporal causality weighting** — Wang, Sankaran & Perdikaris (2022,
  arXiv:2203.07404, "Respecting causality is all you need"). *Have it* (`CausalWeighter`).
- **Curriculum / expanding time-windows (≈ sequence-to-sequence)** — Krishnapriyan,
  Gholami, Zhe, Kirby & Mahoney, *NeurIPS* 2021 ("Characterizing possible failure
  modes in PINNs"). *Have it* (curriculum windows).
- **Gradient/loss balancing** — Wang, Teng & Perdikaris (2021, gradient
  pathologies). *Have it* (`GradientBalancing`).
- **Residual-adaptive resampling (RAD/R3)** — Wu et al. 2023; Daw et al. 2023.
  *Have it* (`ResidualAdaptiveResampler`).
- **Fourier-feature input embedding — REQUIRED, must ADD.** This is the single
  highest-leverage missing piece for the oscillatory ringdown: Tancik et al. 2020;
  and, directly on-point, Ding et al. (2024, arXiv:2409.03536) use Fourier-feature
  PINNs for **time-domain wave** simulation precisely to beat spectral bias on
  high-frequency propagation. The SW `pinn.py` did **not** have RFF; C.3 adds it.
- **FP64 throughout — MANDATE.** Xu et al. (2025, arXiv:2505.10949, "FP64 is All
  You Need") show FP32 + L-BFGS *prematurely* satisfies its convergence test and
  freezes the net in a spurious "failure mode"; FP64 removes it. Our config is
  already `dtype: float64`; C.3 keeps it and the gate treats FP32 as out of spec.

**4. Documented fallback architecture (bounds the one-architecture abort policy).**
If a plain MLP + Fourier-features + causal weighting stalls at C.3, the *single*
pre-registered alternative is a spectral/operator backbone shown to handle wave
equations: **NeuSA** (Neuro-Spectral Architectures, *NeurIPS* 2025,
arXiv:2509.04966 — spectral basis + Neural-ODE, explicitly for linear/nonlinear
wave equations, overcomes spectral bias and enforces causality) or **PINNsFormer**
(2023). Naming it here keeps the abort policy honest: at most one fallback, not an
open-ended grind.

**5. Value-proposition correction (honesty).** Per Cornell 2022, a **single-config**
PINN is *slower* than one FD/Leaver solve — so **C.3 is an accuracy go/no-go, not a
speed claim**. The genuine speed win is **amortised** and only claimed at C.4/C.5:
train once, evaluate any spin cheaply (viability shown by Gu 2026's single-net
spin sweep and the P²INN parametric scheme, Cho et al. 2024, arXiv:2408.09446).
C.5 must benchmark PINN inference against *many* FD solves, never one.

---

## C.0 (SUPERSEDED by C.0′ — historical hybrid record, no model code)

> **Superseded 2026-06-11.** The hybrid decisions below are retained for honesty
> and provenance (committed `e24e1b5`/`f678ef4`). The waveform-accelerator framing
> and the complex-field observation (Decision 4) carry over to the PINN; the
> coarse-FD + FNO-residual *architecture* and its coarse-up gate do **not**. Read
> C.0′ above for the live plan.

**File.** `kerr/notes/phase_c_plan.md` (this file).

**What Phase C is.** A learned surrogate that produces the Kerr ringdown
waveform across spin and initial data *faster than the fine FD solver*, from
which QNMs are then extracted by the **already-validated Phase B extractor**
(`kerr/src/extractor_m4.py`). It is **not** a new QNM oracle — the `qnm`
package and the FD solver remain ground truth. It is the parametric,
amortised waveform model that the FD solver is not.

**Lineage.** This is "Phase 6" of the signed-off hybrid proposal
(`proposals/hybrid_fd_fno_proposal.md`, A1–A7, 2026-05-29) and "Phase C" of
`kerr/README.md`. The two diverge in one deliberate way, and C.0 must resolve
it (Decision 2): the proposal locked a single architecture (coarse-FD + FNO
residual); the README widened Phase C to *"not a straight port … allowed to be
a fundamentally different architecture … inspired by our Schwarzschild hybrid,
adapted and improved for Kerr."* We follow the **wider README framing** — a
candidate slate, not a single locked model — while keeping the SW-proven hybrid
as the primary candidate.

**Acceptance gate (README, verbatim, do NOT soften).**
> Kerr surrogate beats coarse-up baseline by $\ge 10\times$ in field RMSD on the
> canonical Kerr evaluation set, **OR** suppresses worst-case $|\Delta\tau/\tau|$
> on the population tail by at least the same factor the Schwarzschild hybrid did
> ($30\%\to5\%$). One of the two must hold.

**Abort policy (README, verbatim).** If Phase C fails its gate after **one
architecture** has been trained and evaluated: ship Phase A+B only. No
second-architecture rescue within this paper; other architectures become future
work. (This bounds the "not limited to one model" directive: we may *prototype*
a slate, but the paper commits to the first that clears the gate, and we do not
grind indefinitely.)

---

### Decisions

**1. The surrogate's job = waveform accelerator, not a frequency regressor.**
The model predicts the *field* $\psi(\sigma,\tau)$ (or its fine-grid correction);
QNMs come out by running the Phase B plateau/ESPRIT extractor on the predicted
field, exactly as for the FD waveform. Rejected alternative: a direct
$(a/M,\text{ID})\!\to\!(M\omega,\tau)$ regressor. Reasons: (i) it would *replace*
the verified extractor with an unvalidated black box and break the "extract from
the waveform, like SW" requirement; (ii) it cannot produce the time-domain
waveform the paper also wants; (iii) the gate is written in field-RMSD and
$\tau$-from-ringdown terms, which presume a predicted field. The accelerator
framing keeps the worst-case-bounded invariant below.

**2. Architecture slate, with a primary.** The gate is architecture-agnostic
(field RMSD / $\tau$-tail), so we hold a small slate and let Phase B's hard
regimes + the SW precedent choose:
  - **Primary — coarse-FD + FNO residual** (the SW-proven design,
    `kerr/src/hybrid_fno.py` already copied). Predicts
    $\delta=\psi_{\rm fine}-\mathrm{up}(\psi_{\rm coarse})$ on the fine grid;
    $\psi_{\rm hybrid}=\mathrm{up}(\psi_{\rm coarse})+\delta$. **Key invariant:
    if the net outputs zero, the fallback is the upsampled coarse solver, so the
    worst-case error is bounded by coarse-FD error** — a guarantee the pure FNO
    lacks. This is why it is primary.
  - **Reference baseline — pure FNO** $(a/M,\text{ID})\!\to\!\psi_{\rm fine}$
    (`kerr/src/fno_model.py` copied). Not a candidate to ship; it is the thing
    the hybrid must *clearly beat* (proposal A-§6.4 story risk), and it bounds
    "is the residual actually easier than the full field?"
  - **Alternates (prototyped only if the primary stalls):** a complex-valued /
    two-channel FNO variant (see Decision 4), or a DeepONet-style
    branch/trunk operator. Named, not committed.

**3. Spin range $[0,0.95]$, not the proposal's $[0,0.99]$.** Ground truth only
exists where Phase B validated it. The near-extremal tail was *already* the
hardest FD regime at $a/M=0.95$ (needed the `envelope_tail_cap` fix), and we
have no audited solver above $0.95$. Training/evaluating a surrogate against
un-validated FD data would launder solver error into the ML result. $[0,0.99]$
is deferred until/unless Phase B is extended; recorded here so the deviation is
explicit.

**4. The genuine "adapted and improved for Kerr": the field is COMPLEX.** The SW
hybrid target $\Phi$ was real (1 channel). The Kerr Teukolsky $\psi$ is complex
for $a>0$ (imag/real $\sim1$; frame-dragging $\propto am$), and *weakly* complex
at low spin (imag/real $0.2$–$0.6$ — the regime Phase B found hardest). So the
Kerr surrogate must carry **two real target channels $(\mathrm{Re}\,\delta,
\mathrm{Im}\,\delta)$** (or a complex-valued FNO). This is the concrete Kerr
adaptation, not cosmetic: the model must learn the Re/Im phase relationship that
encodes the oscillation. Conditioning adds **spin $a/M$ as an input channel**
(the SW model had no spin); the angular separation constant $\lambda(a,\omega)$
is *not* fed to the net (Phase B's audit showed it is a weak, sub-leading input).

**5. Data design — the missing artefact, built first.** Phase B saved *scalars
only*; no field corpus exists. C.1/C.2 build it:
  - **Sweep** $(a/M, r_0, w)$ on a Sobol quasi-random grid. **Amplitude $A$ is
    dropped: the Teukolsky equation is linear, so $\psi(2A)=2\psi(A)$ holds to
    $0.0$ relative error (C.1 spot-check) — $A$ is a redundant axis.** Spin
    $a/M\in[0,0.95]$; initial-pulse params centred on the Phase B-validated pulse
    ($r_0=10M$, $w=1M$) and widened to **$r_0\in[8,11]M$, $w\in[1.0,1.5]M$** —
    the box fixed by the spot-check so the **coarsest** grid ($N=201$) keeps
    $\ge5$ points across $[r_0-w,r_0+w]$ on *every* draw (a narrower $w=0.75$
    collapses to 3 points at high $r_0$/spin and under-resolves the pulse).
  - **Per sample, run the Phase B operator three times:** fine ($N=801$, the B.9
    production grid) and **both** coarse grids — $N=401$ ($k=2$) and $N=201$
    ($k=4$) — so the $k$-ablation uses *identical* parameter draws and the fine
    solve is done once. The $\sigma$-grids **nest exactly** (spot-check:
    $801[::2]\equiv401$, $801[::4]\equiv201$) and share $\sigma_{\rm KO}=0.2$.
    Store the fine + both coarse **complex** fields + the parameter vector +
    per-sample reference $(M\omega,\tau)$ from `qnm`. Upsampling coarse$\to$fine
    deferred to the data-pipe (matches parent `src/hybrid_data_pipe.py`).
  - **Counts:** 1024 train / 128 val / 128 test (powers of two for Sobol balance;
    $\approx$ proposal §9 Phase 1's 1000/100/100). Generation is cheap — $111.7$ s
    per sample for all three grids at the hardest spin ($a/M=0.9$) $\Rightarrow$
    $\sim$40 core-hours for the 1280-sample corpus, $\sim$1.2 h on 32 `icelake`
    cores.
  - **Format:** `.npz` (matches all `kerr/outputs/phase_b/*`), gitignored.
  - **Baseline + eval (C.3):** mirror `scripts/eval_hybrid_sw.py` —
    compare **hybrid vs coarse-up vs fine-truth**, report field RMSD/L² *and*
    QNM error via the Phase B extractor. The **canonical Kerr evaluation set**
    (gate term) = the held-out test split; the **population tail** (gate term) =
    its worst-case $|\Delta\tau/\tau|$, reported because Phase B showed $\tau$ and
    the weakly-complex low-spin band are the hard cases.

**C.0 addendum — stability spot-check (empirical, 2026-06-10).** Five facts
measured on the Phase B operator before locking the C.1 ranges: (i) the grids
nest exactly, $801[::2]\equiv401$ and $801[::4]\equiv201$; (ii) temporal
resolution is huge on every grid ($\ge1600$ pts/period at $a/M=0.95$), so the
coarse error is **spatial**, not temporal aliasing; (iii) the box $r_0\in[8,11]$,
$w\in[1.0,1.5]$ keeps $\ge5$ points across the pulse on the coarsest grid
($N=201$, worst corner exactly 5); (iv) the field is bit-exactly linear in $A$
($f(2A)-2f(A)=0$), so $A$ is dropped as a sweep axis; (v) cost is $111.7$ s per
sample for all three grids $\Rightarrow\sim$1.2 h on 32 cores for 1280 samples.

---

### Honest risks (PINN, recorded now, not after they bite)

1. **PINNs are hard for oscillatory, long-time, hyperbolic problems.** This is
   the headline risk — the SW forward PINN was mediocre at late-time ringdown for
   exactly this reason. *Mitigations (all proven in the literature and partly in
   `src/pinn.py`):* (a) the **hyperboloidal** formulation removes the BCs and puts
   scri at a grid endpoint; (b) **Fourier-feature input embedding** to defeat
   spectral bias on the QNM oscillation; (c) **causal / time-windowed training**
   (`CausalWeighting`, Wang 2022, already in `src/pinn.py`) so the net respects
   time-ordering; (d) Adam→L-BFGS. If single-config (C.3) cannot hit the gate
   with these, the whole approach is in doubt — that is *why C.3 is the cheap
   CPU go/no-go before any GPU spend.*
2. **Complex field, two coupled channels.** $\Psi=p+iq$ with a phase relation set
   by the complex coefficients; the weakly-complex low-spin band ($a/M\in[0.05,
   0.2]$, Phase B's hardest, $\le0.3\%$) is the likely failure point. *Mitigation:*
   report per-spin, not just population-mean; the imaginary part is generated
   dynamically, so check $q$ does not collapse to $0$ at low spin.
3. **Parametric stiffness.** Enforcing the residual over a *distribution* of
   $(a/M,r_0,w)$ (C.4) is harder than a single config (C.3). *Mitigation:* prove
   C.3 first; in C.4 use gradient balancing (`GradientBalancing`, Wang 2021,
   already in `src/pinn.py`) across the param batch; curriculum from low to high
   spin if needed.
4. **Frozen separation constant.** $c_\Psi$ depends on $\lambda(a)=
   \texttt{teukolsky\_lambda}(a,\ldots,\omega_{\rm ref}(a))$, frozen at the
   reference QNM. This is the same single-mode convention Phase B's audit found to
   be a *weak, sub-leading* input (V3 non-circularity), so freezing it per-spin is
   safe; recorded so it is not silently assumed.
5. **Near-extremal tail.** $a/M=0.95$ ringdown gives way to tail/near-degenerate
   overtone by $\sim6.8\tau$; the PINN inherits this. The extractor's
   `envelope_tail_cap` already handles it on predicted fields.
6. **GPU dependency.** C.4 needs an A100 allocation — **to be confirmed before
   C.4**, not assumed. C.3 is CPU and gates the GPU spend.

---

## C.1 — Kerr field corpus generator  ✓ DONE (now the validation oracle)
**File.** `kerr/scripts/build_kerr_dataset.py`, `kerr/src/kerr_dataset.py`
(committed `6816a45`). **Role under C.0′:** the FD fields are the **independent
oracle** the PINN is validated against, **not** training data.
**Implemented.** Sobol sweep over $(a/M,r_0,w)$ (amplitude dropped — linear PDE);
per sample evolve the Phase B operator at fine $N=801$ + both coarse $N=401/201$,
recording the full **complex** field $\psi(\sigma,\tau)$ on one shared canonical
$\tau$-axis ($T_{\rm store}=220M$, $\Delta\tau=0.25M$, $N_\tau=881$). Re-uses the
B.9 primitives verbatim. (The coarse grids are now only needed for an *optional*
FD-cost reference in C.5, not for a hybrid prior; kept since already built.)
**Acceptance (met).** `--selftest` PASS at $a/M\in\{0,0.5,0.9\}$: fine reproduces
`qnm` + the B.9 gate to $\le1.7\times10^{-5}$; coarse finite and extracts QNM
within $5\times10^{-3}$.

## C.2 — authoritative corpus (SLURM, CPU)  ✓ DONE (oracle built + verified)
**File.** `kerr/scripts/slurm_kerr_dataset.sh` (committed `8e12772`); output
`kerr/outputs/phase_c/dataset_{train,val,test}.npz` (gitignored).
**Implemented + verified.** Job `30357375` COMPLETED (32 `icelake` cores, 1 h
07 m). `--verify-corpus` **PASS**: train 1024 / val 128 / test 128, all fields
finite, QNMs finite, shapes consistent, **splits disjoint**; spin span
$[1.4\times10^{-4},\,0.950]$; manifest at
`kerr/outputs/phase_c/corpus_manifest.json`.
**Acceptance (met).** Strict corpus verification PASS + manifest emitted.
**C.0′ note:** these splits become the PINN's **held-out validation/test** sets;
the PINN never trains on the fields, so "train/val/test" here just labels which
spins/IDs are used to *grade* generalisation.

## C.3 — Teukolsky residual module + single-config PINN proof (CPU)
**File.** `kerr/src/teukolsky_residual.py` (torch coefficient/residual module),
`kerr/src/kerr_pinn.py` (DeepXDE model builder: net + IC + residual wiring),
`kerr/scripts/train_kerr_pinn.py`. **Framework: DeepXDE** (reuse the SW harness in
`src/pinn.py` — copy to `kerr/src/pinn.py` — for the proven causal/grad-balance/
resampler/curriculum machinery; a clean-room PyTorch core is deferred per C.0′ §
"framework decision").
**Implements.** (a) A torch module that evaluates the closed-form coefficients
$\lambda_{\rm in},\lambda_{\rm out},c_\Pi,c_\Phi,c_\Psi$ at arbitrary
$(\sigma,a/M)$ — **ported verbatim** from `build_teukolsky_op` (validated forms),
unit-tested to match it to $\le10^{-10}$ on the corpus grid. (b) The complex
second-order residual $\mathcal{R}[\Psi]$ via autodiff (two real channels:
$p,q$), plus the time-symmetric Gaussian IC loss; **no BC term** (hyperboloidal
outflow), **no FD field in the loss**. (c) A **single-config** PINN (one fixed
$(a/M,r_0,w)$) with **Fourier-feature input embedding (REQUIRED — Tancik 2020 /
Ding 2024 for time-domain waves; the SW harness lacks it, add it)** + causal
weighting (Wang 2022) + curriculum windows (Krishnapriyan 2021), Adam→L-BFGS, and
**FP64 throughout (mandate — Xu 2025; FP32 is out of spec)**. Prove it at $a/M=0$
(real field, sanity) and $a/M=0.7$ (genuinely complex).
**Acceptance (accuracy go/no-go, NOT speed — a single config is expected slower
than one FD solve, Cornell 2022).** For each of the two configs, the PINN's scri
waveform matches the corresponding **FD corpus sample** to field rel-$L^2 \le 5\%$,
**and** the QNM extracted from the PINN waveform (Phase B extractor) matches
`qnm`/Leaver ($M\omega \le 1\%$). Coefficient module matches `build_teukolsky_op`
to $\le10^{-10}$. Written to `kerr/outputs/phase_c/pinn_single_{a0,a07}.json`. One
commit. **This is the CPU go/no-go before any GPU spend** — if a single config
cannot be solved by a PINN, stop and reconsider (then the pre-registered fallback
in C.0′ §4, NeuSA/PINNsFormer, before abort).

## C.4 — parametric PINN across spin (GPU)
**File.** `kerr/scripts/train_kerr_pinn.py` (extended), config
`kerr/configs/kerr_pinn.yaml`.
**Implements.** Add $(a/M,r_0,w)$ as **input channels**; enforce the residual +
IC over a *distribution* of parameters sampled jointly with the collocation
points (the spin-dependence of the coefficients is what teaches generalisation).
Gradient balancing (Wang 2021) across the parameter batch; curriculum low→high
spin if needed. `--smoke` proves the loop on CPU first.
**Acceptance.** Training converges; on a **held-out validation spin** (not in the
training parameter draw) the waveform rel-$L^2$ and extracted $M\omega$ are within
the C.3 single-config tolerances. If it converges only by memorising training
spins (held-out fails), STOP and reconsider before the full eval. One commit.

## C.5 — evaluation against the gate + ablations
**File.** `kerr/scripts/eval_kerr_pinn.py`; output `kerr/outputs/phase_c/eval/`.
**Implements.** PINN vs FD-truth vs Leaver on the **held-out test split**: field
rel-$L^2$, $M\omega$/$\tau$ errors, **worst-case $|\Delta\tau/\tau|$ on the
population tail**; per-spin breakdown (low-spin weakly-complex band called out).
**Speedup**: solver-free PINN inference wall-time vs a fine-FD solve. Optional:
sensitivity to the frozen $\lambda(a)$.
**Acceptance (LOCKED gate, see C.0′; may tighten not loosen at C.3).** On
held-out spins, field rel-$L^2 \le 5\%$ **and** $M\omega \le 1\%$ / $\tau \le 5\%$
vs Leaver — **both** must hold. If it fails after a genuine training+tuning
effort: invoke the abort policy (ship A+B). One commit.

## C.6 — paper integration (deferred to write-up)
**File.** note only; `.tex` after C.5 passes. New Results subsection: a
**solver-free, spin-conditioned PINN** that produces Kerr $(2,2)$ ringdown across
spin with no FD target, QNMs extracted from the predicted waveform; framed as the
first rung of a **higher-dimensional PDE solver** (1+1D → 2+1D Kerr Teukolsky
without $\ell$-separation is the natural next rung). Higher modes / PINO over
arbitrary IC-fields named as future work. Deferred — listed so it is not
forgotten.

---

## Phase C dependency chain (PINN)
```
C.0′ pivot  →  C.1 corpus gen ✓  →  C.2 corpus(SLURM) ✓  ─┐  (oracle, done)
                                                          │
   C.3 residual module + single-config PINN proof (CPU) ←─┘  GO/NO-GO
                                   │
                          C.4 parametric PINN (GPU)
                                   │
                          C.5 gate + ablations + speedup  →  C.6 paper
```
The FD corpus (C.1–C.2) exists **before** any model — but as the *oracle*, not
training data. C.3 is the cheap CPU proof that gates the GPU spend. No paper prose
before C.5 passes.

## Scope guard (unchanged from Phase B)
Commit only `kerr/` files (parent `../` tree is off-limits). GPU work (C.4) needs
an A100 allocation — confirm before C.4. The PINN is the committed model; the
abort policy (ship A+B) forbids an open-ended architecture search. PINO over
arbitrary IC-fields is documented future work, not part of this paper.

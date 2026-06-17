# SW Hybrid Fix — Plan: Coarse-Anchored, Label-Free Physics Refinement

**Status:** proposed (2026-06-13). Single-config SW first; parametric operator later.
**One-line goal:** demonstrate that a *cheap* coarse FD prior + a *label-free*
physics-informed refiner produces (a) an accurate ringdown **waveform** and
(b) accurate **QNMs (ω, τ)** extracted from it — without ever fitting fine FD data.

---

## 1. Why we are doing this (the two problems it solves at once)

1. **SW hybrid is cosmetic for the QNM.** Eval summaries
   (`outputs/hybrid/*/eval/summary.json`): the FNO correction cuts the *field*
   error 6–18× (rl2 0.4–1.2% → 0.06%) but leaves the **QNM unchanged** (ω still
   ~0.5%, sometimes worse). The ML earns its keep on the waveform, not the science.
   Root cause: plain global field MSE against fine FD (i) is dominated by the
   high-amplitude prompt, not the low-amplitude ringdown, and (ii) cannot beat the
   fine-FD's own ~0.5% extraction error (you can't out-fit your label).

2. **Supervised-on-fine-FD does not transfer to higher-D.** The endgame is
   higher-dimensional / Kerr evolution where **fine FD labels cannot be generated
   at all**. A method that regresses against fine FD is therefore self-defeating in
   the target regime. The only training signal that survives is the **PDE residual**.

**The fix (both at once):** replace the fine-FD data loss with a **PDE residual**
(scales to any dimension) **anchored to a cheap coarse prior** (kills the null
space / phase ambiguity that otherwise collapses a from-scratch PINN — the C.3
failure). This is **physics-informed super-resolution for PDEs**: coarse field =
low-res image, PDE residual = the constraint that recovers the high-frequency
detail correctly instead of from paired data.

**Terminology discipline:** it is *label-free*, **not** *unsupervised*. The PDE
residual and the coarse anchor are both supervision; we only drop the *fine-FD
label*. The anchor is intended to be load-bearing — the *hypothesis* is that it
cures the collapse — but this is **UNTESTED as of 2026-06-15**, see the honesty
note below.

> **Anchor track-record (honest, 2026-06-15).** "Anchor" is overloaded in this
> repo; only one variant is the coarse-prior anchor of this plan, and it is NOT
> yet demonstrated to work:
> - **Coarse-prior anchor** `α‖u−u_prior‖²` (THIS plan): the only proper run
>   (`outputs/hybrid/fno_sw_physics_k4`) has a trained model but **no eval**; the
>   two diagnostic variants (`..._diag`, `..._diag_norad`) logged the **anchor
>   term = 0** (residual-only) and sat at field rel-L2 ≈ 5% = the bare k4 prior,
>   i.e. did **not** beat it. So "cures the collapse" is a hypothesis, not a result.
> - **The currently-running SW jobs (`30589279`/`30589280`) do NOT test this
>   anchor.** They are Richardson super-resolution, weights
>   `(data,physics,anchor)=(1.0,0.0,0.0)` — anchor OFF. They validate the
>   *Richardson* label-free target, a different mechanism (see §below).
> - **Continuity anchors** (curriculum-PINN window stitching, weight 10): the
>   paper conclusion records these performed **WORSE** than the hard-partition
>   stitch. A *different* kind of anchor, but the reason to be skeptical.
> - **RAD anchor-retention** / **B.9 "anchor spins"**: unrelated uses of the word
>   (collocation-point retention; reference spin values) — not loss terms.
>
> Net: the coarse-prior anchor is **plausible but unproven**; the Stage-1 run that
> would actually test it has not been executed (the running jobs test Richardson).

---

## 2. Success definition (LOCKED — both metrics, vs fine FD, beating the bare prior)

A run "works" only if the refiner beats the **bare coarse prior** (control C0) on
**both** axes, on the same single config, scored against the fine FD oracle + Leaver.

**Primary = waveform** (this is what coarsening actually degrades; see Stage 0):

| axis | metric | target |
|---|---|---|
| **waveform (primary)** | rel-L2 of refined field vs fine FD | **< bare-prior rel-L2**, aim < 1% |
| QNM (validation) | ω, τ via the **M1–M5 suite** (`src/qnm.py`), judged at **x_q=2** | stays **≤ bare-prior error**, does not regress |

QNM extraction uses the **existing M1–M5 methods** exactly as `scripts/eval_hybrid_sw.py`
does (M1 FFT+log-linear, M2 NLS, M3 ESPRIT, M4 two-mode window-scan, M5 2-D scan).
M4/M5 are the stability-scanned methods — they already handle the late-time window
robustly, so there is **no manual extraction-window tuning**. Judge τ at x_q=2 (clean);
x_q=10 τ is a known M2 single-window artifact that M4/M5 avoid.

Controls reported alongside every run:
- **C0** = bare coarse prior (the thing we rescue).
- **C1** = current supervised hybrid (`fno_sw_k2_h64`) — the incumbent.
- **Reference** = fine FD field + Leaver (ω=0.3737, τ=11.241 for ℓ=2).

**Cost is a first-class number, not an afterthought.** Always report:
- prior **build** cost: a k× coarse FD solve is ~k² cheaper in 1+1D (k4 ≈ 16×, k8 ≈ 64×).
- refiner **train/infer** cost vs one fine FD solve.
- headline = **accuracy at a fraction of fine-FD compute** (a Pareto point), which
  is the hybrid's only real reason to exist. A prior that is already <1% is NOT a
  compelling demonstration — we must rescue a *visibly degraded, genuinely cheap* one.

---

## 3. Architecture choice for Stage 1: PINN + autograd (not FNO)

For a **single config** the right tool is a PINN (MLP on (x,t) → u), not the FNO:

- **Exact residual, no stencil landmine.** Autograd gives ∂²u exactly. The existing
  `loss_pde_residual` in [src/fno_model.py](src/fno_model.py) uses a **2nd-order
  grid stencil** — using that as a *loss* would re-inject the very dispersion error
  we are trying to remove. Autograd sidesteps this entirely. (If we ever compute the
  residual on a grid, it MUST use the high-order DRP7/central stencils from
  [scripts/exploration/coarse_stencil_isolation.py](scripts/exploration/coarse_stencil_isolation.py).)
- **Reuses the SW PINN harness** ([src/pinn.py](src/pinn.py)).
- **Doubles as the C.3 fix:** C.3's bare PINN collapsed to zero; anchoring it to a
  coarse prior is the same mechanism we need here. One experiment, two problems.

Two equivalent ways to wire the prior in (identical for a single fixed prior):
- **(A) Anchor** (primary, = PINO instance-finetune form):
  `u_θ = MLP(x,t)` with hard-IC ansatz; `L = L_res + α·L_anchor + (BC)`.
- **(B) Correction ansatz** (the operator-stage form): `u_θ = up(u_coarse) + s²·N_θ`;
  residual on the full `u_θ`. Needs differentiable interpolation of the prior.

Use **(A)** for Stage 1 (simplest; prior only needs point-evaluation, not
differentiation; maps exactly to PINO's `L_pde + α·L_op`).

---

## 4. Stage 0 — characterize the cheap prior (FIRST; cheap, login-node, no training)

Pick the **honest cheap prior** = the coarsening that leaves *visible room to
improve on BOTH metrics*. Measure, do not assume:

- Solve the canonical config (M=1, ℓ=2 Zerilli, A=1, x₀=4M, σ=5M, x∈[−50,150],
  t∈[0,50]) at **plain 2nd-order** k4 (dx=0.8, dt=0.4) and k8 (dx=1.6, dt=0.8).
- Score each bare prior vs fine FD (dx=0.2) + Leaver: field rel-L2 **and** ω/τ at
  x_q=10M, 2M.
- Expectation: plain k4 ≈ 5% field; QNM degraded (dispersion corrupts ringdown
  phase). If k4's QNM is already <1%, escalate to k8.
- **Deliverable:** the chosen prior (k4 default; k8 if k4 too good) + a table of its
  native (field, ω, τ) error and its k² cost saving. This is the bar to beat.

(Do NOT use the DRP7 high-order coarse prior here — it is already ~0.45% and
recreates the "prior too good" problem. Plain low-order coarse is the honest start.)

---

## 5. Stage 1 — single-config SW refiner (the decisive test)

**Config:** the one characterized in Stage 0 (canonical M=1).

**Model:** PINN MLP (reuse [src/pinn.py](src/pinn.py)), tanh, FP64. Hard-IC ansatz
(exact Gaussian IC u₀(x); `u = u₀ + s²·N_θ`, s=t/T) so IC + zero-velocity are exact
→ no IC loss, and it removes the phase ambiguity structurally.

**Loss (start MINIMAL — add reactively, per the C.3 over-engineering lesson):**

| term | weight (start) | role | notes |
|---|---|---|---|
| PDE residual (Zerilli, **autograd**) | λ_res = 1.0 | the physics; the only term that scales to higher-D | u_tt − u_xx + V u |
| coarse-prior anchor ‖u_θ − u_prior‖² | α = 1.0 | replaces fine-FD loss; kills null space → cures collapse | prior interpolated to collocation pts |
| hard IC | — (ansatz) | exact, removes phase ambiguity | no loss term |
| outflow BC (both ends) | λ_bc = 1.0 | SW truncated domain needs it; **Kerr hyperboloidal will not** | may be anchor-implied; add explicit only if edges drift |
| ringdown / time-weighted | **OFF** | targets late low-amplitude ringdown | **add only if QNM lags after the minimal run** |
| grad-enhanced residual | **OFF** | marginal; more weight-tuning | likely never |

**Key knob:** α (anchor strength). Too low → collapse risk returns; too high → you
inherit the prior's error (can't beat C0). Sweep α ∈ {0.1, 1, 10} if the first run
fails a gate. This is the one parameter that matters most.

**Optimizer:** Adam → L-BFGS (SW recipe: ~10k Adam + L-BFGS), FP64. Anchor stays on
through both phases (it is the stabilizer, unlike a warm-up).

**Acceptance:** §2 gate — beat C0 on waveform rel-L2 AND ω AND τ; report vs C1 and cost.

**Decisive-failure value:** if residual+anchor+IC cannot beat the bare prior on the
QNM, that is itself a publishable/strategic finding — it says physics refinement
cannot clear the bar here, *before* any GPU/operator spend.

---

## 5b. G1b — the anchor go/no-go (EXECUTABLE SPEC, 2026-06-15)

**Why this section exists.** The running SW jobs (`30589279`/`30589280`) test the
**Richardson** target with `(data,physics,anchor)=(1.0,0.0,0.0)` — **anchor OFF**.
They do NOT test the coarse-prior anchor that the Kerr Stage-2 port (§6) depends on.
This is the missing test: *does a label-free, PDE-residual-trained refiner anchored
to the cheap coarse prior actually beat the bare prior?*

**KEY — the model already exists, so Step 0 costs ZERO training.**
`outputs/hybrid/fno_sw_physics_k4/model_best.pt` was trained with the exact recipe
([configs/hybrid_sw_physics_k4.yaml](../configs/hybrid_sw_physics_k4.yaml): data=0,
physics=1.0, anchor=0.1, label-free, hard-IC gating) but was **never evaluated**.

**Form note (honest, do not gloss).** The SW code is **Form (B)**:
`u = prior + corr`, anchor = `‖corr‖²` (a shrink-toward-prior penalty), with hard-IC
gating `corr = s²·raw`. The Kerr Stage-2 plan (§6.2) is **Form (A)** (free net,
anchor `‖u−prior‖²`, prior NOT in the ansatz, to preserve solver-freedom). Form (B)
is the *stronger* anchor (structural, not just a loss), so: a Form-(B) **PASS** is
**necessary-but-not-sufficient** evidence for the Form-(A) Kerr port; a Form-(B)
**FAIL** strongly discourages porting the anchor at all. Say this in any writeup.

**Structural worry to keep in mind.** The anchor `‖corr‖²` can only pull *toward*
the prior; the only term that can push *past* it is the fine-grid PDE residual. But
the coarse prior already nearly satisfies the fine 2nd-order residual (the
`fno_sw_physics_k4_diag` / `_diag_norad` runs sat at field rel-L2 ≈ 5% = the bare
k4 prior with the residual at ~1e-6). So the live risk is **"refiner just inherits
the prior"**, not collapse-to-zero (hard-IC blocks zero). Step 0 measures exactly this.

### Controls (all scored vs fine FD + Leaver in ONE eval)
The eval script already emits all three, so no extra runs:
- **C0 = bare k4 prior** (coarse-up baseline) = `field.rl2_baseline` + `*_base_*`
  QNM in `summary.json`. The bar. **Free in every eval.**
- **A = physics-only** (`anchor_weight=0`): isolates whether the residual alone
  moves off the prior. (`_diag` evidence hints NO.)
- **B = physics + anchor 0.1** = the existing `fno_sw_physics_k4`. The test article.

### Steps
0. **(free) Evaluate the existing checkpoint.**
   - login smoke (<1 min, confirm it runs):
     `scripts/eval_hybrid_sw.py --config configs/hybrid_sw_physics_k4.yaml --n 5 --xq 2`
   - full: same minus `--n`, on **SLURM** (inference-only → CPU icelake is fine).
   - **PASS iff** `field.rl2_hybrid < field.rl2_baseline` (beats prior on waveform)
     **AND** `xq2_hyb_M4_omega_pct_err ≤ xq2_base_M4_omega_pct_err` (QNM not
     regressed). Both read from the SAME `summary.json`.
1. **If Step 0 PASSES** → G1b green: the anchor works on SW. Proceed to Kerr
   Stage-2 (still gated on G2). **No new training.**
2. **If Step 0 FAILS** → run the 1-variable A/B to attribute: A (`anchor_weight=0`)
   and B (`anchor_weight ∈ {0.1, 1, 10}`), label-free physics, **same seed/budget,
   ONLY anchor_weight varies**. Gate = best B beats C0 on field AND keeps QNM ≤ C0;
   report A to show the anchor's marginal effect. If even the best anchor cannot
   beat C0 → the anchor does **NOT** cure the gap on SW; **do NOT port it to Kerr**;
   the Kerr collapse rescue must come from elsewhere (Fourier G2 / curriculum). That
   is itself a decisive, publishable finding (§5 "decisive-failure value").

### Compute & discipline
- Eval = inference-only, no GPU. Step-2 training = the established FNO GPU path
  (ampere), short (100 Adam + 30 L-BFGS epochs; `_diag` was ~15 s/epoch).
- **SLURM mandate:** full eval + any training → SLURM, never login (login only the
  `--n 5` smoke).
- Step 0 changes nothing; Step 2 changes ONLY `anchor_weight`. The clean A/B the
  C.3 over-engineering lesson demands.

---

## 6. Stage 2 — parametric operator → Kerr (LATER; gated on Stage-1 + Fourier verdict)

**Status:** on-paper wiring only (drafted 2026-06-15). NO code, NO compute until
both gates in §6.1 are green. Written so it is ready to execute the moment the SW
Stage-1 A/B (`30589279`/`30589280`) and the Kerr Fourier a/M=0 go/no-go
(`30593612`) report.

### 6.1 Two decision gates (both green before any Stage-2 code)

| gate | source | green = | if red |
|---|---|---|---|
| **G1a — Richardson (running)** | SW jobs `30589279`/`30589280` | Richardson refiner beats bare k4 on waveform AND keeps QNM ≤ prior | Richardson target insufficient; lean on the anchor (G1b) |
| **G1b — anchor (THE port gate)** | §5b anchor go/no-go (eval existing `fno_sw_physics_k4` FIRST) | physics+anchor refiner beats bare k4 on waveform AND keeps QNM ≤ prior | anchor does NOT cure the gap → do NOT port the anchor to Kerr |
| **G2 — pure-PINN status** | Kerr Fourier a/M=0 (`30593612`) | Fourier PINN already PASSES a/M=0 | anchor is the rescue for the collapse → highest value precisely *here* |

**G1b, not G1a, gates the Kerr anchor port** (Stage-2 uses the anchor, not
Richardson). G1a is a parallel label-free data point. Run G1b's Step 0 (a free eval
of an already-trained model) before assuming the anchor works — see §5b.

Honest branch: Stage-2 (anchor) is the **rescue path for the C.3 collapse**, so it
matters most when **G2 is RED** (Fourier alone does not cure a/M=0). If G2 is green
the pure PINN stands alone and the anchor becomes a spin-generalisation aid, not a
rescue.

### 6.2 Reconciliation with the C.0′ pivot (DO NOT resurrect the rejected hybrid)

C.0′ ([kerr/notes/phase_c_plan.md](../kerr/notes/phase_c_plan.md)) dropped the
**supervised** coarse-FD→FNO→fine-FD hybrid for three reasons: (1) not solver-free
at inference, (2) capped by its fine-FD teacher, (3) no higher-D transfer. Stage-2
walks back **none** of these, because it borrows only the *anchor mechanism*, not
the architecture:

- **Operator = the parametric PINN** (`kerr/src/kerr_pinn.py`), NOT an FNO. C.0′
  already ruled out FNO/PINO for Kerr (FFT → Gibbs at scri/horizon; 3-scalar family
  wastes function-space machinery; autodiff is geometry-agnostic and scales past
  3+1D). Stage-2 keeps that ruling.
- **Form (A), not Form (B).** Inject the coarse prior as a **training-time anchor
  loss** `α‖Ψ_θ − Ψ_prior‖²`, NOT baked into the ansatz `up(Ψ_coarse)+δ`. The anchor
  is a regulariser **discarded at inference** — deployment still evaluates
  `Ψ_θ(σ,τ;a/M)` from the equation alone. Solver-freedom (C.0′ defect 1) is
  preserved. This is the deliberate upgrade over this section's old Form-(B) stub,
  which would have re-introduced an inference-time coarse solve.
- **Label-free.** Loss = Teukolsky **PDE residual**
  ([kerr/src/teukolsky_residual.py](../kerr/src/teukolsky_residual.py), 2.5e-16) +
  anchor; the anchor target is a **cheap coarse Teukolsky solve**, never a fine-FD
  field. No fine FD in the loss → C.0′ defect 2 does not bite (the network can drop
  *below* the coarse prior's error, since it minimises the true residual, not a
  teacher MSE).
- **Higher-D transfer** (C.0′ defect 3) intact: the residual is identical in any
  dimension; the anchor is only a finite-time stabiliser to escape the trivial-zero
  basin, not a 1+1D-specific crutch.

So Stage-2 = "the proven SW anchor mechanism, ported as a **training-time
regulariser** onto the C.0′ pure PINN." We do not rebuild the hybrid.

### 6.3 Wiring (file-level; extend, do not fork)

- **Residual:** `kerr/src/teukolsky_residual.py` — unchanged.
- **Network + domain:** `kerr/src/kerr_pinn.py` — add ONE loss term
  (`α·‖Ψ_θ − Ψ_prior‖²` at collocation points) + a prior-evaluation hook. Fourier
  features, hard-IC ansatz, hyperboloidal domain, no-BC: all unchanged.
- **Prior source:** a cheap coarse Teukolsky solve from the Phase-B core
  (`kerr/src/teukolsky_minimal_gauge.py`). **First check whether the C.1/C.2 corpus
  (`kerr/outputs/phase_c/`) already carries a coarse tier** — if so, no new solver
  code, only a differentiable-interpolation hook. Do NOT build a coarse-prior
  builder until this is confirmed.
- **Driver:** `kerr/scripts/train_kerr_pinn.py` — add `--anchor-prior PATH` and
  `--anchor-weight α`, default OFF, so the baseline/Fourier runs stay byte-identical.
- **Config:** extend the existing schema → new `kerr/configs/kerr_pinn_anchor.yaml`.

### 6.4 Loss (start MINIMAL — same discipline as §5)

| term | weight (start) | role |
|---|---|---|
| Teukolsky PDE residual (autograd, 2 real channels) | λ_res = 1.0 | the physics; only term that scales to higher-D |
| coarse-prior anchor ‖Ψ_θ − Ψ_prior‖² (**training only**) | α = 1.0 | escapes the trivial-zero basin; discarded at inference |
| hard IC | — (ansatz) | exact; no loss term |
| BC | **none** | hyperboloidal outflow is automatic (Kerr's free win over SW) |
| Fourier features | on | spectral-bias mitigation (the G2 lever) |

**Key knob:** α, exactly as SW §5. Sweep α ∈ {0.1, 1, 10} only if a/M=0 still
collapses. Anchor stays on through Adam→L-BFGS (stabiliser, not warm-up).

### 6.5 Acceptance (the LOCKED Kerr Phase-C gate — unchanged, not softened)

- **First** clear **a/M=0** (the current blocker): scri rel-L2 ≤ 5% AND M*omega ≤
  1% vs Leaver — the *same* gate the bare and Fourier PINNs face, so the anchor is a
  clean one-variable A/B.
- **Then** held-out spins: field rel-L2 ≤ 5% AND M*omega ≤ 1% AND τ ≤ 5% vs Leaver.
- **Always report the bare-coarse-prior control** (C0). The anchor must let the PINN
  **beat** the prior, not merely inherit it — else we have validated the coarse
  solver, not the operator.

### 6.6 Honest risks specific to Stage-2

- **Anchor may not fully cure the collapse.** If a/M=0 still collapses with the
  training-time anchor, the only fallback is Form (B) (prior in the ansatz) — which
  **costs solver-freedom** (C.0′ defect 1 returns). Documented last resort, used
  only if Form (A) + Fourier both fail, and flagged as such in any writeup.
- **"Cosmetic QNM" repeat.** Phase B showed the FD QNM is robust to coarsening; the
  coarse prior likely already carries M*omega. Expect the anchor to earn its keep on
  the **waveform**, with the QNM as a "stays-green" validation — exactly the SW
  Stage-0 finding. Say so honestly.
- **Anchor masking a dead network.** A strong anchor can pass the gate while N_θ
  does nothing. Mitigation: the C0 control above + report the residual reached at
  α=0 vs α>0.
- **Compute discipline.** a/M=0 go/no-go is CPU (SLURM, `FERGUSSON-SL3-CPU`); no GPU
  until a/M=0 passes, per the Phase-C ladder.

### 6.7 Sequencing (one variable at a time)

1. Wait for **G1** (SW Stage-1) and **G2** (Fourier a/M=0).
2. If G2 RED and G1 GREEN → wire §6.3, run the a/M=0 anchor go/no-go (CPU/SLURM)
   with **Fourier ON + anchor ON**, vs the Fourier-only control. One new variable.
3. If a/M=0 passes → a/M=0.7, then held-out spins (this *is* the PINO "virtual
   instances" test — the parametric PINN already enforces the residual over a
   continuous a/M distribution, no FD per spin), then C.4 GPU.
4. PINO over arbitrary IC *fields* stays the documented future option (C.0′ caveat),
   NOT built here.

---

## 7. Risks & mitigations

- **Collapse to zero (the C.3 failure).** Mitigation: the prior anchor; raise α.
- **Residual dispersion poisoning the loss.** Mitigation: PINN+autograd (exact) in
  Stage 1; high-order DRP7 stencils if ever grid-based.
- **"Prior too good" → no story.** Mitigation: Stage 0 picks a genuinely degraded
  cheap prior (k4/k8 plain).
- **Over-engineering the loss.** Mitigation: start with 3 terms (res+anchor+IC); add
  ringdown/time-weighted ONLY if QNM lags. Never all 7 at once.
- **Grading the prior as if free.** Mitigation: cost (k² saving + train/infer) is a
  required column in every result.

## 8. Explicitly deferred (do NOT do now)

- **MPINN multifidelity** (few-fine-points correlation, [1903.00104]): attacks the
  build-cost flaw, which the *already-built* SW corpus has already paid. Adds 2–4
  coupled subnets + a relaxation parameter = more failure modes. Revisit only after
  the single-config refiner works, as a data-efficiency optimization.
- **Differentiable solver-in-the-loop** ([2007.00016]): heavier than needed; its
  distribution-shift critique does not bite our one-shot (non-autoregressive) setting.

## 9. Immediate next action

Stage 0: write one login-node script that solves the canonical config at fine /
plain-k4 / plain-k8, and prints the (field rel-L2, ω, τ) table + k² cost for each.
Decide the prior, then build Stage 1.

---

## STAGE 0 RESULT (2026-06-13) — DONE

Script: `scripts/exploration/prior_characterization.py` (M1–M5, login-safe).
Output: `outputs/exploration/prior_char/prior_char.json`. Canonical M=1 Zerilli,
plain 2nd-order, vs fine (dx=0.2) + Leaver. Best-of-M1–M5 at x_q=2:

| prior | cost (work) | **field rel-L2** | best ω% | best τ% |
|---|---|---|---|---|
| k2 | 4× | 0.0098 | 0.003% | 0.011% |
| **k4** | **16×** | **0.0493** | 0.041% | 0.125% |
| k8 | 63× | 0.1996 | 0.180% | 0.110% |

**Findings:** (1) the **field rel-L2 is the clean coarsening signal** (1%/5%/20%) —
this is the refiner's job. (2) The **QNM is robust** to coarsening (M4/M5 keep ω,τ
≤0.2% even at k8): it lives in low frequencies that survive coarsening. So the
refiner's product is the **high-fidelity waveform**; the QNM is validation that
stays green. (3) The earlier "7% τ" was an M2 single-window artifact; **M4/M5
dissolve it** — use the full suite, judge τ at x_q=2.

**Decision: rescue the k4 prior** (16× cheaper, 4.9% field error = real room, QNM
already excellent). Stage 1 gate: refiner field rel-L2 ≪ 4.9% (toward <1%), M1–M5
ω/τ ≤ k4 floor, all **label-free**, beating bare k4 (C0).

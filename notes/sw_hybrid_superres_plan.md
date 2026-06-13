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
label*. The anchor is load-bearing (it is what cures the collapse), not decorative.

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

## 6. Stage 2 — parametric operator (LATER, only after Stage 1 passes)

Promote the refiner to an **operator** (FNO/PINO, form (B): `u = up(u_coarse)+δ_θ`,
trained on the **PDE residual**, not fine-FD MSE), conditioned first on M (SW), then
on Kerr spin a/M using the **validated Teukolsky residual (2.5e-16)**
([kerr/src/teukolsky_residual.py](kerr/src/teukolsky_residual.py)). Add PINO's
**virtual instances** (extra spins, no FD at all) and **instance-wise fine-tuning +
anchor loss** at query time. This is the label-free operator that scales in dimension.

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

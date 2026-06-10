# Phase C — Kerr neural surrogate (task breakdown)

The machine-learning phase. Builds a learned waveform surrogate on top of the
**validated, audited** Phase B finite-difference Teukolsky solver
(`kerr/src/teukolsky_minimal_gauge.py`, $s=-2$, $\ell=m=2$, hyperboloidal,
$a/M\in[0,0.95]$). Phase B passed its B.9 gate (population-mean $M\omega_{220}$
error $4.7\times10^{-4}$) and was independently verified non-circular and
$2$nd-order convergent. Phase C is the analogue, for Kerr, of the parent
repo's Schwarzschild **hybrid coarse-FD + FNO-residual** surrogate
(`src/hybrid_*.py`, `scripts/*hybrid*`).

Each task is one commit, one test, one acceptance check. Do not merge two.
Phase C needs **GPU** (CSD3 `ampere`/A100) for training — unlike Phases A/B,
which were CPU-only. Dataset generation stays CPU (`icelake`). `kerr/` stays
self-contained — reusable parent modules are copied/extended in `kerr/src/`,
never imported from the parent `src/`.

---

## C.0 — architecture & data decision record (no model code)

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

### Honest risks (recorded now, not after they bite)

1. **Residual may not be smoother than the field** (proposal risk 1). If coarse
   FD aliases the ringdown, $\delta$ oscillates at the QNM frequency and is as
   hard to learn as $\psi$ itself — the hybrid then won't beat the pure FNO.
   *Mitigation:* ablate $k\in\{2,4\}$ in C.2. The spot-check ruled out *temporal*
   aliasing as the mechanism — even $N=201$ keeps $\sim$1600 pts/period at
   $a/M=0.95$ — so the coarse error is **spatial** (pulse under-resolution), which
   is exactly why the C.1 box holds $\ge5$ pts on $N=201$. If $\delta$ is still
   too rough at $k=4$, fall back to $k=2$.
2. **Complex doubles the target.** Two channels with a learned phase relation is
   strictly harder than the SW real case; the weakly-complex low-spin band (where
   Phase B already had its largest $0.3\%$ errors) is the likely failure point.
   *Mitigation:* report per-spin, not just population-mean; consider a
   phase-aware loss only if MSE stalls there.
3. **Near-extremal tail.** $a/M=0.95$ ringdown gives way to tail/near-degenerate
   overtone by $\sim6.8\tau$; the surrogate inherits this. The extractor's
   `envelope_tail_cap` already handles it on predicted fields, but the net may
   mis-amplitude the short clean window.
4. **GPU dependency.** Phases A/B were CPU-only; C.4/C.5 need an A100 allocation
   (proposal A6 says the existing pure-FNO allocation suffices — **to be
   confirmed before C.4**, not assumed).
5. **Story risk.** If the hybrid only *matches* the pure FNO, it is not a
   headline (proposal §6.4). The C.5 gate's "clearly beats coarse-up" is the
   guard; the abort policy is the backstop.

---

## C.1 — Kerr coarse/fine dataset generator
**File.** `kerr/scripts/build_kerr_dataset.py`, `kerr/src/kerr_dataset.py`.
**Implements.** Sobol sweep over $(a/M, r_0, w)$ (amplitude dropped — linear
PDE); per sample run the Phase B `evolve` at fine $N=801$ and **both** coarse
grids $N=401$ ($k=2$) / $N=201$ ($k=4$), store the fine + both coarse **complex**
fields, parameter vector, and `qnm` reference $(M\omega,\tau)$. Re-uses
`kv3_qnm.evolve` and `qnm_kerr_reference` verbatim (no new physics path).
`--smoke` = a handful of samples at low $N$ for a login-node spot-check.
**Acceptance.** Spot-check $\ge5$ samples: fine field reproduces a direct B.9
single-spin run to machine precision at matching params; coarse field is finite
(no NaN/boundary growth) and within sensible $L^2$ of upsampled-to-fine; the
saved per-sample $(M\omega,\tau)$ matches a fresh `qnm` call. One commit.

## C.2 — authoritative corpus (SLURM, CPU)
**File.** `kerr/scripts/slurm_kerr_dataset.sh`, output
`kerr/outputs/phase_c/dataset_{train,val,test}.npz` (each holding fine + both
coarse grids; gitignored).
**Implements.** Full 1280-sample generation (1024/128/128), fine + both coarse
($k\in\{2,4\}$) per sample, on `icelake`.
**Acceptance.** Both `.npz` complete, all samples finite, train/val/test split
saved with fixed seed; a manifest (counts, ranges, $k$, grids, wall-time)
printed and committed as a small text/JSON record. One commit.

## C.3 — baseline + evaluation harness
**File.** `kerr/scripts/eval_kerr_surrogate.py` (mirrors
`scripts/eval_hybrid_sw.py`), `kerr/src/kerr_data_pipe.py`.
**Implements.** Upsample coarse$\to$fine; assemble FNO input channels (incl.
$a/M$); compute the **coarse-up baseline** field RMSD and its QNM error via the
Phase B extractor; define the canonical eval = test split and the population-tail
metric. No trained model yet — this fixes the numbers the surrogate must beat.
**Acceptance.** Baseline table printed (coarse-up vs fine-truth: field RMSD,
$M\omega$ err, $\tau$ err, worst-case $|\Delta\tau/\tau|$) and saved to
`kerr/outputs/phase_c/baseline.json`. Sanity: fine-truth QNM errors reproduce the
B.9 numbers at matching spins. One commit.

## C.4 — Kerr surrogate architecture + training (GPU)
**File.** `kerr/scripts/train_kerr_surrogate.py`, config
`kerr/configs/kerr_surrogate_k{2,4}.yaml`; extends `kerr/src/hybrid_fno.py` for
the **2-channel complex** target + spin conditioning.
**Implements.** Primary candidate (coarse-FD + complex-FNO residual). Adam
$\to$ L-BFGS; MSE on $(\mathrm{Re},\mathrm{Im})\,\delta$. `--smoke` trains a few
epochs on the smoke dataset to prove the loop on CPU before the GPU run.
**Acceptance.** Val MSE converges and is **below the equivalent pure-FNO** on the
same sweep (apples-to-apples; proposal §9 Phase 2). If not, STOP and reconsider
before spending the eval. One commit.

## C.5 — evaluation against the gate + ablations
**File.** reuse `eval_kerr_surrogate.py` with `--model`; output
`kerr/outputs/phase_c/eval_k{2,4}/`.
**Implements.** Hybrid vs coarse-up vs fine-truth on the test split: field RMSD,
QNM errors, **worst-case $|\Delta\tau/\tau|$ on the population tail**; $k\in\{2,4\}$
ablation; speedup (coarse-FD + inference vs fine-FD wall-time).
**Acceptance (README gate, verbatim).** Hybrid beats coarse-up by $\ge10\times$
in field RMSD **OR** suppresses worst-case $|\Delta\tau/\tau|$ by the SW factor
($30\%\to5\%$). If it fails after this one architecture: invoke the abort policy.
One commit.

## C.6 — paper integration (deferred to write-up)
**File.** note only; `.tex` after C.5 passes. New Results subsection extending
the SW hybrid result to Kerr $(2,2)$ across spin; higher modes named as future
work. Deferred — listed so it is not forgotten.

---

## Phase C dependency chain
```
C.0 decisions  →  C.1 dataset gen  →  C.2 corpus(SLURM,CPU)  →  C.3 baseline/eval
                                                                       │
                                                  C.4 train(GPU)  ←────┘
                                                       │
                                                  C.5 gate + ablations  →  C.6 paper
```
Data + baseline (C.1–C.3) exist **before** any model is trained — the same order
the Schwarzschild side was built. No paper prose before C.5 passes.

## Scope guard (unchanged from Phase B)
Commit only `kerr/` files (parent `../` tree is off-limits). GPU work uses the
existing CSD3 allocation (confirm before C.4). Architecture slate may be
*prototyped*, but the paper ships the first candidate that clears the C.5 gate;
the abort policy forbids an open-ended model search.

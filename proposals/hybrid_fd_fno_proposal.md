# Hybrid FD + FNO waveform model — method proposal

Status: DRAFT for user approval. No code or paper prose to be written until this is signed off.

---

## 1. Problem statement

Existing modalities in this work have complementary weaknesses:

- **Fine RK4 FD** (Δx=0.2M, Δt=0.1M): reference accuracy (extractor floor ~0.02% in Mω) but per-waveform cost; no parametric generalisation, every new (M, A₀, x₀, σ₀) needs a fresh integration.
- **Pure FNO**: amortises across parameters and is ~10³× faster at inference, but the operator residual currently floors QNM extraction at ~0.2% in Mω, an order of magnitude above the FD extractor floor.
- **PINN**: physics-consistent but slow to train, no amortisation, and inverse-problem dominated.

**Goal.** A single waveform surrogate that (i) is faster than fine FD at inference, (ii) reaches the fine-FD extractor floor in QNM accuracy, (iii) extends to Kerr by changing only the FD core. Validated on Schwarzschild ℓ=2 first, then on Kerr (a, ℓ, m) within the same paper.

## 2. Proposed architecture

**Coarse-FD + FNO residual correction.** Lineage: Kochkov et al. 2021 PNAS (Navier–Stokes); Um et al. 2020 NeurIPS solver-in-the-loop; Pathak et al. 2022 (weather).

```
  parameters (M, A0, x0, σ0)         [SW]
              │              (a, M, ℓ, m, A0, x0, σ0)   [Kerr]
              ▼
       ┌──────────────┐
       │ Coarse RK4   │   Δx_c = k·Δx_f, Δt_c = k·Δt_f
       │  Zerilli /   │   k ∈ {2, 4} chosen empirically
       │  Teukolsky   │
       └──────┬───────┘
              │ Φ_c(x,t) coarse field, upsampled to fine grid by spectral interp
              ▼
       ┌──────────────┐
       │     FNO      │   inputs:  [Φ_c upsampled, params broadcast as channels]
       │  (2D modes)  │   output:  δΦ(x,t)  on fine grid
       └──────┬───────┘
              │
              ▼
        Φ_hybrid = Φ_c_upsampled + δΦ        (target: Φ_f from fine RK4)
```

Key invariant: if FNO output is zero, fallback is the upsampled coarse solver. Worst-case error is bounded by coarse-FD error. No such floor exists for the pure FNO.

## 3. Training protocol

**SW phase.**
- Dataset: sweep (M, A₀, x₀, σ₀) on a quasi-random (Sobol) grid; N_train ≈ 500–1000, N_val ≈ 100, N_test ≈ 100. Re-use existing `solve_fd` for both coarse and fine.
- Loss: MSE on δΦ over (x,t) grid + optional auxiliary M4-frequency consistency term, weight TBD by ablation.
- Architecture: 2D FNO, ~10 modes per axis, 4–6 layers. Smaller than current pure-FNO since target is a smoother residual.
- Optimiser: Adam → L-BFGS finetune.

**Kerr phase.**
- Teukolsky radial+time equation with the Sasaki-Nakamura transform (numerically well-behaved). Spin-weighted spheroidal harmonics handled by mode-by-mode separation; this paper restricts to ℓ=m=2 to keep scope finite.
- Dataset: sweep over (a/M ∈ [0, 0.99]), fixed ℓ=m=2, with same initial-data sweep.
- Same FNO architecture, retrained.

## 4. Validation plan

For both SW and Kerr:

1. **Field accuracy**: pointwise error heatmap, L² over (x,t), MAD/RMSD against fine FD on held-out test set. Compare hybrid vs pure FNO vs upsampled coarse.
2. **QNM frequency accuracy**: M4 plateau extraction on hybrid waveforms, compare Mω / τ against Leaver. Target: hybrid M4 error ≤ 2× fine-FD M4 error (currently 0.02% / 0.06% at xq=2).
3. **Speedup**: wall-time per waveform: hybrid (coarse FD + FNO inference) vs fine FD. Target: ≥ 5× at SW, ≥ 10× at Kerr (Kerr fine FD is more expensive, so speedup grows).
4. **Generalisation**: held-out (M, a) combinations and out-of-distribution initial data (e.g. σ₀ outside training range) to expose extrapolation failures.

## 5. Computational budget (rough, to be refined)

| Stage | Cost | Wallclock estimate |
|---|---|---|
| SW dataset generation (1000 × fine + coarse) | CPU | 1–2 days on CSD3 |
| SW FNO training | 1 A100 | 4–12 h |
| Kerr Teukolsky solver implementation | dev | ~1 week of focused work |
| Kerr dataset generation (1000 × fine + coarse) | CPU | 3–5 days |
| Kerr FNO training | 1 A100 | 8–24 h |
| Validation + plotting | mixed | 2–3 days |

**Total**: ~3–4 weeks of focused work, ~1 week if Kerr is dropped to a companion paper.

## 6. Honest risks and where this might fail

1. **Residual may not be smoother than the full waveform** at the QNM ringdown band. If coarse FD already captures the ringdown frequency well but mis-amplitudes it, δΦ is mostly amplitude correction → easy. If coarse FD aliases the ringdown, δΦ is oscillatory at the QNM frequency → as hard to learn as the full waveform, and the hybrid won't beat pure FNO. **Mitigation**: ablate k ∈ {2, 4} early; if k=4 aliases, fall back to k=2 (smaller speedup, but architecture still works).
2. **Teukolsky in 1+1D is not trivial**. The standard frequency-domain Teukolsky code (e.g. BHPToolkit) is well-tested but time-domain Teukolsky requires the SN transform or hyperboloidal compactification to be stable. Implementing this is a real chunk of work and is the dominant uncertainty in the timeline. **Mitigation**: prototype on Schwarzschild-limit Teukolsky (a→0) first to validate against the existing Zerilli pipeline.
3. **Kerr ℓ=m=2 only is a real scope limitation**, but defensible. Higher-ℓ Kerr modes can be relegated to "future work" honestly.
4. **The story risk**: if hybrid only matches pure FNO and doesn't clearly beat it, the architecture is not publishable as the headline. **Mitigation**: validate the SW phase fully before committing to Kerr; if SW hybrid doesn't beat pure FNO, abort Kerr.

## 7. What I need from you before writing any code

Please answer / approve:

- **Q1.** Architecture choice confirmed: coarse-FD + FNO residual? Or do you want me to also explore (B) coarse-FD + neural-ODE-style timestep correction, (C) FNO as initial guess for a few FD refinement steps, before locking in?
- **Q2.** Coarse-grid factor k: I propose ablating k ∈ {2, 4}. OK?
- **Q3.** Scope of Kerr: ℓ=m=2 only, spin sweep a/M ∈ [0, 0.99]. Or broader?
- **Q4.** Teukolsky implementation: are you OK with me writing it from scratch (using JMS-style hyperboloidal slicing for stability), or do you want to wrap an existing library (BHPToolkit / GremlinEq)? Wrapping is faster but adds an external dependency.
- **Q5.** Order of operations: SW hybrid first end-to-end, validate, only THEN start Kerr — agreed? (This protects against the failure mode in §6.4.)
- **Q6.** Compute: this needs an A100 allocation on CSD3 for FNO training and a chunk of CPU time for dataset generation. Do we have that?
- **Q7.** Paper structure once it lands: I propose the existing Schwarzschild PINN/FNO/M4–M5 content becomes the *baseline* (Section 4 Results: baselines), with new Sections 5 (Hybrid method) and 6 (Hybrid Results: SW and Kerr). The current narrative shifts from "we built extractors and surrogates" to "we built a hybrid waveform model; baselines and extractors are auxiliary contributions." Agree?

---

## 8. Committed answers (signed off 2026-05-29)

- **A1.** Coarse-FD + FNO residual. Options (B) neural-ODE timestep correction and (C) FNO-as-initial-guess were dropped after deeper analysis: (B) accumulates error sequentially with no architectural payoff for a hyperbolic PDE; (C) doesn't fit time-stepping (nothing to "refine into"). Justification for (A) is internal, not lineage-based: at coarse Δt = 0.4M the QNM period (~17M) is still resolved at ~40 points/period, so coarse FD preserves ω₀ and only the amplitude/phase drifts. The residual δΦ = Φ_fine − Φ_coarse is therefore a smooth amplitude/phase correction, genuinely easier to learn than the full waveform.
- **A2.** Ablate k ∈ {2, 4}; add k = 3 only if results between 2 and 4 disagree qualitatively. Justification: k = 4 already leaves ~10 points/period at the QNM ringdown; k ≥ 8 would alias the ringdown frequency, breaking the residual-is-smoother assumption from A1. k = 2 / 4 spans the meaningful range (4× / 16× per-step speedup in 2D).
- **A3.** Kerr scope: (ℓ, m) = (2, 2) only, spin sweep a/M ∈ [0, 0.99]. (3,3) and (4,4) named explicitly as next-paper scope. Literature standard for first surrogate/ML ringdown papers is (2,2)-only; higher-mode extensions are routinely deferred.
- **A4.** Teukolsky: wrap an existing library (BHPToolkit / qnm package) for the frequency-domain QNM oracle used as validation target; hand-code the time-domain Teukolsky integrator with JMS-style hyperboloidal slicing for the FD core. Rationale: frequency-domain Kerr QNMs are not novel and there is no upside to rewriting; the time-domain integrator is the actual research artefact and must live in our codebase.
- **A5.** SW hybrid end-to-end and validated before any Kerr work.
- **A6.** Same CSD3 GPU allocation already used for the pure-FNO training; no new resource ask.
- **A7.** Hybrid is *just another method*. Add as the fifth method in the existing Methods structure (alongside Forward PINN, Inverse PINN, Curriculum PINN, FNO) and as a new subsection in Results. No reorganisation of existing sections; the paper's "compare several waveform methods on Schwarzschild ℓ=2, extract QNMs" architecture is preserved. Kerr (2,2) results form a final Results subsection demonstrating method generalisation.

---

## 9. Implementation plan (file-level, in execution order)

Gating: each phase must produce its named deliverable and pass its named acceptance check before the next phase starts. No paper prose written until Phase 4 acceptance passes.

### Phase 1 — Coarse-FD harness and dataset (SW)
**New files:**
- `src/hybrid_dataset.py`: `build_coarse_fine_dataset(cfg) -> (Φ_c, Φ_f, params)` driver. Re-uses existing `src.fd_solver.solve_fd` twice per sample (coarse and fine grids); fine is the existing reference grid Δx=0.2M, Δt=0.1M; coarse is k× coarser on both axes with CFL-respecting Δt.
- `configs/hybrid_sw_dataset.yaml`: sweep ranges. M ∈ [0.8, 1.2], A₀ ∈ [0.5, 1.5], x₀ ∈ [2, 6]M, σ₀ ∈ [3, 7]M. 1000 train / 100 val / 100 test Sobol points.
- `scripts/build_hybrid_dataset.py`: CLI wrapper, writes HDF5.

**Deliverable:** `outputs/hybrid/dataset_sw_k{2,4}.h5` with paired (Φ_c upsampled to fine grid by 2D spline, Φ_f, parameter vector).
**Acceptance:** spot-check 5 samples: fine RK4 matches existing Zerilli reference to existing tolerances; coarse RK4 is stable (no NaN, no boundary growth); upsampled coarse field within sensible L² of fine.

### Phase 2 — Hybrid FNO architecture and training
**New files:**
- `src/hybrid_fno.py`: 2D FNO. Inputs: upsampled Φ_c + parameter channels broadcast to (x,t) shape. Output: δΦ on fine grid. ~10–12 Fourier modes per axis, 4 layers, width 32. Smaller than the pure-FNO since residual is smoother.
- `scripts/train_hybrid_fno.py`: training driver. Adam → L-BFGS finetune. Loss: MSE on δΦ.
- `configs/hybrid_sw_train.yaml`.

**Deliverable:** `outputs/hybrid/fno_sw_k{2,4}/model.pt` + training-curve plots.
**Acceptance:** validation MSE on δΦ converges and is below the equivalent pure-FNO trained on the same parameter sweep (apples-to-apples comparison). If it isn't, abort and reconsider.

### Phase 3 — SW evaluation and QNM extraction
**New files:**
- `scripts/eval_hybrid_sw.py`: on held-out test set, compare hybrid vs pure FNO vs upsampled coarse vs fine FD. Compute (i) field accuracy: pointwise error heatmap, L², MAD, RMSD. (ii) QNM extraction: run existing `qnm_method_4_two_mode` on hybrid waveforms at canonical xq ∈ {2, 10}, compare to Leaver.

**Deliverable:** `outputs/hybrid/eval_sw/`: full plot set matching the cross-modality consistency rule (heatmap with LogNorm/magma, ringdown overlay linear and log, snapshot row), plus `errors.json` with the four-number table per modality.
**Acceptance criteria (these are the go/no-go for paper inclusion):**
- Hybrid M4 ω-error ≤ 2× fine-FD M4 ω-error at xq=2 (target: ≲ 0.04%).
- Hybrid wall-time per inference ≤ ½ × fine-FD wall-time.
- Hybrid clearly beats pure FNO on M4 error at the same parameter sweep.

If any acceptance check fails: STOP, report back, do not proceed to Kerr, do not write paper prose.

### Phase 4 — SW paper integration
**Files edited:**
- `paper/sections/methods.tex`: new subsection "Hybrid coarse-FD + FNO residual surrogate" after the existing FNO subsection. Same structural pattern as the other methods.
- `paper/sections/results.tex`: new subsection presenting the SW hybrid numbers using the existing table/figure conventions.
- `paper/refs.bib`: add Kochkov 2021, Um 2020, Pathak 2022.

**Acceptance:** PDF builds clean, no undefined refs/citations, cross-modality plots present (heatmap + ringdown overlay + snapshots), prose matches the established declarative voice, no em-dashes / colons-in-prose / internal-note language.

### Phase 5 — Kerr Teukolsky FD core
**New files:**
- `src/teukolsky_fd.py`: time-domain Teukolsky for (s=−2, ℓ=m=2) on JMS hyperboloidal slicing. Inputs: a/M, M, initial-data params. Output: Ψ₄(x,t) on the same kind of (x,t) grid as Zerilli.
- `src/qnm_kerr_reference.py`: thin wrapper around the `qnm` Python package (Stein 2019) for Leaver-equivalent Kerr (ℓ, m, n) frequencies. This is the wrap-not-rewrite from A4.

**Deliverable:** working `solve_teukolsky(cfg)` returning a stable waveform for a/M ∈ {0, 0.5, 0.9} validated against the wrapped Kerr QNM oracle via M4 extraction.
**Acceptance:** time-domain M4 extraction from `solve_teukolsky` matches `qnm` package frequencies to within the established Schwarzschild Zerilli extractor floor (~0.05% in Mω at the dominant mode). a→0 limit reproduces the existing Zerilli result to plotting precision.

### Phase 6 — Kerr hybrid training and evaluation
Same structure as Phases 1–3 with `solve_teukolsky` substituted for the fine FD core and a spin sweep added to the parameter dataset.
**Acceptance:** same go/no-go shape as Phase 3.

### Phase 7 — Kerr paper integration
New Results subsection extending the SW hybrid result to Kerr (2,2) across spin. Discussion paragraph naming higher-mode extensions as next paper.

---

## 10. What I will NOT do without re-asking

- Touch any existing file outside `src/hybrid_*.py`, `src/teukolsky_fd.py`, `src/qnm_kerr_reference.py`, `scripts/*hybrid*`, `scripts/eval_hybrid_*`, `configs/hybrid_*.yaml`, until paper integration phases.
- Write any paper prose before the matching evaluation phase passes its acceptance check.
- Add a sixth method, a second oracle, or any cross-cutting "infrastructure" not listed in §9.
- Make scope changes (extra ℓ, extra modes, extra Kerr regimes) without explicit user sign-off.

If any acceptance check fails I will stop and report, not patch around it.


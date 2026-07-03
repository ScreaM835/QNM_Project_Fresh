# QNM Project — Results Record (captured 2026-06-30)

> Written after the CSD3 HPC went offline indefinitely. The HPC held all trained
> models, corpora, and `report.json`/`per_sample.json` outputs, which are now
> inaccessible. **These numbers are reconstructed from the Copilot chat history**,
> which is our only surviving source of truth. We are continuing **locally** on a
> Windows laptop from a fresh clone of GitHub `main` (`f41bd9b`).

---

## 1. Kerr hybrid FNO — latest & best (decoupled, label-free)

**Thesis.** Replicate the *supervised* dual-win (good field **and** good QNM) using a
**cheap, label-free Richardson target** instead of fine-grid labels — because
"Richardson is cheaper than a fine solve" is the method's whole justification.

**Design ("decoupled").**
- **Model input / prior** = `N=101`, **2nd-order** coarse solve (`k8`). Deliberately
  coarse so it retains large QNM headroom.
- **Target / label** = **4th-order Richardson** of the finer pair:
  `psi_target = (16*u_401 - u_201)/15`, both rungs solved at 4th order. Field error
  0.04–0.21%, QNM 0.1–0.6%, cost ~0.32x a fine solve, **no fine solve in any label**.
- Decoupling = the input grid (101) is **separate** from the target's Richardson
  rungs (401/201).

**Corpus `phase_c_l4_decoupled`.** ell=4; Sobol over `a/M in [0,0.95]`, `r0 in [8,11]`,
`w in [1,1.5]`; train 1024 / val 128 / test 128. Grids: fine 801, k2=401 (o4),
k4=201 (o4), k8=101 (o2 prior). float32 storage. Training: best val MSE 1.39e-4 @
epoch 196/200, 7.7M params.

**Test results (medians, 128 held-out samples).** All values in %; "fine" = the
fine field's own extraction = practical floor. Richardson *target* field error = 0.026%.

| a/M       | n  | field prior→hyb | Mw prior→hyb (fine) | tau prior→hyb (fine) |
|-----------|----|-----------------|---------------------|----------------------|
| [0, 0.3]  | 41 | 51.2 → 0.70     | 2.92 → 3.43 (2.46)  | 3.53 → 9.83 (2.82)   |
| [0.3,0.6] | 40 | 59.1 → 0.71     | 1.31 → 0.92 (0.58)  | 3.98 → 8.54 (4.18)   |
| [0.6,0.8] | 27 | 65.7 → 0.86     | 1.05 → 0.56 (0.40)  | 1.75 → 3.42 (0.76)   |
| [0.8,0.95]| 20 | 73.6 → 1.08     | 8.16 → 0.68 (0.31)  | 21.42 → 8.62 (0.47)  |
| overall   |128 | 59.5 → 0.78     | 1.82 → 1.16 (0.64)  | 3.87 → 6.57 (2.03)   |

**Gates.** field_pass = **True** (0.78% << 5%). qnm_pass = **False** (Mw median 1.16% > 1%,
dragged by the low-spin bin where the *fine* field itself only reaches 2.46% — an
extraction-floor regime, no real headroom). overall_pass = False.

**Headline wins.**
- Field: ~60–67x at every spin, uniform.
- Mw: rescued where there is headroom — high spin **8.2% → 0.68%** (~ fine's 0.31%).
- **Floor theory confirmed:** target field 0.026% << hybrid 0.78% → the model hit its
  **input-limited floor (~0.77%)** and did *not* saturate the target. The cheap label
  was "good enough" precisely because it beat the floor.

**Three-way comparison (the point of the exercise).**

| teacher                          | field   | Mw (high spin) | tau   | fine solve? | cost   |
|----------------------------------|---------|----------------|-------|-------------|--------|
| old order-2 Richardson (coupled) | 15% ❌  | 1.80%          | 8.83% | no          | ~0.08x |
| supervised (fine labels)         | 0.77%   | 0.80%          | 6.28% | **yes**     | 1.0x   |
| decoupled order-4 Richardson     | 0.78%   | 0.79%          | 6.57% | **no**      | ~0.32x |

→ The decoupled run **matches supervised on field and Mw, label-free, ~3x cheaper
than a fine solve.**

---

## 2. The tau problem — diagnosed; fix designed but NOT yet implemented

tau is the one weak axis (hybrid 6.57% vs fine 2.03%; worse than the prior at low/mid spin).

- **tau extraction is envelope-based** (M1 = log-envelope slope, M2 = damped-cosine NLS),
  *not* 1/frequency. The pipeline is consistent across prior/hybrid/fine — not a bug.
- **Root cause = dynamic range, not noise.** The late ringdown is *clean, learnable*
  signal (the Richardson **label** matches fine to 0.0–0.2% there), but the FNO never
  learns it: amplitude-weighted L2 gives it ~1e-10 of the loss, and the single per-sample
  scale `s = rms(|prior|)` is peak-dominated, so the late ringdown sits at ~1e-5 in
  normalized space — invisible to the network and the loss.
- **Field-error localization** (hybrid vs fine, by time window at scri): global 0.7–1.1%,
  early ring 1–2%, **late ring [8–14] tau_ref = 79–175%**.
- Amplitude decay (|psi|/peak): 8 tau_ref 4.7e-3, 10 5.9e-4, 12 8.7e-5, 14 1.2e-5
  (still ~280x above the float32 floor 1e-7).

**Proposed fix (designed, not built).** Per-time **envelope normalization**
`w(tau) = max(rms_sigma|prior(tau,.)|, eps*peak)`; normalize I/O by it; standard L2 in
that space. Principled floor **eps ~ 1e-5**, derived as `eta_float32 / delta_model =
1.2e-7 / 1e-2`. The SW `|grad prior|` gate would *backfire* (it suppresses the
low-gradient late ringdown). Generalization is safe because `w` is computed from the
prior, which is recomputed for every input.

---

## 3. Earlier Kerr runs (context, superseded by the decoupled run)

- **ell=2 (`phase_c`):** field prior 8.35% → hybrid 0.79% (~10x); QNM net-negative
  (ensemble: prior 0.119% → hybrid 0.510%) — prior already at the QNM floor, **no
  headroom at ell=2**. Field win only.
- **ell=4 `phase_c_l4_n101`, order-2 Richardson:** field 15% (FAIL — student saturated a
  bad target). Motivated the order-4 fix.
- **ell=4 `phase_c_l4_n101`, supervised:** field 0.77%, Mw high-spin 18.3 → 0.80% — the
  dual-win, but uses fine labels (the thing the decoupled run replaced).

---

## 4. Schwarzschild / Zerilli — FNO, hybrid, PINN

- **Pure FNO surrogate** (`src/fno_dataset.py`): learns `G:(Phi0, Pi0, V, M) -> Phi(x,t)`,
  4-channel broadcast input, supervised (+ PDE-residual/physics terms on the PINN /
  inverse side).
- **SW hybrid FNO** (`src/hybrid_fno.py`): 5-channel, with a multiplicative `|grad prior|`
  output gate `g = |grad prior| / (|grad prior| + scale)`, `field = prior + g*FNO`,
  confining the correction to wavefronts.

---

## 5. Code lost with the HPC (re-implemented locally 2026-06-30 from chat memory)

Not in clone `f41bd9b` (verified by grep); re-created locally:

- `kerr/src/fd_stencils.py`: `d1_4` (4th-order central FD; Fornberg one-sided closures
  at the 2 boundary nodes each side; needs n>=5).
- `kerr/src/dissipation.py`: `ko_dissipation_6` (6th-difference KO, +sigma/64 * D6,
  zero outer 3 cells).
- `kerr/src/kerr_dataset.py`: `GRID_ORDER` module global + `order` param in
  `evolve_full_field` + per-grid order in `_evolve_one`.
- `kerr/src/hybrid_data_pipe.py`: `assemble(W_prior, prior_key='k8', richardson_p)`
  generalization + generic `load_split` (loads every grid present, incl. k8).
- `kerr/scripts/build_kerr_dataset_lscan.py`: `set_grid_order` + `--grid-order` CLI.
- `kerr/scripts/train_eval_hybrid_kerr.py`: `cfg.data.prior_grid` + `richardson_p`
  wiring -> builds `W_prior`, passes to `assemble`.
- `kerr/configs/hybrid_kerr_l4_decoupled.yaml`: `prior_grid=k8`, `richardson_p=4`.

Also lost (cannot reconstruct — data, not code): all trained models, corpora,
`report.json`, `per_sample.json`, figures.

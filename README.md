# Hybrid Neural Methods for Black-Hole Ringdown

This repository accompanies the project report
*Hybrid Fourier Neural Operators for Time-Domain Black-Hole Perturbations
and Quasinormal-Mode Extraction*. It reproduces the time-domain PINN calculation of Patel,
Aykutalp and Laguna, improves that PINN, and develops a hybrid Fourier neural
operator (hFNO) for Schwarzschild and Kerr perturbations.

## Scientific Scope

The report contains four numerical tests:

1. A faithful PINN reproduction on the Schwarzschild `l=2` Zerilli and
   Regge-Wheeler equations.
2. An enhanced Zerilli PINN using residual-greedy sampling and a longer
   L-BFGS phase.
3. A Schwarzschild hFNO that corrects a coarse finite-difference trajectory
   towards a label-free Richardson target across `(M, x0, sigma)`.
4. The hybrid framework extended to the complex hyperboloidal Teukolsky
   field and retrained across `(a/M, r0, w)` for the `(l,m,n)=(4,2,0)` mode.

QNMs are extracted from the PINN, hFNO and fine-reference fields using
`src/qnm.py` and `kerr/src/qnm_ensemble_kerr.py`. The coarse Kerr solve is
reported only as the hFNO input and field-accuracy baseline. QNM errors are
benchmarked against Leaver values supplied directly or through the `qnm`
package.

## Main Results

| Test | Principal result |
|---|---|
| Faithful PINN reproduction | RL2 `2.61%` (Zerilli) and `2.67%` (Regge-Wheeler), versus `28.06%` and `23.59%` reported by Patel et al. |
| Enhanced PINN | RL2 `0.58%`; best-of-suite QNM errors `0.029%` in frequency and `1.99%` in damping time |
| Schwarzschild hFNO | Mean RL2 `13.39% -> 1.85%`; field MSE reduced by `46x` over 100 test configurations |
| Kerr hFNO | Median RL2 `59.5% -> 0.78%`; high-spin hFNO frequency error `0.38%` over 128 test configurations |

For Kerr, the retrospective Leaver-closest errors are `0.33%` in frequency
and `1.52%` in damping time for the hFNO fields, compared with `0.05%` and
`0.27%` for the fine fields. The report also gives every extractor separately
because these summary values depend on post-selection.

## Environment

```text
python -m venv .venv
```

Activate the environment, then install dependencies:

```text
pip install -r requirements.txt
```

On Windows PowerShell, PINN runs also require UTF-8 console output:

```powershell
$env:DDE_BACKEND = "pytorch"
$env:PYTHONIOENCODING = "utf-8"
```

Run the small end-to-end check before any full experiment:

```text
python scripts/run_pinn.py --config configs/quick_test.yaml
python scripts/extract_qnm.py --config configs/quick_test.yaml --source fd
```

## Schwarzschild PINNs

```text
python scripts/run_pinn.py --config configs/zerilli_l2_paper.yaml
python scripts/run_pinn.py --config configs/regge_wheeler_l2_paper.yaml
python scripts/run_pinn.py --config configs/zerilli_l2_greedy_f03_lbfgs30k.yaml
```

These commands generate the FD comparison, trained PINN fields, metrics and
paper-style plots under `outputs/pinn/`. QNM extraction can be rerun with
`scripts/extract_qnm.py`; use `--help` for the optional ESPRIT and plateau
scans.

## Schwarzschild hFNO

Build the two nested-grid corpora:

```text
python scripts/build_hybrid_dataset.py --config configs/hybrid_sw_dataset_t100.yaml --k 4 --out outputs/hybrid/dataset_sw_k4_t100.npz
python scripts/build_hybrid_dataset.py --config configs/hybrid_sw_dataset_t100.yaml --k 2 --out outputs/hybrid/dataset_sw_k2_t100.npz
```

Train, evaluate and regenerate figures:

```text
python scripts/train_hybrid_fno.py --config configs/hybrid_sw_gate_s1em3_t100.yaml
python scripts/eval_hybrid_sw.py --config configs/hybrid_sw_gate_s1em3_t100.yaml --xq 10 --t_end 100
python scripts/eval_hybrid_protocol1.py --config configs/hybrid_sw_gate_s1em3_t100.yaml --dataset-config configs/hybrid_sw_dataset_t100.yaml --t_end_m4 100
python scripts/make_hybrid_paper_figs.py --config configs/hybrid_sw_gate_s1em3_t100.yaml --dataset-cfg configs/hybrid_sw_dataset_t100.yaml
```

The fine evolution has `1001 x 1000` spatial point/time-step updates and the
deployed `k=4` prior has `251 x 250`, a finite-difference point-update ratio of
`15.95`. This is not an end-to-end runtime measurement because interpolation
and FNO evaluation are additional deployment costs.

## Kerr hFNO

The Kerr pipeline and its artifact requirements are documented in
[`kerr/README.md`](kerr/README.md). The principal entry points are:

```text
python kerr/scripts/build_kerr_dataset_lscan.py --help
python kerr/scripts/train_eval_hybrid_kerr.py --config kerr/configs/hybrid_kerr_l4_decoupled.yaml
```

## Artifact Availability

The tracked Schwarzschild hFNO directory
`outputs/hybrid/fno_sw_gate_s1em3_t100/` contains its checkpoints, metrics and
figures. The tracked Kerr directory `kerr/outputs/_run2_download/` contains the
reported `history.json`, `report.json`, `per_sample.json` and figure set.

The full Kerr training/validation/test corpora and `model.pt` are too large for
this repository and are not present in the current checkout. They can be rebuilt
with `kerr/scripts/build_kerr_dataset_lscan.py` and
`kerr/scripts/train_eval_hybrid_kerr.py`. Consequently, the reported Kerr
metrics and figures are auditable from the tracked JSON files, while complete
field regeneration requires rebuilding or separately supplying those large
artifacts.

## Build the Report

From `paper/`, run:

```text
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The separate executive summary is built with:

```text
pdflatex executive_summary.tex
```

## Use of Auto-Generation Tools

### GitHub Copilot

GitHub Copilot was used supportively for code completion, debugging and
documentation, and for drafting assistance, proofreading and wording
suggestions during preparation of the report.

### ChatGPT

ChatGPT was consulted occasionally for technical explanations and plotting
suggestions. All suggestions from auto-generation tools were reviewed by the
author and, where applicable, tested and revised before inclusion.

## Repository Layout

- `src/`: Schwarzschild solvers, PINNs, neural operators and extractors
- `scripts/`: dataset, training, evaluation and plotting entry points
- `configs/`: reproducible Schwarzschild experiment configurations
- `kerr/`: hyperboloidal Teukolsky solver and Kerr hFNO pipeline
- `outputs/`: tracked report metrics, figures and selected checkpoints
- `paper/`: report, executive summary and bibliography
- `results/`: consolidated metric records used for consistency checks



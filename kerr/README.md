# Kerr Hyperboloidal Teukolsky hFNO

This directory contains the Kerr extension reported in the project. It evolves
the complex spin `s=-2` Teukolsky field on a horizon-penetrating hyperboloidal
slice and applies the same coarse-prior-plus-residual framework used by the
Schwarzschild hFNO.

The reported experiment targets the fundamental `(l,m,n)=(4,2,0)` mode across
`a/M in [0,0.95]`. It is an adaptation and separate training of the hybrid
framework, not transfer of Schwarzschild network weights.

## Numerical Pipeline

Each Sobol sample `(a/M, r0, w)` is evolved to `tau/M=220` on nested spatial
grids:

| Role | Grid | Spatial discretisation |
|---|---:|---|
| Fine evaluation field | `N=801` | fourth-order centred plus sixth-difference KO |
| Richardson rung | `N=401` | fourth-order centred plus sixth-difference KO |
| Richardson rung | `N=201` | fourth-order centred plus sixth-difference KO |
| Deployment prior | `N=101` | second-order centred plus fourth-difference KO |

All grids use RK4 time integration with CFL safety `0.4`, KO strength `0.2`,
and a common stored cadence `Delta tau=0.25M`. Initial data are a
time-symmetric Gaussian of amplitude one with `r0/M in [8,11]` and
`w/M in [1,1.5]`.

The label-free target is

```text
u_star = (16 * upsample(u_401) - upsample(u_201)) / 15
```

The hFNO receives the upsampled `N=101` complex prior, the initial pulse and
the broadcast spin. It predicts a two-channel real/imaginary residual. The
fine `N=801` field is used only for evaluation.

## Reported Run

Configuration: `configs/hybrid_kerr_l4_decoupled.yaml`

- Data split: `1024/128/128`, disjoint Sobol draws with seed `0`
- Training seed: `1234`
- FNO: four layers, modes `(64,24)`, hidden width `48`
- Trainable parameters: approximately `7.7 million`
- Training: 200 Adam epochs

Held-out test medians:

| Field quantity | Prior | hFNO | Richardson reference |
|---|---:|---:|---:|
| Relative L2 error | `59.5%` | `0.78%` | `0.008%` |

QNM extraction is reported for the hFNO and fine-evaluation fields only:

| Extracted quantity | hFNO | Fine reference |
|---|---:|---:|
| Frequency error | `0.33%` | `0.05%` |
| Damping-time error | `1.52%` | `0.27%` |

For the high-spin bin `a/M in [0.8,0.95]`, the hFNO frequency error is
`0.38%`. These QNM values are retrospective Leaver-closest summaries. The
report gives the full per-extractor results and finite-result counts.

## Build the Corpus

Run from the repository root. The following command builds the training split
with the exact grid ladder used in the report:

```text
python kerr/scripts/build_kerr_dataset_lscan.py --ell 4 --m 2 --split train --seed 0 --ks 2 4 8 --coarse-n "2:401,4:201,8:101" --grid-order "1:4,2:4,4:4,8:2" --out kerr/outputs/phase_c_l4_decoupled/dataset_train.npz --workers 1
```

Repeat with `--split val` and `--split test`, changing the output filename.
On Windows, use `--workers 1`; Linux runs may use multiple workers.

## Train and Evaluate

```text
python kerr/scripts/train_eval_hybrid_kerr.py --config kerr/configs/hybrid_kerr_l4_decoupled.yaml
```

To reevaluate an existing checkpoint without retraining:

```text
python kerr/scripts/train_eval_hybrid_kerr.py --config kerr/configs/hybrid_kerr_l4_decoupled.yaml --eval-only
```

The job writes `model.pt`, `history.json`, `report.json`, `per_sample.json` and
the paper-style figures beneath the configured output directory.

## Tracked and External Artifacts

`outputs/_run2_download/` contains the reportable `history.json`,
`report.json`, `per_sample.json` and figure set used by the manuscript.

The full `dataset_train.npz`, `dataset_val.npz`, `dataset_test.npz` and
`model.pt` are not tracked because of their size. Complete field-level figure
regeneration therefore requires rebuilding the corpus and model with the
commands above or supplying those artifacts separately. The tracked JSON files
are sufficient to audit every aggregate number in the Kerr Results table.

## Layout

- `src/teukolsky_minimal_gauge.py`: complex hyperboloidal evolution system
- `src/kerr_dataset.py`: nested-grid corpus and Sobol splits
- `src/hybrid_data_pipe.py`: spatial upsampling and Richardson assembly
- `src/hybrid_fno.py`: complex residual FNO
- `src/qnm_ensemble_kerr.py`: Kerr QNM extraction suite
- `scripts/build_kerr_dataset_lscan.py`: corpus builder
- `scripts/train_eval_hybrid_kerr.py`: train, evaluate and plot entry point
- `scripts/replot_kerr_paper_figs.py`: regenerate figures from saved artifacts
- `configs/hybrid_kerr_l4_decoupled.yaml`: reported configuration

To regenerate the paper figures from separately supplied artifacts:

```text
python kerr/scripts/replot_kerr_paper_figs.py --dataset <dataset_test.npz> --model <model.pt>
```

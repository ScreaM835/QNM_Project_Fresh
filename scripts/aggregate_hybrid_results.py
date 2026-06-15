#!/usr/bin/env python3
"""Aggregate all reportable HYBRID (coarse-FD + FNO residual) results into one JSON.

Mirrors scripts/aggregate_results.py (which covers the forward/inverse PINN) but
for the hybrid super-resolution runs. Reads each run's already-produced
``outputs/hybrid/<run>/eval/summary.json`` (field rel-L2/MSE vs fine-FD truth and
the full base/hybrid/fine QNM percent-error tables for M1-M5 at xq in {2,10}) plus
``outputs/hybrid/<run>/history.json`` (training trajectory + provenance), and
writes ``outputs/hybrid/reportable_hybrid_results.json``.

Nothing is re-run; values are copied straight from the eval artifacts so the JSON
can be cross-checked against the figures and the paper tables.

Usage:
    python scripts/aggregate_hybrid_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

# --------------------------------------------------------------
# Truth values (Leaver continued-fraction Schwarzschild ell=2 fundamental),
# identical to scripts/aggregate_results.py and the eval script.
OMEGA_TRUE = 0.3737
TAU_TRUE = 11.241
M_TRUE = 1.0

REPO = Path(__file__).resolve().parents[1]
HYB = REPO / "outputs" / "hybrid"
CONFIGS = REPO / "configs"

# Runs to aggregate, in reporting order. The label-free Richardson run is the
# headline; the supervised run is its A/B ceiling on the SAME k4 prior; the
# remaining three are the earlier supervised hybrids kept for context.
RUNS = [
    ("richardson_labelfree", "fno_sw_richardson",
     "configs/hybrid_sw_richardson.yaml",
     "Label-free Richardson target (4*up2-up4)/3 on the k4 prior; no fine FD in loss."),
    ("richardson_supervised", "fno_sw_richardson_supervised",
     "configs/hybrid_sw_richardson_supervised.yaml",
     "A/B control: fine-FD label on the SAME k4 prior; ceiling for the label-free run."),
    ("k2_h64", "fno_sw_k2_h64",
     "configs/hybrid_sw_train_k2_h64.yaml",
     "Supervised hybrid on the k2 prior (hidden=64)."),
    ("drp7_k2_h64", "fno_sw_drp7_k2_h64",
     "configs/hybrid_sw_drp7_k2_h64.yaml",
     "Supervised hybrid on the k2 prior built with the drp7 coarse stencil."),
    ("drp7_k4", "fno_sw_drp7_k4",
     "configs/hybrid_sw_drp7_k4.yaml",
     "Supervised hybrid on the k4 prior built with the drp7 coarse stencil."),
]

METHODS = ["M1", "M2", "M3", "M4", "M5"]
XQS = ["xq2", "xq10"]
SOURCES = ["base", "hyb", "fine"]  # bare prior / hybrid / fine-FD floor


def load_json(path: Path):
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _ratio(num, den):
    try:
        num = float(num); den = float(den)
    except (TypeError, ValueError):
        return None
    if den == 0 or den != den:
        return None
    return num / den


def collect_field(summary: dict) -> dict:
    """Field accuracy vs the held-out fine-FD truth (eval-only, never in loss)."""
    f = summary.get("field", {})
    rl2_h = f.get("rl2_hybrid")
    rl2_b = f.get("rl2_baseline")
    mse_h = f.get("mse_hybrid")
    mse_b = f.get("mse_baseline")
    return {
        "n_eval": summary.get("n_eval"),
        "rl2_hybrid": rl2_h,
        "rl2_baseline_bare_prior": rl2_b,
        "rl2_improvement_factor": _ratio(rl2_b, rl2_h),
        "mse_hybrid": mse_h,
        "mse_baseline_bare_prior": mse_b,
        "mse_improvement_factor": _ratio(mse_b, mse_h),
    }


def _qnm_nested(flat: dict) -> dict:
    """Reshape the flat ``xq2_hyb_M4_omega_pct_err`` keys into a nested dict
    xq -> source -> method -> {omega_pct_err, tau_pct_err}. Missing keys are
    copied as None so every (xq, source, method, quantity) slot is present."""
    out = {}
    for xq in XQS:
        out[xq] = {}
        for src in SOURCES:
            out[xq][src] = {}
            for M in METHODS:
                out[xq][src][M] = {
                    "omega_pct_err": flat.get(f"{xq}_{src}_{M}_omega_pct_err"),
                    "tau_pct_err": flat.get(f"{xq}_{src}_{M}_tau_pct_err"),
                }
    return out


def collect_qnm(summary: dict) -> dict:
    """Full three-way (base/hyb/fine) QNM percent-error tables.

    Keeps BOTH the flat per-statistic dicts (median/mean/max, exactly as written
    by eval_hybrid_sw.py, so nothing is lost) AND a nested median view for easy
    reading and for the clean-window headline below.
    """
    med = summary.get("qnm_pct_err_median", {})
    mean = summary.get("qnm_pct_err_mean", {})
    mx = summary.get("qnm_pct_err_max", {})
    return {
        "window": summary.get("qnm_window"),
        "theory": summary.get("qnm_theory"),
        "median_nested": _qnm_nested(med),
        "flat_median": med,
        "flat_mean": mean,
        "flat_max": mx,
    }


def collect_training(history: dict | None) -> dict | None:
    """First (epoch-0 ~ bare prior), best, and last field val rel-L2 plus the
    per-phase epoch counts, wall time and provenance from history.json."""
    if history is None:
        return None
    recs = history.get("history") if isinstance(history, dict) else history
    if not recs:
        return None
    vl = [r.get("val_l2_ratio") for r in recs if r.get("val_l2_ratio") is not None]
    n_adam = sum(1 for r in recs if r.get("phase") == "adam")
    n_lbfgs = sum(1 for r in recs if r.get("phase") == "lbfgs")
    wall = sum(float(r.get("wall_s", 0.0)) for r in recs)
    return {
        "config_path": history.get("config_path"),
        "dataset_meta": history.get("dataset_meta"),
        "model_params": history.get("model_params"),
        "best_val_mse": history.get("best_val_mse"),
        "epochs_adam": n_adam,
        "epochs_lbfgs": n_lbfgs,
        "wall_s_total": wall,
        "val_l2_ratio_first": vl[0] if vl else None,
        "val_l2_ratio_best": min(vl) if vl else None,
        "val_l2_ratio_last": vl[-1] if vl else None,
    }


def read_target_mode_and_fno(cfg_path: Path):
    """target_mode (default 'supervised') and the FNO hyperparameters."""
    cfg = load_json_yaml(cfg_path)
    if cfg is None:
        return None, None
    ds = cfg.get("dataset", {})
    target_mode = str(ds.get("target_mode", "supervised"))
    fno = cfg.get("fno", {})
    return target_mode, fno or None


def load_json_yaml(path: Path):
    """Minimal YAML loader via the project's config loader (kept import-local so
    this script has no hard dependency when a config is absent)."""
    if not path.exists():
        return None
    try:
        import yaml
        with path.open() as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def clean_window_headline(qnm_nested_median: dict) -> dict:
    """The xq=2 (clean window) M4/M5 omega & tau for bare prior / hybrid / fine
    floor -- the numbers that show the QNM is already at the floor and that the
    hybrid neither beats nor (robustly) harms it."""
    xq2 = qnm_nested_median.get("xq2", {})
    out = {}
    for M in ("M4", "M5", "M2"):
        out[M] = {
            "omega_pct_err": {
                "bare_prior": xq2.get("base", {}).get(M, {}).get("omega_pct_err"),
                "hybrid": xq2.get("hyb", {}).get(M, {}).get("omega_pct_err"),
                "fine_floor": xq2.get("fine", {}).get(M, {}).get("omega_pct_err"),
            },
            "tau_pct_err": {
                "bare_prior": xq2.get("base", {}).get(M, {}).get("tau_pct_err"),
                "hybrid": xq2.get("hyb", {}).get(M, {}).get("tau_pct_err"),
                "fine_floor": xq2.get("fine", {}).get(M, {}).get("tau_pct_err"),
            },
        }
    return out


def acceptance_gate(qnm_nested_median: dict) -> dict:
    """Built-in eval gate: hybrid M4 omega median <= 2x the fine-FD M4 omega
    floor (computed here from the medians since the eval script only printed it)."""
    xq2 = qnm_nested_median.get("xq2", {})
    hyb = xq2.get("hyb", {}).get("M4", {}).get("omega_pct_err")
    fine = xq2.get("fine", {}).get("M4", {}).get("omega_pct_err")
    passed = None
    if hyb is not None and fine is not None:
        passed = bool(hyb <= 2.0 * fine)
    return {
        "rule": "hybrid M4 omega median pct-err <= 2 x fine-FD M4 omega floor (xq=2)",
        "hybrid_M4_omega_pct_err": hyb,
        "fine_M4_omega_pct_err": fine,
        "passed": passed,
    }


def main():
    out = {
        "truth": {
            "M_true": M_TRUE,
            "omega_true": OMEGA_TRUE,
            "tau_true": TAU_TRUE,
            "source": "Leaver 1985 continued-fraction Schwarzschild l=2 fundamental",
        },
        "notes": (
            "Hybrid = coarse FD prior + FNO residual super-resolution. Field rel-L2 "
            "is measured against the held-out fine-FD field (eval-only, never in the "
            "loss). 'baseline' = the bare upsampled coarse prior (does the FNO help?). "
            "'fine' = the fine-FD discretisation's own QNM floor. QNM judged at xq=2 "
            "(clean window); xq=10 carries a documented tail-contamination artifact "
            "(see fine-floor tau there). All percent-errors are vs the Leaver truth."
        ),
        "runs": {},
    }

    for label, run, cfg_rel, desc in RUNS:
        run_dir = HYB / run
        summary = load_json(run_dir / "eval" / "summary.json")
        if summary is None:
            print(f"  [skip] {label}: no eval/summary.json")
            continue
        history = load_json(run_dir / "history.json")
        target_mode, fno = read_target_mode_and_fno(REPO / cfg_rel)

        qnm = collect_qnm(summary)
        nested_med = qnm["median_nested"]
        figs_dir = run_dir / "figs"
        figs = sorted(p.name for p in figs_dir.glob("*.png")) if figs_dir.is_dir() else []

        out["runs"][label] = {
            "run": run,
            "out_dir": str(run_dir.relative_to(REPO)),
            "config": cfg_rel,
            "description": desc,
            "target_mode": target_mode,
            "fno": fno,
            "field": collect_field(summary),
            "qnm_clean_window_xq2_headline": clean_window_headline(nested_med),
            "qnm_acceptance_gate": acceptance_gate(nested_med),
            "qnm": qnm,
            "training": collect_training(history),
            "figures": figs,
        }
        f = out["runs"][label]["field"]
        print(f"  [ok] {label:24s} field rl2 {f['rl2_baseline_bare_prior']:.4f}"
              f" -> {f['rl2_hybrid']:.5f}  ({f['rl2_improvement_factor']:.1f}x)")

    out_path = HYB / "reportable_hybrid_results.json"
    with out_path.open("w") as fh:
        json.dump(out, fh, indent=2, sort_keys=False, default=str)
    print(f"Wrote {out_path}")
    print(f"  runs aggregated: {len(out['runs'])}")


if __name__ == "__main__":
    main()

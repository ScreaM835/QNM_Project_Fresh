#!/usr/bin/env python3
"""Aggregate all reportable results for the paper into a single JSON.

Reads the per-run outputs already produced by run_pinn.py and
extract_qnm.py and writes outputs/reportable_results.json.

The keys mirror the tables and equations of paper/sections/results.tex
so values can be cross-checked without re-running anything.
"""

from __future__ import annotations

import json
from pathlib import Path

# --------------------------------------------------------------
# Truth values (Leaver continued-fraction Schwarzschild ell=2 fundamental)
# Same constants used by extract_qnm.py
OMEGA_TRUE = 0.3737
TAU_TRUE = 11.241
M_TRUE = 1.0
M_INIT = 1.2

REPO = Path(__file__).resolve().parents[1]
PINN = REPO / "outputs" / "pinn"
QNM = REPO / "outputs" / "qnm"

FORWARD_RUN = "zerilli_l2_greedy_f03_lbfgs30k"

INVERSE_VARIANTS = {
    "baseline": "zerilli_l2_inverse_qnm",
    "A_tring18": "zerilli_l2_inverse_qnm_tring18",
    "B_lring100": "zerilli_l2_inverse_qnm_lring100",
    "C_nring1000": "zerilli_l2_inverse_qnm_nring1000",
    "D_combo": "zerilli_l2_inverse_qnm_combo",
    "E_2mode": "zerilli_l2_inverse_qnm_2mode",
}


def load_json(path: Path):
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def pct_err(value, truth):
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    return 100.0 * abs(v - truth) / abs(truth)


def collect_qnm(run_dir: Path):
    """Pick up the five extraction-method outputs from a qnm/<run>/ dir."""
    out = {}

    m1 = load_json(run_dir / "pinn_method1.json")
    if m1 is not None:
        out["M1_fft_loglinear"] = {
            "omega": m1.get("omega"),
            "tau": m1.get("tau"),
            "omega_pct_err": m1.get("omega_pct_err"),
            "tau_pct_err": m1.get("tau_pct_err"),
            "window": "[10,50]M",
        }

    m2 = load_json(run_dir / "pinn_method2.json")
    if m2 is not None:
        out["M2_nls_wide"] = {
            "omega": m2.get("omega"),
            "tau": m2.get("tau"),
            "omega_pct_err": m2.get("omega_pct_err"),
            "tau_pct_err": m2.get("tau_pct_err"),
            "window": "[10,50]M",
        }

    m2p = load_json(run_dir / "pinn_method2_on_m4_plateau.json")
    if m2p is not None:
        out["M2_nls_on_m4_plateau"] = {
            "omega": m2p.get("omega"),
            "tau": m2p.get("tau"),
            "omega_pct_err": m2p.get("omega_pct_err"),
            "tau_pct_err": m2p.get("tau_pct_err"),
            "window": m2p.get("window") or "M4 plateau window",
        }

    m3 = load_json(run_dir / "pinn_method3_esprit.json")
    if m3 is not None:
        amps = m3.get("all_amps") or []
        out["M3_esprit"] = {
            "K": m3.get("K"),
            "omega": m3.get("omega"),
            "tau": m3.get("tau"),
            "omega_pct_err": m3.get("omega_pct_err"),
            "tau_pct_err": m3.get("tau_pct_err"),
            "max_amp": max(amps) if amps else None,
            "ill_conditioned": bool(amps and max(amps) < 1e-10),
        }

    m4 = load_json(run_dir / "pinn_method4_two_mode.json")
    if m4 is not None:
        out["M4_two_mode_plateau"] = {
            "omega": m4.get("omega"),
            "omega_std": m4.get("omega_std"),
            "tau": m4.get("tau"),
            "tau_std": m4.get("tau_std"),
            "omega_pct_err": m4.get("omega_pct_err"),
            "tau_pct_err": m4.get("tau_pct_err"),
            "t0_plateau_min": m4.get("t0_plateau_min"),
            "t0_plateau_max": m4.get("t0_plateau_max"),
        }

    m5 = load_json(run_dir / "pinn_method5_2d_scan.json")
    if m5 is not None:
        out["M5_two_mode_2d_plateau"] = {
            "omega": m5.get("omega"),
            "omega_std": m5.get("omega_std"),
            "tau": m5.get("tau"),
            "tau_std": m5.get("tau_std"),
            "omega_pct_err": m5.get("omega_pct_err"),
            "tau_pct_err": m5.get("tau_pct_err"),
            "t0_plateau_min": m5.get("t0_plateau_min"),
            "t0_plateau_max": m5.get("t0_plateau_max"),
            "te_plateau_min": m5.get("te_plateau_min"),
            "te_plateau_max": m5.get("te_plateau_max"),
        }

    return out


def collect_inverse_template(param_history_path: Path):
    """Final values of the learnable scalars at end of training."""
    ph = load_json(param_history_path)
    if ph is None:
        return None

    def last(key):
        v = ph.get(key)
        if not v:
            return None
        return float(v[-1])

    if "omega_history" in ph:
        # single-mode template (baseline/A/B/C/D)
        return {
            "kind": "single_mode",
            "M_learned": last("M_history"),
            "omega_learned": last("omega_history"),
            "tau_learned": last("tau_history"),
            "A_ring": last("A_ring_history"),
            "phi0": last("phi0_history"),
        }
    if "omega0_history" in ph:
        # two-mode template (E)
        return {
            "kind": "two_mode",
            "M_learned": last("M_history"),
            "mode0": {
                "omega_learned": last("omega0_history"),
                "tau_learned": last("tau0_history"),
                "A": last("A0_history"),
                "phi0": last("phi0_history"),
            },
            "mode1": {
                "omega_learned": last("omega1_history"),
                "tau_learned": last("tau1_history"),
                "A": last("A1_history"),
                "phi0": last("phi0_history"),
            },
        }
    return None


def main():
    out = {
        "truth": {
            "M_true": M_TRUE,
            "M_init": M_INIT,
            "omega_true": OMEGA_TRUE,
            "tau_true": TAU_TRUE,
            "source": "Leaver 1985 continued-fraction Schwarzschild l=2 fundamental",
        },
    }

    # ---- forward field metrics (Eq. 14) ----
    fwd_metrics = load_json(PINN / FORWARD_RUN / "metrics.json") or {}
    out["forward_field_metrics"] = {
        "run": FORWARD_RUN,
        "RMSD": fwd_metrics.get("rmsd") or fwd_metrics.get("RMSD"),
        "MAD": fwd_metrics.get("mad") or fwd_metrics.get("MAD"),
        "RL2": fwd_metrics.get("rl2") or fwd_metrics.get("RL2"),
    }

    # ---- forward QNM extraction (Table tab:fwd-qnm) ----
    out["forward_qnm"] = {
        "run": FORWARD_RUN,
        "methods": collect_qnm(QNM / FORWARD_RUN),
    }

    # ---- inverse: per-variant (Tables tab:inv-summary, tab:inv-learned) ----
    inverse = {}
    for label, run in INVERSE_VARIANTS.items():
        pinn_dir = PINN / run
        qnm_dir = QNM / run

        metrics = load_json(pinn_dir / "metrics.json") or {}
        rmsd = metrics.get("rmsd") or metrics.get("RMSD")
        mad = metrics.get("mad") or metrics.get("MAD")
        rl2 = metrics.get("rl2") or metrics.get("RL2")

        template = collect_inverse_template(pinn_dir / "param_history.json")
        m_learned = template.get("M_learned") if template else None

        qnm_methods = collect_qnm(qnm_dir)

        # Headline per Sec.3.6 reporting convention:
        #   omega = M4 plateau mean (with sigma_omega as quality flag)
        #   tau   = M2 wide-window
        m4 = qnm_methods.get("M4_two_mode_plateau") or {}
        m2w = qnm_methods.get("M2_nls_wide") or {}

        inverse[label] = {
            "run": run,
            "field_metrics": {"RMSD": rmsd, "MAD": mad, "RL2": rl2},
            "mass": {
                "M_learned": m_learned,
                "M_true": M_TRUE,
                "M_pct_err": pct_err(m_learned, M_TRUE),
            },
            "headline_omega_M4": {
                "omega": m4.get("omega"),
                "sigma_omega": m4.get("omega_std"),
                "omega_pct_err": m4.get("omega_pct_err"),
                "t0_plateau_min": m4.get("t0_plateau_min"),
                "t0_plateau_max": m4.get("t0_plateau_max"),
                "plateau_failed": (
                    m4.get("omega") is None
                    or (m4.get("omega") != m4.get("omega"))  # NaN
                    or (
                        m4.get("omega_std") is not None
                        and m4.get("omega") not in (None, 0)
                        and abs(m4["omega_std"] / m4["omega"]) > 0.4
                    )
                ),
            },
            "headline_tau_M2_wide": {
                "tau": m2w.get("tau"),
                "tau_pct_err": m2w.get("tau_pct_err"),
                "window": "[10,50]M",
            },
            "learned_template": template,
            "qnm_methods": qnm_methods,
        }

    out["inverse"] = inverse

    # ---- A_ring--tau degeneracy diagnostic (Table tab:tau-degeneracy) ----
    degeneracy = {}
    for label, payload in inverse.items():
        tmpl = payload["learned_template"] or {}
        if tmpl.get("kind") == "single_mode":
            tau_learned = tmpl.get("tau_learned")
        elif tmpl.get("kind") == "two_mode":
            tau_learned = (tmpl.get("mode0") or {}).get("tau_learned")
        else:
            tau_learned = None
        tau_m2_wide = payload["headline_tau_M2_wide"]["tau"]
        bias_gap = (
            abs(tau_learned - tau_m2_wide)
            if (tau_learned is not None and tau_m2_wide is not None)
            else None
        )
        degeneracy[label] = {
            "tau_learned": tau_learned,
            "tau_M2_wide": tau_m2_wide,
            "tau_true": TAU_TRUE,
            "bias_gap": bias_gap,
        }
    out["tau_degeneracy"] = degeneracy

    # ---- write ----
    out_path = REPO / "outputs" / "reportable_results.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2, sort_keys=False, default=str)
    print(f"Wrote {out_path}")
    print(f"  forward run: {FORWARD_RUN}")
    print(f"  inverse variants: {len(inverse)}")


if __name__ == "__main__":
    main()

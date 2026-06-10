"""B.9 — full Kerr spin sweep of the fundamental (2,2,0).

Runs the validated KV.3 fundamental extractor (the COMPLEX single-mode 2D
plateau scan, observer self-selection + median gate; a=0 uses the real path)
across a dense spin grid a/M in [0, 0.95], each compared to the `qnm` package.
Produces the appendix curve + table (saved to an .npz) and the population gate.

This is NOT a new physics path: every per-spin extraction is exactly the B.8
`run_fundamental` machinery, imported verbatim from `kv3_qnm` so the sweep and
the three canonical-spin gate cannot drift apart. B.9 only adds the population
view: are ALL spins (not just {0,0.5,0.9}) extracted to gate tolerance, is the
recovered M*omega_220(a) monotonic, and are there isolated blow-ups?

Acceptance (README gate, verbatim, do not soften):
  population-mean M*omega_220 error across the full [0,0.95] sweep <= 0.2%;
  monotonic, no isolated blow-ups (any catastrophic draw diagnosed as
  extractor-window vs operator, exactly as the Phase A draw-71 audit).

Runs on SLURM (slurm_kerr_sweep.sh). --quick is a 6-point login smoke check at
N=401 and is explicitly NON-authoritative.
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_THIS, "..", "..")))

from kerr.scripts.kv3_qnm import (
    evolve,
    extract_fundamental,
    gate_estimate,
    tau_final_for,
    err,
    ELL,
    MM,
    SIGMA_KO,
)
from kerr.src.qnm_kerr_reference import kerr_qnm

# --- Sweep configuration ------------------------------------------------------
N_SWEEP = 801                  # production resolution (B.8: 801 gives ~1e-4)
SPIN_LO, SPIN_HI, SPIN_N = 0.0, 0.95, 20

# --- Gate thresholds (README, do not soften) ----------------------------------
MEAN_ERR_GATE = 2.0e-3         # population-mean M*omega_220 err <= 0.2%
MONO_TOL = 1.0e-3              # adjacent-spin omega spacing (~0.01-0.03) >> this;
#                                a dip below -MONO_TOL is a genuine non-monotonicity
BLOWUP_ERR = 1.0e-2            # single-spin err above this = catastrophic draw

OUTDIR = os.path.abspath(os.path.join(_THIS, "..", "outputs", "phase_b"))


def extract_one(a_over_M, N):
    """One spin: evolve + per-observer single-mode plateau scan + median gate.

    Identical to a single N-iteration of B.8 `run_fundamental`, returned as
    plain numbers for the population view.
    """
    ref = kerr_qnm(a_over_M=a_over_M, ell=ELL, m=MM, n=0)
    is_real = (a_over_M == 0.0)
    tau_ref = ref.tau_over_M
    tau_final = tau_final_for(tau_ref)
    op, taus, series, info = evolve(a_over_M, N, tau_final)
    per_obs = []
    for label, y in series.items():
        sm = extract_fundamental(taus, y, is_real=is_real,
                                 tau_ref=tau_ref, tau_final=tau_final)
        per_obs.append((label, sm.get("omega", np.nan),
                        sm.get("tau", np.nan), sm.get("omega_std", np.nan)))
    om, ta, used = gate_estimate(per_obs)
    return ref, om, ta, used, info


def run_sweep(spins, N):
    print(f"=== B.9 Kerr fundamental (2,2,0) spin sweep  N={N} "
          f"({len(spins)} spins) ===", flush=True)
    print(f"  a/M in [{spins[0]:.3f}, {spins[-1]:.3f}]  KO={SIGMA_KO}  "
          f"gate: mean Mw err <= {MEAN_ERR_GATE:.1e}\n", flush=True)
    rows = []
    t_wall = time.time()
    for a in spins:
        ref, om, ta, used, info = extract_one(a, N)
        e_om = err(om, ref.M_omega_R)
        e_ta = err(ta, ref.tau_over_M)
        finite = bool(info["finite_final"])
        flag = "PASS" if (finite and np.isfinite(e_om) and e_om <= BLOWUP_ERR) else "CHECK"
        print(f"  a/M={a:.3f}  Mw_ref={ref.M_omega_R:.6f} Mw_ext={om:.6f} "
              f"(err {e_om:.2e})  tau_ref={ref.tau_over_M:.4f} tau_ext={ta:.4f} "
              f"(err {e_ta:.2e})  finite={finite} n_obs={len(used)}  {flag}",
              flush=True)
        rows.append({
            "a": float(a), "Mw_ref": float(ref.M_omega_R), "Mw_ext": float(om),
            "e_om": float(e_om), "tau_ref": float(ref.tau_over_M),
            "tau_ext": float(ta), "e_ta": float(e_ta), "finite": finite,
            "n_obs": len(used),
        })

    # --- population gate -------------------------------------------------------
    errs = np.array([r["e_om"] for r in rows], dtype=float)
    finite_errs = errs[np.isfinite(errs)]
    mean_err = float(np.mean(finite_errs)) if finite_errs.size else np.nan
    max_err = float(np.max(finite_errs)) if finite_errs.size else np.nan

    om_ext = np.array([r["Mw_ext"] for r in rows], dtype=float)
    diffs = np.diff(om_ext)
    mono = bool(np.all(np.isfinite(diffs)) and np.all(diffs > -MONO_TOL))
    bad_mono = [(rows[i]["a"], rows[i + 1]["a"], float(diffs[i]))
                for i in range(len(diffs)) if not (diffs[i] > -MONO_TOL)]

    blowups = [r for r in rows
               if (not r["finite"]) or (not np.isfinite(r["e_om"]))
               or (r["e_om"] > BLOWUP_ERR)]
    all_finite = all(r["finite"] for r in rows)

    ok = (np.isfinite(mean_err) and mean_err <= MEAN_ERR_GATE
          and mono and len(blowups) == 0)

    print(f"\n=== B.9 sweep summary ({time.time()-t_wall:.0f}s wall) ===", flush=True)
    print(f"  population-mean Mw err = {mean_err:.2e}  (gate <= {MEAN_ERR_GATE:.1e})  "
          f"-> {'OK' if mean_err <= MEAN_ERR_GATE else 'CHECK'}", flush=True)
    print(f"  max single-spin Mw err = {max_err:.2e}  (at a/M="
          f"{rows[int(np.nanargmax(errs))]['a']:.3f})", flush=True)
    print(f"  monotonic M*omega(a): {mono}"
          + ("" if mono else f"  (dips: {bad_mono})"), flush=True)
    print(f"  all finite (no blow-ups): {all_finite}", flush=True)
    if blowups:
        print("  CATASTROPHIC DRAWS to diagnose (extractor-window vs operator):",
              flush=True)
        for r in blowups:
            print(f"    a/M={r['a']:.3f} err={r['e_om']:.2e} finite={r['finite']} "
                  f"n_obs={r['n_obs']}", flush=True)

    _save_sweep_npz(rows, N, os.path.join(OUTDIR, "kerr_sweep.npz"))
    print(f"\n  B.9 GATE: {'PASS' if ok else 'CHECK'}", flush=True)
    return rows, ok


def _save_sweep_npz(rows, N, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = ["a", "Mw_ref", "Mw_ext", "e_om", "tau_ref", "tau_ext", "e_ta", "n_obs"]
    data = {k: np.array([r[k] for r in rows], dtype=float) for k in keys}
    data["finite"] = np.array([r["finite"] for r in rows], dtype=bool)
    data["N"] = float(N)
    np.savez(path, **data)
    print(f"  saved sweep -> {path}", flush=True)


if __name__ == "__main__":
    if "--quick" in sys.argv:
        spins = np.linspace(0.0, 0.9, 6)   # 6-point login smoke check
        N = 401
    else:
        spins = np.linspace(SPIN_LO, SPIN_HI, SPIN_N)
        N = N_SWEEP
    run_sweep(spins, N)

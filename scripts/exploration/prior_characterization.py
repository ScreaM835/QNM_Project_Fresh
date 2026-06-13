"""Stage 0 of the SW hybrid super-resolution plan: characterize the CHEAP coarse
FD prior we intend to rescue.

Question this answers: at what coarsening factor k does the plain 2nd-order FD
prior become *visibly degraded on BOTH metrics* (waveform field error AND the
extracted QNM omega/tau), so that a label-free physics refiner has real room to
improve on it? A prior that is already <1% on the QNM is not a compelling
demonstration; we need an honest, genuinely cheap, genuinely degraded start.

For the canonical Schwarzschild/Zerilli ringdown (M=1, l=2, A=1, x0=4, sigma=5,
x in [-50,150], t in [0,50]) we solve at:
    fine  : dx=0.2,  dt=0.1   (the oracle / supervised target)
    k2    : dx=0.4,  dt=0.2   (~4x cheaper)
    k4    : dx=0.8,  dt=0.4   (~16x cheaper)
    k8    : dx=1.6,  dt=0.8   (~64x cheaper)
all with the SAME 2nd-order central scheme + Sommerfeld BCs as production
(NOT the high-order DRP7 coarse prior, which is ~0.45% and recreates the
"prior too good" problem this stage exists to avoid).

For each we report, against the fine oracle + Leaver (omega=0.3737, tau=11.241):
  * waveform field rel-L2  (coarse cubic-upsampled onto the fine grid; this is
    literally the "bare prior" C0 the refiner is anchored to)
  * QNM omega / tau % error at the observer radii x_q = 10 M and 2 M
    (extracted from each solve's own native-resolution time series at exactly
    x_q via cubic spatial interpolation, so the observer location is identical
    across resolutions)
  * cost: measured wall-time and the theoretical k^2 saving (1+1D explicit RK4
    work ~ Nx*Nt ~ 1/(dx*dt))

The fine-grid QNM error is the supervised floor (the best a fine-FD-fitted model
could do); the coarse QNM error is the bar a label-free refiner must beat.

Login-node safe: a handful of small FD solves, < ~1 min, no training, no GPU,
no writes outside outputs/exploration/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
from scipy.interpolate import RectBivariateSpline

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", ".."))
sys.path.insert(0, _REPO)

from src.fd_solver import solve_fd
from src.qnm import (
    qnm_method_1,
    qnm_method_2,
    qnm_method_3_esprit,
    qnm_method_4_window_scan,
    qnm_method_5_2d_scan,
    percentage_errors,
    theory_ref,
)

METHODS = ["M1", "M2", "M3", "M4", "M5"]

# Canonical config (matches configs/hybrid_sw_dataset.yaml base/fine grid).
BASE_CFG: Dict = {
    "physics": {"M": 1.0, "l": 2, "potential": "zerilli", "pde_sign": "standard"},
    "domain": {"xmin": -50.0, "xmax": 150.0, "tmin": 0.0, "tmax": 50.0},
    "initial_data": {"A": 1.0, "x0": 4.0, "sigma": 5.0,
                     "velocity_profile": "outgoing"},
    "fd": {"dx": 0.2, "dt": 0.1, "scheme": "rk4_mol"},
}

FINE_DX, FINE_DT = 0.2, 0.1
OUTDIR = os.path.abspath(os.path.join(_REPO, "outputs", "exploration", "prior_char"))


def _cfg_at_k(k: int) -> Dict:
    cfg = json.loads(json.dumps(BASE_CFG))   # deep copy
    cfg["fd"]["dx"] = FINE_DX * k
    cfg["fd"]["dt"] = FINE_DT * k
    return cfg


def _solve_timed(cfg: Dict) -> Dict:
    t0 = time.perf_counter()
    sol = solve_fd(cfg)
    sol["wall_s"] = time.perf_counter() - t0
    return sol


def _series_at_xq(sol: Dict, xq: float) -> np.ndarray:
    """Time series at exactly x=xq via cubic spatial interpolation, on the
    solve's own native time grid (reflects that solve's temporal accuracy)."""
    spl = RectBivariateSpline(sol["t"], sol["x"], sol["phi"], kx=3, ky=3)
    return spl.ev(sol["t"], np.full_like(sol["t"], xq))


def _field_rel_l2_vs_fine(coarse: Dict, fine: Dict) -> float:
    """rel-L2 of the coarse field cubic-upsampled onto the fine (t,x) grid,
    over the time span the coarse solve covers (the bare-prior C0 error)."""
    spl = RectBivariateSpline(coarse["t"], coarse["x"], coarse["phi"], kx=3, ky=3)
    tmax_c = float(coarse["t"][-1])
    tmask = fine["t"] <= tmax_c + 1e-9
    tf = fine["t"][tmask]
    approx = spl(tf, fine["x"])               # (len tf, Nx_fine)
    ref = fine["phi"][tmask]
    num = float(np.sqrt(np.sum((approx - ref) ** 2)))
    den = float(np.sqrt(np.sum(ref ** 2)))
    return num / den if den > 0 else float("inf")


def _safe(fn, *args, **kwargs) -> Dict[str, float]:
    """Call a QNM extractor, return {omega, tau} or NaNs on failure."""
    try:
        r = fn(*args, **kwargs)
        return {"omega": float(r.get("omega", float("nan"))),
                "tau": float(r.get("tau", float("nan")))}
    except Exception:
        return {"omega": float("nan"), "tau": float("nan")}


def _qnm_errs(sol: Dict, xq: float, t_start: float, t_end: float
              ) -> Dict[str, Dict[str, float]]:
    """Run the canonical M1-M5 suite (identical to scripts/eval_hybrid_sw.py) on
    the time series at x=xq, and return per-method dimensionless values + pct
    errors vs Leaver."""
    y = _series_at_xq(sol, xq)
    t_end_eff = min(t_end, float(sol["t"][-1]))
    raw = {
        "M1": _safe(qnm_method_1, sol["t"], y, t_start, t_end_eff),
        "M2": _safe(qnm_method_2, sol["t"], y, t_start, t_end_eff),
        "M3": _safe(qnm_method_3_esprit, sol["t"], y, t_start, t_end_eff, K=4),
        "M4": _safe(qnm_method_4_window_scan, sol["t"], y,
                    t_start_min=t_start, t_start_max=t_start + 8.0,
                    t_end=t_end_eff, n_starts=12, potential="zerilli", ell=2),
        "M5": _safe(qnm_method_5_2d_scan, sol["t"], y,
                    t_start_min=t_start, t_start_max=t_start + 8.0,
                    t_end_min=t_end_eff - 10.0, t_end_max=t_end_eff,
                    n_starts=8, n_ends=5, potential="zerilli", ell=2),
    }
    out: Dict[str, Dict[str, float]] = {}
    for name, res in raw.items():
        err = percentage_errors(res, potential="zerilli", ell=2, M=1.0)
        out[name] = {"omega": res["omega"], "tau": res["tau"],
                     "omega_pct": err.get("omega_pct_err", float("nan")),
                     "tau_pct": err.get("tau_pct_err", float("nan"))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 0: cheap-prior characterization")
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 4, 8],
                    help="coarsening factors to characterize (fine = k=1)")
    ap.add_argument("--xq", type=float, nargs="+", default=[10.0, 2.0],
                    help="observer radii for QNM extraction")
    ap.add_argument("--t-start", type=float, default=12.0)
    ap.add_argument("--t-end", type=float, default=48.0)
    ap.add_argument("--out", type=str,
                    default=os.path.join(OUTDIR, "prior_char.json"))
    args = ap.parse_args()

    ref = theory_ref("zerilli", 2)
    print(f"Leaver ref: omega={ref['omega']:.4f}  tau={ref['tau']:.3f}", flush=True)
    print(f"QNM window t in [{args.t_start}, {args.t_end}] "
          f"(capped at each solve's t_max)\n", flush=True)

    # --- fine oracle ---------------------------------------------------------
    fine = _solve_timed(_cfg_at_k(1))
    fine_work = fine["phi"].size
    print(f"fine: dx={FINE_DX} dt={FINE_DT}  grid={fine['phi'].shape}  "
          f"wall={fine['wall_s']:.2f}s", flush=True)
    fine_qnm = {xq: _qnm_errs(fine, xq, args.t_start, args.t_end) for xq in args.xq}
    for xq in args.xq:
        mm = fine_qnm[xq]
        cells = "  ".join(f"{m}:{mm[m]['omega_pct']:.2f}/{mm[m]['tau_pct']:.2f}"
                           for m in METHODS)
        print(f"   [FLOOR] x_q={xq:4.1f} (om%/tau%): {cells}", flush=True)

    rows: List[Dict] = []
    for k in args.ks:
        sol = _solve_timed(_cfg_at_k(k))
        field = _field_rel_l2_vs_fine(sol, fine)
        qnm = {xq: _qnm_errs(sol, xq, args.t_start, args.t_end) for xq in args.xq}
        work_ratio = fine_work / sol["phi"].size
        wall_ratio = fine["wall_s"] / max(sol["wall_s"], 1e-9)
        rows.append({"k": k, "dx": sol["dx"], "dt": sol["dt"],
                     "grid": list(sol["phi"].shape),
                     "field_rel_l2": field,
                     "qnm": {str(xq): qnm[xq] for xq in args.xq},
                     "wall_s": sol["wall_s"], "work_ratio_k2": work_ratio,
                     "wall_ratio": wall_ratio})
        print(f"\nk={k}: dx={sol['dx']:.2f} dt={sol['dt']:.2f}  "
              f"grid={sol['phi'].shape}  cost: {work_ratio:.1f}x cheaper "
              f"(work), {wall_ratio:.1f}x (wall)", flush=True)
        print(f"   field rel-L2 vs fine = {field:.4f}", flush=True)
        for xq in args.xq:
            mm = qnm[xq]
            cells = "  ".join(f"{m}:{mm[m]['omega_pct']:.2f}/{mm[m]['tau_pct']:.2f}"
                               for m in METHODS)
            print(f"   x_q={xq:4.1f} (om%/tau%): {cells}", flush=True)

    # --- verdict hint --------------------------------------------------------
    print("\n=== summary: field rel-L2 | best-method omega% / tau% @x_q=2 ===", flush=True)
    xq_clean = args.xq[-1]   # x_q=2 is the clean radius
    for r in rows:
        mm = r["qnm"][str(xq_clean)]
        best_om = min((mm[m]["omega_pct"] for m in METHODS
                       if np.isfinite(mm[m]["omega_pct"])), default=float("nan"))
        best_tau = min((mm[m]["tau_pct"] for m in METHODS
                        if np.isfinite(mm[m]["tau_pct"])), default=float("nan"))
        print(f"  k={r['k']}: field {r['field_rel_l2']:.4f} | "
              f"best om {best_om:.3f}% / best tau {best_tau:.3f}%  "
              f"(~{r['work_ratio_k2']:.0f}x cheaper)", flush=True)
    print("\nField rel-L2 is the clean coarsening signal (the refiner's job). "
          "QNM via M1-M5 is robust validation. Pick the smallest k with clear "
          "field-error room while staying genuinely cheap.", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    payload = {"leaver": ref, "window": [args.t_start, args.t_end],
               "fine": {"dx": FINE_DX, "dt": FINE_DT, "wall_s": fine["wall_s"],
                        "qnm_floor": {str(xq): fine_qnm[xq] for xq in args.xq}},
               "coarse": rows}
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

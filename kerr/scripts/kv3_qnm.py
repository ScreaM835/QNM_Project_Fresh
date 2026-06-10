"""KV.3 QNM-extraction gate for the Kerr Teukolsky operator (analogue of V.3).

THE correctness crux of Phase B (plan B.8): evolve the full complex Teukolsky
operator (s=-2, l=m=2), record fixed-radius + scri observers, and extract the
QNM with DATA-DRIVEN windows, compared to the `qnm` package, in this order:

  1. a = 0 reduction  -- MUST reproduce Phase A / Schwarzschild:
         M*omega_220 = 0.3737,  tau/M = 11.24
     (the s=-2 Teukolsky a=0 limit is Bardeen-Press, isospectral to
     Regge-Wheeler, so the QNM frequency is identical). If this fails, B.2/B.4
     are wrong: stop and fix the operator, do NOT tune windows.
  2. a/M in {0.5, 0.9} fundamental (2,2,0) vs qnm.
  3. first overtone (2,2,1) at a/M=0.9 (two complex modes).

At a=0 the Bardeen-Press coefficients are real, so real initial data stays real
and the Schwarzschild-validated real-field Method-5 2D plateau scan applies
verbatim. At a>0 frame dragging makes the field genuinely complex; the
fundamental is then read from the complex envelope+phase slopes
(`qnm_complex_phase`) over the same kind of data-driven 2D (t_start,t_end)
plateau, and the overtone from a complex two-mode fit. (Those a>0 paths are
added in the next step; this file currently implements and self-tests the a=0
linchpin.)

Runs on SLURM. The a=0 check is light enough to validate once on the login node.
"""
from __future__ import annotations

import os
import sys
import time
import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_THIS, "..", "..")))

from kerr.src.teukolsky_minimal_gauge import (
    build_teukolsky_op,
    rhs_teuk,
    state_from_psi,
    cfl_dt,
    observer_index,
    scri_index,
)
from kerr.src.fd_stencils import d1_central
from kerr.src.mol_rk4 import integrate_state
from kerr.src.dissipation import ko_dissipation
from kerr.src.extractor_m4 import (
    qnm_method_5_2d_scan,
    qnm_method_2_2d_scan,
    qnm_complex_2d_scan,
    qnm_complex_phase,
    qnm_complex_esprit,
    qnm_method_2,
    envelope_tail_cap,
)
from kerr.src.qnm_kerr_reference import kerr_qnm


M = 1.0
ELL = 2
MM = 2
SAFETY = 0.4
SIGMA_KO = 0.2              # production value (KV.1/KV.2)
ID_R0 = 10.0
ID_WIDTH = 1.0
ID_AMP = 1.0
OBSERVERS_RM = {"scri": None, "r50M": 50.0, "r20M": 20.0, "r10M": 10.0}

# --- Data-driven extraction windows -------------------------------------------
# The fundamental is read from the CLEAN late ringdown, where the overtone
# (tau_ot ~ tau_fund/3) has decayed away. Both the evolution length and the 2D
# (t0, te) plateau-scan window are expressed as multiples of the *reference*
# damping time tau_ref = tau/M from the qnm package, so they scale automatically
# with spin and are never hand-pinned to specific numbers:
#     t0 in [4, 9] tau_ref,  te in [10, 14] tau_ref  (te capped below tau_final)
# Physics: at t = 4 tau_fund the overtone is e^{-(4 tau_fund)/(tau_fund/3)}
# = e^{-12} below the fundamental, so a single-mode fit is unbiased.
N_STARTS, N_ENDS = 8, 6
T0_LO_FAC, T0_HI_FAC = 4.0, 9.0
TE_LO_FAC, TE_HI_FAC = 10.0, 14.0
MIN_WINDOW_FAC = 3.0        # min (te - t0) in units of tau_ref
TAU_FINAL_FAC = 14.0        # evolve to ~14 tau_ref ...
TAU_FINAL_MIN, TAU_FINAL_MAX = 180.0, 220.0   # ... clamped to validated range

# Near-extremal late-time tail cap (a>0 only). The fixed [10,14] tau_ref window
# assumes the QNM dominates the late ringdown; this holds for a/M <= 0.9 but
# fails approaching extremality (a/M >= ~0.93), where the slowly-decaying tail
# and the nearly-degenerate (2,2,1) overtone contaminate the ringdown EARLIER
# (the QNM->tail crossover moves to a smaller tau-multiple). `envelope_tail_cap`
# detects, from each field's own envelope, the latest time it still decays at
# the QNM rate; when that cap falls inside the nominal window the scan is
# confined to the QNM-clean zone below it. The cap requires the decay to slow
# *permanently* (a sustained shallow run, not a transient): at low spin the
# weakly-complex |psi| has near-nodes whose momentary slope dips would otherwise
# false-trip the cap. For a/M <= 0.9 the cap lands at the evolution end, so the
# window is unchanged -- no regression; only the near-extremal tail is capped.
TAIL_CAP_SLOPE_FRAC = 0.7   # cap where envelope decay slows below 0.7x QNM rate
TAIL_CAP_SMOOTH_FAC = 0.6   # half-width (in tau_ref) of the sliding slope fit
TAIL_CAP_PERSIST_FRAC = 1.5  # shallow run must last >=1.5 tau_ref to cap (a
                            # permanent tail, not a weakly-complex near-node dip)
TAIL_MIN_WINDOW_FAC = 2.0   # min (te - t0) in tau_ref when the cap is active
TAIL_TE_LO_FAC = 3.0        # te scan low edge = te_cap - 3 tau_ref when capped

# Gate parameters. A per-observer plateau is "trusted" when its relative scatter
# (omega_std / |omega|) is below STD_TRUST_REL; the gate estimate is the median
# over trusted observers (>=2 required, else fall back to all finite). This is
# fully data-driven: floor-limited (scri) or initial-data-contaminated (r10M)
# observers self-identify through a large plateau scatter and are dropped.
STD_TRUST_REL = 1.0e-3
GATE_TOL = 1.0e-3           # 0.1% on M*omega_220
SPINS = (0.0, 0.5, 0.9)
GATE_SPINS = (0.0, 0.5, 0.9)

# First-overtone (2,2,1) extraction at a/M=0.9 via complex ESPRIT. The overtone
# (tau ~ tau_fund/3) lives only in the EARLY ringdown, so it is read from short
# windows soon after the prompt transient has cleared. Windows scale with the
# fundamental tau and are scanned (data-driven, not hand-pinned); the overtone
# is the non-fundamental physical mode nearest in frequency to the fundamental,
# discriminated by its ~3x faster damping.
OT_SPIN = 0.9
OT_K = 4                    # ESPRIT model order (2 QNM modes + transient/tail)
OT_T0_LO_FAC, OT_T0_HI_FAC = 2.5, 3.2   # window start in units of tau_fund
OT_WIN_FAC = 3.3                         # window length in units of tau_fund
OT_N_STARTS = 6
OT_OMEGA_TOL = 2.0e-2      # |Mw_ot - ref|/ref gate (overtone is weaker, ~near-degenerate)
OT_TAU_RATIO_MAX = 0.55    # tau_ot / tau_fund must be < this (a distinct faster mode)

OUTDIR = os.path.abspath(os.path.join(_THIS, "..", "outputs", "phase_b"))


def tau_final_for(tau_ref):
    return float(min(TAU_FINAL_MAX, max(TAU_FINAL_MIN, TAU_FINAL_FAC * tau_ref)))


def scan_windows(tau_ref, tau_final):
    """Spin-scaled (t0_min, t0_max, te_min, te_max, min_window) for the plateau
    scan, derived from the reference damping time (not hand-pinned)."""
    t0_lo, t0_hi = T0_LO_FAC * tau_ref, T0_HI_FAC * tau_ref
    te_lo = TE_LO_FAC * tau_ref
    te_hi = min(TE_HI_FAC * tau_ref, tau_final - 0.5 * tau_ref)
    return t0_lo, t0_hi, te_lo, te_hi, MIN_WINDOW_FAC * tau_ref


def make_initial_pulse(amp, r0, width):
    def psi0(r):
        return amp * np.exp(-((r - r0) ** 2) / (2.0 * width ** 2))
    return psi0


def evolve(a_over_M, N, tau_final):
    """Evolve the full Teukolsky operator; return (op, taus, series, info).

    `series` maps each observer label to its COMPLEX waveform time series
    (scri = index 0 is the gravitational waveform at infinity).
    """
    ref = kerr_qnm(a_over_M=a_over_M, ell=ELL, m=MM, n=0)
    omega_ref = complex(ref.M_omega_R, ref.M_omega_I)
    op = build_teukolsky_op(
        N=N, a_over_M=a_over_M, M=M, ell=ELL, m=MM,
        omega_ref=omega_ref, include_potential=True,
    )
    dt = cfl_dt(op, safety=SAFETY)
    n_steps = int(np.ceil(tau_final / dt))
    dt = tau_final / n_steps
    record_every = max(1, int(round(0.05 / dt)))

    state0 = state_from_psi(make_initial_pulse(ID_AMP, ID_R0, ID_WIDTH), op, d1_central)
    observers = {}
    for label, rM in OBSERVERS_RM.items():
        observers[label] = scri_index(op) if rM is None else observer_index(op, rM)

    def rhs_fn(s):
        dPsi, dU, dW = rhs_teuk(s, op, d1_central)
        dU = dU + ko_dissipation(s[1], SIGMA_KO)
        dW = dW + ko_dissipation(s[2], SIGMA_KO)
        return dPsi, dU, dW

    t0 = time.time()
    state, taus, series = integrate_state(
        state0, dt, n_steps, rhs_fn,
        observer_field=0, record_every=record_every, observers=observers,
    )
    elapsed = time.time() - t0
    info = dict(
        N=N, dt=dt, n_steps=n_steps, record_every=record_every, elapsed=elapsed,
        finite_final=bool(all(np.all(np.isfinite(x)) for x in state)),
        sigma_obs={k: float(op.sigma[i]) for k, i in observers.items()},
        r_obs={k: float(op.r[i]) for k, i in observers.items()},
        max_abs_imag={k: float(np.max(np.abs(series[k].imag))) for k in observers},
        max_abs_real={k: float(np.max(np.abs(series[k].real))) for k in observers},
    )
    return op, taus, series, info


def err(om, ref_val):
    return abs(om - ref_val) / abs(ref_val) if np.isfinite(om) else np.nan


def extract_fundamental(taus, y, is_real, tau_ref, tau_final):
    """Single-mode 2D plateau scan for the fundamental (2,2,0).

    a=0 (is_real): real damped-cosine scan on Re(psi).
    a>0          : complex envelope+phase scan on the full complex psi, with a
                   data-driven late-time tail cap (envelope_tail_cap) that
                   confines the scan to the QNM-clean zone near extremality.
    Returns the plateau dict (omega, tau, omega_std, tau_std, plateau bounds);
    for a>0 it also carries 'te_cap' (the detected QNM->tail transition time).
    """
    t0_lo, t0_hi, te_lo, te_hi, min_w = scan_windows(tau_ref, tau_final)
    if is_real:
        return qnm_method_2_2d_scan(
            taus, np.real(y), t0_lo, t0_hi, te_lo, te_hi,
            n_starts=N_STARTS, n_ends=N_ENDS, min_window=min_w,
        )
    te_cap = envelope_tail_cap(
        taus, y, tau_ref, t_search_start=t0_lo,
        slope_frac=TAIL_CAP_SLOPE_FRAC, smooth_frac=TAIL_CAP_SMOOTH_FAC,
        persist_frac=TAIL_CAP_PERSIST_FRAC,
    )
    if te_cap < te_hi:   # tail intrudes into the nominal window -> confine to QNM zone
        te_hi = te_cap
        min_w = min(min_w, TAIL_MIN_WINDOW_FAC * tau_ref)
        t0_lo = min(t0_lo, te_cap - min_w - tau_ref)
        t0_hi = min(t0_hi, te_cap - min_w)
        te_lo = max(te_cap - TAIL_TE_LO_FAC * tau_ref, t0_lo + min_w)
    out = qnm_complex_2d_scan(
        taus, y, t0_lo, t0_hi, te_lo, te_hi,
        n_starts=N_STARTS, n_ends=N_ENDS, min_window=min_w,
    )
    out["te_cap"] = float(te_cap)
    return out


def gate_estimate(per_obs):
    """Robust gate estimate from per-observer (label, omega, tau, omega_std).

    Trust observers whose plateau relative scatter is < STD_TRUST_REL; the
    estimate is the median over trusted observers (>=2 required, else median
    over all finite). Returns (omega, tau, used_labels).
    """
    finite = [(l, o, t, s) for (l, o, t, s) in per_obs if np.isfinite(o) and np.isfinite(t)]
    trusted = [(l, o, t, s) for (l, o, t, s) in finite
               if np.isfinite(s) and abs(o) > 0 and (s / abs(o)) < STD_TRUST_REL]
    use = trusted if len(trusted) >= 2 else finite
    if not use:
        return np.nan, np.nan, []
    om = float(np.median([o for (_, o, _, _) in use]))
    ta = float(np.median([t for (_, _, t, _) in use]))
    return om, ta, [l for (l, _, _, _) in use]


def run_fundamental(a_over_M, resolutions, ref=None):
    """Fundamental (2,2,0) extraction at one spin across resolutions.

    a=0 uses the real single-mode 2D plateau scan; a>0 uses the complex one.
    Returns (by_N, ok) where by_N[N] = (omega, tau, used_labels) is the gate
    estimate and ok is the PASS flag for this spin.
    """
    if ref is None:
        ref = kerr_qnm(a_over_M=a_over_M, ell=ELL, m=MM, n=0)
    is_real = (a_over_M == 0.0)
    tau_ref = ref.tau_over_M
    tau_final = tau_final_for(tau_ref)
    t0_lo, t0_hi, te_lo, te_hi, min_w = scan_windows(tau_ref, tau_final)
    kind = "REAL single-mode" if is_real else "COMPLEX single-mode"
    print(f"=== KV.3 fundamental (2,2,0)  a/M={a_over_M:.3f} ===", flush=True)
    print(f"  qnm ref: M*omega_R={ref.M_omega_R:.6f}  M*omega_I={ref.M_omega_I:.6f}  "
          f"tau/M={tau_ref:.4f}", flush=True)
    print(f"  {kind} 2D scan t0=[{t0_lo:.1f},{t0_hi:.1f}] te=[{te_lo:.1f},{te_hi:.1f}] "
          f"tau_final={tau_final:.1f} KO={SIGMA_KO}\n", flush=True)

    by_N = {}
    for N in resolutions:
        op, taus, series, info = evolve(a_over_M, N, tau_final)
        print(f"--- N={N}  dt={info['dt']:.3e} n_steps={info['n_steps']} "
              f"elapsed={info['elapsed']:.1f}s finite={info['finite_final']} ---", flush=True)
        per_obs = []
        for label, y in series.items():
            imag_frac = info["max_abs_imag"][label] / max(info["max_abs_real"][label], 1e-300)
            sm = extract_fundamental(taus, y, is_real=is_real,
                                     tau_ref=tau_ref, tau_final=tau_final)
            om, ta = sm.get("omega", np.nan), sm.get("tau", np.nan)
            om_std = sm.get("omega_std", np.nan)
            per_obs.append((label, om, ta, om_std))
            rel = (om_std / abs(om)) if (np.isfinite(om_std) and np.isfinite(om) and om != 0) else np.nan
            trust = "T" if (np.isfinite(rel) and rel < STD_TRUST_REL) else "."
            cap_str = ""
            if not is_real:
                tc = sm.get("te_cap", float("nan"))
                if np.isfinite(tc) and tc < te_hi:
                    cap_str = f" cap={tc:.0f}"
            print(f"  [{label:5s}] {trust} M*omega={om:.6f} (err {err(om, ref.M_omega_R):.2e}) "
                  f"tau/M={ta:.4f} (err {err(ta, ref.tau_over_M):.2e}) "
                  f"std={om_std:.1e} "
                  f"win=t0[{sm.get('t0_plateau_min', float('nan')):.0f},"
                  f"{sm.get('t0_plateau_max', float('nan')):.0f}]"
                  f"te[{sm.get('te_plateau_min', float('nan')):.0f},"
                  f"{sm.get('te_plateau_max', float('nan')):.0f}]"
                  f"{cap_str}"
                  f"  imag/real={imag_frac:.1e}",
                  flush=True)
        om_g, ta_g, used = gate_estimate(per_obs)
        by_N[N] = (om_g, ta_g, used)
        print(f"  --> N={N} gate estimate (median over {used}): "
              f"M*omega={om_g:.6f} (err {err(om_g, ref.M_omega_R):.2e})  "
              f"tau/M={ta_g:.4f} (err {err(ta_g, ref.tau_over_M):.2e})\n", flush=True)

    fine_om = by_N[resolutions[-1]][0]
    drift = (abs(by_N[resolutions[-1]][0] - by_N[resolutions[0]][0]) / ref.M_omega_R
             if all(np.isfinite(by_N[N][0]) for N in resolutions) else np.nan)
    ok = (np.isfinite(fine_om) and err(fine_om, ref.M_omega_R) <= GATE_TOL
          and np.isfinite(drift) and drift <= GATE_TOL)
    print(f"  a/M={a_over_M:.3f}: finest-grid omega err = {err(fine_om, ref.M_omega_R):.2e} "
          f"(gate <= {GATE_TOL:.0e}); across-N drift = {drift:.2e}  -> "
          f"{'PASS' if ok else 'CHECK'}\n", flush=True)
    return by_N, ok


def check_a0(resolutions=(401, 801, 1601)):
    """The decisive a=0 operator-correctness check (plan B.8 step 1)."""
    return run_fundamental(0.0, resolutions)


def identify_overtone(modes, omega_fund):
    """From ESPRIT physical modes, identify the fundamental and first-overtone.

    ESPRIT can return spurious long-lived low-frequency modes (numerical
    tail/DC); identifying the fundamental as the globally longest-tau mode then
    mislabels them. So we FIRST restrict to a physical band around the
    fundamental frequency, Re(omega) in [0.5, 1.5] omega_fund (overtones of the
    same (l,m) cluster there); the fundamental is the longest-tau in-band mode,
    and the overtone the OTHER in-band mode nearest in frequency, required to
    decay faster (tau_ot < OT_TAU_RATIO_MAX * tau_fund). If no faster in-band
    mode survives (late window: overtone already decayed) we honestly return no
    overtone rather than mislabelling the fundamental.
    """
    band = [m for m in modes
            if 0.5 * omega_fund < m["omega_R"] < 1.5 * omega_fund and m["tau"] > 0]
    if not band:
        return None, None
    band.sort(key=lambda m: -m["tau"])  # longest tau first
    fund = band[0]
    cand = [m for m in band[1:] if m["tau"] < OT_TAU_RATIO_MAX * fund["tau"]]
    if not cand:
        return fund, None
    cand.sort(key=lambda m: abs(m["omega_R"] - omega_fund))
    return fund, cand[0]


def run_overtone(a_over_M=OT_SPIN, resolutions=(401, 801, 1601)):
    """First overtone (2,2,1) at a/M via complex ESPRIT over a scan of early
    windows. The overtone Mw is the median over windows; the gate requires it to
    match the qnm (2,2,1) reference within OT_OMEGA_TOL and to be a clearly
    faster-damped distinct mode, consistent across N.
    """
    q0 = kerr_qnm(a_over_M=a_over_M, ell=ELL, m=MM, n=0)
    q1 = kerr_qnm(a_over_M=a_over_M, ell=ELL, m=MM, n=1)
    tau_f = q0.tau_over_M
    tau_final = tau_final_for(tau_f)
    t0s = np.linspace(OT_T0_LO_FAC * tau_f, OT_T0_HI_FAC * tau_f, OT_N_STARTS)
    win = OT_WIN_FAC * tau_f
    print(f"=== KV.3 first overtone (2,2,1)  a/M={a_over_M:.3f} ===", flush=True)
    print(f"  qnm ref: (2,2,0) Mw={q0.M_omega_R:.6f} tau/M={tau_f:.4f}   "
          f"(2,2,1) Mw={q1.M_omega_R:.6f} tau/M={q1.tau_over_M:.4f}", flush=True)
    print(f"  ESPRIT K={OT_K}  t0=[{t0s[0]:.1f},{t0s[-1]:.1f}] win={win:.1f} "
          f"(on r20M)  tau_final={tau_final:.1f}\n", flush=True)

    by_N = {}
    for N in resolutions:
        op, taus, series, info = evolve(a_over_M, N, tau_final)
        y = series["r20M"]
        ot_oms, ot_taus, f_oms, f_taus = [], [], [], []
        print(f"--- N={N}  dt={info['dt']:.3e} elapsed={info['elapsed']:.1f}s ---", flush=True)
        for t0 in t0s:
            es = qnm_complex_esprit(taus, y, float(t0), float(t0 + win), K=OT_K)
            fund, ot = identify_overtone(es.get("modes", []), q0.M_omega_R)
            if fund is not None:
                f_oms.append(fund["omega_R"]); f_taus.append(fund["tau"])
            if ot is not None:
                ot_oms.append(ot["omega_R"]); ot_taus.append(ot["tau"])
                tag = ""
            else:
                tag = "  (no overtone)"
            fo = fund["omega_R"] if fund else np.nan
            ft = fund["tau"] if fund else np.nan
            oo = ot["omega_R"] if ot else np.nan
            ot_t = ot["tau"] if ot else np.nan
            print(f"  [{t0:4.0f},{t0+win:4.0f}] fund(Mw={fo:.4f},tau={ft:.3f}) "
                  f"ot(Mw={oo:.4f},tau={ot_t:.3f}){tag}", flush=True)
        ot_om = float(np.median(ot_oms)) if ot_oms else np.nan
        ot_ta = float(np.median(ot_taus)) if ot_taus else np.nan
        n_res = len(ot_oms)
        by_N[N] = (ot_om, ot_ta, n_res)
        print(f"  --> N={N} overtone median (n={n_res}/{len(t0s)} windows): "
              f"Mw={ot_om:.6f} (err {err(ot_om, q1.M_omega_R):.2e})  "
              f"tau/M={ot_ta:.4f} (err {err(ot_ta, q1.tau_over_M):.2e})\n", flush=True)

    fine_om, fine_ta, n_res = by_N[resolutions[-1]]
    drift = (abs(by_N[resolutions[-1]][0] - by_N[resolutions[0]][0]) / q1.M_omega_R
             if all(np.isfinite(by_N[N][0]) for N in resolutions) else np.nan)
    resolved = (np.isfinite(fine_om) and err(fine_om, q1.M_omega_R) <= OT_OMEGA_TOL
                and np.isfinite(fine_ta) and fine_ta < OT_TAU_RATIO_MAX * tau_f
                and n_res >= max(2, OT_N_STARTS // 2)
                and np.isfinite(drift) and drift <= OT_OMEGA_TOL)
    print(f"  a/M={a_over_M:.3f}: overtone Mw err = {err(fine_om, q1.M_omega_R):.2e} "
          f"(gate <= {OT_OMEGA_TOL:.0e}); tau/M={fine_ta:.3f} (<{OT_TAU_RATIO_MAX*tau_f:.1f}); "
          f"across-N drift = {drift:.2e}  -> "
          f"{'RESOLVED' if resolved else 'NOT RESOLVED'}\n", flush=True)
    return by_N, resolved


def diag_a0(N=801):
    """Single-mode diagnostic: is the a=0 late ringdown a clean fundamental at
    M*omega=0.3737, tau/M=11.24?  Scans single-damped-cosine fits (Method 2)
    over several late windows.  If omega_R plateaus at 0.3737 the operator is
    correct and any Method-5 scatter is an over-parametrised two-mode fit, not
    an operator bug.
    """
    ref = kerr_qnm(a_over_M=0.0, ell=ELL, m=MM, n=0)
    print(f"=== KV.3 a=0 SINGLE-MODE diagnostic (N={N}) ===", flush=True)
    print(f"  ref: M*omega_R={ref.M_omega_R:.6f}  tau/M={ref.tau_over_M:.4f}\n", flush=True)
    op, taus, series, info = evolve(0.0, N, tau_final_for(ref.tau_over_M))
    print(f"  dt={info['dt']:.3e} n_steps={info['n_steps']} "
          f"elapsed={info['elapsed']:.1f}s finite={info['finite_final']}\n", flush=True)
    windows = [(40, 90), (40, 100), (50, 110), (50, 120), (60, 130), (70, 150), (90, 160)]
    for label, y in series.items():
        yr = np.real(y)
        peak = float(np.max(np.abs(yr)))
        print(f"  [{label:5s}] r={info['r_obs'][label]:8.3f}  peak|psi|={peak:.3e}", flush=True)
        for (ta, tb) in windows:
            try:
                m2 = qnm_method_2(taus, yr, float(ta), float(tb))
                om, tau = m2["omega"], m2["tau"]
                # late-window signal level (relative to peak) to flag floor
                m = (taus >= ta) & (taus <= tb)
                lvl = float(np.max(np.abs(yr[m]))) / max(peak, 1e-300)
                print(f"      [{ta:3.0f},{tb:3.0f}] M*omega={abs(om):.6f} "
                      f"(err {err(abs(om), ref.M_omega_R):.2e}) tau/M={tau:.4f} "
                      f"(err {err(tau, ref.tau_over_M):.2e})  lvl={lvl:.1e}", flush=True)
            except Exception as exc:
                print(f"      [{ta:3.0f},{tb:3.0f}] fit failed: {exc}", flush=True)
        print("", flush=True)


def diag_overtone(N=801, a_over_M=0.9):
    """Find where complex ESPRIT cleanly resolves the (2,2,1) overtone.

    Scans early windows x model orders K and prints the two longest-lived
    physical modes so the fundamental (2,2,0) and overtone (2,2,1) can be
    identified. The overtone (tau ~ tau_fund/3) is only present early, so it
    must be read from a short window soon after the prompt transient.
    """
    q0 = kerr_qnm(a_over_M=a_over_M, ell=ELL, m=MM, n=0)
    q1 = kerr_qnm(a_over_M=a_over_M, ell=ELL, m=MM, n=1)
    tau_final = tau_final_for(q0.tau_over_M)
    print(f"=== KV.3 OVERTONE diagnostic  a/M={a_over_M:.3f}  N={N} ===", flush=True)
    print(f"  (2,2,0) Mw={q0.M_omega_R:.6f} tau/M={q0.tau_over_M:.4f}   "
          f"(2,2,1) Mw={q1.M_omega_R:.6f} tau/M={q1.tau_over_M:.4f}\n", flush=True)
    op, taus, series, info = evolve(a_over_M, N, tau_final)
    print(f"  dt={info['dt']:.3e} n_steps={info['n_steps']} elapsed={info['elapsed']:.1f}s\n",
          flush=True)
    y = series["r20M"]
    windows = [(20, 50), (20, 60), (25, 65), (30, 70), (30, 80), (35, 75), (40, 90)]
    for K in (2, 3, 4):
        print(f"  --- K={K} (model order) ---", flush=True)
        for (ta, tb) in windows:
            es = qnm_complex_esprit(taus, y, float(ta), float(tb), K=K)
            modes = es.get("modes", [])
            txt = []
            for mmode in modes[:3]:
                txt.append(f"(Mw={mmode['omega_R']:.4f},tau={mmode['tau']:.3f},"
                           f"|A|={mmode['amp']:.2e})")
            print(f"    [{ta:3.0f},{tb:3.0f}] " + "  ".join(txt), flush=True)
        print("", flush=True)


def _save_summary_npz(results_by_N, ot_by_N, resolutions, path):
    """Persist the gate estimates so the report / SLURM run leaves an artifact."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {"resolutions": np.asarray(resolutions, dtype=float)}
    for a, by_N in results_by_N.items():
        ref = kerr_qnm(a_over_M=a, ell=ELL, m=MM, n=0)
        data[f"fund_a{a:.3f}_omega"] = np.array(
            [by_N[N][0] for N in resolutions], dtype=float)
        data[f"fund_a{a:.3f}_tau"] = np.array(
            [by_N[N][1] for N in resolutions], dtype=float)
        data[f"fund_a{a:.3f}_omega_ref"] = float(ref.M_omega_R)
        data[f"fund_a{a:.3f}_tau_ref"] = float(ref.tau_over_M)
    ref1 = kerr_qnm(a_over_M=OT_SPIN, ell=ELL, m=MM, n=1)
    data[f"ot_a{OT_SPIN:.3f}_omega"] = np.array(
        [ot_by_N[N][0] for N in resolutions], dtype=float)
    data[f"ot_a{OT_SPIN:.3f}_tau"] = np.array(
        [ot_by_N[N][1] for N in resolutions], dtype=float)
    data[f"ot_a{OT_SPIN:.3f}_omega_ref"] = float(ref1.M_omega_R)
    data[f"ot_a{OT_SPIN:.3f}_tau_ref"] = float(ref1.tau_over_M)
    np.savez(path, **data)
    print(f"  saved summary -> {path}", flush=True)


if __name__ == "__main__":
    res = (401, 801) if "--quick" in sys.argv else (401, 801, 1601)
    if "--diag" in sys.argv:
        diag_a0(N=801)
    elif "--otdiag" in sys.argv:
        diag_overtone(N=801)
    elif "--overtone" in sys.argv:
        run_overtone(OT_SPIN, res)
    elif "--spin" in sys.argv:
        a = float(sys.argv[sys.argv.index("--spin") + 1])
        run_fundamental(a, res)
    else:
        results = {}
        fund_by_N = {}
        for a in SPINS:
            by_N, ok = run_fundamental(a, res)
            results[a] = ok
            fund_by_N[a] = by_N
        ot_by_N, ot_ok = run_overtone(OT_SPIN, res)
        print("=== KV.3 summary ===", flush=True)
        for a in SPINS:
            print(f"  a/M={a:.3f} fundamental (2,2,0): {'PASS' if results[a] else 'CHECK'}",
                  flush=True)
        print(f"  a/M={OT_SPIN:.3f} overtone (2,2,1): "
              f"{'RESOLVED' if ot_ok else 'NOT RESOLVED'}", flush=True)
        all_ok = all(results.values()) and ot_ok
        _save_summary_npz(fund_by_N, ot_by_N, res,
                          os.path.join(OUTDIR, "kv3_qnm_summary.npz"))
        print(f"\n  KV.3 GATE: {'PASS' if all_ok else 'CHECK'}", flush=True)

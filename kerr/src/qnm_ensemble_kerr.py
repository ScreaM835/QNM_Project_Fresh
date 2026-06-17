"""Multi-method QNM consistency ensemble, adapted from the SW m1-m5 to Kerr.

WHY THIS EXISTS
---------------
The Schwarzschild report never trusted one estimator: it read the QNM from the
AGREEMENT of several independent methods (m1 FFT+log-envelope, m2 damped-cosine
NLS, m3 ESPRIT, m4 two-mode NLS, m5 2D plateau scan). The consensus is the
measurement; the spread across methods is the honest error bar. A number is only
believed when methods with *independent failure modes* land in the same place.
A single estimator cannot self-diagnose -- e.g. the single complex-phase fit
returned 173% on a clean ell=4, a/M=0.5 field with no way to flag it.

WHY IT TRANSFERS TO KERR
------------------------
A single Kerr QNM at scri is psi(tau) = A exp(-i omega tau), omega = omega_R -
i omega_I (omega_I>0). Its real part is

    Re psi(tau) = |A| exp(-omega_I tau) cos(omega_R tau + arg A),

which is *exactly* the real damped cosine the SW m1-m5 were built to fit. So the
real ensemble applies unchanged to Re(psi) AND, independently, to Im(psi) (two
real damped cosines sharing one (omega_R, omega_I) -> a built-in cross-check).
The genuinely-complex field additionally admits the complex-native estimators
qnm_complex_phase (= complex m1) and qnm_complex_esprit (= complex multi-mode
m3/m4), which use the phase winding and resolve the (.,.,1) overtone explicitly.

For a/M = 0 the complex members are skipped and the ensemble reduces EXACTLY to
the SW real {m2, m3, m5} consensus -- a continuity check against the report.

OUTPUT
------
``extract_qnm_kerr_ensemble`` returns the robust consensus (omega_R, tau) with
MAD outlier rejection, the across-method spread (the consistency/error bar), the
number of methods that survived, and the full per-method table for audit. Wide
spread == the field/QNM is untrustworthy, which is exactly what an honest hybrid
evaluation must surface instead of a silent single-method artifact.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from kerr.src.extractor_m4 import (
    qnm_method_1,
    qnm_method_2_2d_scan,
    qnm_method_3_esprit,
    qnm_method_4_two_mode,
    qnm_method_5_2d_scan,
    qnm_complex_2d_scan,
    qnm_complex_esprit,
    envelope_tail_cap,
)

# Window / scan constants mirrored from kv3_qnm so the ensemble uses the SAME
# spin-scaled, data-driven fitting band as the validated single-method path.
N_STARTS, N_ENDS = 8, 6
T0_LO_FAC, T0_HI_FAC = 4.0, 9.0
TE_LO_FAC, TE_HI_FAC = 10.0, 14.0
MIN_WINDOW_FAC = 3.0
TAIL_CAP_SLOPE_FRAC = 0.7
TAIL_CAP_SMOOTH_FAC = 0.6
TAIL_CAP_PERSIST_FRAC = 1.5
TAIL_MIN_WINDOW_FAC = 2.0
TAIL_TE_LO_FAC = 3.0

# ESPRIT model order: 2 QNM conjugate content + tail/overtone slack.
ESPRIT_K_REAL = 4
ESPRIT_K_CPLX = 3

# --- Target-aware (mode-selective) extraction (l >= 4 / known-mode path) ------
# At high l/high spin a dense ladder of slowly-decaying late-time TAIL modes
# (Mw_R ~ 0.3-0.7, |Mw_I| ~ 0.015) sits BELOW the fast QNM (|Mw_I| ~ 0.09) in the
# late [10,14]*tau_ref band, so the single-mode estimators lock onto the tail and
# the consensus suffers a CORRELATED failure (a tight spread around the WRONG
# mode). When the physical (l, m, n) is known (always, in a labelled corpus), the
# multi-mode estimators (real/complex ESPRIT, two-mode NLS) can isolate the pole
# nearest that Leaver target and the single-mode members are gated by mode ID.
# The mid-ring band [TARGET_T0_LO, TARGET_T1_HI]*tau_ref is where the fast QNM is
# still above the tail; it is scanned for robustness.
TARGET_T0_LO_FAC, TARGET_T0_HI_FAC = 2.0, 5.0
TARGET_T1_LO_FAC, TARGET_T1_HI_FAC = 7.0, 12.0
TARGET_N_T0, TARGET_N_T1 = 3, 3
TARGET_MIN_WINDOW_FAC = 3.0
TARGET_ESPRIT_K_REAL = 6
TARGET_ESPRIT_K_CPLX = (3, 4, 5, 6)
# A member enters the consensus only if it is measuring the target mode, i.e. its
# omega_R is within TARGET_REL_GATE of the Leaver target. Neighbouring (l,2,0)
# modes are >~20% apart -- far outside this gate -- while the operator/grid error
# on the TRUE mode is only a few %, so the gate performs mode IDENTIFICATION, not
# value forcing. The reported error bar is the SPREAD across the surviving
# INDEPENDENT methods; sel_dist confirms the poles were genuinely AT the target.
TARGET_REL_GATE = 0.12


def _scan_windows(tau_ref: float, tau_final: float):
    t0_lo, t0_hi = T0_LO_FAC * tau_ref, T0_HI_FAC * tau_ref
    te_lo = TE_LO_FAC * tau_ref
    te_hi = min(TE_HI_FAC * tau_ref, tau_final - 0.5 * tau_ref)
    return t0_lo, t0_hi, te_lo, te_hi, MIN_WINDOW_FAC * tau_ref


def _apply_tail_cap(tau, psi, tau_ref, t0_lo, t0_hi, te_lo, te_hi, min_w):
    """Confine the window to the QNM-clean zone (high-spin tail), as in kv3."""
    te_cap = envelope_tail_cap(
        tau, psi, tau_ref, t_search_start=t0_lo,
        slope_frac=TAIL_CAP_SLOPE_FRAC, smooth_frac=TAIL_CAP_SMOOTH_FAC,
        persist_frac=TAIL_CAP_PERSIST_FRAC,
    )
    if te_cap < te_hi:
        te_hi = te_cap
        min_w = min(min_w, TAIL_MIN_WINDOW_FAC * tau_ref)
        t0_lo = min(t0_lo, te_cap - min_w - tau_ref)
        t0_hi = min(t0_hi, te_cap - min_w)
        te_lo = max(te_cap - TAIL_TE_LO_FAC * tau_ref, t0_lo + min_w)
    return t0_lo, t0_hi, te_lo, te_hi, min_w, float(te_cap)


def _esprit_real_median(tau, yr, t0_lo, te_lo, te_hi, tau_ref) -> float:
    """Median dominant-mode |omega| over a few real-ESPRIT sub-windows."""
    oms = []
    starts = np.linspace(t0_lo, t0_lo + 2.0 * tau_ref, 3)
    for t0 in starts:
        r = qnm_method_3_esprit(tau, yr, float(t0), float(te_hi),
                                K=ESPRIT_K_REAL)
        om = r.get("omega", np.nan)
        if np.isfinite(om):
            oms.append(abs(float(om)))
    return float(np.median(oms)) if oms else float("nan")


def _consensus(vals: List[float], rel_reject: float = 0.12
               ) -> Tuple[float, float, int]:
    """Robust center + spread with median-absolute-deviation outlier rejection.

    This is the step that catches a lone bad estimator: the median is robust to
    a minority of wild members, and any value farther than max(rel_reject*|med|,
    3*MAD) from the median is dropped before the final median/spread. The
    returned spread (std of survivors) is the consistency error bar.
    """
    v = np.array([x for x in vals if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), 0
    med = float(np.median(v))
    mad = 1.4826 * float(np.median(np.abs(v - med)))
    tol = max(rel_reject * abs(med), 3.0 * mad)
    keep = np.abs(v - med) <= tol
    vk = v[keep] if keep.any() else v
    return float(np.median(vk)), float(np.std(vk)), int(vk.size)


# --- Target-aware (mode-selective) helpers -----------------------------------
# These isolate, from each multi-mode estimator's pole list, the pole NEAREST a
# known Leaver (l, m, n) target. This is legitimate mode identification (the
# corpus is labelled by mode); each estimator still reports its own measured
# frequency, so the across-method spread remains an honest error bar.

def _target_window_pairs(tau_ref: float, tau_final: float
                         ) -> List[Tuple[float, float]]:
    """Mid-ring (t0, t1) scan where the fast high-l QNM is above the tail."""
    t0s = np.linspace(TARGET_T0_LO_FAC * tau_ref, TARGET_T0_HI_FAC * tau_ref,
                      TARGET_N_T0)
    hi = min(TARGET_T1_HI_FAC * tau_ref, tau_final - 0.5 * tau_ref)
    t1s = np.linspace(TARGET_T1_LO_FAC * tau_ref, hi, TARGET_N_T1)
    return [(float(t0), float(t1)) for t0 in t0s for t1 in t1s
            if t1 - t0 >= TARGET_MIN_WINDOW_FAC * tau_ref]


def _select_nearest_real(omegas, taus, target_R: float
                         ) -> Tuple[float, float, float]:
    """Pole (omega_R>0) nearest target_R from aligned (omegas, taus) lists."""
    best_o, best_t, best_d = float("nan"), float("nan"), float("inf")
    for o, t in zip(omegas, taus):
        o = float(o)
        if not np.isfinite(o) or o <= 0:
            continue
        d = abs(o - target_R)
        if d < best_d:
            best_o = o
            best_t = float(t) if np.isfinite(t) else float("nan")
            best_d = d
    return best_o, best_t, best_d


def _select_nearest_complex(modes, target: complex
                            ) -> Tuple[float, float, float]:
    """Pole nearest ``target`` (= omega_R - i*omega_I) in the complex plane."""
    best_o, best_t, best_d = float("nan"), float("nan"), float("inf")
    for m in modes:
        oR = m.get("omega_R", float("nan"))
        if not np.isfinite(oR) or oR <= 0:
            continue
        tk = m.get("tau", float("nan"))
        oI = (1.0 / tk) if (np.isfinite(tk) and tk > 0) else 0.0
        d = abs((oR - target.real) + 1j * (-oI - target.imag))
        if d < best_d:
            best_o = float(oR)
            best_t = float(tk) if np.isfinite(tk) else float("nan")
            best_d = float(d)
    return best_o, best_t, best_d


def _m3_target(tau, yr, pairs, target_R, K=TARGET_ESPRIT_K_REAL
               ) -> Tuple[float, float, float]:
    """Real ESPRIT, nearest-target pole, median over the window scan."""
    os_, ts_, ds_ = [], [], []
    for (t0, t1) in pairs:
        r = qnm_method_3_esprit(tau, yr, t0, t1, K=K, use_analytic=True)
        o, t, d = _select_nearest_real(r.get("all_omegas", []),
                                       r.get("all_taus", []), target_R)
        if np.isfinite(o):
            os_.append(o); ts_.append(t); ds_.append(d)
    if not os_:
        return float("nan"), float("nan"), float("nan")
    return (float(np.median(os_)), float(np.nanmedian(ts_)),
            float(np.median(ds_)))


def _m4_target(tau, yr, pairs, target_R) -> Tuple[float, float, float]:
    """Two-mode NLS, nearest-target of the two fitted modes, median over scan."""
    os_, ts_ = [], []
    for (t0, t1) in pairs:
        r = qnm_method_4_two_mode(tau, yr, t0, t1)
        cand_o = [r.get("omega", float("nan")), r.get("omega1", float("nan"))]
        cand_t = [r.get("tau", float("nan")), r.get("tau1", float("nan"))]
        o, t, _ = _select_nearest_real(cand_o, cand_t, target_R)
        if np.isfinite(o):
            os_.append(o); ts_.append(t)
    if not os_:
        return float("nan"), float("nan"), float("nan")
    return float(np.median(os_)), float(np.nanmedian(ts_)), float("nan")


def _cesprit_target(tau, psi, pairs, target,
                    K_list=TARGET_ESPRIT_K_CPLX) -> Tuple[float, float, float]:
    """Complex ESPRIT, nearest-target pole, median over window x order scan."""
    os_, ts_, ds_ = [], [], []
    for K in K_list:
        for (t0, t1) in pairs:
            r = qnm_complex_esprit(tau, psi, t0, t1, K=K)
            modes = r.get("all_modes") or r.get("modes") or []
            o, t, d = _select_nearest_complex(modes, target)
            if np.isfinite(o):
                os_.append(o); ts_.append(t); ds_.append(d)
    if not os_:
        return float("nan"), float("nan"), float("nan")
    return (float(np.median(os_)), float(np.nanmedian(ts_)),
            float(np.median(ds_)))


def extract_qnm_kerr_ensemble(
    tau: np.ndarray,
    psi: np.ndarray,
    a_over_M: float,
    tau_ref: float,
    tau_final: float,
    return_detail: bool = False,
    omega_target: complex | None = None,
) -> Dict[str, object]:
    """Robust (omega_R, tau) from the multi-method Kerr consensus.

    Parameters
    ----------
    tau        : (Ntau,) time axis (uniform).
    psi        : (Ntau,) complex scri waveform.
    a_over_M   : spin (selects the real-only reduction at a=0).
    tau_ref    : reference damping time tau/M (qnm package) -> sets the band.
    tau_final  : last time available (caps te).
    return_detail : also return the per-method table.
    omega_target : optional complex Leaver target (M*omega_R - i*M*omega_I,
        omega_I>0) of the mode being measured. When given (and a>0), the
        ensemble switches to the MODE-SELECTIVE path: multi-mode estimators
        (real/complex ESPRIT, two-mode NLS) isolate the pole nearest this target
        over a mid-ring window scan, and single-mode members are gated by mode
        identity. This fixes the high-l correlated tail-lock failure (the late
        single-mode band otherwise agrees on the slow tail, not the QNM). With
        ``omega_target=None`` the original (l=2-validated) path is unchanged.

    Returns dict with: omega (consensus omega_R), tau (consensus damping),
    omega_std / tau_std (across-method spread), n_omega / n_tau (survivors),
    te_cap, and (if return_detail) ``methods`` = list of (label, omega_R, tau).
    The mode-selective path also returns ``targeted=True``, ``sel_dist`` (median
    complex distance of the selected poles to the target -- the confidence
    guard) and ``n_single_dropped`` (single-mode members rejected by the gate).
    """
    tau = np.asarray(tau, dtype=float)
    psi = np.asarray(psi, dtype=complex)
    is_real = abs(a_over_M) < 1e-6

    t0_lo, t0_hi, te_lo, te_hi, min_w = _scan_windows(tau_ref, tau_final)
    te_cap = float(tau_final)
    if not is_real:
        t0_lo, t0_hi, te_lo, te_hi, min_w, te_cap = _apply_tail_cap(
            tau, psi, tau_ref, t0_lo, t0_hi, te_lo, te_hi, min_w)

    if (omega_target is not None) and (not is_real):
        return _ensemble_targeted(
            tau, psi, tau_ref, tau_final, te_cap, omega_target,
            t0_lo, t0_hi, te_lo, te_hi, min_w, return_detail)

    methods: List[Tuple[str, float, float]] = []

    def add(label, om, ta):
        methods.append((label, float(om) if np.isfinite(om) else np.nan,
                        float(ta) if np.isfinite(ta) else np.nan))

    # --- Real-part ensemble (always); imag-part too when the field is complex.
    parts = [("re", psi.real)]
    if not is_real:
        parts.append(("im", psi.imag))

    for pname, yr in parts:
        yr = np.asarray(yr, dtype=float)
        # m2: damped-cosine NLS, 2D plateau scan (robust single-mode).
        s2 = qnm_method_2_2d_scan(tau, yr, t0_lo, t0_hi, te_lo, te_hi,
                                  n_starts=N_STARTS, n_ends=N_ENDS,
                                  min_window=min_w)
        add(f"m2_{pname}", abs(s2.get("omega", np.nan)), s2.get("tau", np.nan))
        # m3: ESPRIT, median over sub-windows (independent algebra, no NLS).
        add(f"m3_{pname}",
            _esprit_real_median(tau, yr, t0_lo, te_lo, te_hi, tau_ref),
            np.nan)
        # m5: two-mode NLS 2D scan (overtone-aware; seeded from m2, data-driven).
        s5 = qnm_method_5_2d_scan(tau, yr, t0_lo, t0_hi, te_lo, te_hi,
                                  n_starts=N_STARTS, n_ends=N_ENDS,
                                  ell=2, min_window=min_w)
        add(f"m5_{pname}", abs(s5.get("omega", np.nan)), s5.get("tau", np.nan))

    # --- Complex-native members (a>0 only): use phase winding + multi-mode.
    if not is_real:
        cp = qnm_complex_2d_scan(tau, psi, t0_lo, t0_hi, te_lo, te_hi,
                                 n_starts=N_STARTS, n_ends=N_ENDS,
                                 min_window=min_w)
        add("cphase", abs(cp.get("omega", np.nan)), cp.get("tau", np.nan))
        ce = qnm_complex_esprit(tau, psi, t0_lo, te_hi, K=ESPRIT_K_CPLX)
        modes = ce.get("modes", [])
        if modes:
            add("cesprit", abs(modes[0]["omega_R"]), modes[0]["tau"])
        else:
            add("cesprit", np.nan, np.nan)

    om_vals = [m[1] for m in methods]
    ta_vals = [m[2] for m in methods]
    omega, omega_std, n_om = _consensus(om_vals)
    tau_c, tau_std, n_ta = _consensus(ta_vals)

    out: Dict[str, object] = {
        "omega": omega, "tau": tau_c,
        "omega_std": omega_std, "tau_std": tau_std,
        "n_omega": n_om, "n_tau": n_ta,
        "te_cap": te_cap,
    }
    if return_detail:
        out["methods"] = methods
    return out


def _ensemble_targeted(
    tau, psi, tau_ref, tau_final, te_cap, omega_target,
    t0_lo, t0_hi, te_lo, te_hi, min_w, return_detail,
) -> Dict[str, object]:
    """Mode-selective consensus around a known Leaver ``omega_target``.

    Independent MULTI-mode estimators (real ESPRIT on Re/Im, two-mode NLS on
    Re/Im, complex ESPRIT) each isolate the pole nearest the target over a
    mid-ring window scan -> they measure the SAME physical mode with independent
    algebra, so their spread is an honest error bar. Single-mode members (m2 on
    Re/Im, cphase) are computed on the usual late band and admitted to the
    consensus ONLY if they fall on the target mode (gate); at high l they lock
    onto the slow tail and are dropped (their removal is the cure for the
    correlated single-mode failure). ``sel_dist`` reports how close the selected
    poles actually were to the target (small => genuinely found, not snapped).
    """
    target_R = float(np.real(omega_target))
    # QNM convention omega_R - i*omega_I (omega_I > 0 for a decaying mode):
    wt = complex(target_R, -abs(float(np.imag(omega_target))))
    pairs = _target_window_pairs(tau_ref, tau_final)

    methods: List[Tuple[str, float, float]] = []
    sel_dists: List[float] = []

    # --- Independent target-selected multi-mode estimators (one each) ---
    for pname, yr in (("re", np.asarray(psi.real, float)),
                      ("im", np.asarray(psi.imag, float))):
        o3, t3, d3 = _m3_target(tau, yr, pairs, target_R)
        methods.append((f"m3sel_{pname}", o3, t3))
        if np.isfinite(d3):
            sel_dists.append(d3)
        o4, t4, _ = _m4_target(tau, yr, pairs, target_R)
        methods.append((f"m4sel_{pname}", o4, t4))
    oc, tc, dc = _cesprit_target(tau, psi, pairs, wt)
    methods.append(("cesprit_sel", oc, tc))
    if np.isfinite(dc):
        sel_dists.append(dc)

    n_multi = len(methods)

    # --- Single-mode cross-checks (late band; gated by mode identity) ---
    s2re = qnm_method_2_2d_scan(tau, np.asarray(psi.real, float),
                                t0_lo, t0_hi, te_lo, te_hi,
                                n_starts=N_STARTS, n_ends=N_ENDS, min_window=min_w)
    s2im = qnm_method_2_2d_scan(tau, np.asarray(psi.imag, float),
                                t0_lo, t0_hi, te_lo, te_hi,
                                n_starts=N_STARTS, n_ends=N_ENDS, min_window=min_w)
    cp = qnm_complex_2d_scan(tau, psi, t0_lo, t0_hi, te_lo, te_hi,
                             n_starts=N_STARTS, n_ends=N_ENDS, min_window=min_w)
    single = [
        ("m2_re", abs(s2re.get("omega", np.nan)), s2re.get("tau", np.nan)),
        ("m2_im", abs(s2im.get("omega", np.nan)), s2im.get("tau", np.nan)),
        ("cphase", abs(cp.get("omega", np.nan)), cp.get("tau", np.nan)),
    ]
    methods.extend(single)

    # --- Mode-identity gate + consensus ---
    gate = TARGET_REL_GATE * abs(target_R)
    kept = [(l, o, t) for (l, o, t) in methods
            if np.isfinite(o) and abs(o - target_R) <= gate]
    single_labels = {"m2_re", "m2_im", "cphase"}
    n_drop = sum(1 for (l, o, t) in methods
                 if l in single_labels and np.isfinite(o)
                 and abs(o - target_R) > gate)

    if len(kept) >= 2:
        omega, omega_std, n_om = _consensus([o for (_, o, _) in kept])
        tau_c, tau_std, n_ta = _consensus(
            [t for (_, _, t) in kept if np.isfinite(t)])
    else:
        # Nothing resolved the target mode -> low confidence. Report the
        # nearest-target multi-mode picks ungated so the caller sees a large
        # sel_dist / spread rather than a silent snap to target.
        mm = [o for (l, o, t) in methods[:n_multi] if np.isfinite(o)]
        omega, omega_std, n_om = (_consensus(mm) if mm
                                  else (float("nan"), float("nan"), 0))
        tau_c, tau_std, n_ta = float("nan"), float("nan"), 0

    out: Dict[str, object] = {
        "omega": omega, "tau": tau_c,
        "omega_std": omega_std, "tau_std": tau_std,
        "n_omega": n_om, "n_tau": n_ta,
        "te_cap": te_cap,
        "targeted": True,
        "sel_dist": float(np.median(sel_dists)) if sel_dists else float("nan"),
        "n_single_dropped": int(n_drop),
    }
    if return_detail:
        out["methods"] = methods
    return out


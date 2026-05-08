from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.signal import find_peaks
from scipy.fft import rfft, rfftfreq
from scipy.optimize import curve_fit
from scipy.signal import hilbert


def _damped_cos(t: np.ndarray, A: float, tau: float, omega: float, phi: float) -> np.ndarray:
    return A * np.exp(-t / tau) * np.cos(omega * t + phi)


def _fft_estimate_omega(tt: np.ndarray, yy: np.ndarray, pad_factor: int = 64) -> float:
    """
    Estimate omega (rad / unit time) from an FFT of the time series.

    Implementation details (chosen to make FFT-based estimates usable on short windows):
    - subtract mean (remove DC),
    - apply a Hann window (reduce spectral leakage),
    - zero-pad by `pad_factor` (increase frequency sampling density),
    - parabolic interpolation around the FFT magnitude peak.

    This is consistent with the spirit of the paper’s “Fourier transform” approach, but avoids
    the coarse frequency-bin artifact that occurs when the fit window is short.
    """
    tt = np.asarray(tt, dtype=float)
    yy = np.asarray(yy, dtype=float)
    dt = float(tt[1] - tt[0])

    N = tt.size
    w = np.hanning(N)
    y0 = (yy - np.mean(yy)) * w

    # choose FFT length as next power of 2 >= pad_factor*N
    Nfft = 1
    while Nfft < pad_factor * N:
        Nfft *= 2

    Y = np.abs(rfft(y0, n=Nfft))
    freqs = rfftfreq(Nfft, d=dt)  # cycles per unit time

    # ignore the zero bin
    k = int(np.argmax(Y[1:])) + 1

    # parabolic interpolation using k-1, k, k+1
    if 1 <= k < (Y.size - 1):
        alpha, beta, gamma = Y[k - 1], Y[k], Y[k + 1]
        denom = alpha - 2.0 * beta + gamma
        if denom != 0:
            p = 0.5 * (alpha - gamma) / denom
        else:
            p = 0.0
    else:
        p = 0.0

    k_interp = k + p
    freq_interp = k_interp * freqs[1]  # linear in k for uniform FFT bins
    omega = 2.0 * np.pi * freq_interp
    return float(omega)


def qnm_method_1(t: np.ndarray, y: np.ndarray, t_start: float, t_end: float) -> Dict[str, float]:
    """
    Method 1 (per the target paper):
      - FFT to estimate ω,
      - log-linear fit of the envelope maxima to estimate τ.

    Returns:
      { "omega": ω, "tau": τ }
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = (t >= t_start) & (t <= t_end)
    tt = t[mask]
    yy = y[mask]

    omega = _fft_estimate_omega(tt, yy)

    peaks, _ = find_peaks(np.abs(yy))
    if peaks.size < 2:
        return {"omega": float(omega), "tau": float("nan")}

    tp = tt[peaks]
    ap = np.abs(yy[peaks])

    # log(ap) = c - tp/tau
    coeff = np.polyfit(tp, np.log(ap + 1e-30), deg=1)
    slope = float(coeff[0])
    tau = -1.0 / slope if slope < 0 else float("nan")

    return {"omega": float(omega), "tau": float(tau)}


def qnm_method_2(t: np.ndarray, y: np.ndarray, t_start: float, t_end: float) -> Dict[str, float]:
    """
    Method 2: direct nonlinear fit of a damped cosine:
        y(t) ≈ A exp(-t/τ) cos(ω t + φ).

    Returns:
      { "omega": ω, "tau": τ, "A": A, "phi": φ }
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = (t >= t_start) & (t <= t_end)
    tt = t[mask]
    yy = y[mask]

    m1 = qnm_method_1(t, y, t_start, t_end)
    omega0 = m1["omega"]
    tau0 = m1["tau"] if np.isfinite(m1["tau"]) else 10.0

    A0 = float(np.max(np.abs(yy)))
    phi0 = 0.0

    popt, _ = curve_fit(
        _damped_cos,
        tt,
        yy,
        p0=[A0, tau0, omega0, phi0],
        maxfev=50000,
    )
    A, tau, omega, phi = popt
    return {"omega": float(omega), "tau": float(tau), "A": float(A), "phi": float(phi)}


# ---------------------------------------------------------------------------
# Theoretical QNM reference values  (Leaver 1985, l=2 Schwarzschild)
# These are the same values cited in Patel, Laguna & Shoemaker Table 3.
# ---------------------------------------------------------------------------
THEORY = {
    "zerilli": {2: {"omega": 0.3737, "tau": 11.241}},
    "regge_wheeler": {2: {"omega": 0.3737, "tau": 11.241}},
}


def percentage_errors(
    result: Dict[str, float],
    potential: str = "zerilli",
    ell: int = 2,
) -> Dict[str, float]:
    """
    Compute percentage errors of extracted ω and τ relative to theoretical
    QNM values, matching the convention used in Patel et al. Table 3:

        % error = |extracted - theory| / theory × 100

    Parameters
    ----------
    result : dict with keys "omega" and "tau"
    potential : "zerilli" or "regge_wheeler"
    ell : angular mode number

    Returns
    -------
    dict with keys "omega_pct_err", "tau_pct_err", "omega_theory", "tau_theory"
    """
    ref = THEORY.get(potential, {}).get(ell)
    if ref is None:
        return {"omega_pct_err": float("nan"), "tau_pct_err": float("nan")}

    omega_err = abs(result["omega"] - ref["omega"]) / ref["omega"] * 100.0
    tau_err = abs(result["tau"] - ref["tau"]) / ref["tau"] * 100.0 if np.isfinite(result.get("tau", float("nan"))) else float("nan")

    return {
        "omega_pct_err": float(omega_err),
        "tau_pct_err": float(tau_err),
        "omega_theory": ref["omega"],
        "tau_theory": ref["tau"],
    }


# ---------------------------------------------------------------------------
# Method 3: ESPRIT (Estimation of Signal Parameters via Rotational Invariance
# Techniques). Models the signal as a sum of K complex exponentials
#     y_n = sum_k c_k * z_k^n,    z_k = exp((-1/tau_k + i*omega_k) * dt)
# and recovers {z_k} via SVD of a Hankel matrix and an eigendecomposition of a
# small KxK rotation matrix. Linear, non-iterative, multi-mode.
# ---------------------------------------------------------------------------

def _esprit_core(y: np.ndarray, dt: float, K: int, L: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Core ESPRIT routine. Operates on a (possibly complex) uniformly-sampled
    signal `y` with spacing `dt`, fitting K complex exponentials.

    Returns a dict with arrays of poles, omegas, taus, and amplitudes.
    """
    y = np.asarray(y).astype(complex)
    N = y.size
    if L is None:
        L = N // 2
    M = N - L  # Hankel has shape (L+1) x M ; need L+M = N-1 ... we use H of shape (L, M)
    # Build Hankel: H[i, j] = y[i + j], shape (L, M), with L + M - 1 = N - 1
    L = N // 2
    M = N - L
    H = np.empty((L, M), dtype=complex)
    for i in range(L):
        H[i, :] = y[i : i + M]

    # SVD and truncate to rank K (signal subspace)
    U, _S, _Vh = np.linalg.svd(H, full_matrices=False)
    Us = U[:, :K]

    # Rotational invariance: U_up * Psi = U_down
    U_up = Us[:-1, :]
    U_dn = Us[1:, :]
    Psi, *_ = np.linalg.lstsq(U_up, U_dn, rcond=None)

    # Eigenvalues of Psi are the poles z_k
    z = np.linalg.eigvals(Psi)

    # Convert poles to physical (omega, tau) via z = exp((-1/tau + i*omega) * dt)
    log_z = np.log(z)
    omega = np.imag(log_z) / dt
    tau = np.where(np.real(log_z) < 0, -dt / np.real(log_z), np.inf)

    # Amplitudes: solve linear LS  V c = y, where V[n, k] = z_k^n
    n_idx = np.arange(N)
    V = z[np.newaxis, :] ** n_idx[:, np.newaxis]
    c, *_ = np.linalg.lstsq(V, y, rcond=None)

    return {
        "z": z,
        "omega": omega,
        "tau": tau,
        "amp": c,
    }


def qnm_method_3_esprit(
    t: np.ndarray,
    y: np.ndarray,
    t_start: float,
    t_end: float,
    K: int = 4,
    use_analytic: bool = True,
) -> Dict[str, float]:
    """
    Method 3: ESPRIT extraction of the dominant ringdown mode.

    Parameters
    ----------
    t, y       : time and signal arrays (uniformly sampled in t).
    t_start, t_end : window over which to extract.
    K          : model order (number of complex exponentials). For a single
                 real-valued damped cosine K=2 (conjugate pair); use K=4 to
                 also resolve the first overtone or capture a tail mode.
    use_analytic : if True, work with the analytic (Hilbert-transformed)
                 complex signal so each physical mode appears once instead of
                 as a conjugate pair (improves conditioning).

    Returns
    -------
    dict with keys
      "omega", "tau" : dominant-mode estimates (largest |amp|, finite tau,
                       positive omega convention)
      "K"            : model order used
      "all_omegas"   : list of all extracted omegas
      "all_taus"     : list of all extracted taus
      "all_amps"     : list of all |c_k|
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = (t >= t_start) & (t <= t_end)
    tt = t[mask]
    yy = y[mask]
    if tt.size < max(8, 2 * K):
        return {"omega": float("nan"), "tau": float("nan"), "K": K,
                "all_omegas": [], "all_taus": [], "all_amps": []}

    # Verify uniform spacing
    dts = np.diff(tt)
    dt = float(dts.mean())
    if not np.allclose(dts, dt, rtol=1e-6, atol=1e-9):
        # interpolate onto uniform grid
        tt_uni = np.linspace(tt[0], tt[-1], tt.size)
        yy = np.interp(tt_uni, tt, yy)
        tt = tt_uni
        dt = float(tt[1] - tt[0])

    sig = hilbert(yy) if use_analytic else yy.astype(complex)

    res = _esprit_core(sig, dt=dt, K=K)
    omegas = np.asarray(res["omega"])
    taus = np.asarray(res["tau"])
    amps = np.abs(np.asarray(res["amp"]))

    # Pick dominant physical mode: finite positive tau, positive omega
    # (when use_analytic=True the conjugate is suppressed so omega has a sign;
    # otherwise pick |omega|).
    valid = np.isfinite(taus) & (taus > 0)
    if use_analytic:
        valid &= (omegas > 0)
    if not np.any(valid):
        # Fallback: just take largest amplitude regardless
        idx_sorted = np.argsort(-amps)
    else:
        # Sort the valid modes by amplitude (descending)
        idx_valid = np.where(valid)[0]
        idx_sorted = idx_valid[np.argsort(-amps[idx_valid])]

    dom = int(idx_sorted[0])
    omega_dom = float(abs(omegas[dom])) if not use_analytic else float(omegas[dom])
    tau_dom = float(taus[dom])

    return {
        "omega": omega_dom,
        "tau": tau_dom,
        "K": K,
        "all_omegas": [float(x) for x in omegas],
        "all_taus": [float(x) for x in taus],
        "all_amps": [float(x) for x in amps],
    }


# ---------------------------------------------------------------------------
# Method 4: two-mode NLS (fundamental + first overtone) with start-time
# stability scan. This is the Giesler, Isi, Scheel & Teukolsky (2019) recipe:
# fit  sum_{n=0}^{1} A_n exp(-t/tau_n) cos(omega_n t + phi_n)  over a sliding
# start-time window and report the plateau mean as the extracted value.
# ---------------------------------------------------------------------------

def _two_mode(
    t: np.ndarray,
    A0: float, tau0: float, omega0: float, phi0: float,
    A1: float, tau1: float, omega1: float, phi1: float,
) -> np.ndarray:
    return (A0 * np.exp(-t / tau0) * np.cos(omega0 * t + phi0)
            + A1 * np.exp(-t / tau1) * np.cos(omega1 * t + phi1))


# Theoretical first-overtone (n=1) values for Schwarzschild l=2 (Leaver 1985).
# Used as initial guess only; the fit is free to move away.
_THEORY_OVERTONE = {
    "zerilli":       {2: {"omega": 0.3467, "tau": 3.651}},
    "regge_wheeler": {2: {"omega": 0.3467, "tau": 3.651}},
}


def qnm_method_4_two_mode(
    t: np.ndarray,
    y: np.ndarray,
    t_start: float,
    t_end: float,
    potential: str = "zerilli",
    ell: int = 2,
) -> Dict[str, float]:
    """
    Method 4 (single window): two-mode NLS fit of fundamental + first overtone.

    Returns the fundamental-mode parameters under the keys
        "omega", "tau", "A", "phi"
    plus the overtone parameters under
        "omega1", "tau1", "A1", "phi1".
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = (t >= t_start) & (t <= t_end)
    tt = t[mask]
    yy = y[mask]
    if tt.size < 16:
        return {"omega": float("nan"), "tau": float("nan")}

    # Initial guess for fundamental: use Method 2's single-mode fit.
    m2 = qnm_method_2(t, y, t_start, t_end)
    omega0_g = m2["omega"]
    tau0_g = m2["tau"] if np.isfinite(m2["tau"]) else 11.0
    A0_g = m2["A"]
    phi0_g = m2["phi"]

    # Initial guess for overtone: theory values, modest amplitude
    ovt = _THEORY_OVERTONE.get(potential, {}).get(ell)
    if ovt is None:
        omega1_g = 2.0 * omega0_g
        tau1_g = tau0_g / 3.0
    else:
        omega1_g = ovt["omega"]
        tau1_g = ovt["tau"]
    A1_g = 0.1 * abs(A0_g)
    phi1_g = 0.0

    p0 = [A0_g, tau0_g, omega0_g, phi0_g, A1_g, tau1_g, omega1_g, phi1_g]

    # Bounds: keep tau positive and frequencies positive; cap to physically
    # plausible ranges for Schwarzschild l=2 (avoid pathological optima).
    lo = [-5*abs(A0_g)-1, 0.5, 0.05, -2*np.pi, -5*abs(A0_g)-1, 0.2, 0.05, -2*np.pi]
    hi = [ 5*abs(A0_g)+1, 100.0, 5.0,  2*np.pi,  5*abs(A0_g)+1, 50.0, 5.0,  2*np.pi]

    try:
        popt, _ = curve_fit(
            _two_mode, tt, yy, p0=p0, bounds=(lo, hi), maxfev=200000,
        )
    except Exception:
        return {"omega": float("nan"), "tau": float("nan")}

    A0f, tau0f, omega0f, phi0f, A1f, tau1f, omega1f, phi1f = popt

    # Identify which of the two fitted modes is the fundamental.
    # Convention: fundamental has the longer tau (less damped).
    if tau1f > tau0f:
        A0f, tau0f, omega0f, phi0f, A1f, tau1f, omega1f, phi1f = (
            A1f, tau1f, omega1f, phi1f, A0f, tau0f, omega0f, phi0f,
        )

    return {
        "omega": float(omega0f), "tau": float(tau0f),
        "A": float(A0f), "phi": float(phi0f),
        "omega1": float(omega1f), "tau1": float(tau1f),
        "A1": float(A1f), "phi1": float(phi1f),
    }


def qnm_method_4_window_scan(
    t: np.ndarray,
    y: np.ndarray,
    t_start_min: float,
    t_start_max: float,
    t_end: float,
    n_starts: int = 16,
    potential: str = "zerilli",
    ell: int = 2,
    plateau_frac: float = 0.5,
) -> Dict[str, object]:
    """
    Method 4 (with stability scan): run two-mode NLS over a sweep of start
    times t_0 in [t_start_min, t_start_max] and report the plateau mean.

    The "plateau" is identified as the contiguous block of the lowest-scatter
    `plateau_frac` fraction of consecutive fits (rolling-stddev minimum).

    Returns dict with:
      "omega", "tau"            : plateau mean of fundamental
      "omega_std", "tau_std"    : plateau stddev (systematic uncertainty)
      "t_starts"                : array of start times tried
      "omegas", "taus"          : per-window fundamental fits
      "omegas1", "taus1"        : per-window overtone fits
      "plateau_idx"             : indices used for the plateau mean
      "t0_plateau_min/max"      : start-time bounds of the plateau
    """
    t_starts = np.linspace(t_start_min, t_start_max, n_starts)
    omegas, taus = [], []
    omegas1, taus1 = [], []
    for t0 in t_starts:
        r = qnm_method_4_two_mode(t, y, float(t0), t_end,
                                  potential=potential, ell=ell)
        omegas.append(r.get("omega", float("nan")))
        taus.append(r.get("tau", float("nan")))
        omegas1.append(r.get("omega1", float("nan")))
        taus1.append(r.get("tau1", float("nan")))

    omegas_a = np.asarray(omegas, dtype=float)
    taus_a = np.asarray(taus, dtype=float)

    # Rolling stddev to find the most stable contiguous block.
    win = max(3, int(round(plateau_frac * n_starts)))
    valid = np.isfinite(omegas_a) & np.isfinite(taus_a)
    if valid.sum() < win:
        return {
            "omega": float("nan"), "tau": float("nan"),
            "omega_std": float("nan"), "tau_std": float("nan"),
            "t_starts": t_starts.tolist(),
            "omegas": omegas, "taus": taus,
            "omegas1": omegas1, "taus1": taus1,
            "plateau_idx": [],
            "t0_plateau_min": float("nan"), "t0_plateau_max": float("nan"),
        }

    # Combined stability score (relative stddev of omega + tau)
    best_score = np.inf
    best_start = 0
    for i in range(0, n_starts - win + 1):
        o_blk = omegas_a[i:i+win]
        ta_blk = taus_a[i:i+win]
        if not (np.all(np.isfinite(o_blk)) and np.all(np.isfinite(ta_blk))):
            continue
        score = (np.std(o_blk) / abs(np.mean(o_blk))
                 + np.std(ta_blk) / abs(np.mean(ta_blk)))
        if score < best_score:
            best_score = score
            best_start = i

    idx = list(range(best_start, best_start + win))
    o_pl = omegas_a[idx]
    t_pl = taus_a[idx]
    return {
        "omega": float(np.mean(o_pl)),
        "tau": float(np.mean(t_pl)),
        "omega_std": float(np.std(o_pl)),
        "tau_std": float(np.std(t_pl)),
        "t_starts": t_starts.tolist(),
        "omegas": omegas, "taus": taus,
        "omegas1": omegas1, "taus1": taus1,
        "plateau_idx": idx,
        "t0_plateau_min": float(t_starts[idx[0]]),
        "t0_plateau_max": float(t_starts[idx[-1]]),
    }


def qnm_method_5_2d_scan(
    t: np.ndarray,
    y: np.ndarray,
    t_start_min: float,
    t_start_max: float,
    t_end_min: float,
    t_end_max: float,
    n_starts: int = 10,
    n_ends: int = 6,
    potential: str = "zerilli",
    ell: int = 2,
    plateau_frac_t0: float = 0.5,
    plateau_frac_te: float = 0.5,
    min_window: float = 8.0,
) -> Dict[str, object]:
    """
    Method 5: two-dimensional stability scan over both the fit start time t_0
    and the fit end time t_end.

    Generalises ``qnm_method_4_window_scan``: for each (t_0, t_end) on the
    rectangular grid we run the two-mode NLS fit and store the recovered
    fundamental (omega, tau). The "plateau rectangle" is the contiguous block
    of size (w_t0 x w_te) with the smallest combined relative scatter in
    (omega, tau). Cells with t_end - t_0 < min_window are excluded.

    Returns dict with:
        "omega", "tau"          : plateau-rectangle mean of fundamental
        "omega_std", "tau_std"  : plateau-rectangle stddev (systematic)
        "t_starts", "t_ends"    : 1-D grid axes (length n_starts, n_ends)
        "omegas_grid"           : (n_ends, n_starts) array of omega fits
        "taus_grid"             : (n_ends, n_starts) array of tau fits
        "plateau_t0_idx"        : list of t_starts indices in the plateau
        "plateau_te_idx"        : list of t_ends indices in the plateau
        "t0_plateau_min/max"    : start-time bounds of plateau rectangle
        "te_plateau_min/max"    : end-time bounds of plateau rectangle
    """
    t_starts = np.linspace(t_start_min, t_start_max, n_starts)
    t_ends = np.linspace(t_end_min, t_end_max, n_ends)

    omegas_grid = np.full((n_ends, n_starts), np.nan, dtype=float)
    taus_grid = np.full((n_ends, n_starts), np.nan, dtype=float)

    for j, te in enumerate(t_ends):
        for i, t0 in enumerate(t_starts):
            if (te - t0) < min_window:
                continue
            r = qnm_method_4_two_mode(
                t, y, float(t0), float(te), potential=potential, ell=ell
            )
            omegas_grid[j, i] = r.get("omega", float("nan"))
            taus_grid[j, i] = r.get("tau", float("nan"))

    # Find best stability rectangle of size (w_te x w_t0).
    w_t0 = max(2, int(round(plateau_frac_t0 * n_starts)))
    w_te = max(2, int(round(plateau_frac_te * n_ends)))

    best_score = np.inf
    best_ij = None
    for j in range(0, n_ends - w_te + 1):
        for i in range(0, n_starts - w_t0 + 1):
            o_blk = omegas_grid[j:j+w_te, i:i+w_t0]
            ta_blk = taus_grid[j:j+w_te, i:i+w_t0]
            if not (np.all(np.isfinite(o_blk)) and np.all(np.isfinite(ta_blk))):
                continue
            score = (np.std(o_blk) / abs(np.mean(o_blk))
                     + np.std(ta_blk) / abs(np.mean(ta_blk)))
            if score < best_score:
                best_score = score
                best_ij = (j, i)

    if best_ij is None:
        return {
            "omega": float("nan"), "tau": float("nan"),
            "omega_std": float("nan"), "tau_std": float("nan"),
            "t_starts": t_starts.tolist(), "t_ends": t_ends.tolist(),
            "omegas_grid": omegas_grid.tolist(),
            "taus_grid": taus_grid.tolist(),
            "plateau_t0_idx": [], "plateau_te_idx": [],
            "t0_plateau_min": float("nan"), "t0_plateau_max": float("nan"),
            "te_plateau_min": float("nan"), "te_plateau_max": float("nan"),
        }

    j0, i0 = best_ij
    t0_idx = list(range(i0, i0 + w_t0))
    te_idx = list(range(j0, j0 + w_te))
    o_pl = omegas_grid[j0:j0+w_te, i0:i0+w_t0]
    t_pl = taus_grid[j0:j0+w_te, i0:i0+w_t0]

    return {
        "omega": float(np.mean(o_pl)),
        "tau": float(np.mean(t_pl)),
        "omega_std": float(np.std(o_pl)),
        "tau_std": float(np.std(t_pl)),
        "t_starts": t_starts.tolist(),
        "t_ends": t_ends.tolist(),
        "omegas_grid": omegas_grid.tolist(),
        "taus_grid": taus_grid.tolist(),
        "plateau_t0_idx": t0_idx,
        "plateau_te_idx": te_idx,
        "t0_plateau_min": float(t_starts[t0_idx[0]]),
        "t0_plateau_max": float(t_starts[t0_idx[-1]]),
        "te_plateau_min": float(t_ends[te_idx[0]]),
        "te_plateau_max": float(t_ends[te_idx[-1]]),
    }

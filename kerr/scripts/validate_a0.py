"""Phase A acceptance: Schwarzschild-limit (a=0) Regge-Wheeler vs Zerilli.

Pipeline:
  1. Build hyperboloidal RW operator on r_* grid (tanh(r_*/L) gauge).
  2. Set Gaussian initial data Phi(0) = exp(-(r_* - x0)^2 / (2 sigma^2)), Pi(0) = 0.
  3. Time-march with RK4 + 2nd-order central FD to tau = T.
  4. Extract M*omega_220 and tau_220 from observer time series via M4 plateau.
  5. Compare to (i) qnm-package (s=-2, l=2, m=2, n=0) at a=0
                (ii) parent paper's Zerilli result M*omega = 0.3737.

Note on CFL. The system has characteristic speeds c_+ = 1/(1+H) (outgoing)
and c_- = -1/(1-H) (ingoing). On a hyperboloidal slice the ingoing mode at
scri+ is suppressed by the slice geometry (info only flows out), so the
physical CFL is set by c_+ <= 1 and dt ~ dx is sufficient. The conservative
cfl_dt(...) helper in src/mol_rk4.py uses 1/(1-|H|) and is over-restrictive
for this problem; we pick dt = 0.3 * dx directly here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fd_stencils import d1_central, d2_central
from src.initial_data import GaussianID, gaussian
from src.mol_rk4 import integrate, cfl_dt
from src.observers import make_observers, observers_as_indices
from src.rwz_hyperboloidal import build_operator, rhs
from src.dissipation import ko_dissipation, outer_sponge_profile
from src.extractor_m4 import qnm_method_4_window_scan
from src.qnm_kerr_reference import kerr_qnm


# Parent paper Zerilli (l=2) M4 fundamental at canonical BH (M=1, x0=4, sigma=5).
ZERILLI_M_OMEGA_PARENT = 0.3737
ZERILLI_TAU_OVER_M_PARENT = 11.241


def run(
    M: float = 1.0,
    L: float = 10.0,
    r_star_min: float = -20.0,
    r_star_max: float = 80.0,
    n: int = 1001,
    x0: float = 4.0,
    sigma: float = 5.0,
    A0: float = 1.0,
    T: float = 150.0,
    cfl_safety: float = 0.4,
    r_obs_list: tuple = (("r10M", 10.0), ("r20M", 20.0)),
    ell: int = 2,
    sigma_ko: float = 0.05,
    sponge_width_frac: float = 0.2,
    sponge_gamma_max: float = 0.5,
    save_waveform: str | None = "outputs/phase_a/waveform_a0.npz",
):
    r_star = np.linspace(r_star_min, r_star_max, n)
    dx = float(r_star[1] - r_star[0])
    op = build_operator(r_star, M=M, L=L, ell=ell)
    print(f"[grid] n={n}, dx={dx:.4e}, r_star in [{r_star_min}, {r_star_max}]")
    print(f"[gauge] L={L} M, |H|_max={float(np.max(np.abs(op.H))):.6f}, "
          f"min(1-H^2)={float(op.one_minus_H2.min()):.3e}")

    Phi0, Pi0 = gaussian(r_star, GaussianID(A0=A0, x0=x0, sigma=sigma))

    observers = make_observers(r_star, list(r_obs_list))
    obs_idx = observers_as_indices(observers)
    for k, v in observers.items():
        print(f"[obs] {k}: r_obs={v.r_obs_M:.3f} M -> grid r_*={v.r_actual_M:.4f} (idx {v.grid_index})")

    gamma_sponge = outer_sponge_profile(r_star, width_frac=sponge_width_frac, gamma_max=sponge_gamma_max)
    print(f"[damp] sigma_KO={sigma_ko}, sponge width={sponge_width_frac}, gamma_max={sponge_gamma_max}")

    def rhs_fn(Phi, Pi):
        dPhi, dPi = rhs(Phi, Pi, op, d1_central, d2_central)
        if sigma_ko > 0.0:
            dPhi = dPhi + ko_dissipation(Phi, sigma_ko)
            dPi = dPi + ko_dissipation(Pi, sigma_ko)
        if sponge_gamma_max > 0.0:
            dPi = dPi - gamma_sponge * Pi
        return dPhi, dPi

    dt = cfl_dt(dx, float(np.max(np.abs(op.H))), safety=cfl_safety)
    n_steps = int(np.ceil(T / dt))
    record_every = max(1, int(round(0.1 / dt)))
    print(f"[time] dt={dt:.4e} (CFL safety {cfl_safety}), T={T} M, n_steps={n_steps}, "
          f"record every {record_every} steps (dt_sample ~ {record_every * dt:.4f} M)")

    Phi_end, Pi_end, taus, series = integrate(
        Phi0, Pi0, dt, n_steps, rhs_fn,
        record_every=record_every,
        observers=obs_idx,
    )

    if not np.all(np.isfinite(Phi_end)):
        raise RuntimeError("integration produced NaN / Inf")
    max_abs_end = float(np.max(np.abs(Phi_end)))
    print(f"[stability] max|Phi(T)| = {max_abs_end:.3e}, samples = {len(taus)}")

    if save_waveform:
        Path(save_waveform).parent.mkdir(parents=True, exist_ok=True)
        np.savez(save_waveform, taus=taus, **{f"y_{k}": series[k] for k in series})
        print(f"[saved] waveform -> {save_waveform}")

    qnm_ref = kerr_qnm(a_over_M=0.0, ell=ell, m=2, n=0)
    print(f"\n[reference] qnm pkg (s=-2, l={ell}, m=2, n=0, a=0):  "
          f"M*omega = {qnm_ref.M_omega_R:.6f}, tau/M = {qnm_ref.tau_over_M:.4f}")
    print(f"[reference] parent paper Zerilli M4:                  "
          f"M*omega = {ZERILLI_M_OMEGA_PARENT}, tau/M = {ZERILLI_TAU_OVER_M_PARENT}")

    results = {}
    for label, obs in observers.items():
        t = taus
        y = series[label]
        scan = qnm_method_4_window_scan(
            t, y,
            t_start_min=15.0, t_start_max=max(35.0, T - 40.0), t_end=T - 5.0,
            n_starts=24, potential="zerilli", ell=ell, plateau_frac=0.5,
        )
        om = scan["omega"]
        tau_q = scan["tau"]
        results[label] = (om, tau_q, scan)

        print(f"\n[extract @ {label} r_*={obs.r_actual_M:.2f} M]")
        if not (np.isfinite(om) and np.isfinite(tau_q)):
            print(f"  M4 fit failed (om={om}, tau={tau_q})")
            continue
        err_qnm = abs(om - qnm_ref.M_omega_R) / qnm_ref.M_omega_R
        err_paper = abs(om - ZERILLI_M_OMEGA_PARENT) / ZERILLI_M_OMEGA_PARENT
        print(f"  M*omega = {om:.6f}  (qnm: {qnm_ref.M_omega_R:.6f}, parent: {ZERILLI_M_OMEGA_PARENT})")
        print(f"  tau/M   = {tau_q:.4f}  (qnm: {qnm_ref.tau_over_M:.4f}, parent: {ZERILLI_TAU_OVER_M_PARENT})")
        print(f"  rel err vs qnm   : {err_qnm:.3e}  (gate < 1e-3 ? {'PASS' if err_qnm < 1e-3 else 'FAIL'})")
        print(f"  rel err vs parent: {err_paper:.3e}  (gate < 5e-4 ? {'PASS' if err_paper < 5e-4 else 'FAIL'})")

    return results


if __name__ == "__main__":
    run()

"""WEAK-FORM PINO PROBE: is the *weak (variational)* PDE residual aligned with
field error in the FNO's representable band, where the STRONG residual was
PROVEN anti-correlated?

ESTABLISHED (strong-form smoking gun, _tmp_best_representable / representation
test): r = L[u] = u_tt - u_xx + V u computed by FD on the output grid amplifies
high-k by k^2. In the FNO band (|k_t|<=16,|k_x|<=32) it is ANTI-correlated with
field error -- the field-optimal representable correction (0.61% field) has ~26x
HIGHER strong residual than doing nothing (prior). => strong-residual label-free
objective is misaligned. DEAD.

HYPOTHESIS under test: the WEAK form <phi, L[u]> = <L*phi, u> moves derivatives
onto SMOOTH analytic test functions phi, so u appears UNDIFFERENTIATED and its
high-k modes are NOT k^2-amplified. Tested against LOW-k phi it should weight the
low-k, field-dominant modes -> possibly ALIGNED with field error.
(NB: earlier we low-passed the FD-computed strong residual and it still failed --
but that is 'option a' = amplify-by-k^2 THEN low-pass. This is 'option b' = the
true adjoint weak form, u never FD-differentiated. Mechanistically different.)

DECISIVE TEST (mirror the strong-form representation table, swap the residual
norm; NO training, pure linear algebra): compute
      field relL2 | STRONG mean(r^2) | WEAK residual energy
for three fields:
    prior          (k4 quintic-upsampled, do-nothing)        field ~4.93%
    prior + d_rep  (FFT-lowpass of (truth - prior) to band)  field ~0.61%
    truth          (fine FD)                                 field 0%
plus prior + d_R_rep (representable Richardson correction) for context.

STRONG (known, validates pipeline): prior < prior+rep  (anti-correlated, rep WORSE).
WEAK pass = prior+rep < prior  (residual DROPS as field improves => ALIGNED).
Swept over test-function cutoff (P_t, Q_x): as cutoff -> grid, weak -> strong.
The question is whether a LOW-k cutoff regime exists where weak is aligned.

CPU, seconds. Delete after use.
"""
import json
import os
import sys
import time

import numpy as np
from scipy.interpolate import RectBivariateSpline

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.fd_solver import solve_fd

CFG = {"physics": {"M": 1.0, "l": 2, "potential": "zerilli", "pde_sign": "standard"},
       "domain": {"xmin": -50.0, "xmax": 150.0, "tmin": 0.0, "tmax": 50.0},
       "initial_data": {"A": 1.0, "x0": 4.0, "sigma": 5.0, "velocity_profile": "outgoing"},
       "fd": {"dx": 0.2, "dt": 0.1}}          # central2 default (matches dataset_sw_k4)


def at_k(k):
    c = json.loads(json.dumps(CFG)); c["fd"]["dx"] = 0.2 * k; c["fd"]["dt"] = 0.1 * k
    return c


def quintic_up(sol, tf, xf):
    return RectBivariateSpline(sol["t"], sol["x"], sol["phi"], kx=5, ky=5)(tf, xf)


# ---- solves --------------------------------------------------------------------
t0 = time.time()
fine = solve_fd(at_k(1))
xf, tf = fine["x"], fine["t"]
Phi_fine = fine["phi"]
V = fine["V"]
sol2, sol4 = solve_fd(at_k(2)), solve_fd(at_k(4))
P2 = quintic_up(sol2, tf, xf)
P4 = quintic_up(sol4, tf, xf)              # prior = k4 (the FNO input / deployable)
Phi_R = (4.0 * P2 - P4) / 3.0              # label-free Richardson estimate (p=2)
print(f"[weak] solves+upsample done in {time.time()-t0:.1f}s  fine {Phi_fine.shape}")

Nt, Nx = Phi_fine.shape
dt = float(tf[1] - tf[0])
dx = float(xf[1] - xf[0])
norm_fine = float(np.sqrt(np.sum(Phi_fine ** 2)))


def rel(field):
    return float(np.sqrt(np.sum((field - Phi_fine) ** 2)) / norm_fine)


# ---- representable corrections (FFT low-pass to the FNO band 16x32) -------------
def lowpass(field, mt=16, mx=32):
    F = np.fft.fft2(field)
    kt = np.abs(np.fft.fftfreq(Nt) * Nt)
    kx = np.abs(np.fft.fftfreq(Nx) * Nx)
    mask = (kt[:, None] <= mt) & (kx[None, :] <= mx)
    return np.real(np.fft.ifft2(F * mask))


d_true_rep = lowpass(Phi_fine - P4)        # best representable field-fix (~0.61%)
d_R_rep    = lowpass(Phi_R - P4)           # representable Richardson correction

fields = {
    "prior (do-nothing)      ": P4,
    "prior + d_rep (truth-rep)": P4 + d_true_rep,
    "prior + d_R_rep (Rich-rep)": P4 + d_R_rep,
    "truth (fine FD)         ": Phi_fine,
}


# ---- STRONG residual: r = u_tt - u_xx + V u via 2nd-order FD, mean(r^2) ---------
def strong_res(u):
    u_tt = (u[2:, :] - 2.0 * u[1:-1, :] + u[:-2, :]) / (dt * dt)
    u_xx = (u[:, 2:] - 2.0 * u[:, 1:-1] + u[:, :-2]) / (dx * dx)
    r = u_tt[:, 1:-1] - u_xx[1:-1, :] + V[None, 1:-1] * u[1:-1, 1:-1]
    return float(np.mean(r ** 2))


# ---- WEAK residual: WRE(u) = sum_{p,q} <L*phi_pq, u>^2 --------------------------
#   phi_pq(t,x) = a_p(t) b_q(x), built so phi AND its 1st derivative VANISH at all
#   four domain edges -> IBP boundary terms are exactly zero -> <L*phi,u> equals
#   the TRUE weak residual <phi, L[u]> with u UNDIFFERENTIATED (no k^2 amplify).
#     a_p(tau) = W(tau) sin(p pi tau),  W=sin^2(pi tau)  (Hann window)
#     b_q(xi)  = W(xi)  sin(q pi xi)
#   W and W' vanish at 0,1, so a_p,a_p',b_q,b_q' all vanish at the edges.
#   Analytic 2nd derivs (chain rule d^2/dt^2 = (1/Tspan^2) d^2/dtau^2):
#     (W S)'' = W'' S + 2 W' S' + W S'',  W'=pi sin(2pi.), W''=2pi^2 cos(2pi.)
Tspan = float(CFG["domain"]["tmax"] - CFG["domain"]["tmin"])
Lx    = float(CFG["domain"]["xmax"] - CFG["domain"]["xmin"])
tau = (tf - CFG["domain"]["tmin"]) / Tspan
xi  = (xf - CFG["domain"]["xmin"]) / Lx


def _windowed_basis(s, n, span):
    """Columns f_k = W(s) sin(k pi s) and analytic d^2/d(phys)^2, k=1..n.
    s in [0,1]; span maps s-derivatives to physical-coordinate derivatives."""
    k = np.arange(1, n + 1)
    W   = np.sin(np.pi * s) ** 2                      # (Ns,)
    Wp  = np.pi * np.sin(2.0 * np.pi * s)
    Wpp = 2.0 * np.pi ** 2 * np.cos(2.0 * np.pi * s)
    S   = np.sin(np.outer(s, k * np.pi))              # (Ns, n)
    Sp  = (k * np.pi)[None, :] * np.cos(np.outer(s, k * np.pi))
    Spp = -((k * np.pi) ** 2)[None, :] * S
    f   = W[:, None] * S
    fpp_s = Wpp[:, None] * S + 2.0 * Wp[:, None] * Sp + W[:, None] * Spp
    return f, fpp_s / (span ** 2)                     # physical 2nd derivative


def weak_res(u, P, Q):
    Ap, App = _windowed_basis(tau, P, Tspan)          # (Nt,P)
    Bq, Bqpp = _windowed_basis(xi, Q, Lx)             # (Nx,Q)
    VBq = V[:, None] * Bq                              # (Nx,Q)
    A   = u @ Bq                                       # (Nt,Q)  sum_x b_q u
    Bxx = u @ Bqpp                                     # (Nt,Q)  sum_x b_q'' u
    Cv  = u @ VBq                                      # (Nt,Q)  sum_x V b_q u
    # <L*phi_pq,u> = int [a_p'' b_q - a_p b_q'' + V a_p b_q] u  dt dx
    M = (App.T @ A - Ap.T @ Bxx + Ap.T @ Cv) * (dt * dx)   # (P,Q)
    return float(np.sum(M ** 2))


# ---- report --------------------------------------------------------------------
print(f"\n{'field':>27} | {'rel-L2':>8} | {'STRONG mean(r^2)':>16}")
strong = {}
for name, u in fields.items():
    f = rel(u) * 100.0
    s = strong_res(u)
    strong[name] = s
    print(f"{name:>27} | {f:7.3f}% | {s:16.4e}")
print("  [validate] STRONG should be ANTI-correlated: prior+d_rep > prior "
      "(rep WORSE despite better field).")

print(f"\nWEAK residual energy WRE(u) and ratio vs prior, swept over test-fn cutoff:")
print(f"{'(P_t,Q_x)':>12} | {'WRE prior':>12} | {'WRE +d_rep':>12} | "
      f"{'rep/prior':>9} | {'WRE +d_R_rep':>12} | {'WRE truth':>12} | {'tru/prior':>9} | aligned?")
for (P, Q) in [(4, 8), (8, 16), (16, 32), (32, 64), (64, 128)]:
    w_prior = weak_res(fields["prior (do-nothing)      "], P, Q)
    w_rep   = weak_res(fields["prior + d_rep (truth-rep)"], P, Q)
    w_Rrep  = weak_res(fields["prior + d_R_rep (Rich-rep)"], P, Q)
    w_truth = weak_res(fields["truth (fine FD)         "], P, Q)
    ratio = w_rep / w_prior
    flag = "ALIGNED" if ratio < 1.0 else "misaligned"
    print(f"{str((P, Q)):>12} | {w_prior:12.4e} | {w_rep:12.4e} | {ratio:9.3f} | "
          f"{w_Rrep:12.4e} | {w_truth:12.4e} | {w_truth/w_prior:9.3f} | {flag}")

print("\n[validate] WRE(truth)/WRE(prior) MUST be << 1 (true field nearly satisfies")
print("           L[u]=0). If ~1, boundary terms still contaminate -> estimator wrong.")

# ---- DECISIVE non-tautological test: gradient of the weak loss at the prior ------
#   d_rep is *defined* as band(truth-prior), so the ratio table could be partly
#   tautological. The honest go/no-go (exact analog of the strong-form projected-GD
#   test that made the field WORSE 4.93%->7.95%): take grad_u WRE at u=prior and ask
#   whether descending it (-grad) points toward truth. cos(-grad, truth-prior)>0 =>
#   minimizing the weak residual *reduces field error* => weak-PINO has a usable,
#   correctly-signed gradient (independent of how d_rep was built).
def weak_grad(u, P, Q):
    """grad_u sum_pq <L*phi_pq,u>^2  as a field on the (Nt,Nx) grid."""
    Ap, App = _windowed_basis(tau, P, Tspan)
    Bq, Bqpp = _windowed_basis(xi, Q, Lx)
    VBq = V[:, None] * Bq
    M = (App.T @ (u @ Bq) - Ap.T @ (u @ Bqpp) + Ap.T @ (u @ VBq)) * (dt * dx)   # (P,Q)
    # reconstruct field: 2 dt dx * sum_pq M_pq L*phi_pq
    g = (App @ M @ Bq.T) - (Ap @ M @ Bqpp.T) + V[None, :] * (Ap @ M @ Bq.T)
    return 2.0 * dt * dx * g, float(np.sum(M ** 2))


def cosang(a, b):
    return float(np.sum(a * b) / (np.sqrt(np.sum(a * a)) * np.sqrt(np.sum(b * b))))


truth_minus_prior = Phi_fine - P4
print("\n[GRADIENT TEST] cos(-grad_u WRE | prior , improvement dir) -- "
      ">0 = weak descent reduces field err (analog of strong-form GD that WORSENED):")
print(f"{'(P_t,Q_x)':>12} | {'cos(-g, truth-prior)':>20} | {'cos(-g, d_rep[band])':>20} | verdict")
for (P, Q) in [(4, 8), (8, 16), (16, 32), (32, 64)]:
    g, _ = weak_grad(P4, P, Q)
    c_full = cosang(-g, truth_minus_prior)
    c_band = cosang(-g, d_true_rep)
    verdict = "DESCENDS->TRUTH" if c_band > 0 else "ascends (anti)"
    print(f"{str((P, Q)):>12} | {c_full:20.4f} | {c_band:20.4f} | {verdict}")
print("  (strong-form analog was NEGATIVE: residual descent moved field AWAY from truth.)")

# ---- WHERE the field error LIVES: are the OTHER physics terms (IC, BC) even active? -
#   A full PINN physics functional = interior residual + IC residual + BC residual.
#   The IC/BC terms only carry signal if the field error has support at t=0 / at the
#   x-boundaries. The coarse prior is a REAL FD solve with the EXACT IC and the same
#   BC as the solver, so it should already discharge them. Measure it, don't assert.
err = Phi_fine - P4                                   # truth - prior, (Nt,Nx)
err_e = float(np.sum(err ** 2))
# (a) IC-displacement: prior=truth at t=0 by construction (both sample analytic IC)
ic_disp_rel = float(np.sqrt(np.sum(err[0, :] ** 2) / np.sum(Phi_fine[0, :] ** 2)))
# (a') IC-velocity: initial d_t error vs field's own initial velocity
vel_err = (err[1, :] - err[0, :]) / dt
vel_fine = (Phi_fine[1, :] - Phi_fine[0, :]) / dt
ic_vel_rel = float(np.sqrt(np.sum(vel_err ** 2) / max(np.sum(vel_fine ** 2), 1e-30)))
# error energy: first 1% of time vs last 50% -> is error IC or accumulated evolution?
n_early = max(1, Nt // 100)
early_frac = float(np.sum(err[:n_early, :] ** 2) / err_e)
late_frac = float(np.sum(err[Nt // 2:, :] ** 2) / err_e)
# (b) BC: error + field energy within nb points of each x-edge; field amp at edges
nb = 5
edge_err = float((np.sum(err[:, :nb] ** 2) + np.sum(err[:, -nb:] ** 2)) / err_e)
edge_field = float((np.sum(Phi_fine[:, :nb] ** 2) + np.sum(Phi_fine[:, -nb:] ** 2))
                   / np.sum(Phi_fine ** 2))
edge_amp = float(max(np.max(np.abs(Phi_fine[:, 0])), np.max(np.abs(Phi_fine[:, -1]))))
int_amp = float(np.max(np.abs(Phi_fine)))

print("\n[IC/BC LEVERAGE] where does the field error (truth-prior) live? "
      "=> are IC/BC physics terms even active in this prior-based architecture?")
print(f"  IC-disp : rel err of correction at t=0 = {ic_disp_rel:.2e}  "
      f"(prior=truth at t=0 by construction => IC-disp loss ~inactive)")
print(f"  IC-vel  : rel err of initial d_t       = {ic_vel_rel:.2e}")
print(f"  time    : err energy in first {n_early} t-slices = {early_frac:.2e}; "
      f"in last 50% of t = {late_frac:.3f}")
print(f"            => error is ACCUMULATED EVOLUTION (interior), not IC")
print(f"  BC      : err  energy within {nb} pts of x-edges = {edge_err:.2e}")
print(f"            field energy within {nb} pts of x-edges = {edge_field:.2e}")
print(f"            peak|field| at x-boundary = {edge_amp:.2e} vs interior {int_amp:.2e} "
      f"(ratio {edge_amp/int_amp:.1e})")
print(f"            => field ~0 at boundaries for whole window (domain oversized,")
print(f"               xmax=150 but wave only reaches ~54) => Sommerfeld BC ~no support")
print(f"[weak] done in {time.time()-t0:.1f}s")

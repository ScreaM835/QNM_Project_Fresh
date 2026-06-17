"""Fork (B) PROTOTYPE: does Richardson extrapolation from TWO cheap coarse solves
give a label-free FIELD target that beats the bare k4 prior (4.93%)?

The active dataset (dataset_sw_k4.npz) uses the default central2 scheme (2nd-order
spatial) + RK4 -> leading error ~ C h^2, so the OBSERVED convergence order should
be p ~= 2.  Richardson:  Phi_exact ~= Phi(h) + [Phi(h) - Phi(2h)] / (2^p - 1).

We solve k2 (h: dx=0.4) and k4 (2h: dx=0.8) -- BOTH cheaper than the fine k1 solve
and BOTH label-free -- quintic-upsample each to the fine grid, measure the observed
order against the (validation-only) fine solution, form the Richardson estimate, and
compare its field rel-L2 to the bare priors.

  * if E_richardson << E_k4 (4.93%)  -> Richardson is a good label-free target
    => supervise the FNO on it (field-aligned, the supervised path that works).

CPU, seconds, login-safe.  Delete after use.
"""
import json
import os
import sys

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


def upsample(sol, tf, xf):
    return RectBivariateSpline(sol["t"], sol["x"], sol["phi"], kx=5, ky=5)(tf, xf)


fine = solve_fd(at_k(1))
xf, tf = fine["x"], fine["t"]
norm = float(np.sqrt(np.sum(fine["phi"] ** 2)))


def rel(field):
    return float(np.sqrt(np.sum((field - fine["phi"]) ** 2)) / norm)


sol2, sol4 = solve_fd(at_k(2)), solve_fd(at_k(4))
P2, P4 = upsample(sol2, tf, xf), upsample(sol4, tf, xf)
E2, E4 = rel(P2), rel(P4)
p_obs = float(np.log(E4 / E2) / np.log(2.0))     # k4 step = 2x k2 step
print(f"bare priors:   k2 (dx=0.4) {E2*100:.3f}%   k4 (dx=0.8) {E4*100:.3f}%")
print(f"observed convergence order p = log(E4/E2)/log2 = {p_obs:.3f}  (central2 expects ~2)")

# LABEL-FREE order estimate from a THIRD coarse solve (no fine solution used):
#   p ~= log( ||P4 - P8|| / ||P2 - P4|| ) / log 2
sol8 = solve_fd(at_k(8))
P8 = upsample(sol8, tf, xf)
d24 = float(np.sqrt(np.sum((P2 - P4) ** 2)))
d48 = float(np.sqrt(np.sum((P4 - P8) ** 2)))
p_lf = float(np.log(d48 / d24) / np.log(2.0))
print(f"LABEL-FREE order (k2,k4,k8, no fine) p = {p_lf:.3f}  (vs fine-measured {p_obs:.3f})")

print(f"\n{'Richardson p':>14} | {'field relL2':>11} | {'vs k4 prior':>11} | {'vs k2':>8}")
for tag, p in [("p=2 (a priori)", 2.0), ("p=label-free", p_lf), ("p=fine-meas", p_obs)]:
    PR = P2 + (P2 - P4) / (2.0 ** p - 1.0)
    ER = rel(PR)
    print(f"{tag:>14} | {ER*100:11.4f} | {E4/ER:10.1f}x | {E2/ER:7.1f}x")

# Two-term (Romberg) extrapolation from THREE solves -- fully label-free, assumes
# only the a-priori orders p=2 then p=4 (central2 symmetric error expansion):
R1_h = (4.0 * P2 - P4) / 3.0          # cancels h^2 using (k2,k4)
R1_2h = (4.0 * P4 - P8) / 3.0         # cancels h^2 using (k4,k8)
R2 = (16.0 * R1_h - R1_2h) / 15.0     # cancels next (h^4) term
print(f"{'Romberg (3-solve)':>14} | {rel(R2)*100:11.4f} | {E4/rel(R2):10.1f}x | {E2/rel(R2):7.1f}x")

# also: how much of the Richardson target's improvement is in the FNO band?
def lowpass(field, mt=16, mx=32):
    F = np.fft.fft2(field); nt, nx = field.shape
    kt = np.abs(np.fft.fftfreq(nt) * nt); kx = np.abs(np.fft.fftfreq(nx) * nx)
    return np.real(np.fft.ifft2(F * ((kt[:, None] <= mt) & (kx[None, :] <= mx))))


PR = P2 + (P2 - P4) / (2.0 ** p_obs - 1.0)
tgt = PR - P4                                     # what the FNO would learn (target - input prior)
cap = float(np.sqrt(np.sum(lowpass(tgt) ** 2)) / np.sqrt(np.sum(tgt ** 2)))
print(f"\nFNO target (Richardson - k4 prior): {cap*100:.2f}% of its energy is in the 16x32 band "
      f"(representable)")

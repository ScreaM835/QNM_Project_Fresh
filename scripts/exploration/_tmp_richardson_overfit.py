"""Fork (B) DIAGNOSTIC: can the REAL hybrid FNO learn the Richardson target?

This is the go/no-go before building the production Richardson pipeline. We take
the validated label-free Richardson field target (two cheap coarse solves k2,k4,
quintic-upsampled, Phi_R = (4 P2 - P4)/3) and OVERFIT the actual hybrid FNO
(src/hybrid_fno.build_hybrid_fno, ~1.1M params, 16x32 modes) on a SINGLE M=1
sample with the exact supervised convention used by the trainer:

    correction = FNO(X)                       # 5-channel input, no gate
    field      = prior_k4 + correction
    loss       = MSE(correction, Phi_R - prior_k4)

Three numbers tell the whole story, tracked over epochs:
  * err_vs_richardson -> 0     : FNO can REPRESENT + FIT the target  (representability)
  * err_vs_fine -> ~0.36%      : honest field error floor = Richardson ceiling
  * (prior err_vs_fine 4.93%)  : the bar we must beat

PASS = err_vs_fine descends from 4.93% toward ~0.4% and plateaus there (it cannot
go below the target's own 0.36% error vs the true fine field). FAIL = stuck near
4.93% (cannot optimise) or the target itself is not representable.

CPU, single sample, minutes.  Delete after use.
"""
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.interpolate import RectBivariateSpline

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.fd_solver import solve_fd
from src.hybrid_fno import build_hybrid_fno, count_parameters, HYBRID_IN_CHANNELS

torch.manual_seed(0)
np.random.seed(0)

CFG = {"physics": {"M": 1.0, "l": 2, "potential": "zerilli", "pde_sign": "standard"},
       "domain": {"xmin": -50.0, "xmax": 150.0, "tmin": 0.0, "tmax": 50.0},
       "initial_data": {"A": 1.0, "x0": 4.0, "sigma": 5.0, "velocity_profile": "outgoing"},
       "fd": {"dx": 0.2, "dt": 0.1}}          # central2 default (matches dataset_sw_k4)

FNO_CFG = {"fno": {"modes_t": 16, "modes_x": 32, "hidden_channels": 32,
                   "n_layers": 4, "domain_padding": 0.10,
                   "positional_embedding": "grid"}}


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
print(f"[overfit] solves done in {time.time()-t0:.1f}s  fine {Phi_fine.shape}")

P2 = quintic_up(sol2, tf, xf)
P4 = quintic_up(sol4, tf, xf)             # = input prior channel 0 (k4, deployable)
Phi_R = (4.0 * P2 - P4) / 3.0             # label-free Richardson estimate (p=2)

norm_fine = float(np.sqrt(np.sum(Phi_fine ** 2)))
norm_R = float(np.sqrt(np.sum(Phi_R ** 2)))


def rel(a, b, nrm):
    return float(np.sqrt(np.sum((a - b) ** 2)) / nrm)


print(f"[overfit] bare k4 prior  err_vs_fine = {rel(P4, Phi_fine, norm_fine)*100:.3f}%")
print(f"[overfit] Richardson tgt err_vs_fine = {rel(Phi_R, Phi_fine, norm_fine)*100:.3f}%  "
      f"(this is the label-free CEILING)")

# ---- assemble the 5-channel input EXACTLY like src/hybrid_data_pipe -------------
Nt, Nx = P4.shape
dt_c = float(sol4["t"][1] - sol4["t"][0])
Phi0 = P4[0, :]                                          # IC displacement (t=0)
Pi0_c = (sol4["phi"][1, :] - sol4["phi"][0, :]) / dt_c   # IC velocity (coarse FD)
Pi0 = np.interp(xf, sol4["x"], Pi0_c)

X = np.zeros((1, HYBRID_IN_CHANNELS, Nt, Nx), dtype=np.float32)
X[0, 0] = P4
X[0, 1] = np.broadcast_to(Phi0, (Nt, Nx))
X[0, 2] = np.broadcast_to(V, (Nt, Nx))
X[0, 3] = 1.0                                            # M
X[0, 4] = np.broadcast_to(Pi0, (Nt, Nx))

# Two delta targets that differ by only ~0.36% of ||fine||:
#   richardson : label-free  (Phi_R - P4)        <- the method under test
#   supervised : KNOWN-GOOD  (Phi_fine - P4)     <- positive control; real
#                pipeline fits this to 0.07%. If the toy stalls on THIS, the
#                single-sample toy itself is the bottleneck (not the target).
Y_rich = (Phi_R    - P4)[None, None].astype(np.float32)
Y_sup  = (Phi_fine - P4)[None, None].astype(np.float32)

xb = torch.from_numpy(X)
P4_t = torch.from_numpy(P4.astype(np.float32))
Phi_fine_np = Phi_fine.astype(np.float64)
Phi_R_np = Phi_R.astype(np.float64)

STEPS = 200
LOG = 20


def run_target(name, Y_np):
    """Fresh model (same seed => same init), Adam-only, single-sample overfit."""
    torch.manual_seed(0)                       # identical init for both targets
    model = build_hybrid_fno(FNO_CFG)
    yb = torch.from_numpy(Y_np)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    print(f"\n=== target: {name}  (params {count_parameters(model)}) ===")
    print(f"{'step':>5} | {'loss':>11} | {'err_vs_fine':>11} | {'err_vs_Rich':>11}")
    t0 = time.time()
    for s in range(STEPS + 1):
        model.train()
        opt.zero_grad()
        pred = model(xb)
        loss = torch.mean((pred - yb) ** 2)
        if s > 0:
            loss.backward()
            opt.step()
        if s % LOG == 0:
            with torch.no_grad():
                recon = (P4_t + pred[0, 0]).detach().numpy().astype(np.float64)
            ef = rel(recon, Phi_fine_np, norm_fine) * 100.0
            er = rel(recon, Phi_R_np, norm_R) * 100.0
            print(f"{s:5d} | {loss.item():11.4e} | {ef:10.3f}% | {er:10.3f}%",
                  flush=True)
    print(f"[{name}] done in {time.time()-t0:.1f}s")


# Positive control FIRST (supervised, known-good target), then the method.
run_target("supervised (Phi_fine - P4)  KNOWN-GOOD", Y_sup)
run_target("richardson (Phi_R - P4)     label-free", Y_rich)
print("\n[overfit] prior 4.93% | Richardson ceiling 0.365% | supervised real-pipeline floor 0.07%")
print("[overfit] READ: if BOTH stall ~4.9% => single-sample toy is the bottleneck,")
print("          not the target => use real multi-sample training. If supervised")
print("          descends but richardson stalls => richardson target is the problem.")


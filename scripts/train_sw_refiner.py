"""Stage 1 of the SW hybrid super-resolution plan: single-config, LABEL-FREE,
coarse-anchored PINN refiner.

Goal: take a CHEAP coarse FD prior (k4, 16x cheaper than fine) and refine it into
a fine-fidelity waveform using ONLY the physics (Zerilli PDE residual) + an anchor
to the coarse prior + a hard initial-condition ansatz. **No fine-FD data is used
in the loss** -- that is the whole point: this recipe transfers to higher
dimensions where fine FD cannot be generated.

  u_theta(x,t) = u0(x) + (t/T)^2 * N_theta(x,t)         (hard IC: u(.,0)=u0, u_t(.,0)=0)
  L = lam_res * || u_tt - u_xx + V u ||^2               (Zerilli residual, autograd)
    + alpha   * || u_theta - u_prior ||^2               (anchor to cheap coarse prior)
    + lam_bc  * (Sommerfeld outflow at both x-ends)

The anchor replaces the fine-FD data loss and kills the null space (phase
ambiguity) that otherwise collapses a from-scratch PINN. alpha is the key knob:
too small -> collapse risk; too large -> inherit the prior's error.

Evaluation (vs fine FD oracle + Leaver, the bare prior as control C0):
  * waveform field rel-L2 on the fine (x,t) grid           [PRIMARY]
  * QNM omega/tau via the M1-M5 suite at x_q=2             [VALIDATION]

FP64 throughout. Login node is CPU; a short smoke runs here, the real run goes to
SLURM (see --adam/--lbfgs).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Tuple

import numpy as np
import torch
from scipy.interpolate import RectBivariateSpline

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, ".."))
sys.path.insert(0, _REPO)

from src.fd_solver import solve_fd
from src.potentials import V_of_x_torch
from src.initial_data import gaussian_phi
from src.qnm import (
    qnm_method_1, qnm_method_2, qnm_method_3_esprit,
    qnm_method_4_window_scan, qnm_method_5_2d_scan,
    percentage_errors, theory_ref,
)

DTYPE = torch.float64
METHODS = ["M1", "M2", "M3", "M4", "M5"]

# Canonical SW/Zerilli config (matches configs/hybrid_sw_dataset.yaml base grid).
CFG = {
    "physics": {"M": 1.0, "l": 2, "potential": "zerilli", "pde_sign": "standard"},
    "domain": {"xmin": -50.0, "xmax": 150.0, "tmin": 0.0, "tmax": 50.0},
    "initial_data": {"A": 1.0, "x0": 4.0, "sigma": 5.0,
                     "velocity_profile": "outgoing"},
    "fd": {"dx": 0.2, "dt": 0.1, "scheme": "rk4_mol"},
}
FINE_DX, FINE_DT = 0.2, 0.1
M_BH, ELL, POT = 1.0, 2, "zerilli"
A_IC, X0_IC, SIG_IC = 1.0, 4.0, 5.0
XMIN, XMAX, TMIN, TMAX = -50.0, 150.0, 0.0, 50.0


# ---------------------------------------------------------------------------
# FD solves: cheap coarse prior + fine reference (reference used ONLY for eval).
# ---------------------------------------------------------------------------

def _cfg_at_k(k: int) -> Dict:
    c = json.loads(json.dumps(CFG))
    c["fd"]["dx"] = FINE_DX * k
    c["fd"]["dt"] = FINE_DT * k
    return c


def make_prior_interp(k: int):
    """Solve the k-coarse FD prior and return (interp_callable, solve_dict).

    interp(x_t) evaluates the prior field at arbitrary (x,t) points via cubic
    spline (the prior is a fixed target, so no autograd needed through it)."""
    sol = solve_fd(_cfg_at_k(k))
    spl = RectBivariateSpline(sol["t"], sol["x"], sol["phi"], kx=3, ky=3)

    def interp(x: np.ndarray, t: np.ndarray) -> np.ndarray:
        return spl.ev(t, x)
    return interp, sol


# ---------------------------------------------------------------------------
# Network + hard-IC ansatz.
# ---------------------------------------------------------------------------

class MLP(torch.nn.Module):
    def __init__(self, width: int = 128, depth: int = 5):
        super().__init__()
        layers = [torch.nn.Linear(2, width), torch.nn.Tanh()]
        for _ in range(depth - 1):
            layers += [torch.nn.Linear(width, width), torch.nn.Tanh()]
        layers += [torch.nn.Linear(width, 1)]
        self.net = torch.nn.Sequential(*layers).to(DTYPE)
        # normalisation constants -> map (x,t) to ~[-1,1]
        self.register_buffer("x_mid", torch.tensor(0.5 * (XMIN + XMAX), dtype=DTYPE))
        self.register_buffer("x_half", torch.tensor(0.5 * (XMAX - XMIN), dtype=DTYPE))
        self.register_buffer("t_scale", torch.tensor(TMAX, dtype=DTYPE))

    def raw(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        xn = (x - self.x_mid) / self.x_half
        tn = 2.0 * t / self.t_scale - 1.0
        return self.net(torch.cat([xn, tn], dim=1))


def u0_torch(x: torch.Tensor) -> torch.Tensor:
    """Hard-IC base: the exact Gaussian initial pulse (zero initial velocity)."""
    return A_IC * torch.exp(-((x - X0_IC) ** 2) / (SIG_IC ** 2))


def forward_field(model: MLP, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Hard-IC ansatz: u = u0(x) + (t/T)^2 * N(x,t).

    (t/T)^2 and its t-derivative vanish at t=0, so u(.,0)=u0 and u_t(.,0)=0
    EXACTLY for any network output (kills the phase ambiguity structurally)."""
    s = t / TMAX
    return u0_torch(x) + (s * s) * model.raw(x, t)


# ---------------------------------------------------------------------------
# Losses.
# ---------------------------------------------------------------------------

def pde_residual(model: MLP, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Zerilli residual u_tt - u_xx + V u via autograd (EXACT, no stencil)."""
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)
    u = forward_field(model, x, t)
    ones = torch.ones_like(u)
    u_x = torch.autograd.grad(u, x, ones, create_graph=True)[0]
    u_t = torch.autograd.grad(u, t, ones, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t, torch.ones_like(u_t), create_graph=True)[0]
    V = V_of_x_torch(x, M_BH, ELL, POT)
    return u_tt - u_xx + V * u


def bc_residual(model: MLP, t: torch.Tensor) -> torch.Tensor:
    """Sommerfeld outflow at both truncated ends:
       left  (x->-inf): u_t - u_x = 0
       right (x->+inf): u_t + u_x = 0."""
    xL = torch.full_like(t, XMIN).requires_grad_(True)
    xR = torch.full_like(t, XMAX).requires_grad_(True)
    tt = t.requires_grad_(True)
    uL = forward_field(model, xL, tt)
    uR = forward_field(model, xR, tt)
    uL_x = torch.autograd.grad(uL, xL, torch.ones_like(uL), create_graph=True)[0]
    uL_t = torch.autograd.grad(uL, tt, torch.ones_like(uL), create_graph=True,
                               retain_graph=True)[0]
    uR_x = torch.autograd.grad(uR, xR, torch.ones_like(uR), create_graph=True)[0]
    uR_t = torch.autograd.grad(uR, tt, torch.ones_like(uR), create_graph=True)[0]
    return torch.cat([uL_t - uL_x, uR_t + uR_x], dim=0)


# ---------------------------------------------------------------------------
# Collocation sampling.
# ---------------------------------------------------------------------------

def sample_interior(n: int, rng: np.random.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    x = rng.uniform(XMIN, XMAX, size=(n, 1))
    t = rng.uniform(TMIN, TMAX, size=(n, 1))
    return (torch.tensor(x, dtype=DTYPE), torch.tensor(t, dtype=DTYPE))


def anchor_target(interp, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    vals = interp(x.numpy().ravel(), t.numpy().ravel())
    return torch.tensor(vals.reshape(-1, 1), dtype=DTYPE)


# ---------------------------------------------------------------------------
# Evaluation.
# ---------------------------------------------------------------------------

def _safe(fn, *a, **k):
    try:
        r = fn(*a, **k)
        return {"omega": float(r.get("omega", np.nan)), "tau": float(r.get("tau", np.nan))}
    except Exception:
        return {"omega": float("nan"), "tau": float("nan")}


def qnm_all_methods(t: np.ndarray, y: np.ndarray, t_start=12.0, t_end=48.0
                    ) -> Dict[str, Dict[str, float]]:
    te = min(t_end, float(t[-1]))
    raw = {
        "M1": _safe(qnm_method_1, t, y, t_start, te),
        "M2": _safe(qnm_method_2, t, y, t_start, te),
        "M3": _safe(qnm_method_3_esprit, t, y, t_start, te, K=4),
        "M4": _safe(qnm_method_4_window_scan, t, y, t_start_min=t_start,
                    t_start_max=t_start + 8.0, t_end=te, n_starts=12,
                    potential=POT, ell=ELL),
        "M5": _safe(qnm_method_5_2d_scan, t, y, t_start_min=t_start,
                    t_start_max=t_start + 8.0, t_end_min=te - 10.0, t_end_max=te,
                    n_starts=8, n_ends=5, potential=POT, ell=ELL),
    }
    out = {}
    for name, res in raw.items():
        err = percentage_errors(res, potential=POT, ell=ELL, M=M_BH)
        out[name] = {"omega_pct": err.get("omega_pct_err", np.nan),
                     "tau_pct": err.get("tau_pct_err", np.nan)}
    return out


def field_rel_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    num = float(np.sqrt(np.sum((pred - ref) ** 2)))
    den = float(np.sqrt(np.sum(ref ** 2)))
    return num / den if den > 0 else float("inf")


@torch.no_grad()
def predict_grid(model: MLP, x: np.ndarray, t: np.ndarray) -> np.ndarray:
    X, T = np.meshgrid(x, t)
    xt = torch.tensor(X.ravel()[:, None], dtype=DTYPE)
    tt = torch.tensor(T.ravel()[:, None], dtype=DTYPE)
    u = forward_field(model, xt, tt).numpy().reshape(X.shape)
    return u


def series_at_xq(field: np.ndarray, x: np.ndarray, t: np.ndarray, xq: float) -> np.ndarray:
    spl = RectBivariateSpline(t, x, field, kx=3, ky=3)
    return spl.ev(t, np.full_like(t, xq))


# ---------------------------------------------------------------------------
# Training driver.
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Stage 1 coarse-anchored label-free SW refiner")
    ap.add_argument("--k", type=int, default=4, help="coarse prior factor (default k4=16x cheaper)")
    ap.add_argument("--alpha", type=float, default=1.0, help="anchor weight (KEY knob)")
    ap.add_argument("--lam-res", type=float, default=1.0)
    ap.add_argument("--lam-bc", type=float, default=1.0)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--n-domain", type=int, default=20000)
    ap.add_argument("--n-bc", type=int, default=400)
    ap.add_argument("--adam", type=int, default=15000)
    ap.add_argument("--adam-lr", type=float, default=1e-3)
    ap.add_argument("--lbfgs", type=int, default=15000)
    ap.add_argument("--resample-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.smoke:
        args.adam, args.lbfgs, args.n_domain = 400, 0, 2000
        args.width, args.depth = 32, 3

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    ref = theory_ref(POT, ELL)
    print(f"=== Stage 1 refiner: k{args.k} prior, alpha={args.alpha}, "
          f"label-free (no fine-FD loss) ===", flush=True)
    print(f"  Leaver: omega={ref['omega']:.4f} tau={ref['tau']:.3f}", flush=True)

    # cheap coarse prior + fine reference (reference = EVAL ONLY)
    t0 = time.time()
    interp, prior_sol = make_prior_interp(args.k)
    fine_sol = solve_fd(_cfg_at_k(1))
    print(f"  prior k{args.k} grid={prior_sol['phi'].shape}  "
          f"fine grid={fine_sol['phi'].shape}  ({time.time()-t0:.1f}s)", flush=True)

    # bare-prior control C0: upsample prior to fine grid, score it
    prior_on_fine = RectBivariateSpline(
        prior_sol["t"], prior_sol["x"], prior_sol["phi"], kx=3, ky=3
    )(fine_sol["t"], fine_sol["x"])
    c0_field = field_rel_l2(prior_on_fine, fine_sol["phi"])
    c0_y = series_at_xq(prior_on_fine, fine_sol["x"], fine_sol["t"], 2.0)
    c0_qnm = qnm_all_methods(fine_sol["t"], c0_y)
    print(f"  [C0 bare prior] field rel-L2={c0_field:.4f}  "
          f"QNM best om/tau %: "
          f"{min(c0_qnm[m]['omega_pct'] for m in METHODS):.3f}/"
          f"{min(c0_qnm[m]['tau_pct'] for m in METHODS):.3f}", flush=True)

    model = MLP(width=args.width, depth=args.depth)

    # fixed BC collocation in time
    t_bc = torch.tensor(rng.uniform(TMIN, TMAX, size=(args.n_bc, 1)), dtype=DTYPE)

    def closure_terms():
        xc, tc = sample_interior(args.n_domain, rng)
        r = pde_residual(model, xc, tc)
        u_pred = forward_field(model, xc, tc)
        a_tgt = anchor_target(interp, xc, tc)
        bc = bc_residual(model, t_bc)
        l_res = torch.mean(r ** 2)
        l_anc = torch.mean((u_pred - a_tgt) ** 2)
        l_bc = torch.mean(bc ** 2)
        loss = args.lam_res * l_res + args.alpha * l_anc + args.lam_bc * l_bc
        return loss, l_res, l_anc, l_bc

    # --- Adam ---------------------------------------------------------------
    opt = torch.optim.Adam(model.parameters(), lr=args.adam_lr)
    t0 = time.time()
    for it in range(args.adam):
        opt.zero_grad()
        loss, l_res, l_anc, l_bc = closure_terms()
        loss.backward()
        opt.step()
        if it % max(1, args.adam // 20) == 0 or it == args.adam - 1:
            print(f"  [adam {it:5d}] loss={loss.item():.3e} "
                  f"res={l_res.item():.3e} anc={l_anc.item():.3e} "
                  f"bc={l_bc.item():.3e}", flush=True)
    print(f"  adam done ({time.time()-t0:.1f}s)", flush=True)

    # --- L-BFGS (fixed collocation set for a stable quasi-Newton phase) -------
    if args.lbfgs > 0:
        xc, tc = sample_interior(args.n_domain, rng)
        a_tgt = anchor_target(interp, xc, tc)
        lbfgs = torch.optim.LBFGS(
            model.parameters(), max_iter=args.lbfgs, history_size=100,
            tolerance_grad=1e-12, tolerance_change=0, line_search_fn="strong_wolfe")

        def closure():
            lbfgs.zero_grad()
            r = pde_residual(model, xc, tc)
            u_pred = forward_field(model, xc, tc)
            bc = bc_residual(model, t_bc)
            loss = (args.lam_res * torch.mean(r ** 2)
                    + args.alpha * torch.mean((u_pred - a_tgt) ** 2)
                    + args.lam_bc * torch.mean(bc ** 2))
            loss.backward()
            return loss
        t0 = time.time()
        lbfgs.step(closure)
        print(f"  lbfgs done ({time.time()-t0:.1f}s) final loss={closure().item():.3e}",
              flush=True)

    # --- evaluate vs fine FD + Leaver ---------------------------------------
    pred = predict_grid(model, fine_sol["x"], fine_sol["t"])
    f_rel = field_rel_l2(pred, fine_sol["phi"])
    y2 = series_at_xq(pred, fine_sol["x"], fine_sol["t"], 2.0)
    qnm = qnm_all_methods(fine_sol["t"], y2)
    best_om = min(qnm[m]["omega_pct"] for m in METHODS)
    best_tau = min(qnm[m]["tau_pct"] for m in METHODS)

    print(f"\n  --- Stage 1 result (k{args.k}, alpha={args.alpha}, label-free) ---", flush=True)
    print(f"  field rel-L2:  refiner={f_rel:.4f}   C0 bare prior={c0_field:.4f}", flush=True)
    print(f"  QNM @x_q=2 (best of M1-M5):  omega {best_om:.3f}%   tau {best_tau:.3f}%", flush=True)
    for m in METHODS:
        print(f"     {m}: om {qnm[m]['omega_pct']:.3f}%  tau {qnm[m]['tau_pct']:.3f}%", flush=True)
    improved = f_rel < c0_field
    print(f"  ==> waveform {'IMPROVED' if improved else 'NOT improved'} over bare prior "
          f"({c0_field:.4f} -> {f_rel:.4f})\n", flush=True)

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"k": args.k, "alpha": args.alpha, "smoke": args.smoke,
                       "field_rel_l2": f_rel, "c0_field_rel_l2": c0_field,
                       "qnm_best_omega_pct": best_om, "qnm_best_tau_pct": best_tau,
                       "qnm_per_method": qnm,
                       "train": {"adam": args.adam, "lbfgs": args.lbfgs,
                                 "n_domain": args.n_domain,
                                 "width": args.width, "depth": args.depth}}, f, indent=2)
        print(f"  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

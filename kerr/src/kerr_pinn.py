"""Kerr time-domain Teukolsky PINN (Phase C.3) — DeepXDE model + training.

Solves the second-order complex Teukolsky residual (see
``teukolsky_residual.py``) on the hyperboloidal slice with a physics-informed
loss only: **PDE residual, no FD field, no boundary terms** (hyperboloidal
characteristics are outflow at both ends), and the initial data is enforced
*exactly* by a hard-constraint output ansatz, so there is **no IC loss term**
either. The single training signal is the PDE residual.

Framework: DeepXDE (pytorch backend), reusing its geometry / autodiff / Adam +
L-BFGS machinery. The C.3 baseline is a *plain* FNN with no training
mitigations; an optional Fourier-feature embedding (Tancik 2020 / Ding 2024)
can be switched on via the config if spectral bias shows up. FP64 throughout
(Xu 2025).

The model maps ``(sigma, tau) -> (p, q)`` with ``Psi = p + i q``. The hard-IC
ansatz with ``s = tau / T`` is

    p(sigma, tau) = p0(sigma) + s^2 * N_p(sigma, tau)
    q(sigma, tau) =            s^2 * N_q(sigma, tau)

which satisfies ``p(.,0)=p0``, ``q(.,0)=0`` and ``p_tau(.,0)=q_tau(.,0)=0``
(zero-velocity, time-symmetric Gaussian) for *any* network output, because
``s^2`` and its first ``tau``-derivative vanish at ``tau=0``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import deepxde as dde

try:
    from kerr.src.teukolsky_residual import KerrCoeffs, second_order_residual
except ImportError:
    try:
        from .teukolsky_residual import KerrCoeffs, second_order_residual
    except ImportError:
        from teukolsky_residual import KerrCoeffs, second_order_residual


# ---------------------------------------------------------------------------
# Fourier-feature input embedding (Tancik et al. 2020; Ding et al. 2024 for
# time-domain waves). Random Gaussian frequencies, with the TIME scale set from
# the known QNM oscillation so the embedding is centred on the physics.
# ---------------------------------------------------------------------------

def make_fourier_features(
    n_freq: int,
    sig_min: float,
    sig_max: float,
    T: float,
    scale_sigma: float,
    scale_tau: float,
    include_raw: bool = True,
    seed: int = 0,
) -> Tuple[Callable[[torch.Tensor], torch.Tensor], int]:
    """Build a DeepXDE feature transform and report its output dimension.

    Inputs ``(sigma, tau)`` are first normalised to ``[-1, 1]^2``, then mapped to
    ``[sin(x_n B^T), cos(x_n B^T)]`` with ``B`` a fixed ``(n_freq, 2)`` Gaussian
    matrix whose two columns are scaled by ``scale_sigma`` / ``scale_tau``. The
    raw normalised coordinates are optionally appended (helps the low-frequency
    DC trend).
    """
    rng = np.random.default_rng(seed)
    B = np.stack([
        rng.normal(0.0, scale_sigma, size=n_freq),
        rng.normal(0.0, scale_tau, size=n_freq),
    ], axis=1)                                   # (n_freq, 2)
    B_t = torch.as_tensor(B, dtype=torch.float64)

    def transform(x: torch.Tensor) -> torch.Tensor:
        sg = x[:, 0:1]
        ta = x[:, 1:2]
        sg_n = 2.0 * (sg - sig_min) / (sig_max - sig_min) - 1.0
        ta_n = 2.0 * ta / T - 1.0
        xn = torch.cat([sg_n, ta_n], dim=1)      # (N, 2)
        proj = xn @ B_t.t()                      # (N, n_freq)
        feats = [torch.sin(proj), torch.cos(proj)]
        if include_raw:
            feats.append(xn)
        return torch.cat(feats, dim=1)

    n_out = 2 * n_freq + (2 if include_raw else 0)
    return transform, n_out


# ---------------------------------------------------------------------------
# Plain input normalisation (used when Fourier features are off). Maps the raw
# (sigma, tau) collocation coords to [-1, 1]^2 so tanh sees well-conditioned
# inputs (tau can be O(150)). Same normalisation the Fourier path applies
# internally, so toggling Fourier on/off leaves the input scaling unchanged.
# ---------------------------------------------------------------------------

def make_input_norm(
    sig_min: float, sig_max: float, T: float,
) -> Callable[[torch.Tensor], torch.Tensor]:
    def transform(x: torch.Tensor) -> torch.Tensor:
        sg = x[:, 0:1]
        ta = x[:, 1:2]
        sg_n = 2.0 * (sg - sig_min) / (sig_max - sig_min) - 1.0
        ta_n = 2.0 * ta / T - 1.0
        return torch.cat([sg_n, ta_n], dim=1)
    return transform


# ---------------------------------------------------------------------------
# Hard initial-condition output ansatz (enforces all 4 ICs exactly).
# ---------------------------------------------------------------------------

def make_ic_output_transform(
    r_plus: float, A: float, r0: float, w: float, T: float,
    out_scale: float = 1.0,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return a DeepXDE output transform implementing the hard-IC ansatz.

    ``p0(sigma) = A exp(-(r - r0)^2 / (2 w^2))`` with ``r = r_plus / sigma`` is
    the purely-real, time-symmetric Gaussian initial pulse (identical to the FD
    ``make_initial_pulse``). The factor ``s^2`` (``s = tau / T``) makes the
    ansatz reproduce the IC and zero initial velocity for any network output.

    ``out_scale`` is a multiplicative **preconditioner** on the network's
    time-dependent correction. The hyperboloidal field blueshifts as the pulse
    propagates to scri (the scri waveform reaches ~25x the IC amplitude), so a
    last-layer gain of order that growth speeds convergence. It does NOT bias the
    solution — it is the last-layer scale of a universal approximator — and the
    PDE-residual gate is unaffected.
    """
    def transform(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sg = x[:, 0:1]
        ta = x[:, 1:2]
        s = ta / T
        r = r_plus / sg
        p0 = A * torch.exp(-((r - r0) ** 2) / (2.0 * w ** 2))
        s2 = s * s
        p = p0 + s2 * out_scale * y[:, 0:1]
        q = s2 * out_scale * y[:, 1:2]
        return torch.cat([p, q], dim=1)

    return transform


# ---------------------------------------------------------------------------
# PDE residual closure.
# ---------------------------------------------------------------------------

def make_pde(coeffs: KerrCoeffs):
    """Build the DeepXDE ``pde(x, y)`` returning ``[r_re, r_im]``.

    ``y`` already carries the hard-IC ansatz (applied as an output transform),
    so the autodiff derivatives are of the *physical* fields ``p, q``.
    """
    def pde(x, y):
        p = y[:, 0:1]
        q = y[:, 1:2]
        p_s = dde.grad.jacobian(y, x, i=0, j=0)
        p_t = dde.grad.jacobian(y, x, i=0, j=1)
        q_s = dde.grad.jacobian(y, x, i=1, j=0)
        q_t = dde.grad.jacobian(y, x, i=1, j=1)
        p_ss = dde.grad.hessian(y, x, component=0, i=0, j=0)
        p_tt = dde.grad.hessian(y, x, component=0, i=1, j=1)
        p_ts = dde.grad.hessian(y, x, component=0, i=0, j=1)
        q_ss = dde.grad.hessian(y, x, component=1, i=0, j=0)
        q_tt = dde.grad.hessian(y, x, component=1, i=1, j=1)
        q_ts = dde.grad.hessian(y, x, component=1, i=0, j=1)

        sg = x[:, 0:1]
        lam_in = coeffs.lambda_in(sg)
        lam_out = coeffs.lambda_out(sg)
        c_pi = coeffs.c_pi(sg)
        c_phi = coeffs.c_phi(sg)
        c_psi = coeffs.c_psi(sg)

        r_re, r_im = second_order_residual(
            p_t, q_t, p_s, q_s, p_tt, q_tt, p_ss, q_ss, p_ts, q_ts, p, q,
            lam_in, lam_out, c_pi, c_phi, c_psi)

        return [r_re, r_im]

    return pde


# ---------------------------------------------------------------------------
# Model construction.
# ---------------------------------------------------------------------------

@dataclass
class KerrPINNConfig:
    a_over_M: float
    omega_ref: complex
    r_plus: float
    A: float = 1.0
    r0: float = 10.0
    w: float = 1.0
    out_scale: float = 1.0
    sig_min: float = 1e-8
    sig_max: float = 1.0 - 1e-8
    T: float = 150.0
    # network
    hidden: Tuple[int, ...] = (128, 128, 128, 128)
    activation: str = "tanh"
    use_fourier: bool = False             # baseline = plain FNN; opt-in Fourier
    n_fourier: int = 64
    scale_sigma: float = 4.0
    scale_tau: Optional[float] = None     # default: omega_R * T / 2 (set below)
    # collocation
    num_domain: int = 20000
    seed: int = 1234


def build_model(cfg: KerrPINNConfig) -> Tuple[dde.Model, KerrCoeffs]:
    """Construct the DeepXDE model for one fixed spin/initial-data config."""
    dde.config.set_default_float("float64")
    dde.config.set_random_seed(cfg.seed)

    from kerr.src.teukolsky_residual import coeffs_from_spin
    coeffs = coeffs_from_spin(cfg.a_over_M, omega_ref=cfg.omega_ref)

    geom = dde.geometry.Interval(cfg.sig_min, cfg.sig_max)
    timedomain = dde.geometry.TimeDomain(0.0, cfg.T)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    pde = make_pde(coeffs)

    data = dde.data.TimePDE(
        geomtime, pde, [],
        num_domain=cfg.num_domain, num_boundary=0, num_initial=0,
        train_distribution="Hammersley",
    )

    if cfg.use_fourier:
        scale_tau = cfg.scale_tau
        if scale_tau is None:
            scale_tau = max(abs(cfg.omega_ref.real) * cfg.T / 2.0, 1.0)
        feat_tf, n_feat = make_fourier_features(
            cfg.n_fourier, cfg.sig_min, cfg.sig_max, cfg.T,
            cfg.scale_sigma, scale_tau, include_raw=True, seed=cfg.seed,
        )
    else:
        feat_tf = make_input_norm(cfg.sig_min, cfg.sig_max, cfg.T)
        n_feat = 2

    layer_size = [n_feat] + list(cfg.hidden) + [2]
    net = dde.nn.FNN(layer_size, cfg.activation, "Glorot uniform")
    net = net.double()
    net.apply_feature_transform(lambda x: feat_tf(x))
    net.apply_output_transform(
        make_ic_output_transform(cfg.r_plus, cfg.A, cfg.r0, cfg.w, cfg.T,
                                 out_scale=cfg.out_scale))

    model = dde.Model(data, net)
    return model, coeffs


# ---------------------------------------------------------------------------
# Training.
# ---------------------------------------------------------------------------

def train(
    model: dde.Model,
    adam_iters: int = 15000,
    adam_lr: float = 1.0e-3,
    lbfgs_iters: int = 30000,
    loss_weights: Optional[List[float]] = None,
    resample_period: int = 200,
    display_every: int = 500,
) -> Dict[str, list]:
    """Adam → L-BFGS, with periodic collocation resampling."""
    if loss_weights is None:
        loss_weights = [1.0, 1.0]

    callbacks = [dde.callbacks.PDEPointResampler(period=resample_period)]

    model.compile("adam", lr=adam_lr, loss_weights=loss_weights)
    model.train(iterations=adam_iters, callbacks=callbacks,
                display_every=display_every)

    if lbfgs_iters > 0:
        dde.optimizers.set_LBFGS_options(
            maxcor=100, maxiter=lbfgs_iters, ftol=0, gtol=1e-10, maxls=50)
        model.compile("L-BFGS", loss_weights=loss_weights)
        model.train(display_every=display_every)

    return {"loss_history": model.losshistory.loss_train}


# ---------------------------------------------------------------------------
# Evaluation helpers.
# ---------------------------------------------------------------------------

def predict_field(model: dde.Model, sigma: np.ndarray, tau: np.ndarray
                  ) -> np.ndarray:
    """Predict the complex field on the tensor grid ``(tau, sigma)``.

    Returns ``psi`` of shape ``(len(tau), len(sigma))`` complex128, matching the
    FD corpus layout.
    """
    SG, TA = np.meshgrid(sigma, tau)             # (Ntau, Nsig)
    pts = np.stack([SG.ravel(), TA.ravel()], axis=1).astype(np.float64)
    out = model.predict(pts)                     # (Ntau*Nsig, 2)
    p = out[:, 0].reshape(SG.shape)
    q = out[:, 1].reshape(SG.shape)
    return (p + 1j * q)


def scri_waveform(model: dde.Model, tau: np.ndarray, sigma_scri: float
                  ) -> np.ndarray:
    """Complex scri waveform ``psi(sigma_scri, tau)`` (the gravitational wave)."""
    pts = np.stack([np.full_like(tau, sigma_scri), tau], axis=1).astype(np.float64)
    out = model.predict(pts)
    return out[:, 0] + 1j * out[:, 1]


def rel_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    """Relative L2 error ``||pred - ref|| / ||ref||`` over complex arrays."""
    num = float(np.sqrt(np.sum(np.abs(pred - ref) ** 2)))
    den = float(np.sqrt(np.sum(np.abs(ref) ** 2)))
    return num / den if den > 0 else float("inf")

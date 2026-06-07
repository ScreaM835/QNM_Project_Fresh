"""FNO model wrapper for the Zerilli operator-learning experiment.

Thin shim over `neuralop.models.FNO` so the rest of the project does not
import `neuralop` directly.  Raises a clear, actionable error if the package
is missing rather than failing at random call sites.

Tensor convention (matches src/fno_dataset.py):
    input  : (B, IN_CHANNELS, Nt, Nx)  float
    output : (B, 1,            Nt, Nx)  float
"""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from .fno_dataset import IN_CHANNELS, OUT_CHANNELS


def _require_neuralop():
    try:
        from neuralop.models import FNO  # noqa: F401
    except Exception as exc:                   # ImportError or other
        raise RuntimeError(
            "neuraloperator is required for the FNO pipeline.\n"
            "Install it into the project venv:\n"
            "    pip install neuraloperator"
        ) from exc


def build_fno(cfg: Dict[str, Any]) -> nn.Module:
    """Build an FNO from the `fno:` block of a config dict.

    Expected keys (with sensible fallbacks):
        modes_t, modes_x, hidden_channels, n_layers, domain_padding,
        positional_embedding.
    """
    _require_neuralop()
    from neuralop.models import FNO

    fcfg = cfg["fno"]
    n_modes = (int(fcfg.get("modes_t", 24)), int(fcfg.get("modes_x", 48)))
    hidden = int(fcfg.get("hidden_channels", 64))
    n_layers = int(fcfg.get("n_layers", 4))
    domain_padding = float(fcfg.get("domain_padding", 0.10))
    pos_emb = fcfg.get("positional_embedding", "grid")

    model = FNO(
        n_modes=n_modes,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        hidden_channels=hidden,
        n_layers=n_layers,
        domain_padding=domain_padding,
        positional_embedding=pos_emb,
    )
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(cfg: Dict[str, Any],
                t_grid: torch.Tensor | None = None,
                x_grid: torch.Tensor | None = None) -> nn.Module:
    """Dispatcher: plain FNO (default) or FNO + QNM-decoder head (v5).

    Selected by `cfg["fno"]["model_type"]` in {"plain", "qnm_head"} —
    default "plain" keeps full back-compat with v1..v4.
    """
    fcfg = cfg["fno"]
    mtype = str(fcfg.get("model_type", "plain")).lower()
    trunk = build_fno(cfg)
    if mtype == "plain":
        return trunk
    if mtype == "qnm_head":
        if t_grid is None or x_grid is None:
            raise ValueError("build_model(qnm_head) requires t_grid and x_grid.")
        from .fno_qnm_head import FNOWithQNMHead
        qh = fcfg.get("qnm_head", {})
        return FNOWithQNMHead(
            trunk=trunk,
            t_grid=t_grid,
            x_grid=x_grid,
            n_modes=int(qh.get("n_modes", 2)),
            t_split=float(qh.get("t_split", 20.0)),
            crossover_sharpness=float(qh.get("crossover_sharpness", 2.0)),
            ic_embed_dim=int(qh.get("ic_embed_dim", 32)),
            spatial_hidden=int(qh.get("spatial_hidden", 64)),
            omega_init=float(qh.get("omega_init", 0.3)),
            tau_init=float(qh.get("tau_init", 10.0)),
            omega_min=float(qh.get("omega_min", 0.10)),
            omega_max=float(qh.get("omega_max", 0.60)),
            tau_min=float(qh.get("tau_min", 1.0)),
            tau_max=float(qh.get("tau_max", 50.0)),
        )
    raise ValueError(f"Unknown fno.model_type='{mtype}'")


# ---------------------------------------------------------------------------
# Composite-loss building blocks (kept here so train + inverse can share them)
# ---------------------------------------------------------------------------
def loss_field(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def loss_h1(pred: torch.Tensor, target: torch.Tensor, dx: float, dt: float) -> torch.Tensor:
    """First-derivative MSE in t and x (helps fit phase / decay)."""
    dpdt = (pred[..., 1:, :] - pred[..., :-1, :]) / dt
    dydt = (target[..., 1:, :] - target[..., :-1, :]) / dt
    dpdx = (pred[..., :, 1:] - pred[..., :, :-1]) / dx
    dydx = (target[..., :, 1:] - target[..., :, :-1]) / dx
    return torch.mean((dpdt - dydt) ** 2) + torch.mean((dpdx - dydx) ** 2)


def loss_ringdown(
    pred: torch.Tensor,
    target: torch.Tensor,
    t: torch.Tensor,
    x: torch.Tensor,
    t_start: float,
    t_end: float,
    xq: float,
    half_width: float = 1.0,
) -> torch.Tensor:
    """MSE restricted to the ringdown window (t in [t_start, t_end]) and a
    small spatial neighbourhood around the QNM extraction point xq."""
    t_mask = (t >= t_start) & (t <= t_end)
    x_mask = (x >= xq - half_width) & (x <= xq + half_width)
    if t_mask.sum() == 0 or x_mask.sum() == 0:
        return pred.new_tensor(0.0)
    p = pred[..., t_mask, :][..., x_mask]
    y = target[..., t_mask, :][..., x_mask]
    return torch.mean((p - y) ** 2)


def loss_pde_residual(
    pred: torch.Tensor,
    V: torch.Tensor,
    dx: float,
    dt: float,
) -> torch.Tensor:
    """PINO term: residual of  Phi_tt - Phi_xx + V Phi  on the predicted field.

    pred : (B, 1, Nt, Nx)
    V    : (B, Nx)         per-sample Zerilli potential
    """
    u = pred[:, 0]                                   # (B, Nt, Nx)
    # interior 2nd derivatives, central differences
    u_tt = (u[:, 2:, :] - 2.0 * u[:, 1:-1, :] + u[:, :-2, :]) / (dt ** 2)
    u_xx = (u[:, :, 2:] - 2.0 * u[:, :, 1:-1] + u[:, :, :-2]) / (dx ** 2)
    # align shapes: drop the first/last row in t and column in x
    u_tt = u_tt[:, :, 1:-1]
    u_xx = u_xx[:, 1:-1, :]
    u_int = u[:, 1:-1, 1:-1]
    Vb = V[:, None, 1:-1].expand_as(u_int)
    res = u_tt - u_xx + Vb * u_int
    return torch.mean(res ** 2)


def loss_obs_slice(
    pred: torch.Tensor,
    target: torch.Tensor,
    x: torch.Tensor,
    xq: float,
) -> torch.Tensor:
    """L2 on the 1-D observer slice at x = xq.

    This is the slice the downstream QNM extraction reads, so training it
    directly is the most surgical way to push QNM-extraction accuracy down.

    pred, target : (B, 1, Nt, Nx)
    x            : (Nx,)
    """
    ix = int(torch.argmin(torch.abs(x - xq)).item())
    p = pred[..., :, ix]      # (B, 1, Nt)
    y = target[..., :, ix]
    return torch.mean((p - y) ** 2)


def loss_time_weighted(
    pred: torch.Tensor,
    target: torch.Tensor,
    t: torch.Tensor,
    t_ramp_start: float,
    t_ramp_end: float,
    beta: float,
) -> torch.Tensor:
    """Field MSE with a time-dependent weight that boosts late-time errors.

    w(t) = 1                              for t <= t_ramp_start
         = 1 + beta * s                    for t_ramp_start < t < t_ramp_end
         = 1 + beta                        for t >= t_ramp_end
    where s = (t - t_ramp_start)/(t_ramp_end - t_ramp_start).

    The weight is mean-normalised so the overall loss scale stays comparable
    to plain L2. With beta >> 1 this concentrates gradient on the ringdown.
    """
    s = torch.clamp((t - t_ramp_start) / max(t_ramp_end - t_ramp_start, 1e-12),
                    min=0.0, max=1.0)
    w = 1.0 + beta * s                       # (Nt,)
    w = w / w.mean()
    # broadcast to (1, 1, Nt, 1)
    w = w.view(1, 1, -1, 1)
    return torch.mean(w * (pred - target) ** 2)

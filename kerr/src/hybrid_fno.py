"""Hybrid FNO: predicts the discretisation residual delta = Phi_fine - upsample(Phi_coarse).

Channel layout (in_channels = 5):
    ch 0 : upsampled Phi_coarse(x, t)        — the physics-informed prior
    ch 1 : Phi_coarse(x, 0) broadcast        — initial displacement (fine grid)
    ch 2 : V_fine(x; M, l) broadcast over t  — per-sample potential
    ch 3 : M broadcast over (x, t)           — scalar BH mass
    ch 4 : Phi_coarse_t(x, 0) broadcast      — initial velocity (finite-diff from coarse)

Output (out_channels = 1):
    delta(x, t) — additive correction; hybrid prediction is
                  Phi_hybrid = upsample(Phi_coarse) + delta.

We deliberately *do not* re-use src/fno_dataset.IN_CHANNELS so this module is
independent of the pure-FNO surrogate's channel layout.
"""
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

HYBRID_IN_CHANNELS = 5
HYBRID_OUT_CHANNELS = 1


def _require_neuralop() -> None:
    try:
        from neuralop.models import FNO  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "neuraloperator is required for the hybrid FNO pipeline. "
            "Install it into the project venv: pip install neuraloperator"
        ) from exc


def build_hybrid_fno(cfg: Dict[str, Any]) -> nn.Module:
    """Build a 2D FNO whose output is the residual delta on the fine grid.

    Config keys (under `fno:`):
        modes_t, modes_x, hidden_channels, n_layers, domain_padding,
        positional_embedding. Defaults are smaller than the pure-FNO defaults
        because the residual is a smaller-amplitude target.
    """
    _require_neuralop()
    from neuralop.models import FNO

    fcfg = cfg["fno"]
    n_modes = (int(fcfg.get("modes_t", 16)), int(fcfg.get("modes_x", 32)))
    hidden = int(fcfg.get("hidden_channels", 32))
    n_layers = int(fcfg.get("n_layers", 4))
    domain_padding = float(fcfg.get("domain_padding", 0.10))
    pos_emb = fcfg.get("positional_embedding", "grid")

    return FNO(
        n_modes=n_modes,
        in_channels=HYBRID_IN_CHANNELS,
        out_channels=HYBRID_OUT_CHANNELS,
        hidden_channels=hidden,
        n_layers=n_layers,
        domain_padding=domain_padding,
        positional_embedding=pos_emb,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

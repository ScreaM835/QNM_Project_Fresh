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

Optionally (``fno.extra_channels: [grad_prior]``) the model prepends derived
feature channels computed on the fly from channel 0; see ``_PriorFeatureFNO``.
These need no extra data at inference (deterministic function of the prior).

We deliberately *do not* re-use src/fno_dataset.IN_CHANNELS so this module is
independent of the pure-FNO surrogate's channel layout.
"""
from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

HYBRID_IN_CHANNELS = 5
HYBRID_OUT_CHANNELS = 1

# Optional derived feature channels, computed on the fly from the data tensor
# (no extra data needed at inference). Each name maps to the number of channels
# it appends. They are a deterministic function of channel 0 (the upsampled
# coarse prior), so adding them is inference-safe and keeps the 1-coarse-solve
# deployment story intact.
_EXTRA_CHANNEL_WIDTH = {
    "grad_prior": 1,   # |grad(prior)| magnitude: the "where the prior fails" map
}


def _require_neuralop() -> None:
    try:
        from neuralop.models import FNO  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "neuraloperator is required for the hybrid FNO pipeline. "
            "Install it into the project venv: pip install neuraloperator"
        ) from exc


class _PriorFeatureFNO(nn.Module):
    """Wrap an FNO, prepending derived feature channels before the forward pass.

    The features are computed from channel 0 of the input (the upsampled coarse
    prior) every forward call, so they require no extra data at inference and
    stay perfectly consistent between training, evaluation and plotting (all of
    which call ``model(x)`` on the same 5-channel data tensor).

    ``grad_prior`` appends the gradient magnitude ``sqrt(d_t prior^2 +
    d_x prior^2)`` (unit grid spacing). It is large on the wavefronts — exactly
    where the finite-difference prior carries its dispersion error — and ~0 in
    the smooth interior and the causal exterior where the prior is already
    machine-accurate, so it tells the operator where its correction belongs and
    where it should stay zero.
    """

    def __init__(self, fno: nn.Module, extra: List[str],
                 gate: Dict[str, float] | None = None):
        super().__init__()
        self.fno = fno
        self.extra = list(extra)
        self.gate = dict(gate) if gate else None

    @staticmethod
    def _grad_mag(prior: torch.Tensor) -> torch.Tensor:
        g_t = torch.gradient(prior, dim=2)[0]
        g_x = torch.gradient(prior, dim=3)[0]
        return torch.sqrt(g_t * g_t + g_x * g_x + 1e-24)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prior = x[:, 0:1]                       # (B, 1, Nt, Nx)
        gmag = None
        if "grad_prior" in self.extra or self.gate is not None:
            gmag = self._grad_mag(prior)
        feats = [x]
        if "grad_prior" in self.extra:
            feats.append(gmag)
        out = self.fno(torch.cat(feats, dim=1) if len(feats) > 1 else x)
        if self.gate is not None:
            # Multiplicative support gate: field = prior + g * FNO, with
            # g = |grad prior| / (|grad prior| + scale) in [0, 1). g -> 1 on the
            # wavefronts (where the prior carries its dispersion error) and -> 0
            # in the smooth interior / causal exterior (where the prior is
            # already machine-exact), so the correction is confined to the
            # high-gradient regions BY CONSTRUCTION and cannot leak speckle into
            # clean zones. g is also the prior<->FNO stitch: a smooth convex
            # blend between the pure coarse prior (g=0) and prior+full FNO (g=1).
            g = gmag / (gmag + self.gate["scale"])
            out = g * out
        return out


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

    extra = list(fcfg.get("extra_channels", []) or [])
    for name in extra:
        if name not in _EXTRA_CHANNEL_WIDTH:
            raise ValueError(
                f"unknown extra_channels entry {name!r}; "
                f"known: {sorted(_EXTRA_CHANNEL_WIDTH)}"
            )
    n_extra = sum(_EXTRA_CHANNEL_WIDTH[name] for name in extra)

    # Optional multiplicative output gate g(|grad prior|): confines the FNO
    # correction to high-gradient regions and defines the prior<->FNO stitch.
    gate_cfg = fcfg.get("output_gate")
    gate = None
    if gate_cfg:
        scale = float(gate_cfg.get("scale", 1.0e-4))
        if scale <= 0.0:
            raise ValueError(f"output_gate.scale must be > 0 (got {scale})")
        gate = {"scale": scale}

    fno = FNO(
        n_modes=n_modes,
        in_channels=HYBRID_IN_CHANNELS + n_extra,
        out_channels=HYBRID_OUT_CHANNELS,
        hidden_channels=hidden,
        n_layers=n_layers,
        domain_padding=domain_padding,
        positional_embedding=pos_emb,
    )
    if n_extra or gate is not None:
        return _PriorFeatureFNO(fno, extra, gate)
    return fno


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

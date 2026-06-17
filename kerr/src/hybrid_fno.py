"""Kerr hybrid FNO: predicts the normalised discretisation residual of a coarse
Teukolsky solve, with NO gate (plain additive skip).

Design (driven by the measured Kerr residual topology, NOT ported from the
Schwarzschild hybrid -- see ``kerr/scripts/train_eval_hybrid_kerr.py`` notes):

* **No gradient gate.** On the compactified hyperboloidal slice the field is
  active almost everywhere (only ~0.4% of cells are machine-zero, vs >50% for
  the SW tortoise grid) and the residual lives where the field is *large*. The
  SW gate existed to protect a vast machine-exact causal exterior from
  band-limited speckle; that region does not exist here, so a gate would only
  suppress the network where the correction belongs. The reconstruction is the
  plain additive skip ``psi_hyb = up4 + s * FNO`` (done in the training script;
  this module is the pure residual operator).

* **Anisotropic, tau-heavy modes.** The residual needs ~41 Fourier modes along
  tau but only ~9 along sigma (the hyperboloidal slice is radially smooth), so
  ``n_modes = (modes_tau, modes_sigma)`` is deliberately asymmetric -- the
  opposite of treating the two axes symmetrically.

* **Two real channels for the complex field.** Re/Im are carried as separate
  in/out channels (their correlation is ~-0.29, genuinely independent).

Channel layout (in_channels = 4):
    ch 0 : up4.real / s        -- the prior (real part), per-sample normalised
    ch 1 : up4.imag / s        -- the prior (imag part)
    ch 2 : a/M  (broadcast)    -- the spin (changes the operator; not scaled)
    ch 3 : psi0.real / s       -- the initial pulse at tau=0, broadcast over tau

Output (out_channels = 2):
    [delta.real, delta.imag] / s -- normalised additive correction.
"""
from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

HYBRID_IN_CHANNELS = 4
HYBRID_OUT_CHANNELS = 2


def _require_neuralop() -> None:
    try:
        from neuralop.models import FNO  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "neuraloperator is required for the Kerr hybrid FNO pipeline. "
            "Install it into the project venv: pip install neuraloperator"
        ) from exc


def build_hybrid_fno(cfg: Dict[str, Any]) -> nn.Module:
    """Build the 2-D FNO residual operator (no gate) from the ``fno:`` config.

    Config keys (under ``fno:``):
        modes_tau, modes_sigma : retained Fourier modes per axis (tau-heavy).
        hidden_channels        : lifting width.
        n_layers               : number of Fourier layers.
        domain_padding         : scalar or [pad_tau, pad_sigma] for the
                                 non-periodic hyperboloidal domain.
        positional_embedding   : 'grid' (default) appends (tau, sigma) coords.
    """
    _require_neuralop()
    from neuralop.models import FNO

    fcfg = cfg["fno"]
    n_modes = (int(fcfg.get("modes_tau", 64)), int(fcfg.get("modes_sigma", 24)))
    hidden = int(fcfg.get("hidden_channels", 48))
    n_layers = int(fcfg.get("n_layers", 4))

    dp = fcfg.get("domain_padding", [0.08, 0.08])
    if isinstance(dp, (list, tuple)):
        domain_padding: Any = [float(x) for x in dp]
    else:
        domain_padding = float(dp)

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


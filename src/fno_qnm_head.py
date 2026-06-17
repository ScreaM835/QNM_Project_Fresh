"""QNM-decoder hypernetwork wrapper for the FNO trunk.

Architectural idea (v5):
    The trunk FNO predicts the prompt/transient field on the full (Nt, Nx)
    grid as usual.  A small "QNM head" reads the per-sample initial-condition
    profile (channels 0,1 of the FNO input, which carry Phi0(x) and Pi0(x)),
    embeds it, and emits

        - global mode parameters  (omega_n, tau_n)   for n = 0..N-1
        - per-x amplitudes        A_n(x)             (B, Nx, N)
        - per-x phases            phi_n(x)           (B, Nx, N)

    From these we build an analytic damped-sinusoid tail
        phi_tail(x, t) = sum_n A_n(x) * exp(-(t-t_split)+/tau_n)
                        * cos(omega_n*(t-t_split)+ + phi_n(x)).

    The final prediction is a smooth crossover
        phi(x,t) = w(t) * trunk(x,t) + (1-w(t)) * tail(x,t)
    with w(t) = sigmoid((t_split - t)/sharpness).  Before t_split the
    trunk dominates (prompt+transient); after t_split the tail dominates
    and is *constrained* by construction to be a sum of damped sinusoids.

    The predicted (omega_n, tau_n) become DIRECT OUTPUTS of the network --
    no Prony/template fit needed at evaluation time.

This module is opt-in: nothing changes for the plain-FNO pipeline unless
the config selects fno.model_type == "qnm_head".
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
class QNMHead(nn.Module):
    """Emits (omega_n, tau_n, A_n(x), phi_n(x)) from the IC profile.

    Inputs at forward time:
        ic_profile : (B, 2, Nx)   = X[:, 0:2, 0, :]  (Phi0(x), Pi0(x))
        x_grid     : (Nx,)        spatial grid (used for the per-x MLP)
    """

    def __init__(
        self,
        n_modes: int = 2,
        ic_embed_dim: int = 32,
        spatial_hidden: int = 64,
        omega_init: float = 0.3,
        tau_init: float = 10.0,
        omega_min: float = 0.10,
        omega_max: float = 0.60,
        tau_min: float = 1.0,
        tau_max: float = 50.0,
    ) -> None:
        super().__init__()
        self.n_modes = int(n_modes)
        self.ic_embed_dim = int(ic_embed_dim)
        # Bounded ranges for (omega, tau) — see v5-B blowup post-mortem in
        # /memories/model_design.md.  Softplus left omega unbounded, the PDE
        # residual scaled as omega^4, and any stochastic step that grew omega
        # was amplified to NaN by ep ~70.  Sigmoid map is the structural fix.
        self.omega_min = float(omega_min)
        self.omega_max = float(omega_max)
        self.tau_min   = float(tau_min)
        self.tau_max   = float(tau_max)

        # IC encoder: 1D CNN over x, channel-pool to a vector
        self.ic_encoder = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(32, ic_embed_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),  # (B, ic_embed_dim, 1)
        )

        # Global mode parameters: ic_embed -> 2*N (raw pre-sigmoid logits)
        self.mode_net = nn.Sequential(
            nn.Linear(ic_embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2 * self.n_modes),
        )
        # Initial biases so sigmoid map outputs land near (omega_init, tau_init)
        with torch.no_grad():
            self.mode_net[-1].weight.zero_()
            b = self.mode_net[-1].bias
            b[: self.n_modes].fill_(
                self._inverse_sigmoid_map(omega_init, self.omega_min, self.omega_max))
            b[self.n_modes:].fill_(
                self._inverse_sigmoid_map(tau_init, self.tau_min, self.tau_max))

        # Per-x spatial profile: (x_norm, embed) -> (A_n, phi_n)
        self.spatial_net = nn.Sequential(
            nn.Linear(1 + ic_embed_dim, spatial_hidden),
            nn.GELU(),
            nn.Linear(spatial_hidden, spatial_hidden),
            nn.GELU(),
            nn.Linear(spatial_hidden, 2 * self.n_modes),
        )
        # Zero-init last layer so the tail starts ~0; trunk drives everything
        # initially and the QNM head learns from gradients.
        with torch.no_grad():
            self.spatial_net[-1].weight.mul_(0.01)
            self.spatial_net[-1].bias.zero_()

    @staticmethod
    def _inverse_softplus(y: float) -> float:
        # x = log(exp(y) - 1)
        import math
        y = max(float(y), 1e-6)
        return math.log(math.exp(y) - 1.0)

    @staticmethod
    def _inverse_sigmoid_map(y: float, lo: float, hi: float) -> float:
        # y = lo + (hi-lo)*sigmoid(x)  =>  x = logit((y-lo)/(hi-lo))
        import math
        u = (float(y) - float(lo)) / (float(hi) - float(lo))
        u = min(max(u, 1e-6), 1.0 - 1e-6)
        return math.log(u / (1.0 - u))

    def forward(
        self,
        ic_profile: torch.Tensor,   # (B, 2, Nx)
        x_grid: torch.Tensor,       # (Nx,)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = ic_profile.shape[0]
        Nx = ic_profile.shape[-1]
        N = self.n_modes

        emb = self.ic_encoder(ic_profile).squeeze(-1)   # (B, ic_embed_dim)

        mp = self.mode_net(emb)                          # (B, 2N)
        # Bounded sigmoid map: closes the omega^4 PDE-residual feedback loop.
        omega = self.omega_min + (self.omega_max - self.omega_min) * torch.sigmoid(mp[:, :N])
        tau   = self.tau_min   + (self.tau_max   - self.tau_min)   * torch.sigmoid(mp[:, N:])

        x_norm = (x_grid - x_grid.mean()) / (x_grid.std() + 1e-12)  # (Nx,)
        x_in = x_norm.view(1, Nx, 1).expand(B, Nx, 1)
        e_in = emb.view(B, 1, -1).expand(B, Nx, self.ic_embed_dim)
        sp_in = torch.cat([x_in, e_in], dim=-1)          # (B, Nx, 1+E)
        sp_out = self.spatial_net(sp_in)                  # (B, Nx, 2N)
        A    = sp_out[..., :N]                            # (B, Nx, N)
        phi0 = sp_out[..., N:]                            # (B, Nx, N)
        return omega, tau, A, phi0


# ---------------------------------------------------------------------------
class FNOWithQNMHead(nn.Module):
    """Trunk FNO + QNM-decoder tail with a smooth time crossover."""

    def __init__(
        self,
        trunk: nn.Module,
        t_grid: torch.Tensor,
        x_grid: torch.Tensor,
        n_modes: int = 2,
        t_split: float = 20.0,
        crossover_sharpness: float = 2.0,
        ic_embed_dim: int = 32,
        spatial_hidden: int = 64,
        omega_init: float = 0.3,
        tau_init: float = 10.0,
        omega_min: float = 0.10,
        omega_max: float = 0.60,
        tau_min: float = 1.0,
        tau_max: float = 50.0,
    ) -> None:
        super().__init__()
        self.trunk = trunk
        self.head = QNMHead(
            n_modes=n_modes,
            ic_embed_dim=ic_embed_dim,
            spatial_hidden=spatial_hidden,
            omega_init=omega_init,
            tau_init=tau_init,
            omega_min=omega_min,
            omega_max=omega_max,
            tau_min=tau_min,
            tau_max=tau_max,
        )
        self.t_split = float(t_split)
        self.crossover_sharpness = float(crossover_sharpness)
        # Buffers: move with .to(device); not trainable
        self.register_buffer("t_grid", t_grid.float())
        self.register_buffer("x_grid", x_grid.float())
        # Last forward's QNM parameters (for downstream extraction)
        self._last_omega: torch.Tensor | None = None
        self._last_tau:   torch.Tensor | None = None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, C, Nt, Nx)
        B, _, Nt, Nx = X.shape
        N = self.head.n_modes

        trunk_out = self.trunk(X)                         # (B, 1, Nt, Nx)
        ic = X[:, 0:2, 0, :]                              # (B, 2, Nx) — Phi0, Pi0
        omega, tau, A, phi0 = self.head(ic, self.x_grid)  # global + per-x
        self._last_omega = omega.detach()
        self._last_tau   = tau.detach()

        # Build the analytic tail
        t_shift = (self.t_grid - self.t_split).clamp(min=0.0)         # (Nt,)
        # broadcast to (B, N, Nt, Nx) — be careful with memory: Nt*Nx*N*B
        ts = t_shift.view(1, 1, Nt, 1)
        decay = torch.exp(-ts / tau.view(B, N, 1, 1))                  # (B,N,Nt,1)
        phase = (omega.view(B, N, 1, 1) * ts                            # (B,N,Nt,1)
                 + phi0.permute(0, 2, 1).view(B, N, 1, Nx))             # +(B,N,1,Nx)
        osc   = torch.cos(phase)                                        # (B,N,Nt,Nx)
        amp   = A.permute(0, 2, 1).view(B, N, 1, Nx)                    # (B,N,1,Nx)
        tail  = (amp * decay * osc).sum(dim=1, keepdim=True)            # (B,1,Nt,Nx)

        # Smooth crossover: w=1 before t_split, w=0 after
        w = torch.sigmoid((self.t_split - self.t_grid) / self.crossover_sharpness)
        w = w.to(dtype=trunk_out.dtype).view(1, 1, Nt, 1)
        return w * trunk_out + (1.0 - w) * tail

    # Convenience: expose the last-predicted QNM params
    def last_qnm_params(self) -> Dict[str, torch.Tensor] | None:
        if self._last_omega is None:
            return None
        return {"omega": self._last_omega.cpu(),    # (B, N)
                "tau":   self._last_tau.cpu()}      # (B, N)

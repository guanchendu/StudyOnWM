"""
All model components for Hierarchical JEPA + Inverse Dynamics framework.

Modules:
  1. TokenEncoder           — image → latent tokens
  2. EMATargetEncoder       — EMA copy for supervision targets
  3. InverseDynamicsModel   — (z_t, z_{t+1}) → â_t (latent action)
  4. CoarsePredictor        — z_t + â_{0:h} → ẑ_{t+h}
  5. FineTransformerBlock   — self-attention + cross-attention block
  6. FinePredictor          — z_t + â_t + cond(ẑ_{t+h}) → ẑ_{t+1}
  7. ActionEncoder          — real action → latent action (Phase 2)
  8. ActionDecoder          — latent action → real action (Phase 2)
  9. HierarchicalInvDynWorldModel — combines 1-6 into Phase 1 model
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Encoder
# ============================================================


class TokenEncoder(nn.Module):
    """Encode image to latent token sequence: (B, 3, H, W) → (B, num_tokens, D)."""

    def __init__(self, latent_dim: int, num_tokens: int = 4):
        super().__init__()
        self.num_tokens = num_tokens
        self.latent_dim = latent_dim
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.proj = nn.Linear(128 * 2 * 2, num_tokens * latent_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = h.flatten(1)
        h = self.proj(h)
        h = h.view(-1, self.num_tokens, self.latent_dim)
        return self.norm(h)


# ============================================================
# 2. EMA Target Encoder
# ============================================================


class EMATargetEncoder(nn.Module):
    """Exponential Moving Average target encoder (no gradients)."""

    def __init__(self, encoder: TokenEncoder, momentum: float = 0.996):
        super().__init__()
        self.encoder = copy.deepcopy(encoder)
        self.momentum = momentum
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, online_encoder: TokenEncoder):
        for p_ema, p_online in zip(
            self.encoder.parameters(), online_encoder.parameters()
        ):
            p_ema.data.lerp_(p_online.data, 1.0 - self.momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ============================================================
# 3. Inverse Dynamics Model
# ============================================================


class InverseDynamicsModel(nn.Module):
    """Predict latent action from consecutive latent states.

    (z_t, z_{t+1}) → â_t

    Uses information bottleneck: latent_action_dim << latent_dim
    to prevent encoding all of z_{t+1} into â_t.
    """

    def __init__(
        self,
        latent_dim: int,
        latent_action_dim: int = 32,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.latent_action_dim = latent_action_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_action_dim),
        )

    def forward(
        self, z_t: torch.Tensor, z_tp1: torch.Tensor
    ) -> torch.Tensor:
        """
        z_t:   (B, D) — pooled latent of current frame
        z_tp1: (B, D) — pooled latent of next frame
        Returns: â_t (B, latent_action_dim)
        """
        return self.net(torch.cat([z_t, z_tp1], dim=-1))


# ============================================================
# 4. Coarse Predictor
# ============================================================


class CoarsePredictor(nn.Module):
    """Predict z_{t+h} from z_t + latent actions over h steps.

    z_t + [â_0, ..., â_{h-1}] → ẑ_{t+h}
    """

    def __init__(
        self,
        latent_dim: int,
        latent_action_dim: int,
        horizon_h: int,
    ):
        super().__init__()
        input_dim = latent_dim + horizon_h * latent_action_dim
        hidden_dim = latent_dim * 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(
        self,
        z_t: torch.Tensor,
        latent_actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        z_t:            (B, num_tokens, D) — current latent tokens
        latent_actions: (B, h, latent_action_dim) — latent actions for h steps
        Returns:        (B, D)
        """
        z_pooled = z_t.mean(dim=1)
        a_flat = latent_actions.flatten(1)
        return self.net(torch.cat([z_pooled, a_flat], dim=-1))


# ============================================================
# 5. Fine Predictor — Self-Attention + Cross-Attention
# ============================================================


class FineTransformerBlock(nn.Module):
    """Pre-norm Transformer block: Self-Attn → Cross-Attn → FFN."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm_sa = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.norm_ca = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        self.norm_ff = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, coarse_kv: torch.Tensor
    ) -> torch.Tensor:
        """
        x:         (B, N, D) — fine-grained tokens
        coarse_kv: (B, M, D) — coarse condition for cross-attention K/V
        """
        h = self.norm_sa(x)
        x = x + self.self_attn(h, h, h)[0]

        h = self.norm_ca(x)
        x = x + self.cross_attn(query=h, key=coarse_kv, value=coarse_kv)[0]

        h = self.norm_ff(x)
        x = x + self.ffn(h)

        return x


class FinePredictor(nn.Module):
    """Predict z_{t+1} from z_t + latent action, conditioned on coarse waypoint.

    Input tokens: [z_t tokens, action_token] → Self-Attn + Cross-Attn → ẑ_{t+1}
    """

    def __init__(
        self,
        latent_dim: int,
        latent_action_dim: int,
        nhead: int = 4,
        num_layers: int = 3,
        dim_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.action_proj = nn.Sequential(
            nn.Linear(latent_action_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        self.coarse_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        self.blocks = nn.ModuleList(
            [
                FineTransformerBlock(latent_dim, nhead, dim_ff, dropout)
                for _ in range(num_layers)
            ]
        )

        self.output_norm = nn.LayerNorm(latent_dim)
        self.output_proj = nn.Linear(latent_dim, latent_dim)

    def forward(
        self,
        z_t: torch.Tensor,
        latent_action: torch.Tensor,
        coarse_cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        z_t:           (B, num_tokens, D)
        latent_action: (B, latent_action_dim)
        coarse_cond:   (B, D)
        Returns:       (B, D)
        """
        a_token = self.action_proj(latent_action)
        x = torch.cat([z_t, a_token.unsqueeze(1)], dim=1)

        coarse_kv = self.coarse_proj(coarse_cond).unsqueeze(1)

        for block in self.blocks:
            x = block(x, coarse_kv)

        x = x.mean(dim=1)
        return self.output_proj(self.output_norm(x))


# ============================================================
# 6. Action Encoder / Decoder (Phase 2)
# ============================================================


class ActionEncoder(nn.Module):
    """Map real action → latent action space.  a_t → ã_t"""

    def __init__(self, action_dim: int, latent_action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_action_dim),
        )

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        return self.net(action)


class ActionDecoder(nn.Module):
    """Map latent action → real action space.  â_t → a_t"""

    def __init__(self, latent_action_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, latent_action: torch.Tensor) -> torch.Tensor:
        return self.net(latent_action)


# ============================================================
# 7. Full Phase 1 Model
# ============================================================


class HierarchicalInvDynWorldModel(nn.Module):
    """Complete Phase 1 model: Encoder + Inverse Dynamics + Coarse/Fine Predictors.

    Training flow:
      1. Encode all frames with online encoder
      2. Compute latent actions via inverse dynamics: â_t = inv(z_t, z_{t+1})
      3. Coarse prediction: z_0 + [â_0,...,â_{h-1}] → ẑ_h
      4. Fine prediction:   z_0 + â_0 + cond(ẑ_h)  → ẑ_1
      5. Supervise with target encoder outputs
    """

    def __init__(
        self,
        latent_dim: int = 256,
        latent_action_dim: int = 32,
        num_tokens: int = 4,
        horizon_h: int = 5,
        nhead: int = 4,
        num_fine_layers: int = 3,
        dim_ff: int = 512,
        ema_momentum: float = 0.996,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.latent_action_dim = latent_action_dim
        self.horizon_h = horizon_h

        self.encoder = TokenEncoder(latent_dim, num_tokens)
        self.target_encoder = EMATargetEncoder(self.encoder, ema_momentum)

        self.inverse_dynamics = InverseDynamicsModel(
            latent_dim, latent_action_dim
        )

        self.coarse_predictor = CoarsePredictor(
            latent_dim, latent_action_dim, horizon_h
        )

        self.fine_predictor = FinePredictor(
            latent_dim, latent_action_dim, nhead, num_fine_layers, dim_ff, dropout
        )

    @torch.no_grad()
    def update_target_encoder(self):
        self.target_encoder.update(self.encoder)

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """Online encoder: (B, 3, H, W) → (B, num_tokens, D)"""
        return self.encoder(pixels)

    @torch.no_grad()
    def encode_target(self, pixels: torch.Tensor) -> torch.Tensor:
        """Target encoder (no grad): (B, 3, H, W) → (B, num_tokens, D)"""
        return self.target_encoder(pixels)

    def compute_latent_action(
        self, z_t: torch.Tensor, z_tp1: torch.Tensor
    ) -> torch.Tensor:
        """Inverse dynamics: (z_t, z_{t+1}) → â_t.
        Inputs are token sequences, will be pooled internally."""
        z_t_pooled = z_t.mean(dim=1) if z_t.dim() == 3 else z_t
        z_tp1_pooled = z_tp1.mean(dim=1) if z_tp1.dim() == 3 else z_tp1
        return self.inverse_dynamics(z_t_pooled, z_tp1_pooled)

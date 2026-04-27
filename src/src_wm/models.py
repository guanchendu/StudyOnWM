"""
All model components for Hierarchical JEPA + Inverse Dynamics framework.

Architecture (single-scale fine inverse dynamics + hierarchical predictors):
  1. TokenEncoder           — image → latent token sequence (B, num_tokens, D)
  2. EMATargetEncoder       — EMA copy for supervision targets
  3. InverseDynamicsModel   — (z_t, z_{t+1}) → â_t   (single FINE inverse dynamics)
  4. CoarsePredictor        — z_t + [â_0, ..., â_{h-1}] → ẑ_{t+h}   (token output)
  5. FineTransformerBlock   — self-attention + cross-attention block
  6. FinePredictor          — z_t + â_t + cond(ẑ_{t+h}) → ẑ_{t+1}   (token output)
  7. ActionEncoder          — real action → latent action (Phase 2)
  8. ActionDecoder          — latent action → real action (Phase 2)
  9. ProprioDecoder         — latent state → proprio   (Phase 2; used by planner cost)
  10. HierarchicalInvDynWorldModel — combines 1-6 into Phase 1 model
"""

import copy

import torch
import torch.nn as nn


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
# 3. Inverse Dynamics Model — single scale, fine grained
# ============================================================


class InverseDynamicsModel(nn.Module):
    """Predict latent action from consecutive latent states.

    (z_t, z_{t+1}) → â_t

    Bottleneck: latent_action_dim << latent_dim
    prevents encoding all of z_{t+1} into â_t.
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

    def forward(self, z_t: torch.Tensor, z_tp1: torch.Tensor) -> torch.Tensor:
        """z_t, z_tp1: (B, D). Returns â_t (B, latent_action_dim)."""
        return self.net(torch.cat([z_t, z_tp1], dim=-1))


# ============================================================
# 4. Coarse Predictor — outputs full token sequence
# ============================================================


class CoarsePredictor(nn.Module):
    """Predict z_{t+h} from z_t + h concatenated fine latent actions.

    z_t + [â_0, ..., â_{h-1}] → ẑ_{t+h}

    Output keeps the token structure (B, num_tokens, D) so that the
    fine predictor can attend over distinct tokens during rollout
    (not collapsed to a single repeated vector).
    """

    def __init__(
        self,
        latent_dim: int,
        latent_action_dim: int,
        horizon_h: int,
        num_tokens: int,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.latent_dim = latent_dim
        input_dim = latent_dim + horizon_h * latent_action_dim
        hidden_dim = latent_dim * 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_tokens * latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)

    def forward(
        self,
        z_t: torch.Tensor,
        latent_actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        z_t:            (B, num_tokens, D)
        latent_actions: (B, h, latent_action_dim)
        Returns:        (B, num_tokens, D)
        """
        z_pooled = z_t.mean(dim=1)
        a_flat = latent_actions.flatten(1)
        h = self.net(torch.cat([z_pooled, a_flat], dim=-1))
        return self.norm(h.view(-1, self.num_tokens, self.latent_dim))


# ============================================================
# 5. Fine Predictor — Self-Attention + Cross-Attention, token output
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

    def forward(self, x: torch.Tensor, coarse_kv: torch.Tensor) -> torch.Tensor:
        h = self.norm_sa(x)
        x = x + self.self_attn(h, h, h)[0]

        h = self.norm_ca(x)
        x = x + self.cross_attn(query=h, key=coarse_kv, value=coarse_kv)[0]

        h = self.norm_ff(x)
        x = x + self.ffn(h)

        return x


class FinePredictor(nn.Module):
    """Predict z_{t+1} from z_t + latent action, conditioned on coarse waypoint ẑ_{t+h}.

    Input tokens: [z_t tokens, action_token] → Self-Attn + Cross-Attn → ẑ_{t+1}

    Outputs full token sequence (B, num_tokens, D) so multi-step rollout
    keeps a real per-token representation rather than a duplicated vector.
    coarse_cond can be tokens (B, num_tokens, D) or pooled (B, D).
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
        coarse_cond:   (B, num_tokens, D) or (B, D)
        Returns:       (B, num_tokens, D)
        """
        N = z_t.shape[1]
        a_token = self.action_proj(latent_action)
        x = torch.cat([z_t, a_token.unsqueeze(1)], dim=1)

        coarse_kv = self.coarse_proj(coarse_cond)
        if coarse_kv.dim() == 2:
            coarse_kv = coarse_kv.unsqueeze(1)

        for block in self.blocks:
            x = block(x, coarse_kv)

        x = x[:, :N, :]
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
# 7. Proprio Decoder (Phase 2) — used by planner cost
# ============================================================


class ProprioDecoder(nn.Module):
    """Decode latent state to proprioception space: z → proprio.

    Trained in Phase 2 with labeled data on BOTH encoder outputs and
    predictor outputs, so that planning cost computed on rolled-out
    latent states is in distribution.
    """

    def __init__(self, latent_dim: int, proprio_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.proprio_dim = proprio_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proprio_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, num_tokens, D) or (B, D). Returns (B, proprio_dim)."""
        if z.dim() == 3:
            z = z.mean(dim=1)
        return self.net(z)


# ============================================================
# 8. Full Phase 1 Model
# ============================================================


class HierarchicalInvDynWorldModel(nn.Module):
    """Complete Phase 1 model: Encoder + (single) Inverse Dynamics + Coarse/Fine Predictors.

    Training flow (multi-step rollout, K coarse segments × h fine steps):
      1. Encode K*h+1 frames with online encoder
      2. EMA target encoder produces supervision targets
      3. Compute fine latent actions via inverse dynamics: â_t = inv(z_t, z_{t+1})
      4. Coarse rollout (K segments): z_0 → ẑ_h → ẑ_{2h} → ... → ẑ_{Kh}
         (each step: coarse_predictor consumes h concatenated fine actions)
      5. Fine rollout (per segment, with scheduled sampling):
         z_seg_start + â_t + cond(ẑ_{(k+1)h}) → ẑ_{t+1}
      6. State variance regularization (VICReg-style) on z, prevents collapse
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
            latent_dim, latent_action_dim, horizon_h, num_tokens
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

    def pool(self, tokens: torch.Tensor) -> torch.Tensor:
        """Pool token sequence to single vector: (B, N, D) → (B, D)."""
        return tokens.mean(dim=1) if tokens.dim() == 3 else tokens

    def compute_latent_action(
        self, z_t: torch.Tensor, z_tp1: torch.Tensor
    ) -> torch.Tensor:
        """Inverse dynamics: (z_t, z_{t+1}) → â_t.
        Inputs are token sequences; pooled internally."""
        return self.inverse_dynamics(self.pool(z_t), self.pool(z_tp1))

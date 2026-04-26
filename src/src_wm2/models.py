"""
Multi-Scale Inverse Dynamics JEPA World Model — All Model Components.

Key difference from v1 (src_wm):
  - Two inverse dynamics models at different temporal scales
  - CoarseInverseDynamics: (z_t, z_{t+h}) → â^coarse (action abstraction)
  - FineInverseDynamics:   (z_t, z_{t+1}) → â^fine   (primitive action)
  - CoarsePredictor takes a single â^coarse, not h concatenated fine actions
"""

import copy

import torch
import torch.nn as nn


# ============================================================
# 1. Encoder
# ============================================================


class TokenEncoder(nn.Module):
    """Image → latent token sequence: (B, 3, H, W) → (B, num_tokens, D)."""

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


class EMATargetEncoder(nn.Module):
    """EMA target encoder. No gradients, provides supervision targets."""

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
# 2. Inverse Dynamics — Two Scales
# ============================================================


class FineInverseDynamics(nn.Module):
    """Single-step inverse dynamics: (z_t, z_{t+1}) → â^fine.

    Learns primitive action representations.
    Bottleneck (fine_action_dim << latent_dim) prevents encoding
    all information from z_{t+1}.
    """

    def __init__(
        self,
        latent_dim: int,
        fine_action_dim: int = 32,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.fine_action_dim = fine_action_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, fine_action_dim),
        )

    def forward(self, z_t: torch.Tensor, z_tp1: torch.Tensor) -> torch.Tensor:
        """z_t, z_tp1: (B, D) pooled latent vectors. Returns: (B, fine_action_dim)."""
        return self.net(torch.cat([z_t, z_tp1], dim=-1))


class CoarseInverseDynamics(nn.Module):
    """Multi-step inverse dynamics: (z_t, z_{t+h}) → â^coarse.

    Learns action abstractions — a single vector summarizing
    "what happened over h steps" without seeing intermediate frames.

    This is the core novelty: the model is forced to discover
    high-level action representations.
    """

    def __init__(
        self,
        latent_dim: int,
        coarse_action_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.coarse_action_dim = coarse_action_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, coarse_action_dim),
        )

    def forward(self, z_t: torch.Tensor, z_tph: torch.Tensor) -> torch.Tensor:
        """z_t, z_tph: (B, D) pooled latent vectors. Returns: (B, coarse_action_dim)."""
        return self.net(torch.cat([z_t, z_tph], dim=-1))


# ============================================================
# 3. Coarse Predictor
# ============================================================


class CoarsePredictor(nn.Module):
    """Predict z_{t+h} from z_t + â^coarse.

    Unlike v1 which concatenated h fine actions, this takes a single
    coarse action abstraction. Input dimension is independent of h.

    Outputs full token sequence (B, num_tokens, D) for multi-step rollout.
    """

    def __init__(self, latent_dim: int, coarse_action_dim: int, num_tokens: int):
        super().__init__()
        self.num_tokens = num_tokens
        self.latent_dim = latent_dim
        input_dim = latent_dim + coarse_action_dim
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
        self, z_t: torch.Tensor, coarse_action: torch.Tensor
    ) -> torch.Tensor:
        """
        z_t:           (B, num_tokens, D) — current latent tokens
        coarse_action: (B, coarse_action_dim)
        Returns:       (B, num_tokens, D)
        """
        z_pooled = z_t.mean(dim=1)
        h = self.net(torch.cat([z_pooled, coarse_action], dim=-1))
        return self.norm(h.view(-1, self.num_tokens, self.latent_dim))


# ============================================================
# 4. Fine Predictor — Self-Attention + Cross-Attention
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
    """Predict z_{t+1} from z_t + â^fine, conditioned on coarse waypoint ẑ_{t+h}.

    Tokens: [z_t tokens, action_token] → Self-Attn + Cross-Attn from coarse → ẑ_{t+1}

    Outputs full token sequence (B, num_tokens, D) for multi-step rollout.
    coarse_cond can be tokens (B, num_tokens, D) or pooled (B, D).
    """

    def __init__(
        self,
        latent_dim: int,
        fine_action_dim: int,
        nhead: int = 4,
        num_layers: int = 3,
        dim_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.action_proj = nn.Sequential(
            nn.Linear(fine_action_dim, latent_dim),
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
        fine_action: torch.Tensor,
        coarse_cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        z_t:          (B, num_tokens, D)
        fine_action:  (B, fine_action_dim)
        coarse_cond:  (B, num_tokens, D) or (B, D)
        Returns:      (B, num_tokens, D)
        """
        N = z_t.shape[1]
        a_token = self.action_proj(fine_action)
        x = torch.cat([z_t, a_token.unsqueeze(1)], dim=1)

        coarse_kv = self.coarse_proj(coarse_cond)
        if coarse_kv.dim() == 2:
            coarse_kv = coarse_kv.unsqueeze(1)

        for block in self.blocks:
            x = block(x, coarse_kv)

        x = x[:, :N, :]
        return self.output_proj(self.output_norm(x))


# ============================================================
# 5. Action Encoder / Decoder — Two Scales (Phase 2)
# ============================================================


class FineActionEncoder(nn.Module):
    """Real single action → fine latent action.  a_t → ã^fine"""

    def __init__(self, action_dim: int, fine_action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, fine_action_dim),
        )

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        return self.net(action)


class FineActionDecoder(nn.Module):
    """Fine latent action → real single action.  â^fine → a_t"""

    def __init__(self, fine_action_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fine_action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, latent_action: torch.Tensor) -> torch.Tensor:
        return self.net(latent_action)


class CoarseActionEncoder(nn.Module):
    """Real action sequence → coarse latent action.

    [a_t, ..., a_{t+h-1}] → ã^coarse

    Compresses h real actions into one abstract representation.
    """

    def __init__(
        self,
        action_dim: int,
        horizon_h: int,
        coarse_action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        input_dim = action_dim * horizon_h
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, coarse_action_dim),
        )

    def forward(self, actions_seq: torch.Tensor) -> torch.Tensor:
        """actions_seq: (B, h * action_dim) flattened action sequence."""
        return self.net(actions_seq)


class CoarseActionDecoder(nn.Module):
    """Coarse latent action → real action sequence.

    â^coarse → [a_t, ..., a_{t+h-1}]

    Expands one abstract representation into h real actions.
    """

    def __init__(
        self,
        coarse_action_dim: int,
        action_dim: int,
        horizon_h: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        output_dim = action_dim * horizon_h
        self.net = nn.Sequential(
            nn.Linear(coarse_action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, coarse_action: torch.Tensor) -> torch.Tensor:
        """Returns: (B, h * action_dim) flattened action sequence."""
        return self.net(coarse_action)


class ProprioDecoder(nn.Module):
    """Decode latent state to proprioception space: z → proprio.

    Trained in Phase 2 with labeled data. Used during planning to
    convert predicted latent states into proprio for cost computation
    against goal_proprio.

    Trained on both encoder outputs AND predictor outputs to handle
    the distribution seen during planning (where cost is computed on
    rolled-out predictor states).
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
        """z: (B, num_tokens, D) or (B, D). Returns: (B, proprio_dim)."""
        if z.dim() == 3:
            z = z.mean(dim=1)
        return self.net(z)


# ============================================================
# 6. Full Phase 1 Model
# ============================================================


class MultiScaleInvDynWorldModel(nn.Module):
    """Complete Phase 1 model with multi-scale inverse dynamics.

    Components:
      encoder           — online: image → tokens
      target_encoder    — EMA: provides supervision
      fine_inv_dyn      — (z_t, z_{t+1}) → â^fine
      coarse_inv_dyn    — (z_t, z_{t+h}) → â^coarse
      coarse_predictor  — z_t + â^coarse → ẑ_{t+h}
      fine_predictor    — z_t + â^fine + cond(ẑ_{t+h}) → ẑ_{t+1}
    """

    def __init__(
        self,
        latent_dim: int = 256,
        fine_action_dim: int = 32,
        coarse_action_dim: int = 64,
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
        self.fine_action_dim = fine_action_dim
        self.coarse_action_dim = coarse_action_dim
        self.horizon_h = horizon_h

        self.encoder = TokenEncoder(latent_dim, num_tokens)
        self.target_encoder = EMATargetEncoder(self.encoder, ema_momentum)

        self.fine_inv_dyn = FineInverseDynamics(latent_dim, fine_action_dim)
        self.coarse_inv_dyn = CoarseInverseDynamics(latent_dim, coarse_action_dim)

        self.coarse_predictor = CoarsePredictor(latent_dim, coarse_action_dim, num_tokens)
        self.fine_predictor = FinePredictor(
            latent_dim, fine_action_dim, nhead, num_fine_layers, dim_ff, dropout
        )

    @torch.no_grad()
    def update_target_encoder(self):
        self.target_encoder.update(self.encoder)

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.encoder(pixels)

    @torch.no_grad()
    def encode_target(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.target_encoder(pixels)

    def pool(self, tokens: torch.Tensor) -> torch.Tensor:
        """Pool token sequence to single vector: (B, N, D) → (B, D)."""
        return tokens.mean(dim=1) if tokens.dim() == 3 else tokens

    def compute_fine_action(
        self, z_t: torch.Tensor, z_tp1: torch.Tensor
    ) -> torch.Tensor:
        return self.fine_inv_dyn(self.pool(z_t), self.pool(z_tp1))

    def compute_coarse_action(
        self, z_t: torch.Tensor, z_tph: torch.Tensor
    ) -> torch.Tensor:
        return self.coarse_inv_dyn(self.pool(z_t), self.pool(z_tph))

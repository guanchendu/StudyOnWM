"""
Hierarchical JEPA World Model — Coarse-to-Fine Prediction
==========================================================

Architecture:
  - Encoder:          image → latent tokens  (B, num_tokens, D)
  - Target Encoder:   EMA copy, provides supervision targets
  - Coarse Predictor: z_t + actions_{t:t+h} → ẑ_{t+h}    (skip h steps)
  - Fine Predictor:   z_t + a_t + condition(ẑ_{t+h}) → ẑ_{t+1}
                      uses Self-Attention + Cross-Attention fusion

Training:
  L_total = L_fine(ẑ_{t+1}, z_{t+1}^{target})
          + λ * L_coarse(ẑ_{t+h}, z_{t+h}^{target})

  with curriculum learning: condition gradually shifts from GT to predicted coarse.

Planning:
  1. Coarse predictor generates sparse waypoints: z_0 → ẑ_h → ẑ_{2h} → ...
  2. Fine predictor fills in dense steps between adjacent waypoints
"""

import argparse
import copy
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn as nn
import torch.nn.functional as F

from Background.utils import get_column_normalizer, get_img_preprocessor

DEFAULT_CACHE_DIR = "/Users/guanchendu/Code/StudyOnWM/data"


# ============================================================
# 1. Encoder
# ============================================================


class TokenEncoder(nn.Module):
    """Encode image to a sequence of latent tokens (B, num_tokens, D).

    Unlike a flat vector encoder, token-based output enables
    self-attention in the fine predictor.
    """

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
        """x: (B, 3, H, W) → (B, num_tokens, D)"""
        h = self.conv(x)
        h = h.flatten(1)
        h = self.proj(h)
        h = h.view(-1, self.num_tokens, self.latent_dim)
        return self.norm(h)


class EMATargetEncoder(nn.Module):
    """Exponential Moving Average target encoder for JEPA training."""

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
# 2. Coarse Predictor
# ============================================================


class CoarsePredictor(nn.Module):
    """Predict z_{t+h} from z_t (pooled tokens) + action/proprio over h steps.

    Input:
        z_t:      (B, num_tokens, D)  — current latent tokens
        actions:  (B, h, action_dim)  — actions for next h steps
        proprios: (B, h, proprio_dim) — proprios for next h steps

    Output:
        ẑ_{t+h}: (B, D)  — coarse future state (single vector)
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        proprio_dim: int,
        horizon_h: int,
    ):
        super().__init__()
        input_dim = latent_dim + horizon_h * (action_dim + proprio_dim)
        hidden = latent_dim * 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(
        self,
        z_t: torch.Tensor,
        actions: torch.Tensor,
        proprios: torch.Tensor,
    ) -> torch.Tensor:
        z_pooled = z_t.mean(dim=1)  # (B, D)
        a_flat = actions.flatten(1)  # (B, h * action_dim)
        p_flat = proprios.flatten(1)  # (B, h * proprio_dim)
        return self.net(torch.cat([z_pooled, a_flat, p_flat], dim=-1))


# ============================================================
# 3. Fine Predictor — Self-Attention + Cross-Attention
# ============================================================


class FineTransformerBlock(nn.Module):
    """One block of the Fine Predictor.

    Data flow inside the block:
        x  ──→ [Self-Attention]  ──→ [Cross-Attention from coarse] ──→ [FFN] ──→ x'

    Self-Attention:  tokens attend to each other (capture internal structure)
    Cross-Attention: tokens query the coarse waypoint (receive directional guidance)
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        # --- Self-Attention ---
        self.norm_sa = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # --- Cross-Attention (fine queries coarse) ---
        self.norm_ca = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # --- Feed-Forward ---
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
        x:         (B, N, D)  — fine-grained token sequence
        coarse_kv: (B, M, D)  — coarse condition tokens (M can be 1)
        """
        # Pre-norm Self-Attention
        h = self.norm_sa(x)
        x = x + self.self_attn(h, h, h)[0]

        # Pre-norm Cross-Attention: fine (Q) attends to coarse (K, V)
        h = self.norm_ca(x)
        x = x + self.cross_attn(query=h, key=coarse_kv, value=coarse_kv)[0]

        # Pre-norm FFN
        h = self.norm_ff(x)
        x = x + self.ffn(h)

        return x


class FinePredictor(nn.Module):
    """Predict z_{t+1} from z_t + action, conditioned on coarse waypoint.

    Architecture:
        1. z_t tokens (B, num_tokens, D) + action token (B, 1, D) → (B, N+1, D)
        2. Pass through FineTransformerBlock × num_layers
           each block: self-attn over tokens, then cross-attn from coarse_cond
        3. Mean-pool → project → ẑ_{t+1} (B, D)

    Input:
        z_t:         (B, num_tokens, D) — current latent tokens
        action:      (B, action_dim)
        proprio:     (B, proprio_dim)
        coarse_cond: (B, D)            — coarse predictor's output ẑ_{t+h}

    Output:
        ẑ_{t+1}:    (B, D)
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        proprio_dim: int,
        nhead: int = 4,
        num_layers: int = 3,
        dim_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Project action+proprio into a token
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim + proprio_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        # Project coarse condition for cross-attention K/V
        self.coarse_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                FineTransformerBlock(latent_dim, nhead, dim_ff, dropout)
                for _ in range(num_layers)
            ]
        )

        # Output head
        self.output_norm = nn.LayerNorm(latent_dim)
        self.output_proj = nn.Linear(latent_dim, latent_dim)

    def forward(
        self,
        z_t: torch.Tensor,
        action: torch.Tensor,
        proprio: torch.Tensor,
        coarse_cond: torch.Tensor,
    ) -> torch.Tensor:
        # Build input token sequence: [z_t tokens, action_token]
        a_token = self.action_proj(
            torch.cat([action, proprio], dim=-1)
        )  # (B, D)
        x = torch.cat([z_t, a_token.unsqueeze(1)], dim=1)  # (B, N+1, D)

        # Prepare coarse condition as K/V for cross-attention
        coarse_kv = self.coarse_proj(coarse_cond).unsqueeze(1)  # (B, 1, D)

        # Transformer blocks: self-attn + cross-attn
        for block in self.blocks:
            x = block(x, coarse_kv)

        # Pool tokens → output
        x = x.mean(dim=1)  # (B, D)
        return self.output_proj(self.output_norm(x))


# ============================================================
# 4. Full Hierarchical Model
# ============================================================


class HierarchicalJEPAWorldModel(nn.Module):
    """
    Complete hierarchical JEPA world model.

    Components:
        encoder         — online encoder: image → tokens
        target_encoder  — EMA encoder: provides supervision targets
        coarse_predictor— z_t → ẑ_{t+h}
        fine_predictor  — z_t + a_t + cond(ẑ_{t+h}) → ẑ_{t+1}
    """

    def __init__(
        self,
        action_dim: int,
        proprio_dim: int,
        latent_dim: int = 256,
        num_tokens: int = 4,
        horizon_h: int = 5,
        nhead: int = 4,
        num_fine_layers: int = 3,
        dim_ff: int = 512,
        ema_momentum: float = 0.996,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.horizon_h = horizon_h

        self.encoder = TokenEncoder(latent_dim, num_tokens)
        self.target_encoder = EMATargetEncoder(self.encoder, ema_momentum)

        self.coarse_predictor = CoarsePredictor(
            latent_dim, action_dim, proprio_dim, horizon_h
        )
        self.fine_predictor = FinePredictor(
            latent_dim, action_dim, proprio_dim, nhead, num_fine_layers, dim_ff
        )

    @torch.no_grad()
    def update_target_encoder(self):
        self.target_encoder.update(self.encoder)

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """Online encoder: (B, 3, H, W) → (B, num_tokens, D)"""
        return self.encoder(pixels)

    @torch.no_grad()
    def encode_target(self, pixels: torch.Tensor) -> torch.Tensor:
        """Target encoder: (B, 3, H, W) → (B, num_tokens, D), no grad"""
        return self.target_encoder(pixels)


# ============================================================
# 5. Loss Computation
# ============================================================


def compute_hierarchical_loss(
    model: HierarchicalJEPAWorldModel,
    batch: dict[str, torch.Tensor],
    curriculum_ratio: float = 0.0,
    lambda_coarse: float = 1.0,
):
    """
    Compute coarse + fine losses for one batch.

    Data format:
        pixels:  (B, T, C, H, W)   — T frames per sequence
        action:  (B, T, action_dim)
        proprio: (B, T, proprio_dim)

    where T >= horizon_h + 1.

    curriculum_ratio:
        0.0 → fine predictor sees GT coarse condition (training start)
        1.0 → fine predictor sees predicted coarse condition (training end)

    Returns: L_fine, L_coarse, and dict of metrics
    """
    pixels = batch["pixels"].float()
    actions = batch["action"].float()
    proprios = batch["proprio"].float()
    h = model.horizon_h

    # ---- Encode ----
    # Online encode frame 0 (for prediction input)
    z_0 = model.encode(pixels[:, 0])  # (B, num_tokens, D)

    # Target encode frame 1 and frame h (for supervision)
    with torch.no_grad():
        z_1_target = model.encode_target(pixels[:, 1])  # (B, num_tokens, D)
        z_h_target = model.encode_target(pixels[:, h])  # (B, num_tokens, D)
        # Pool target tokens to single vectors for loss computation
        z_1_target_pooled = z_1_target.mean(dim=1)  # (B, D)
        z_h_target_pooled = z_h_target.mean(dim=1)  # (B, D)

    # ---- Coarse Prediction ----
    # Coarse predictor: z_0 + actions[0:h] → ẑ_h
    z_h_coarse = model.coarse_predictor(
        z_0, actions[:, :h], proprios[:, :h]
    )  # (B, D)

    L_coarse = F.smooth_l1_loss(z_h_coarse, z_h_target_pooled)

    # ---- Fine Prediction with Curriculum Conditioning ----
    # Blend between GT target and predicted coarse
    #   training start (ratio=0): condition = z_h_target (accurate)
    #   training end   (ratio=1): condition = z_h_coarse (matches inference)
    with torch.no_grad():
        coarse_cond = (
            curriculum_ratio * z_h_coarse.detach()
            + (1.0 - curriculum_ratio) * z_h_target_pooled
        )

    # Fine predictor: z_0 + a_0 + condition → ẑ_1
    z_1_fine = model.fine_predictor(
        z_0, actions[:, 0], proprios[:, 0], coarse_cond
    )  # (B, D)

    L_fine = F.smooth_l1_loss(z_1_fine, z_1_target_pooled)

    metrics = {
        "fine_loss": L_fine.item(),
        "coarse_loss": L_coarse.item(),
        "total_loss": (L_fine + lambda_coarse * L_coarse).item(),
        "curriculum_ratio": curriculum_ratio,
    }

    return L_fine, L_coarse, metrics


# ============================================================
# 6. Dataset & DataLoader
# ============================================================


def build_dataset(
    cache_dir: str | Path,
    img_size: int,
    num_steps: int,
    frameskip: int,
):
    keys_to_load = ["pixels", "action", "proprio"]
    keys_to_cache = ["action", "proprio"]

    dataset = swm.data.HDF5Dataset(
        name="tworoom",
        keys_to_load=keys_to_load,
        keys_to_cache=keys_to_cache,
        num_steps=num_steps,
        frameskip=frameskip,
        transform=None,
        cache_dir=cache_dir,
    )

    transforms = [
        get_img_preprocessor(
            source="pixels", target="pixels", img_size=img_size
        )
    ]
    for col in ("action", "proprio"):
        transforms.append(get_column_normalizer(dataset, col, col))

    dataset.transform = spt.data.transforms.Compose(*transforms)
    return dataset


# ============================================================
# 7. Training Loop
# ============================================================


def run_epoch(
    model: HierarchicalJEPAWorldModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    total_epochs: int,
    lambda_coarse: float = 1.0,
    curriculum_warmup_fraction: float = 0.7,
):
    is_train = optimizer is not None
    model.train(is_train)

    # Curriculum schedule: linearly increase from 0 → 1
    curriculum_ratio = min(
        1.0, epoch / max(1, total_epochs * curriculum_warmup_fraction)
    )

    sum_fine = 0.0
    sum_coarse = 0.0
    sum_total = 0.0
    count = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.set_grad_enabled(is_train):
            L_fine, L_coarse, metrics = compute_hierarchical_loss(
                model, batch, curriculum_ratio, lambda_coarse
            )
            loss = L_fine + lambda_coarse * L_coarse

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            model.update_target_encoder()

        bs = batch["pixels"].shape[0]
        count += bs
        sum_fine += metrics["fine_loss"] * bs
        sum_coarse += metrics["coarse_loss"] * bs
        sum_total += metrics["total_loss"] * bs

    return {
        "fine_loss": sum_fine / count,
        "coarse_loss": sum_coarse / count,
        "total_loss": sum_total / count,
        "curriculum_ratio": curriculum_ratio,
    }


# ============================================================
# 8. Planning (Inference)
# ============================================================


@torch.no_grad()
def hierarchical_rollout(
    model: HierarchicalJEPAWorldModel,
    start_pixels: torch.Tensor,
    actions: torch.Tensor,
    proprios: torch.Tensor,
) -> dict[str, list[torch.Tensor]]:
    """
    Hierarchical rollout for planning.

    Given a starting observation and a full action/proprio sequence,
    produce both coarse waypoints and fine-grained trajectory.

    Args:
        start_pixels: (1, 3, H, W)
        actions:      (1, T, action_dim)   — full action sequence
        proprios:     (1, T, proprio_dim)

    Returns:
        dict with:
          "waypoints":  list of coarse predictions  [ẑ_h, ẑ_{2h}, ...]
          "trajectory": list of fine predictions     [ẑ_1, ẑ_2, ..., ẑ_T]
    """
    model.eval()
    h = model.horizon_h
    T = actions.shape[1]
    num_coarse_steps = T // h

    # ---- Phase 1: Coarse — generate waypoints ----
    z_current = model.encode(start_pixels)  # (1, num_tokens, D)
    waypoints = []

    for i in range(num_coarse_steps):
        a_block = actions[:, i * h : (i + 1) * h]
        p_block = proprios[:, i * h : (i + 1) * h]
        z_waypoint = model.coarse_predictor(
            z_current, a_block, p_block
        )  # (1, D)
        waypoints.append(z_waypoint)
        # Expand back to tokens for next coarse step
        z_current = z_waypoint.unsqueeze(1).expand(
            -1, model.encoder.num_tokens, -1
        )

    # ---- Phase 2: Fine — fill in between waypoints ----
    z_current = model.encode(start_pixels)  # reset to start
    trajectory = []

    for i in range(num_coarse_steps):
        coarse_cond = waypoints[i]  # (1, D) — target waypoint

        for step in range(h):
            t = i * h + step
            if t >= T:
                break
            z_next = model.fine_predictor(
                z_current, actions[:, t], proprios[:, t], coarse_cond
            )  # (1, D)
            trajectory.append(z_next)
            # Expand to tokens for next fine step
            z_current = z_next.unsqueeze(1).expand(
                -1, model.encoder.num_tokens, -1
            )

    return {"waypoints": waypoints, "trajectory": trajectory}


# ============================================================
# 9. Main — Training Script
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Hierarchical JEPA World Model on TwoRoom"
    )
    parser.add_argument(
        "--cache-dir", type=str, default=DEFAULT_CACHE_DIR
    )
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--horizon-h", type=int, default=5)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--num-tokens", type=int, default=4)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-fine-layers", type=int, default=3)
    parser.add_argument("--dim-ff", type=int, default=512)
    parser.add_argument("--lambda-coarse", type=float, default=1.0)
    parser.add_argument("--ema-momentum", type=float, default=0.996)
    parser.add_argument(
        "--curriculum-warmup", type=float, default=0.7,
        help="Fraction of training to linearly increase curriculum ratio",
    )
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # ---- Dataset ----
    # num_steps must be > horizon_h to provide both fine and coarse targets
    assert args.num_steps > args.horizon_h, (
        f"num_steps ({args.num_steps}) must be > horizon_h ({args.horizon_h})"
    )

    dataset = build_dataset(
        cache_dir=args.cache_dir,
        img_size=args.img_size,
        num_steps=args.num_steps,
        frameskip=args.frameskip,
    )

    rnd_gen = torch.Generator().manual_seed(args.seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[args.train_split, 1 - args.train_split],
        generator=rnd_gen,
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        generator=rnd_gen,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    # ---- Model ----
    sample = next(iter(train_loader))
    action_dim = sample["action"].shape[-1]
    proprio_dim = sample["proprio"].shape[-1]

    model = HierarchicalJEPAWorldModel(
        action_dim=action_dim,
        proprio_dim=proprio_dim,
        latent_dim=args.latent_dim,
        num_tokens=args.num_tokens,
        horizon_h=args.horizon_h,
        nhead=args.nhead,
        num_fine_layers=args.num_fine_layers,
        dim_ff=args.dim_ff,
        ema_momentum=args.ema_momentum,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # ---- Print Info ----
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print("=" * 72)
    print("Hierarchical JEPA World Model — TwoRoom")
    print("=" * 72)
    print(f"device          : {device}")
    print(f"dataset size    : {len(dataset)}")
    print(f"train / val     : {len(train_set)} / {len(val_set)}")
    print(f"sequence length : {args.num_steps} (frameskip={args.frameskip})")
    print(f"coarse horizon  : {args.horizon_h}")
    print(f"latent dim      : {args.latent_dim}")
    print(f"num tokens      : {args.num_tokens}")
    print(f"fine layers     : {args.num_fine_layers}")
    print(f"attention heads : {args.nhead}")
    print(f"action dim      : {action_dim}")
    print(f"proprio dim     : {proprio_dim}")
    print(f"total params    : {num_params:,}")
    print(f"trainable params: {num_trainable:,}")
    print("=" * 72)

    # ---- Training ----
    output_dir = Path("outputs") / "hierarchical_jepa"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            epoch=epoch,
            total_epochs=args.epochs,
            lambda_coarse=args.lambda_coarse,
            curriculum_warmup_fraction=args.curriculum_warmup,
        )

        val_m = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            optimizer=None,
            epoch=epoch,
            total_epochs=args.epochs,
            lambda_coarse=args.lambda_coarse,
            curriculum_warmup_fraction=args.curriculum_warmup,
        )

        scheduler.step()

        print(
            f"[Epoch {epoch:03d}] "
            f"train: fine={train_m['fine_loss']:.4f} "
            f"coarse={train_m['coarse_loss']:.4f} "
            f"total={train_m['total_loss']:.4f} | "
            f"val: fine={val_m['fine_loss']:.4f} "
            f"coarse={val_m['coarse_loss']:.4f} "
            f"total={val_m['total_loss']:.4f} | "
            f"curriculum={train_m['curriculum_ratio']:.2f}"
        )

        if val_m["total_loss"] < best_val:
            best_val = val_m["total_loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": vars(args),
                    "action_dim": action_dim,
                    "proprio_dim": proprio_dim,
                    "best_val_loss": best_val,
                    "epoch": epoch,
                },
                output_dir / "best_hierarchical_jepa.pt",
            )

    print(f"\nBest val loss: {best_val:.4f}")
    print(f"Checkpoint saved to: {output_dir / 'best_hierarchical_jepa.pt'}")


if __name__ == "__main__":
    main()

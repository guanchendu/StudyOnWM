"""
Phase 1: Self-supervised pre-training of Hierarchical JEPA + (single) Inverse Dynamics.

Architecture:
  - Single fine inverse dynamics: â_t = inv(z_t, z_{t+1})
  - Coarse predictor: z_t + [â_0, ..., â_{h-1}] → ẑ_{t+h}    (token output)
  - Fine predictor:   z_t + â_t + cond(ẑ_{t+h})    → ẑ_{t+1}  (token output)

Training (multi-step rollout, K coarse segments × h fine steps each):
  1. Encode K*h+1 frames with online encoder
  2. EMA target encoder produces supervision targets
  3. Compute fine latent actions via inverse dynamics for all K*h pairs
  4. Multi-step coarse rollout: K segments, gradients flow through chain
  5. Multi-step fine rollout per segment with scheduled sampling
     (matches planning where fine predictions are chained from prior outputs)
  6. State variance regularization on z (VICReg-style) to prevent collapse

Usage:
  cd /Users/guanchendu/Code/StudyOnWM
  conda run -n wm python -m src.src_wm.train_phase1 --epochs 100
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn as nn
import torch.nn.functional as F

from Background.utils import get_column_normalizer, get_img_preprocessor
from src.src_wm.models import HierarchicalInvDynWorldModel

DEFAULT_CACHE_DIR = "/Users/guanchendu/Code/StudyOnWM/data"


# ============================================================
# Dataset
# ============================================================


def build_dataset(
    cache_dir: str | Path,
    img_size: int,
    num_steps: int,
    frameskip: int,
):
    """Build tworoom dataset. Phase 1 does not use action labels."""
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
        get_img_preprocessor(source="pixels", target="pixels", img_size=img_size)
    ]
    for col in ("action", "proprio"):
        transforms.append(get_column_normalizer(dataset, col, col))

    dataset.transform = spt.data.transforms.Compose(*transforms)
    return dataset


# ============================================================
# Regularizers
# ============================================================


def compute_action_reg(
    actions: torch.Tensor, variance_threshold: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """L2 + variance reg for latent actions. Prevents trivial collapse."""
    l2_reg = actions.pow(2).mean()
    per_dim_var = actions.var(dim=0)
    var_reg = F.relu(variance_threshold - per_dim_var).mean()
    return l2_reg, var_reg, per_dim_var


def compute_state_reg(
    z_pooled: torch.Tensor,
    variance_threshold: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """VICReg-style variance + covariance regularization on STATE latents.

    Prevents the encoder/EMA-target from collapsing to near-constant z
    (which would let smooth_l1(z_pred, z_target) → 0 trivially).

    Args:
        z_pooled: (N, D) pooled state vectors aggregated across the batch and time.

    Returns:
        var_reg, cov_reg, per_dim_std
    """
    eps = 1e-4
    per_dim_std = (z_pooled.var(dim=0) + eps).sqrt()
    var_reg = F.relu(variance_threshold - per_dim_std).mean()

    z_centered = z_pooled - z_pooled.mean(dim=0, keepdim=True)
    n, d = z_centered.shape
    cov = (z_centered.T @ z_centered) / max(1, n - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_reg = off_diag.pow(2).sum() / d

    return var_reg, cov_reg, per_dim_std


# ============================================================
# Loss Computation
# ============================================================


def compute_phase1_loss(
    model: HierarchicalInvDynWorldModel,
    batch: dict[str, torch.Tensor],
    curriculum_ratio: float = 0.0,
    coarse_segments: int = 2,
    lambda_coarse: float = 1.0,
    lambda_reg_action: float = 0.01,
    lambda_state_var: float = 1.0,
    lambda_state_cov: float = 0.04,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Phase 1 loss with multi-step rollout at both scales.

    Pixels (B, T, C, H, W), T >= K*h + 1. No action labels used.

    Steps:
      1. Encode all K*h+1 frames (online encoder)
      2. Encode supervision targets (EMA target encoder, no grad)
      3. Compute fine latent actions for all K*h pairs via inverse dynamics
      4. Multi-step coarse rollout (K segments) — gradients flow through chain
      5. Multi-step fine rollout per segment with scheduled sampling
      6. Action L2 + variance reg
      7. State variance + covariance reg (VICReg-style)
    """
    pixels = batch["pixels"].float()
    h = model.horizon_h
    K = coarse_segments
    total_steps = K * h + 1

    # ---- Step 1: Online encoder for all frames ----
    all_z = [model.encode(pixels[:, t]) for t in range(total_steps)]

    # ---- Step 2: Target encoder (no grad). all_z_target[t-1] ↔ pixels[:, t] ----
    with torch.no_grad():
        all_z_target = [
            model.encode_target(pixels[:, t]) for t in range(1, total_steps)
        ]

    # ---- Step 3: Fine inverse dynamics for all K*h pairs ----
    latent_actions = []
    for t in range(K * h):
        a_hat = model.compute_latent_action(all_z[t], all_z[t + 1])
        latent_actions.append(a_hat)

    # ---- Step 4+5: Unified chained rollout matching planning ----
    # Planning structure (planning.py::hierarchical_rollout):
    #   for each segment k:
    #     z_waypoint = coarse_predictor(z_current, seg_actions_k)
    #     z_t = z_current
    #     for t in 0..h-1:
    #         z_t = fine_predictor(z_t, action_t, z_waypoint)
    #     z_current = z_t          # ← next segment's coarse AND fine start from FINE output
    #
    # We mirror that here so train/planning rollout distributions match:
    #   - segment 0 starts from real encoder z_0
    #   - segment k>0 (both coarse and fine) starts from prev segment's last fine output
    teacher_forcing_ratio = max(0.0, 1.0 - curriculum_ratio)

    coarse_losses = []
    fine_losses = []
    z_current = all_z[0]
    for k in range(K):
        # Coarse waypoint for this segment
        seg_actions = torch.stack(
            latent_actions[k * h : (k + 1) * h], dim=1
        )  # (B, h, latent_action_dim)
        z_waypoint = model.coarse_predictor(z_current, seg_actions)
        coarse_target = all_z_target[(k + 1) * h - 1]
        coarse_losses.append(F.smooth_l1_loss(z_waypoint, coarse_target))

        # Coarse condition for fine: blend predicted waypoint vs target (curriculum)
        with torch.no_grad():
            coarse_cond = (
                curriculum_ratio * z_waypoint.detach()
                + (1.0 - curriculum_ratio) * coarse_target
            )

        # Fine chained rollout in segment, starting from z_current
        z_t = z_current
        for t in range(h):
            global_t = k * h + t
            z_t = model.fine_predictor(z_t, latent_actions[global_t], coarse_cond)
            fine_losses.append(F.smooth_l1_loss(z_t, all_z_target[global_t]))

            # Scheduled sampling within segment
            if t < h - 1 and torch.rand(1).item() < teacher_forcing_ratio:
                z_t = all_z[global_t + 1]

        # Next segment's coarse AND fine start from this segment's fine output
        # (matches planning).  No segment-boundary teacher forcing — planning
        # has no GT to swap in, so we don't either.
        z_current = z_t

    L_coarse = sum(coarse_losses) / len(coarse_losses)
    L_fine = sum(fine_losses) / len(fine_losses)

    # ---- Step 6: Action regularization ----
    all_actions_cat = torch.cat(latent_actions, dim=0)
    a_l2, a_var_reg, a_per_dim_var = compute_action_reg(all_actions_cat)
    L_reg_action = a_l2 + a_var_reg

    # ---- Step 7: State regularization (VICReg-style) ----
    # Aggregate online-encoder z over all frames in the batch.
    all_z_stack = torch.stack(all_z, dim=1)            # (B, T, N_tok, D)
    B_, T_, N_tok, D_ = all_z_stack.shape
    z_pooled_all = all_z_stack.mean(dim=2).reshape(B_ * T_, D_)  # (B*T, D)
    s_var_reg, s_cov_reg, s_per_dim_std = compute_state_reg(z_pooled_all)

    # ---- Total ----
    L_total = (
        L_fine
        + lambda_coarse * L_coarse
        + lambda_reg_action * L_reg_action
        + lambda_state_var * s_var_reg
        + lambda_state_cov * s_cov_reg
    )

    metrics = {
        "fine_loss": L_fine.item(),
        "coarse_loss": L_coarse.item(),
        "reg_action": L_reg_action.item(),
        "state_var_reg": s_var_reg.item(),
        "state_cov_reg": s_cov_reg.item(),
        "total_loss": L_total.item(),
        "action_norm": all_actions_cat.norm(dim=-1).mean().item(),
        "action_var": a_per_dim_var.mean().item(),
        "state_std_mean": s_per_dim_std.mean().item(),
        "state_std_min": s_per_dim_std.min().item(),
        "curriculum_ratio": curriculum_ratio,
        "teacher_forcing_ratio": teacher_forcing_ratio,
    }

    return L_total, metrics


# ============================================================
# Training Loop
# ============================================================


def run_epoch(
    model: HierarchicalInvDynWorldModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    total_epochs: int,
    coarse_segments: int = 2,
    lambda_coarse: float = 1.0,
    lambda_reg_action: float = 0.01,
    lambda_state_var: float = 1.0,
    lambda_state_cov: float = 0.04,
    curriculum_warmup_fraction: float = 0.7,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    curriculum_ratio = min(
        1.0, epoch / max(1, total_epochs * curriculum_warmup_fraction)
    )

    accum = {}
    count = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.set_grad_enabled(is_train):
            loss, metrics = compute_phase1_loss(
                model, batch, curriculum_ratio,
                coarse_segments,
                lambda_coarse,
                lambda_reg_action,
                lambda_state_var,
                lambda_state_cov,
            )

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            model.update_target_encoder()

        bs = batch["pixels"].shape[0]
        count += bs
        for k, v in metrics.items():
            accum[k] = accum.get(k, 0.0) + v * bs

    return {k: v / count for k, v in accum.items()}


# ============================================================
# Main
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 1: Hierarchical JEPA + (single) Inverse Dynamics"
    )
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--horizon-h", type=int, default=5)
    parser.add_argument("--coarse-segments", type=int, default=2,
                        help="Chained coarse predictions during training (K_train)")
    parser.add_argument("--num-steps", type=int, default=11,
                        help="Frames per sample, must be >= coarse_segments * horizon_h + 1")
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--latent-action-dim", type=int, default=32)
    parser.add_argument("--num-tokens", type=int, default=4)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-fine-layers", type=int, default=3)
    parser.add_argument("--dim-ff", type=int, default=512)
    parser.add_argument("--lambda-coarse", type=float, default=1.0)
    parser.add_argument("--lambda-reg-action", type=float, default=0.01)
    parser.add_argument("--lambda-state-var", type=float, default=1.0)
    parser.add_argument("--lambda-state-cov", type=float, default=0.04)
    parser.add_argument("--ema-momentum", type=float, default=0.996)
    parser.add_argument("--curriculum-warmup", type=float, default=0.7)
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

    assert args.num_steps >= args.coarse_segments * args.horizon_h + 1, (
        f"num_steps ({args.num_steps}) must be >= "
        f"coarse_segments * horizon_h + 1 "
        f"({args.coarse_segments * args.horizon_h + 1})"
    )

    dataset = build_dataset(
        args.cache_dir, args.img_size, args.num_steps, args.frameskip,
    )

    rnd_gen = torch.Generator().manual_seed(args.seed)
    train_set, val_set = spt.data.random_split(
        dataset, [args.train_split, 1 - args.train_split], generator=rnd_gen,
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=args.num_workers, generator=rnd_gen,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        drop_last=False, num_workers=args.num_workers,
    )

    model = HierarchicalInvDynWorldModel(
        latent_dim=args.latent_dim,
        latent_action_dim=args.latent_action_dim,
        num_tokens=args.num_tokens,
        horizon_h=args.horizon_h,
        nhead=args.nhead,
        num_fine_layers=args.num_fine_layers,
        dim_ff=args.dim_ff,
        ema_momentum=args.ema_momentum,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    num_p = sum(p.numel() for p in model.parameters())
    num_t = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 72)
    print("Phase 1: Hierarchical JEPA + (single) Inverse Dynamics")
    print("=" * 72)
    print(f"device             : {device}")
    print(f"dataset            : {len(dataset)} (train {len(train_set)} / val {len(val_set)})")
    print(f"sequence           : {args.num_steps} steps (frameskip={args.frameskip})")
    print(f"horizon (h)        : {args.horizon_h}")
    print(f"coarse segments K  : {args.coarse_segments}")
    print(f"latent dim         : {args.latent_dim}")
    print(f"latent action dim  : {args.latent_action_dim}")
    print(f"num tokens         : {args.num_tokens}")
    print(f"params             : {num_p:,} total, {num_t:,} trainable")
    print("=" * 72)

    output_dir = Path("outputs") / "hierarchical_invdyn"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(
            model, train_loader, device, optimizer,
            epoch, args.epochs,
            args.coarse_segments,
            args.lambda_coarse, args.lambda_reg_action,
            args.lambda_state_var, args.lambda_state_cov,
            args.curriculum_warmup,
        )
        val_m = run_epoch(
            model, val_loader, device, None,
            epoch, args.epochs,
            args.coarse_segments,
            args.lambda_coarse, args.lambda_reg_action,
            args.lambda_state_var, args.lambda_state_cov,
            args.curriculum_warmup,
        )
        scheduler.step()

        print(
            f"[{epoch:03d}] "
            f"train: fine={train_m['fine_loss']:.4f} coarse={train_m['coarse_loss']:.4f} "
            f"avar={train_m['action_var']:.3f} sstd={train_m['state_std_mean']:.3f}"
            f"({train_m['state_std_min']:.3f}) | "
            f"val: fine={val_m['fine_loss']:.4f} coarse={val_m['coarse_loss']:.4f} | "
            f"cur={train_m['curriculum_ratio']:.2f} tf={train_m['teacher_forcing_ratio']:.2f}"
        )

        if val_m["total_loss"] < best_val:
            best_val = val_m["total_loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": vars(args),
                    "best_val_loss": best_val,
                    "epoch": epoch,
                },
                output_dir / "best_phase1.pt",
            )

    print(f"\nBest val: {best_val:.4f} → {output_dir / 'best_phase1.pt'}")


if __name__ == "__main__":
    main()

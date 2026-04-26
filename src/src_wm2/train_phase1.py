"""
Phase 1: Self-supervised pre-training with multi-scale inverse dynamics.

No action labels needed. Two inverse dynamics models learn at different scales:
  - FineInverseDynamics:   (z_t, z_{t+1}) → â^fine   (primitive actions)
  - CoarseInverseDynamics: (z_t, z_{t+h}) → â^coarse (action abstractions)

Usage:
  cd /Users/guanchendu/Code/StudyOnWM/src
  conda run -n wm python -m src_wm2.train_phase1 --epochs 100
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
from src_wm2.models import MultiScaleInvDynWorldModel

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
# Regularization
# ============================================================


def compute_action_reg(
    actions: torch.Tensor, variance_threshold: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute L2 + variance regularization for latent actions.

    Args:
        actions: (N, action_dim) — all latent actions in the batch

    Returns:
        l2_reg, var_reg, per_dim_variance
    """
    l2_reg = actions.pow(2).mean()
    per_dim_var = actions.var(dim=0)
    var_reg = F.relu(variance_threshold - per_dim_var).mean()
    return l2_reg, var_reg, per_dim_var


# ============================================================
# Loss Computation
# ============================================================


def compute_phase1_loss(
    model: MultiScaleInvDynWorldModel,
    batch: dict[str, torch.Tensor],
    curriculum_ratio: float = 0.0,
    coarse_segments: int = 2,
    lambda_coarse: float = 1.0,
    lambda_reg_fine: float = 0.01,
    lambda_reg_coarse: float = 0.01,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Phase 1 loss with multi-scale inverse dynamics + multi-step rollout
    at BOTH scales. Aligns training distribution with planning.

    Data: pixels (B, T, C, H, W), T >= K*h + 1. No action labels used.

    Steps:
      1. Encode all frames (online encoder)
      2. Fine inv dyn from GT pairs (all K*h steps)
      3. Coarse inv dyn from GT pairs (all K segments)
      4. Multi-step coarse rollout: ẑ_h, ẑ_{2h}, ..., ẑ_{Kh}
      5. Multi-step fine rollout per segment with scheduled sampling.
         Segment k starts from ẑ_{kh} (k>0) or all_z[0] (k=0),
         matching planning where waypoints come from coarse predictor.
    """
    pixels = batch["pixels"].float()
    h = model.horizon_h
    K = coarse_segments
    total_steps = K * h + 1

    # ---- Step 1: Online encoder for all frames needed ----
    all_z = []
    for t in range(total_steps):
        all_z.append(model.encode(pixels[:, t]))

    # ---- Step 2: Target encoder for supervision ----
    # all_z_target[t-1] corresponds to pixels[:, t]
    with torch.no_grad():
        all_z_target = []
        for t in range(1, total_steps):
            all_z_target.append(model.encode_target(pixels[:, t]))

    # ---- Step 3: Fine inverse dynamics from GT pairs (all K*h steps) ----
    fine_actions = []
    for t in range(K * h):
        a_fine = model.compute_fine_action(all_z[t], all_z[t + 1])
        fine_actions.append(a_fine)

    # ---- Step 4: Coarse inverse dynamics from GT pairs (K segments) ----
    coarse_actions = []
    for k in range(K):
        a_c = model.compute_coarse_action(all_z[k * h], all_z[(k + 1) * h])
        coarse_actions.append(a_c)

    # ---- Step 5: Multi-step coarse rollout ----
    # Segment 0: real encoder input → ẑ_h
    # Segment k>0: predictor output input → ẑ_{(k+1)h}
    # Gradients flow through the chain (no detach) — teaches robustness
    # to compounding error.
    z_coarse_preds = []
    z = all_z[0]
    coarse_losses = []
    for k in range(K):
        z = model.coarse_predictor(z, coarse_actions[k])
        z_coarse_preds.append(z)
        target = all_z_target[(k + 1) * h - 1]
        coarse_losses.append(F.smooth_l1_loss(z, target))
    L_coarse = sum(coarse_losses) / len(coarse_losses)

    # ---- Step 6: Multi-step fine rollout per segment (with scheduled sampling) ----
    # Teacher forcing decreases as curriculum increases:
    #   early training: ratio≈1 (mostly use GT z_t for next step's input)
    #   late training: ratio≈0 (chain own predictions, like planning)
    teacher_forcing_ratio = max(0.0, 1.0 - curriculum_ratio)

    fine_losses = []
    for k in range(K):
        # Coarse_cond for segment k: blend curriculum predicted vs target
        with torch.no_grad():
            coarse_cond = (
                curriculum_ratio * z_coarse_preds[k].detach()
                + (1.0 - curriculum_ratio) * all_z_target[(k + 1) * h - 1]
            )

        # Starting state: real encoder for k=0, predictor output for k>0
        # (matches planning where waypoints come from coarse predictor)
        if k == 0:
            z_pred = all_z[0]
        else:
            z_pred = z_coarse_preds[k - 1]

        for t in range(h):
            global_t = k * h + t
            z_pred = model.fine_predictor(
                z_pred, fine_actions[global_t], coarse_cond
            )
            fine_losses.append(F.smooth_l1_loss(z_pred, all_z_target[global_t]))

            # Scheduled sampling: replace next-step input with GT
            if t < h - 1 and torch.rand(1).item() < teacher_forcing_ratio:
                z_pred = all_z[global_t + 1]

    L_fine = sum(fine_losses) / len(fine_losses)

    # ---- Step 7: Regularization ----
    all_fine_cat = torch.cat(fine_actions, dim=0)
    fine_l2, fine_var_reg, fine_var = compute_action_reg(all_fine_cat)
    L_reg_fine = fine_l2 + fine_var_reg

    all_coarse_cat = torch.cat(coarse_actions, dim=0)
    coarse_l2, coarse_var_reg, coarse_var = compute_action_reg(
        all_coarse_cat, variance_threshold=0.1
    )
    L_reg_coarse = coarse_l2 + coarse_var_reg

    # ---- Total ----
    L_total = (
        L_fine
        + lambda_coarse * L_coarse
        + lambda_reg_fine * L_reg_fine
        + lambda_reg_coarse * L_reg_coarse
    )

    metrics = {
        "fine_loss": L_fine.item(),
        "coarse_loss": L_coarse.item(),
        "reg_fine": L_reg_fine.item(),
        "reg_coarse": L_reg_coarse.item(),
        "total_loss": L_total.item(),
        "fine_action_var": fine_var.mean().item(),
        "coarse_action_var": coarse_var.mean().item(),
        "fine_action_norm": all_fine_cat.norm(dim=-1).mean().item(),
        "coarse_action_norm": all_coarse_cat.norm(dim=-1).mean().item(),
        "curriculum_ratio": curriculum_ratio,
        "teacher_forcing_ratio": teacher_forcing_ratio,
    }

    return L_total, metrics


# ============================================================
# Training Loop
# ============================================================


def run_epoch(
    model: MultiScaleInvDynWorldModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    total_epochs: int,
    coarse_segments: int = 2,
    lambda_coarse: float = 1.0,
    lambda_reg_fine: float = 0.01,
    lambda_reg_coarse: float = 0.01,
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
                lambda_coarse, lambda_reg_fine, lambda_reg_coarse,
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
        description="Phase 1: Multi-scale inverse dynamics pre-training"
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
    parser.add_argument("--fine-action-dim", type=int, default=32)
    parser.add_argument("--coarse-action-dim", type=int, default=64)
    parser.add_argument("--num-tokens", type=int, default=4)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-fine-layers", type=int, default=3)
    parser.add_argument("--dim-ff", type=int, default=512)
    parser.add_argument("--lambda-coarse", type=float, default=1.0)
    parser.add_argument("--lambda-reg-fine", type=float, default=0.01)
    parser.add_argument("--lambda-reg-coarse", type=float, default=0.01)
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
        f"coarse_segments * horizon_h + 1 ({args.coarse_segments * args.horizon_h + 1})"
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

    model = MultiScaleInvDynWorldModel(
        latent_dim=args.latent_dim,
        fine_action_dim=args.fine_action_dim,
        coarse_action_dim=args.coarse_action_dim,
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
    print("Phase 1: Multi-Scale Inverse Dynamics Pre-training")
    print("=" * 72)
    print(f"device             : {device}")
    print(f"dataset            : {len(dataset)} (train {len(train_set)} / val {len(val_set)})")
    print(f"sequence           : {args.num_steps} steps (frameskip={args.frameskip})")
    print(f"coarse horizon (h) : {args.horizon_h}")
    print(f"latent dim         : {args.latent_dim}")
    print(f"fine action dim    : {args.fine_action_dim}")
    print(f"coarse action dim  : {args.coarse_action_dim}")
    print(f"params             : {num_p:,} total, {num_t:,} trainable")
    print("=" * 72)

    output_dir = Path("outputs") / "multiscale_invdyn"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(
            model, train_loader, device, optimizer,
            epoch, args.epochs,
            args.coarse_segments,
            args.lambda_coarse, args.lambda_reg_fine, args.lambda_reg_coarse,
            args.curriculum_warmup,
        )
        val_m = run_epoch(
            model, val_loader, device, None,
            epoch, args.epochs,
            args.coarse_segments,
            args.lambda_coarse, args.lambda_reg_fine, args.lambda_reg_coarse,
            args.curriculum_warmup,
        )
        scheduler.step()

        print(
            f"[{epoch:03d}] "
            f"train: fine={train_m['fine_loss']:.4f} coarse={train_m['coarse_loss']:.4f} "
            f"fvar={train_m['fine_action_var']:.4f} cvar={train_m['coarse_action_var']:.4f} | "
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

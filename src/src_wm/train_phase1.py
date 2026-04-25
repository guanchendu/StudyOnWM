"""
Phase 1: Self-supervised pre-training (no action labels needed).

Training flow per batch:
  1. Encode all frames with online encoder → z_0, z_1, ..., z_h
  2. Encode supervision targets with target encoder → z_1^target, z_h^target
  3. Compute latent actions via inverse dynamics: â_t = inv(z_t, z_{t+1})
  4. Coarse prediction: z_0 + [â_0,...,â_{h-1}] → ẑ_h,  supervised by z_h^target
  5. Fine prediction:   z_0 + â_0 + cond(ẑ_h)  → ẑ_1,  supervised by z_1^target
  6. Regularize latent actions to prevent collapse

Usage:
  cd /Users/guanchendu/Code/StudyOnWM/src
  python -m src_wm.train_phase1 --horizon-h 5 --num-steps 8 --epochs 100
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
import torch.nn.functional as F

from Background.utils import get_column_normalizer, get_img_preprocessor
from src_wm.models import HierarchicalInvDynWorldModel

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
    """Build tworoom dataset. We load action/proprio for Phase 2 compatibility,
    but Phase 1 training does NOT use them — latent actions come from inverse dynamics."""
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
# Loss Computation
# ============================================================


def compute_phase1_loss(
    model: HierarchicalInvDynWorldModel,
    batch: dict[str, torch.Tensor],
    curriculum_ratio: float = 0.0,
    lambda_coarse: float = 1.0,
    lambda_reg: float = 0.01,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute Phase 1 losses: L_fine + λ_coarse * L_coarse + λ_reg * L_reg.

    Data format:
        pixels: (B, T, C, H, W) where T >= horizon_h + 1

    No action labels are used. Latent actions come from inverse dynamics.
    """
    pixels = batch["pixels"].float()
    B, T = pixels.shape[:2]
    h = model.horizon_h

    # ---- Step 1: Encode ALL frames with online encoder ----
    # We need z_0, z_1, ..., z_h for inverse dynamics (all need gradients)
    all_z = []
    for t in range(h + 1):
        all_z.append(model.encode(pixels[:, t]))  # (B, num_tokens, D)

    # ---- Step 2: Target encoder for supervision ----
    with torch.no_grad():
        z_1_target = model.encode_target(pixels[:, 1]).mean(dim=1)  # (B, D)
        z_h_target = model.encode_target(pixels[:, h]).mean(dim=1)  # (B, D)

    # ---- Step 3: Compute latent actions via inverse dynamics ----
    latent_actions = []
    for t in range(h):
        a_hat = model.compute_latent_action(all_z[t], all_z[t + 1])  # (B, latent_action_dim)
        latent_actions.append(a_hat)

    latent_actions_stacked = torch.stack(latent_actions, dim=1)  # (B, h, latent_action_dim)

    # ---- Step 4: Coarse prediction ----
    z_h_coarse = model.coarse_predictor(
        all_z[0], latent_actions_stacked
    )  # (B, D)

    L_coarse = F.smooth_l1_loss(z_h_coarse, z_h_target)

    # ---- Step 5: Fine prediction with curriculum conditioning ----
    with torch.no_grad():
        coarse_cond = (
            curriculum_ratio * z_h_coarse.detach()
            + (1.0 - curriculum_ratio) * z_h_target
        )

    z_1_fine = model.fine_predictor(
        all_z[0], latent_actions[0], coarse_cond
    )  # (B, D)

    L_fine = F.smooth_l1_loss(z_1_fine, z_1_target)

    # ---- Step 6: Latent action regularization (prevent collapse) ----
    # L2 norm penalty
    L_reg_l2 = sum(a.pow(2).mean() for a in latent_actions) / len(latent_actions)

    # Variance regularization: ensure each dim has variance across the batch
    all_actions_cat = torch.cat(latent_actions, dim=0)  # (B*h, latent_action_dim)
    per_dim_var = all_actions_cat.var(dim=0)  # (latent_action_dim,)
    variance_threshold = 0.1
    L_reg_var = F.relu(variance_threshold - per_dim_var).mean()

    L_reg = L_reg_l2 + L_reg_var

    # ---- Total loss ----
    L_total = L_fine + lambda_coarse * L_coarse + lambda_reg * L_reg

    metrics = {
        "fine_loss": L_fine.item(),
        "coarse_loss": L_coarse.item(),
        "reg_loss": L_reg.item(),
        "reg_l2": L_reg_l2.item(),
        "reg_var": L_reg_var.item(),
        "total_loss": L_total.item(),
        "latent_action_norm": all_actions_cat.norm(dim=-1).mean().item(),
        "latent_action_var": per_dim_var.mean().item(),
        "curriculum_ratio": curriculum_ratio,
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
    lambda_coarse: float = 1.0,
    lambda_reg: float = 0.01,
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
                model, batch, curriculum_ratio, lambda_coarse, lambda_reg
            )

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        description="Phase 1: Self-supervised pre-training of Hierarchical JEPA + InvDyn"
    )
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--horizon-h", type=int, default=5)
    parser.add_argument("--num-steps", type=int, default=8)
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
    parser.add_argument("--lambda-reg", type=float, default=0.01)
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

    assert args.num_steps > args.horizon_h, (
        f"num_steps ({args.num_steps}) must be > horizon_h ({args.horizon_h})"
    )

    # ---- Dataset ----
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

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 72)
    print("Phase 1: Self-Supervised Pre-training")
    print("Hierarchical JEPA + Inverse Dynamics")
    print("=" * 72)
    print(f"device             : {device}")
    print(f"dataset size       : {len(dataset)}")
    print(f"train / val        : {len(train_set)} / {len(val_set)}")
    print(f"sequence length    : {args.num_steps} (frameskip={args.frameskip})")
    print(f"coarse horizon (h) : {args.horizon_h}")
    print(f"latent dim         : {args.latent_dim}")
    print(f"latent action dim  : {args.latent_action_dim}")
    print(f"num tokens         : {args.num_tokens}")
    print(f"fine layers        : {args.num_fine_layers}")
    print(f"total params       : {num_params:,}")
    print(f"trainable params   : {num_trainable:,}")
    print("=" * 72)

    # ---- Training ----
    output_dir = Path("outputs") / "hierarchical_invdyn"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(
            model, train_loader, device, optimizer,
            epoch, args.epochs, args.lambda_coarse, args.lambda_reg,
            args.curriculum_warmup,
        )

        val_m = run_epoch(
            model, val_loader, device, None,
            epoch, args.epochs, args.lambda_coarse, args.lambda_reg,
            args.curriculum_warmup,
        )

        scheduler.step()

        print(
            f"[Epoch {epoch:03d}] "
            f"train: fine={train_m['fine_loss']:.4f} "
            f"coarse={train_m['coarse_loss']:.4f} "
            f"reg={train_m['reg_loss']:.4f} "
            f"act_var={train_m['latent_action_var']:.4f} | "
            f"val: fine={val_m['fine_loss']:.4f} "
            f"coarse={val_m['coarse_loss']:.4f} | "
            f"cur={train_m['curriculum_ratio']:.2f}"
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

    print(f"\nBest val loss: {best_val:.4f}")
    print(f"Checkpoint: {output_dir / 'best_phase1.pt'}")


if __name__ == "__main__":
    main()

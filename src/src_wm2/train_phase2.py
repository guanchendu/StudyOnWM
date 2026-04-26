"""
Phase 2: Two-scale action space alignment.

Freeze Phase 1. Train four lightweight networks:
  Fine scale:   FineActionEncoder   (a_t → ã^fine)
                FineActionDecoder   (â^fine → a_t)
  Coarse scale: CoarseActionEncoder ([a_t,...,a_{t+h-1}] → ã^coarse)
                CoarseActionDecoder (â^coarse → [a_t,...,a_{t+h-1}])

Usage:
  cd /Users/guanchendu/Code/StudyOnWM/src
  conda run -n wm python -m src_wm2.train_phase2 \
    --phase1-ckpt outputs/multiscale_invdyn/best_phase1.pt \
    --label-fraction 0.1
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
from src_wm2.models import (
    CoarseActionDecoder,
    CoarseActionEncoder,
    FineActionDecoder,
    FineActionEncoder,
    MultiScaleInvDynWorldModel,
)

DEFAULT_CACHE_DIR = "/Users/guanchendu/Code/StudyOnWM/data"


# ============================================================
# Dataset
# ============================================================


def build_dataset(
    cache_dir: str | Path,
    img_size: int,
    num_steps: int,
    frameskip: int,
    label_fraction: float = 1.0,
    seed: int = 42,
):
    """Build dataset for Phase 2. Needs sequences of length >= h+1
    to compute coarse inverse dynamics targets."""
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

    if label_fraction < 1.0:
        n_total = len(dataset)
        n_use = max(1, int(n_total * label_fraction))
        gen = torch.Generator().manual_seed(seed)
        dataset, _ = spt.data.random_split(
            dataset, [n_use / n_total, 1.0 - n_use / n_total], generator=gen,
        )

    return dataset


# ============================================================
# Loss
# ============================================================


def compute_phase2_loss(
    phase1: MultiScaleInvDynWorldModel,
    fine_enc: FineActionEncoder,
    fine_dec: FineActionDecoder,
    coarse_enc: CoarseActionEncoder,
    coarse_dec: CoarseActionDecoder,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute alignment loss at both scales.

    Data: pixels (B, T, C, H, W), action (B, T, action_dim), T >= h+1
    """
    pixels = batch["pixels"].float()
    actions = batch["action"].float()
    h = phase1.horizon_h

    a_t = actions[:, 0]                           # (B, action_dim)
    actions_seq = actions[:, :h].flatten(1)        # (B, h * action_dim)

    # ---- Frozen Phase 1: get latent actions ----
    with torch.no_grad():
        z_0 = phase1.encode(pixels[:, 0])
        z_1 = phase1.encode(pixels[:, 1])
        z_h = phase1.encode(pixels[:, h])

        a_hat_fine = phase1.compute_fine_action(z_0, z_1)       # (B, fine_action_dim)
        a_hat_coarse = phase1.compute_coarse_action(z_0, z_h)   # (B, coarse_action_dim)

    # ---- Fine scale alignment ----
    a_tilde_fine = fine_enc(a_t)
    a_recon_fine = fine_dec(a_hat_fine)
    a_cycle_fine = fine_dec(a_tilde_fine)

    L_align_fine = F.mse_loss(a_tilde_fine, a_hat_fine)
    L_decode_fine = F.mse_loss(a_recon_fine, a_t)
    L_cycle_fine = F.mse_loss(a_cycle_fine, a_t)

    # ---- Coarse scale alignment ----
    a_tilde_coarse = coarse_enc(actions_seq)
    a_recon_coarse = coarse_dec(a_hat_coarse)
    a_cycle_coarse = coarse_dec(a_tilde_coarse)

    L_align_coarse = F.mse_loss(a_tilde_coarse, a_hat_coarse)
    L_decode_coarse = F.mse_loss(a_recon_coarse, actions_seq)
    L_cycle_coarse = F.mse_loss(a_cycle_coarse, actions_seq)

    # ---- Total ----
    L_fine_total = L_align_fine + L_decode_fine + 0.5 * L_cycle_fine
    L_coarse_total = L_align_coarse + L_decode_coarse + 0.5 * L_cycle_coarse
    L_total = L_fine_total + L_coarse_total

    metrics = {
        "align_fine": L_align_fine.item(),
        "decode_fine": L_decode_fine.item(),
        "cycle_fine": L_cycle_fine.item(),
        "align_coarse": L_align_coarse.item(),
        "decode_coarse": L_decode_coarse.item(),
        "cycle_coarse": L_cycle_coarse.item(),
        "total_loss": L_total.item(),
    }

    return L_total, metrics


# ============================================================
# Training Loop
# ============================================================


def run_epoch(
    phase1: MultiScaleInvDynWorldModel,
    fine_enc: FineActionEncoder,
    fine_dec: FineActionDecoder,
    coarse_enc: CoarseActionEncoder,
    coarse_dec: CoarseActionDecoder,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    is_train = optimizer is not None
    phase1.eval()
    fine_enc.train(is_train)
    fine_dec.train(is_train)
    coarse_enc.train(is_train)
    coarse_dec.train(is_train)

    accum = {}
    count = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.set_grad_enabled(is_train):
            loss, metrics = compute_phase2_loss(
                phase1, fine_enc, fine_dec, coarse_enc, coarse_dec, batch,
            )

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
        description="Phase 2: Two-scale action alignment"
    )
    parser.add_argument("--phase1-ckpt", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--label-fraction", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim-fine", type=int, default=128)
    parser.add_argument("--hidden-dim-coarse", type=int, default=256)
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

    # ---- Load Phase 1 ----
    ckpt = torch.load(args.phase1_ckpt, map_location="cpu")
    p1 = ckpt["config"]

    phase1 = MultiScaleInvDynWorldModel(
        latent_dim=p1["latent_dim"],
        fine_action_dim=p1["fine_action_dim"],
        coarse_action_dim=p1["coarse_action_dim"],
        num_tokens=p1["num_tokens"],
        horizon_h=p1["horizon_h"],
        nhead=p1["nhead"],
        num_fine_layers=p1["num_fine_layers"],
        dim_ff=p1["dim_ff"],
        ema_momentum=p1["ema_momentum"],
    ).to(device)
    phase1.load_state_dict(ckpt["model_state_dict"])
    phase1.eval()
    phase1.requires_grad_(False)

    # ---- Dataset ----
    num_steps = p1["horizon_h"] + 1
    dataset = build_dataset(
        args.cache_dir, args.img_size, num_steps, args.frameskip,
        args.label_fraction, args.seed,
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

    # ---- Action dim from data ----
    sample = next(iter(train_loader))
    action_dim = sample["action"].shape[-1]
    horizon_h = p1["horizon_h"]
    fine_action_dim = p1["fine_action_dim"]
    coarse_action_dim = p1["coarse_action_dim"]

    # ---- Phase 2 models ----
    fine_enc = FineActionEncoder(action_dim, fine_action_dim, args.hidden_dim_fine).to(device)
    fine_dec = FineActionDecoder(fine_action_dim, action_dim, args.hidden_dim_fine).to(device)
    coarse_enc = CoarseActionEncoder(
        action_dim, horizon_h, coarse_action_dim, args.hidden_dim_coarse
    ).to(device)
    coarse_dec = CoarseActionDecoder(
        coarse_action_dim, action_dim, horizon_h, args.hidden_dim_coarse
    ).to(device)

    all_params = (
        list(fine_enc.parameters()) + list(fine_dec.parameters())
        + list(coarse_enc.parameters()) + list(coarse_dec.parameters())
    )
    optimizer = torch.optim.Adam(all_params, lr=args.lr)

    total_p2_params = sum(p.numel() for p in all_params)

    print("=" * 72)
    print("Phase 2: Two-Scale Action Alignment")
    print("=" * 72)
    print(f"device             : {device}")
    print(f"phase1 checkpoint  : {args.phase1_ckpt}")
    print(f"label fraction     : {args.label_fraction:.1%}")
    print(f"dataset            : {len(dataset)} (train {len(train_set)} / val {len(val_set)})")
    print(f"action dim         : {action_dim}")
    print(f"fine action dim    : {fine_action_dim}")
    print(f"coarse action dim  : {coarse_action_dim}")
    print(f"phase2 params      : {total_p2_params:,}")
    print("=" * 72)

    output_dir = Path("outputs") / "multiscale_invdyn"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(
            phase1, fine_enc, fine_dec, coarse_enc, coarse_dec,
            train_loader, device, optimizer,
        )
        val_m = run_epoch(
            phase1, fine_enc, fine_dec, coarse_enc, coarse_dec,
            val_loader, device, None,
        )

        print(
            f"[{epoch:03d}] "
            f"train: af={train_m['align_fine']:.4f} df={train_m['decode_fine']:.4f} "
            f"ac={train_m['align_coarse']:.4f} dc={train_m['decode_coarse']:.4f} | "
            f"val: af={val_m['align_fine']:.4f} ac={val_m['align_coarse']:.4f}"
        )

        if val_m["total_loss"] < best_val:
            best_val = val_m["total_loss"]
            torch.save(
                {
                    "fine_enc_state": fine_enc.state_dict(),
                    "fine_dec_state": fine_dec.state_dict(),
                    "coarse_enc_state": coarse_enc.state_dict(),
                    "coarse_dec_state": coarse_dec.state_dict(),
                    "config": vars(args),
                    "phase1_config": p1,
                    "action_dim": action_dim,
                    "best_val_loss": best_val,
                    "epoch": epoch,
                },
                output_dir / "best_phase2.pt",
            )

    print(f"\nBest val: {best_val:.4f} → {output_dir / 'best_phase2.pt'}")


if __name__ == "__main__":
    main()

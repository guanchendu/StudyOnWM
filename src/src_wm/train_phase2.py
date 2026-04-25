"""
Phase 2: Action space alignment with small amount of labeled data.

Freeze all Phase 1 parameters. Train only:
  - ActionEncoder:  a_t (real action) → ã_t (latent action space)
  - ActionDecoder:  â_t (latent action) → a_t (real action)

Loss:
  L_align  = ||ã_t - â_t||²     (action encoder output matches inverse dynamics output)
  L_decode = ||decode(â_t) - a_t||²  (action decoder can reconstruct real action)

Usage:
  cd /Users/guanchendu/Code/StudyOnWM/src
  python -m src_wm.train_phase2 --phase1-ckpt outputs/hierarchical_invdyn/best_phase1.pt
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
from src_wm.models import (
    ActionDecoder,
    ActionEncoder,
    HierarchicalInvDynWorldModel,
)

DEFAULT_CACHE_DIR = "/Users/guanchendu/Code/StudyOnWM/data"


# ============================================================
# Dataset
# ============================================================


def build_dataset(
    cache_dir: str | Path,
    img_size: int,
    frameskip: int,
    label_fraction: float = 1.0,
    seed: int = 42,
):
    """Build dataset for Phase 2. Uses num_steps=2 (only need consecutive pairs).
    Optionally subsample to simulate small labeled dataset."""
    keys_to_load = ["pixels", "action", "proprio"]
    keys_to_cache = ["action", "proprio"]

    dataset = swm.data.HDF5Dataset(
        name="tworoom",
        keys_to_load=keys_to_load,
        keys_to_cache=keys_to_cache,
        num_steps=2,
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
            dataset,
            lengths=[n_use / n_total, 1.0 - n_use / n_total],
            generator=gen,
        )

    return dataset


# ============================================================
# Loss Computation
# ============================================================


def compute_phase2_loss(
    phase1_model: HierarchicalInvDynWorldModel,
    action_enc: ActionEncoder,
    action_dec: ActionDecoder,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute alignment loss using paired (o_t, a_t, o_{t+1}).

    Phase 1 model is frozen — only action_enc and action_dec receive gradients.
    """
    pixels = batch["pixels"].float()  # (B, 2, C, H, W)
    actions = batch["action"].float()  # (B, 2, action_dim)

    a_t = actions[:, 0]  # (B, action_dim) — real action at time t

    # ---- Get latent action from Phase 1 inverse dynamics (frozen) ----
    with torch.no_grad():
        z_t = phase1_model.encode(pixels[:, 0])      # (B, num_tokens, D)
        z_tp1 = phase1_model.encode(pixels[:, 1])    # (B, num_tokens, D)
        a_hat = phase1_model.compute_latent_action(z_t, z_tp1)  # (B, latent_action_dim)

    # ---- Action Encoder: real → latent ----
    a_tilde = action_enc(a_t)  # (B, latent_action_dim)

    # ---- Action Decoder: latent → real ----
    a_recon = action_dec(a_hat)  # (B, action_dim)

    # ---- Losses ----
    L_align = F.mse_loss(a_tilde, a_hat)
    L_decode = F.mse_loss(a_recon, a_t)

    # Cycle consistency: encode then decode should recover real action
    a_cycle = action_dec(a_tilde)
    L_cycle = F.mse_loss(a_cycle, a_t)

    L_total = L_align + L_decode + 0.5 * L_cycle

    metrics = {
        "align_loss": L_align.item(),
        "decode_loss": L_decode.item(),
        "cycle_loss": L_cycle.item(),
        "total_loss": L_total.item(),
    }

    return L_total, metrics


# ============================================================
# Training Loop
# ============================================================


def run_epoch(
    phase1_model: HierarchicalInvDynWorldModel,
    action_enc: ActionEncoder,
    action_dec: ActionDecoder,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    is_train = optimizer is not None
    phase1_model.eval()
    action_enc.train(is_train)
    action_dec.train(is_train)

    accum = {}
    count = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.set_grad_enabled(is_train):
            loss, metrics = compute_phase2_loss(
                phase1_model, action_enc, action_dec, batch
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
        description="Phase 2: Action alignment with small labeled dataset"
    )
    parser.add_argument(
        "--phase1-ckpt", type=str, required=True,
        help="Path to Phase 1 checkpoint (best_phase1.pt)",
    )
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--label-fraction", type=float, default=0.1,
                        help="Fraction of labeled data to use (e.g., 0.05 = 5%%)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
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

    # ---- Load Phase 1 model (frozen) ----
    ckpt = torch.load(args.phase1_ckpt, map_location="cpu")
    p1_cfg = ckpt["config"]

    phase1_model = HierarchicalInvDynWorldModel(
        latent_dim=p1_cfg["latent_dim"],
        latent_action_dim=p1_cfg["latent_action_dim"],
        num_tokens=p1_cfg["num_tokens"],
        horizon_h=p1_cfg["horizon_h"],
        nhead=p1_cfg["nhead"],
        num_fine_layers=p1_cfg["num_fine_layers"],
        dim_ff=p1_cfg["dim_ff"],
        ema_momentum=p1_cfg["ema_momentum"],
    ).to(device)
    phase1_model.load_state_dict(ckpt["model_state_dict"])
    phase1_model.eval()
    phase1_model.requires_grad_(False)

    # ---- Dataset ----
    dataset = build_dataset(
        cache_dir=args.cache_dir,
        img_size=args.img_size,
        frameskip=args.frameskip,
        label_fraction=args.label_fraction,
        seed=args.seed,
    )

    rnd_gen = torch.Generator().manual_seed(args.seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[args.train_split, 1 - args.train_split],
        generator=rnd_gen,
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=args.num_workers, generator=rnd_gen,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        drop_last=False, num_workers=args.num_workers,
    )

    # ---- Determine action_dim from data ----
    sample = next(iter(train_loader))
    action_dim = sample["action"].shape[-1]
    latent_action_dim = p1_cfg["latent_action_dim"]

    # ---- Phase 2 models ----
    action_enc = ActionEncoder(action_dim, latent_action_dim, args.hidden_dim).to(device)
    action_dec = ActionDecoder(latent_action_dim, action_dim, args.hidden_dim).to(device)

    params = list(action_enc.parameters()) + list(action_dec.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    print("=" * 72)
    print("Phase 2: Action Space Alignment")
    print("=" * 72)
    print(f"device             : {device}")
    print(f"phase1 checkpoint  : {args.phase1_ckpt}")
    print(f"label fraction     : {args.label_fraction:.1%}")
    print(f"dataset size       : {len(dataset)}")
    print(f"train / val        : {len(train_set)} / {len(val_set)}")
    print(f"action dim         : {action_dim}")
    print(f"latent action dim  : {latent_action_dim}")
    enc_params = sum(p.numel() for p in action_enc.parameters())
    dec_params = sum(p.numel() for p in action_dec.parameters())
    print(f"action encoder params: {enc_params:,}")
    print(f"action decoder params: {dec_params:,}")
    print("=" * 72)

    # ---- Training ----
    output_dir = Path("outputs") / "hierarchical_invdyn"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(
            phase1_model, action_enc, action_dec,
            train_loader, device, optimizer,
        )
        val_m = run_epoch(
            phase1_model, action_enc, action_dec,
            val_loader, device, None,
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"train: align={train_m['align_loss']:.4f} "
            f"decode={train_m['decode_loss']:.4f} "
            f"cycle={train_m['cycle_loss']:.4f} | "
            f"val: align={val_m['align_loss']:.4f} "
            f"decode={val_m['decode_loss']:.4f}"
        )

        if val_m["total_loss"] < best_val:
            best_val = val_m["total_loss"]
            torch.save(
                {
                    "action_encoder_state_dict": action_enc.state_dict(),
                    "action_decoder_state_dict": action_dec.state_dict(),
                    "config": vars(args),
                    "phase1_config": p1_cfg,
                    "action_dim": action_dim,
                    "best_val_loss": best_val,
                    "epoch": epoch,
                },
                output_dir / "best_phase2.pt",
            )

    print(f"\nBest val loss: {best_val:.4f}")
    print(f"Checkpoint: {output_dir / 'best_phase2.pt'}")


if __name__ == "__main__":
    main()

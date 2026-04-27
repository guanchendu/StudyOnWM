"""
Phase 2: Action space alignment + Proprio decoder training.

Freeze all Phase 1 parameters. Train three lightweight networks:
  - ActionEncoder:  a_t (real action) → ã_t (latent action space)
  - ActionDecoder:  â_t (latent action) → a_t (real action)
  - ProprioDecoder: z (latent state)   → proprio  (used by planner cost)

Losses:
  L_align    = ||ã_t - â_t||²
  L_decode   = ||decode(â_t) - a_t||²
  L_cycle    = ||decode(encode(a_t)) - a_t||²
  L_proprio  = MSE on encoder outputs + MSE on predictor outputs
               (latter is critical: planning cost is computed on rolled-out z)

Sequences of length >= h+1 are loaded so we can reach z_h via the coarse
predictor and supervise proprio_dec on its output too.

Usage:
  cd /Users/guanchendu/Code/StudyOnWM/src
  conda run -n wm python -m src_wm.train_phase2 \
    --phase1-ckpt outputs/hierarchical_invdyn/best_phase1.pt \
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
from src_wm.models import (
    ActionDecoder,
    ActionEncoder,
    HierarchicalInvDynWorldModel,
    ProprioDecoder,
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
    """Phase 2 needs sequences of length >= h+1 for proprio_dec on predictor outputs."""
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
            dataset,
            lengths=[n_use / n_total, 1.0 - n_use / n_total],
            generator=gen,
        )

    return dataset


# ============================================================
# Loss Computation
# ============================================================


def compute_phase2_loss(
    phase1: HierarchicalInvDynWorldModel,
    action_enc: ActionEncoder,
    action_dec: ActionDecoder,
    proprio_dec: ProprioDecoder,
    batch: dict[str, torch.Tensor],
    lambda_proprio: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Action alignment + proprio decoding (encoder + predictor outputs).

    Pixels (B, T, C, H, W), action (B, T, action_dim), proprio (B, T, proprio_dim),
    T >= h+1.
    """
    pixels = batch["pixels"].float()
    actions = batch["action"].float()
    proprio = batch["proprio"].float()
    h = phase1.horizon_h

    a_t = actions[:, 0]                    # (B, action_dim)

    # ---- Frozen Phase 1: get latent states + action + predictor outputs ----
    with torch.no_grad():
        z_0 = phase1.encode(pixels[:, 0])          # (B, num_tokens, D)
        z_1 = phase1.encode(pixels[:, 1])
        z_h = phase1.encode(pixels[:, h])

        a_hat = phase1.compute_latent_action(z_0, z_1)  # (B, latent_action_dim)

        # Stack h fine actions for the coarse predictor.
        # We re-use a_hat for all h slots (rough proxy in the unlabeled
        # setting); for labeled data the encoder maps real actions instead,
        # but here proprio_dec only needs predictor outputs, not exact actions.
        seg_actions = []
        for t in range(h):
            z_t_p = phase1.encode(pixels[:, t])
            z_tp1 = phase1.encode(pixels[:, t + 1])
            seg_actions.append(phase1.compute_latent_action(z_t_p, z_tp1))
        seg_actions_stacked = torch.stack(seg_actions, dim=1)  # (B, h, la_dim)

        z_h_pred = phase1.coarse_predictor(z_0, seg_actions_stacked)
        z_1_pred = phase1.fine_predictor(z_0, seg_actions[0], z_h_pred)

    # ---- Action alignment ----
    a_tilde = action_enc(a_t)
    a_recon = action_dec(a_hat)
    a_cycle = action_dec(a_tilde)

    L_align = F.mse_loss(a_tilde, a_hat)
    L_decode = F.mse_loss(a_recon, a_t)
    L_cycle = F.mse_loss(a_cycle, a_t)

    # ---- Proprio decoding (encoder outputs) ----
    L_p_enc = (
        F.mse_loss(proprio_dec(z_0), proprio[:, 0])
        + F.mse_loss(proprio_dec(z_1), proprio[:, 1])
        + F.mse_loss(proprio_dec(z_h), proprio[:, h])
    ) / 3

    # ---- Proprio decoding (predictor outputs) ----
    # Critical: planning cost is computed on rolled-out predictor states,
    # so proprio_dec must be in distribution on those.
    L_p_pred = (
        F.mse_loss(proprio_dec(z_h_pred), proprio[:, h])
        + F.mse_loss(proprio_dec(z_1_pred), proprio[:, 1])
    ) / 2

    L_proprio = L_p_enc + L_p_pred

    # ---- Total ----
    L_action = L_align + L_decode + 0.5 * L_cycle
    L_total = L_action + lambda_proprio * L_proprio

    metrics = {
        "align": L_align.item(),
        "decode": L_decode.item(),
        "cycle": L_cycle.item(),
        "proprio_enc": L_p_enc.item(),
        "proprio_pred": L_p_pred.item(),
        "total_loss": L_total.item(),
    }

    return L_total, metrics


# ============================================================
# Training Loop
# ============================================================


def run_epoch(
    phase1: HierarchicalInvDynWorldModel,
    action_enc: ActionEncoder,
    action_dec: ActionDecoder,
    proprio_dec: ProprioDecoder,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    lambda_proprio: float = 1.0,
) -> dict[str, float]:
    is_train = optimizer is not None
    phase1.eval()
    action_enc.train(is_train)
    action_dec.train(is_train)
    proprio_dec.train(is_train)

    accum = {}
    count = 0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.set_grad_enabled(is_train):
            loss, metrics = compute_phase2_loss(
                phase1, action_enc, action_dec, proprio_dec,
                batch, lambda_proprio,
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
        description="Phase 2: Action alignment + ProprioDecoder"
    )
    parser.add_argument("--phase1-ckpt", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--label-fraction", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Hidden dim for action encoder/decoder")
    parser.add_argument("--proprio-hidden-dim", type=int, default=128,
                        help="Hidden dim for proprio decoder")
    parser.add_argument("--lambda-proprio", type=float, default=1.0)
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

    # ---- Load Phase 1 (frozen) ----
    ckpt = torch.load(args.phase1_ckpt, map_location="cpu")
    p1_cfg = ckpt["config"]

    phase1 = HierarchicalInvDynWorldModel(
        latent_dim=p1_cfg["latent_dim"],
        latent_action_dim=p1_cfg["latent_action_dim"],
        num_tokens=p1_cfg["num_tokens"],
        horizon_h=p1_cfg["horizon_h"],
        nhead=p1_cfg["nhead"],
        num_fine_layers=p1_cfg["num_fine_layers"],
        dim_ff=p1_cfg["dim_ff"],
        ema_momentum=p1_cfg["ema_momentum"],
    ).to(device)
    phase1.load_state_dict(ckpt["model_state_dict"])
    phase1.eval()
    phase1.requires_grad_(False)

    h = p1_cfg["horizon_h"]
    num_steps = h + 1  # need at least h+1 frames for proprio_dec on predictor outputs

    # ---- Dataset ----
    dataset = build_dataset(
        cache_dir=args.cache_dir,
        img_size=args.img_size,
        num_steps=num_steps,
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

    # ---- Read action_dim and proprio_dim from data ----
    sample = next(iter(train_loader))
    action_dim = sample["action"].shape[-1]
    proprio_dim = sample["proprio"].shape[-1]
    latent_action_dim = p1_cfg["latent_action_dim"]
    latent_dim = p1_cfg["latent_dim"]

    # ---- Phase 2 networks ----
    action_enc = ActionEncoder(action_dim, latent_action_dim, args.hidden_dim).to(device)
    action_dec = ActionDecoder(latent_action_dim, action_dim, args.hidden_dim).to(device)
    proprio_dec = ProprioDecoder(latent_dim, proprio_dim, args.proprio_hidden_dim).to(device)

    params = (
        list(action_enc.parameters())
        + list(action_dec.parameters())
        + list(proprio_dec.parameters())
    )
    optimizer = torch.optim.Adam(params, lr=args.lr)

    print("=" * 72)
    print("Phase 2: Action Alignment + Proprio Decoder")
    print("=" * 72)
    print(f"device             : {device}")
    print(f"phase1 checkpoint  : {args.phase1_ckpt}")
    print(f"label fraction     : {args.label_fraction:.1%}")
    print(f"dataset size       : {len(dataset)}")
    print(f"num_steps (h+1)    : {num_steps}")
    print(f"action_dim         : {action_dim}")
    print(f"proprio_dim        : {proprio_dim}")
    print(f"latent_action_dim  : {latent_action_dim}")
    print(f"params: act_enc={sum(p.numel() for p in action_enc.parameters()):,} "
          f"act_dec={sum(p.numel() for p in action_dec.parameters()):,} "
          f"proprio_dec={sum(p.numel() for p in proprio_dec.parameters()):,}")
    print("=" * 72)

    # ---- Training ----
    output_dir = Path("outputs") / "hierarchical_invdyn"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(
            phase1, action_enc, action_dec, proprio_dec,
            train_loader, device, optimizer, args.lambda_proprio,
        )
        val_m = run_epoch(
            phase1, action_enc, action_dec, proprio_dec,
            val_loader, device, None, args.lambda_proprio,
        )

        print(
            f"[{epoch:03d}] "
            f"train: align={train_m['align']:.4f} dec={train_m['decode']:.4f} "
            f"p_enc={train_m['proprio_enc']:.4f} p_pred={train_m['proprio_pred']:.4f} | "
            f"val: align={val_m['align']:.4f} p_pred={val_m['proprio_pred']:.4f}"
        )

        if val_m["total_loss"] < best_val:
            best_val = val_m["total_loss"]
            torch.save(
                {
                    "action_encoder_state_dict": action_enc.state_dict(),
                    "action_decoder_state_dict": action_dec.state_dict(),
                    "proprio_dec_state_dict": proprio_dec.state_dict(),
                    "config": vars(args),
                    "phase1_config": p1_cfg,
                    "action_dim": action_dim,
                    "proprio_dim": proprio_dim,
                    "best_val_loss": best_val,
                    "epoch": epoch,
                },
                output_dir / "best_phase2.pt",
            )

    print(f"\nBest val: {best_val:.4f} → {output_dir / 'best_phase2.pt'}")


if __name__ == "__main__":
    main()

"""
Phase 2: Action space alignment + Proprio decoder training.

Freeze all Phase 1 parameters. Train three lightweight networks:
  - ActionEncoder:  a_t (real action) → ã_t (latent action space)
  - ActionDecoder:  â_t (latent action) → a_t (real action)
  - ProprioDecoder: z (latent state)   → proprio  (used by planner cost)

Critical design (P1 fix): proprio_dec MUST be supervised on the EXACT
predictor-output distribution that planning sees. Planning runs
K-segment chained rollout where each segment's coarse and fine both
start from the PREVIOUS segment's last fine output. So Phase 2 mirrors
that rollout and supervises proprio_dec on every intermediate state.

Two rollouts are run, both supervising proprio_dec:
  (a) IDM-action rollout — uses inv_dyn(z_t, z_{t+1}) as latent actions
      (matches the distribution Phase 1 trained the predictors on)
  (b) Real-action rollout — uses action_encoder(real_action) as latent
      actions (matches the distribution planning actually uses)

Sequences of length >= K*h+1 are loaded so the chained rollout can run
end to end and proprio at every timestep is available for supervision.

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
    """Phase 2 needs sequences of length >= K*h+1 for chained-rollout proprio supervision."""
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
# Chained rollout matching planning
# ============================================================


def chained_rollout_proprio_loss(
    phase1: HierarchicalInvDynWorldModel,
    proprio_dec: ProprioDecoder,
    z_start: torch.Tensor,                     # (B, num_tokens, D)
    latent_actions_per_step: list,             # length K*h, each (B, latent_action_dim)
    proprio_seq: torch.Tensor,                 # (B, K*h+1, proprio_dim)
    K: int,
    h: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run K-segment chained rollout (mirrors planning.hierarchical_rollout)
    and supervise proprio_dec on:
      - every fine-predictor output (target = proprio at the corresponding next-frame timestep)
      - every coarse waypoint        (target = proprio at the segment-end timestep)

    phase1 is frozen (no param updates), but gradients still flow through it
    to update action_encoder and proprio_dec.
    """
    L_p_fine = 0.0
    L_p_wp = 0.0
    fine_count = 0

    z_current = z_start
    for k in range(K):
        seg_actions = torch.stack(
            latent_actions_per_step[k * h : (k + 1) * h], dim=1
        )  # (B, h, latent_action_dim)

        z_waypoint = phase1.coarse_predictor(z_current, seg_actions)
        L_p_wp = L_p_wp + F.mse_loss(
            proprio_dec(z_waypoint), proprio_seq[:, (k + 1) * h]
        )

        z_t = z_current
        for t in range(h):
            global_t = k * h + t
            z_t = phase1.fine_predictor(
                z_t, latent_actions_per_step[global_t], z_waypoint
            )
            # fine_predictor(z_t, action_t) predicts the next state, so proprio
            # target is at index global_t + 1
            L_p_fine = L_p_fine + F.mse_loss(
                proprio_dec(z_t), proprio_seq[:, global_t + 1]
            )
            fine_count += 1

        # Next segment starts from this segment's fine rollout output
        z_current = z_t

    return L_p_fine / max(1, fine_count), L_p_wp / K


# ============================================================
# Loss Computation
# ============================================================


def compute_phase2_loss(
    phase1: HierarchicalInvDynWorldModel,
    action_enc: ActionEncoder,
    action_dec: ActionDecoder,
    proprio_dec: ProprioDecoder,
    batch: dict[str, torch.Tensor],
    K: int,
    lambda_proprio: float = 1.0,
    lambda_real_path: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Action alignment + chained-rollout proprio supervision.

    pixels  (B, T, C, H, W)         T >= K*h + 1
    action  (B, T, action_dim)
    proprio (B, T, proprio_dim)
    """
    pixels = batch["pixels"].float()
    actions = batch["action"].float()
    proprio = batch["proprio"].float()
    h = phase1.horizon_h
    total = K * h + 1

    # ---- Frozen Phase 1: encode all frames + IDM latent actions ----
    with torch.no_grad():
        all_z = [phase1.encode(pixels[:, t]) for t in range(total)]
        idm_actions = [
            phase1.compute_latent_action(all_z[t], all_z[t + 1])
            for t in range(K * h)
        ]

    # ---- Action alignment (single-step, t=0) ----
    a_t = actions[:, 0]
    a_hat = idm_actions[0]
    a_tilde = action_enc(a_t)
    a_recon = action_dec(a_hat)
    a_cycle = action_dec(a_tilde)

    L_align = F.mse_loss(a_tilde, a_hat)
    L_decode = F.mse_loss(a_recon, a_t)
    L_cycle = F.mse_loss(a_cycle, a_t)

    # ---- Proprio: encoder outputs (cheap, broad coverage) ----
    L_p_enc = sum(
        F.mse_loss(proprio_dec(all_z[t]), proprio[:, t]) for t in range(total)
    ) / total

    # ---- Proprio: chained rollout, IDM-action path ----
    # Matches Phase 1 training distribution; phase1 already in no_grad above
    # but proprio_dec must receive grad, so re-run rollout with grad enabled.
    L_p_pred_idm, L_p_wp_idm = chained_rollout_proprio_loss(
        phase1, proprio_dec, all_z[0], idm_actions, proprio, K, h,
    )

    # ---- Proprio: chained rollout, real-action path (planning distribution) ----
    # action_encoder gradient flows here, so it learns to produce latent actions
    # whose downstream rollout proprio matches reality.
    real_actions_seq = actions[:, : K * h]                                # (B, K*h, A)
    B, T_a, A = real_actions_seq.shape
    real_la_flat = action_enc(real_actions_seq.reshape(B * T_a, A))
    real_la = real_la_flat.reshape(B, T_a, -1)
    real_la_per_step = [real_la[:, t] for t in range(T_a)]
    L_p_pred_real, L_p_wp_real = chained_rollout_proprio_loss(
        phase1, proprio_dec, all_z[0], real_la_per_step, proprio, K, h,
    )

    L_proprio = (
        L_p_enc
        + 0.5 * (L_p_pred_idm + L_p_wp_idm)
        + lambda_real_path * 0.5 * (L_p_pred_real + L_p_wp_real)
    )

    L_action = L_align + L_decode + 0.5 * L_cycle
    L_total = L_action + lambda_proprio * L_proprio

    metrics = {
        "align": L_align.item(),
        "decode": L_decode.item(),
        "cycle": L_cycle.item(),
        "proprio_enc": L_p_enc.item(),
        "proprio_idm_fine": L_p_pred_idm.item(),
        "proprio_idm_wp": L_p_wp_idm.item(),
        "proprio_real_fine": L_p_pred_real.item(),
        "proprio_real_wp": L_p_wp_real.item(),
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
    K: int,
    lambda_proprio: float = 1.0,
    lambda_real_path: float = 1.0,
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
                batch, K, lambda_proprio, lambda_real_path,
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
        description="Phase 2: Action alignment + ProprioDecoder (chained-rollout supervision)"
    )
    parser.add_argument("--phase1-ckpt", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--label-fraction", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Hidden dim for action encoder/decoder")
    parser.add_argument("--proprio-hidden-dim", type=int, default=128,
                        help="Hidden dim for proprio decoder")
    parser.add_argument("--coarse-segments", type=int, default=None,
                        help="K. Defaults to Phase 1 config's coarse_segments.")
    parser.add_argument("--lambda-proprio", type=float, default=1.0)
    parser.add_argument("--lambda-real-path", type=float, default=1.0,
                        help="Weight on the real-action rollout proprio loss "
                             "(0 = only IDM rollout supervision)")
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
    ckpt_path = Path(args.phase1_ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Phase 1 checkpoint not found: {ckpt_path}\n"
            f"Run train_phase1.py first to produce it."
        )
    ckpt = torch.load(ckpt_path, map_location="cpu")
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
    K = args.coarse_segments if args.coarse_segments is not None else p1_cfg.get("coarse_segments", 2)
    num_steps = K * h + 1

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

    sample = next(iter(train_loader))
    action_dim = sample["action"].shape[-1]
    proprio_dim = sample["proprio"].shape[-1]
    latent_action_dim = p1_cfg["latent_action_dim"]
    latent_dim = p1_cfg["latent_dim"]

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
    print("Phase 2: Action Alignment + Proprio Decoder (chained rollout)")
    print("=" * 72)
    print(f"device             : {device}")
    print(f"phase1 checkpoint  : {ckpt_path}")
    print(f"label fraction     : {args.label_fraction:.1%}")
    print(f"dataset size       : {len(dataset)}")
    print(f"horizon_h          : {h}")
    print(f"coarse segments K  : {K}")
    print(f"num_steps (K*h+1)  : {num_steps}")
    print(f"action_dim         : {action_dim}")
    print(f"proprio_dim        : {proprio_dim}")
    print(f"latent_action_dim  : {latent_action_dim}")
    print(f"params: act_enc={sum(p.numel() for p in action_enc.parameters()):,} "
          f"act_dec={sum(p.numel() for p in action_dec.parameters()):,} "
          f"proprio_dec={sum(p.numel() for p in proprio_dec.parameters()):,}")
    print("=" * 72)

    output_dir = Path("outputs") / "hierarchical_invdyn"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(
            phase1, action_enc, action_dec, proprio_dec,
            train_loader, device, optimizer, K,
            args.lambda_proprio, args.lambda_real_path,
        )
        val_m = run_epoch(
            phase1, action_enc, action_dec, proprio_dec,
            val_loader, device, None, K,
            args.lambda_proprio, args.lambda_real_path,
        )

        print(
            f"[{epoch:03d}] "
            f"align={train_m['align']:.4f} dec={train_m['decode']:.4f} | "
            f"p_enc={train_m['proprio_enc']:.4f} "
            f"idm_f={train_m['proprio_idm_fine']:.4f} "
            f"real_f={train_m['proprio_real_fine']:.4f} | "
            f"val: idm_f={val_m['proprio_idm_fine']:.4f} "
            f"real_f={val_m['proprio_real_fine']:.4f}"
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
                    "coarse_segments": K,
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

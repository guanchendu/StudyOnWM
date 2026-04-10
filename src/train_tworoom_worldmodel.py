import argparse
import os
from pathlib import Path
# 

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn as nn
import torch.nn.functional as F

from Background.utils import get_column_normalizer, get_img_preprocessor

DEFAULT_CACHE_DIR = "/Users/guanchendu/Code/StudyOnWM/data"


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
            source="pixels",
            target="pixels",
            img_size=img_size,
        )
    ]

    for col in ("action", "proprio"):
        transforms.append(get_column_normalizer(dataset, col, col))

    dataset.transform = spt.data.transforms.Compose(*transforms)
    return dataset


class ConvEncoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim: int, img_size: int):
        super().__init__()
        self.img_size = img_size
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, 128 * 8 * 8),
            nn.ReLU(inplace=True),
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.proj(z).view(z.shape[0], 128, 8, 8)
        x = self.net(x)
        x = F.interpolate(
            x,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        return x


class SimpleWorldModel(nn.Module):
    def __init__(self, action_dim: int, proprio_dim: int, img_size: int, latent_dim: int = 256):
        super().__init__()
        self.encoder = ConvEncoder(latent_dim=latent_dim)
        self.transition = nn.Sequential(
            nn.Linear(latent_dim + action_dim + proprio_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
        )
        self.next_image = ConvDecoder(latent_dim=latent_dim, img_size=img_size)
        self.next_proprio = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, proprio_dim),
        )

    def forward(
        self,
        pixels_t: torch.Tensor,
        action_t: torch.Tensor,
        proprio_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_t = self.encoder(pixels_t)
        z_next = self.transition(torch.cat([z_t, action_t, proprio_t], dim=-1))
        pred_pixels = self.next_image(z_next)
        pred_proprio = self.next_proprio(z_next)
        return pred_pixels, pred_proprio


def flatten_sequence_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    pixels = batch["pixels"].float()
    action = batch["action"].float()
    proprio = batch["proprio"].float()

    return {
        "pixels_t": pixels[:, :-1].reshape(-1, *pixels.shape[2:]),
        "action_t": action[:, :-1].reshape(-1, action.shape[-1]),
        "proprio_t": proprio[:, :-1].reshape(-1, proprio.shape[-1]),
        "pixels_tp1": pixels[:, 1:].reshape(-1, *pixels.shape[2:]),
        "proprio_tp1": proprio[:, 1:].reshape(-1, proprio.shape[-1]),
    }


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    pixel_loss_weight: float,
    proprio_loss_weight: float,
    epoch_idx: int,
    total_epochs: int,
    batch_index: int | None = None,
):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_pixel_loss = 0.0
    total_proprio_loss = 0.0
    total_items = 0

    for idx, raw_batch in enumerate(loader):
        # Skip to target batch if batch_index is specified
        if batch_index is not None and idx != batch_index:
            continue

        batch = flatten_sequence_batch(raw_batch)
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.set_grad_enabled(is_train):
            pred_pixels, pred_proprio = model(
                batch["pixels_t"],
                batch["action_t"],
                batch["proprio_t"],
            )

            pixel_loss = F.smooth_l1_loss(pred_pixels, batch["pixels_tp1"])
            proprio_loss = F.mse_loss(pred_proprio, batch["proprio_tp1"])
            loss = pixel_loss_weight * pixel_loss + proprio_loss_weight * proprio_loss

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        bs = batch["pixels_t"].shape[0]
        total_items += bs
        total_loss += loss.item() * bs
        total_pixel_loss += pixel_loss.item() * bs
        total_proprio_loss += proprio_loss.item() * bs

        # Break after processing target batch if batch_index is specified
        if batch_index is not None:
            break

    return {
        "loss": total_loss / total_items,
        "pixel_loss": total_pixel_loss / total_items,
        "proprio_loss": total_proprio_loss / total_items,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train a simple world model on the tworoom dataset.")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help="Directory containing tworoom.h5",
    )
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pixel-loss-weight", type=float, default=1.0)
    parser.add_argument("--proprio-loss-weight", type=float, default=0.1)
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

    sample = next(iter(train_loader))
    action_dim = sample["action"].shape[-1]
    proprio_dim = sample["proprio"].shape[-1]

    model = SimpleWorldModel(
        action_dim=action_dim,
        proprio_dim=proprio_dim,
        img_size=args.img_size,
        latent_dim=args.latent_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("=" * 72)
    print("Simple TwoRoom World Model")
    print("=" * 72)
    print(f"device        : {device}")
    print(f"dataset size  : {len(dataset)}")
    print(f"train / val   : {len(train_set)} / {len(val_set)}")
    print(f"pixels shape  : {tuple(sample['pixels'].shape)}")
    print(f"action dim    : {action_dim}")
    print(f"proprio dim   : {proprio_dim}")
    print(f"num parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 72)

    best_val = float("inf")
    output_dir = Path("outputs") / "tworoom_worldmodel"
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            pixel_loss_weight=args.pixel_loss_weight,
            proprio_loss_weight=args.proprio_loss_weight,
            epoch_idx=epoch,
            total_epochs=args.epochs,
            batch_index=None,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            optimizer=None,
            pixel_loss_weight=args.pixel_loss_weight,
            proprio_loss_weight=args.proprio_loss_weight,
            epoch_idx=epoch,
            total_epochs=args.epochs,
            batch_index=None,
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_pixel={train_metrics['pixel_loss']:.4f} "
            f"train_prop={train_metrics['proprio_loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_pixel={val_metrics['pixel_loss']:.4f} "
            f"val_prop={val_metrics['proprio_loss']:.4f}"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_path = output_dir / "best_simple_world_model.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": vars(args),
                    "action_dim": action_dim,
                    "proprio_dim": proprio_dim,
                    "best_val_loss": best_val,
                },
                save_path,
            )

    print(f"best checkpoint saved to: {output_dir / 'best_simple_world_model.pt'}")


if __name__ == "__main__":
    main()

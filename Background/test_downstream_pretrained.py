import argparse
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class SmallEncoder(nn.Module):
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.proj = nn.Linear(128, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.features(x))


class SmallAutoEncoder(nn.Module):
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.encoder = SmallEncoder(feature_dim=feature_dim)
        self.decoder_input = nn.Sequential(
            nn.Linear(feature_dim, 128 * 4 * 4),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        h = self.decoder_input(z).view(x.shape[0], 128, 4, 4)
        recon = self.decoder(h)
        return recon, z


class LinearClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, feature_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        return self.head(feats)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_dataloaders(batch_size: int, num_workers: int):
    transform = T.ToTensor()

    train_set = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    test_set = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, test_loader


def pretrain_autoencoder(
    model: SmallAutoEncoder,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_items = 0
        progress = tqdm(loader, desc=f"Pretrain {epoch:02d}/{epochs:02d}", leave=False)

        for images, _ in progress:
            images = images.to(device)
            recon, _ = model(images)
            loss = F.mse_loss(recon, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = images.shape[0]
            total_loss += loss.item() * bs
            total_items += bs
            progress.set_postfix(recon_loss=f"{loss.item():.4f}")

        print(f"[Pretrain][Epoch {epoch:02d}] recon_loss={total_loss / total_items:.4f}")


def train_downstream(
    model: LinearClassifier,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
):
    optimizer = torch.optim.Adam(model.head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_items = 0
        train_progress = tqdm(
            train_loader,
            desc=f"Downstream train {epoch:02d}/{epochs:02d}",
            leave=False,
        )

        for images, labels in train_progress:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = logits.argmax(dim=1)
            bs = labels.shape[0]
            total_loss += loss.item() * bs
            total_correct += (pred == labels).sum().item()
            total_items += bs

            train_progress.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{(pred == labels).float().mean().item():.3f}",
            )

        train_loss = total_loss / total_items
        train_acc = total_correct / total_items
        test_acc = evaluate(model, test_loader, device)
        print(
            f"[Downstream][Epoch {epoch:02d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}"
        )


@torch.no_grad()
def evaluate(model: LinearClassifier, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_correct = 0
    total_items = 0

    progress = tqdm(loader, desc="Evaluate", leave=False)
    for images, labels in progress:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        pred = logits.argmax(dim=1)
        total_correct += (pred == labels).sum().item()
        total_items += labels.shape[0]

    return total_correct / total_items


def parse_args():
    parser = argparse.ArgumentParser(
        description="A small example of using a pretrained model for a downstream task."
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--pretrain-epochs", type=int, default=3)
    parser.add_argument("--downstream-epochs", type=int, default=5)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--downstream-lr", type=float, default=1e-3)
    parser.add_argument(
        "--pretrained-path",
        type=str,
        default="outputs/downstream_demo/pretrained_encoder.pt",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    train_loader, test_loader = build_dataloaders(args.batch_size, args.num_workers)

    save_path = Path(args.pretrained_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    autoencoder = SmallAutoEncoder(feature_dim=args.feature_dim).to(device)

    if save_path.exists():
        autoencoder.encoder.load_state_dict(torch.load(save_path, map_location=device))
        print(f"Loaded pretrained encoder from {save_path}")
    else:
        print("Stage 1: pretraining a small encoder with image reconstruction")
        pretrain_autoencoder(
            model=autoencoder,
            loader=train_loader,
            device=device,
            epochs=args.pretrain_epochs,
            lr=args.pretrain_lr,
        )
        torch.save(autoencoder.encoder.state_dict(), save_path)
        print(f"Saved pretrained encoder to {save_path}")

    print("Stage 2: downstream task = CIFAR-10 classification with a frozen encoder")
    for param in autoencoder.encoder.parameters():
        param.requires_grad = False

    classifier = LinearClassifier(
        encoder=autoencoder.encoder,
        feature_dim=args.feature_dim,
        num_classes=10,
    ).to(device)

    train_downstream(
        model=classifier,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=args.downstream_epochs,
        lr=args.downstream_lr,
    )


if __name__ == "__main__":
    main()

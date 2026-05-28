"""DINOv2 测试 / 评估:

1. linear probe: 冻结 backbone, 训练一个线性分类头, 报 top-1 acc
2. kNN classifier: 用 backbone 抽特征, 在 train set 上做 kNN, 报 top-1 acc

用法:
    python eval.py --ckpt checkpoints/dinov2_vits14_ep99.pt --data /path/to/imagenet --mode linear
    python eval.py --ckpt checkpoints/dinov2_vits14_ep99.pt --data /path/to/imagenet --mode knn
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from model import DINOv2Wrapper, vit_small_patch14


def build_loaders(root: str, batch_size: int = 128):
    norm = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    train_t = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        norm,
    ])
    val_t = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        norm,
    ])
    train_set = ImageFolder(Path(root) / "train", transform=train_t)
    val_set = ImageFolder(Path(root) / "val", transform=val_t)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, len(train_set.classes)


def load_backbone(ckpt_path: str, device: torch.device):
    backbone = vit_small_patch14(img_size=224)
    wrapper = DINOv2Wrapper(backbone)
    state = torch.load(ckpt_path, map_location="cpu")
    wrapper.load_state_dict(state["teacher"])  # 评估用 teacher 权重 (DINOv2 惯例)
    wrapper.to(device).eval()
    return wrapper.backbone


@torch.no_grad()
def extract_features(backbone, loader, device):
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        out = backbone(x)
        feats.append(F.normalize(out["cls_token"], dim=-1).cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


@torch.no_grad()
def knn_eval(backbone, train_loader, val_loader, device, k: int = 20, T: float = 0.07):
    train_f, train_y = extract_features(backbone, train_loader, device)
    val_f, val_y = extract_features(backbone, val_loader, device)
    train_f, train_y = train_f.to(device), train_y.to(device)
    val_f, val_y = val_f.to(device), val_y.to(device)
    n_classes = int(train_y.max().item() + 1)

    correct = 0
    chunk = 256
    for i in range(0, val_f.size(0), chunk):
        q = val_f[i:i + chunk]
        sim = q @ train_f.t()                                    # (chunk, N_train)
        topk_sim, topk_idx = sim.topk(k, dim=1)
        topk_labels = train_y[topk_idx]                          # (chunk, k)
        weights = (topk_sim / T).exp()
        votes = torch.zeros(q.size(0), n_classes, device=device)
        votes.scatter_add_(1, topk_labels, weights)
        pred = votes.argmax(dim=1)
        correct += (pred == val_y[i:i + chunk]).sum().item()
    acc = correct / val_f.size(0)
    print(f"[kNN k={k}] top-1 acc = {acc * 100:.2f}%")
    return acc


def linear_eval(backbone, train_loader, val_loader, num_classes: int, device, epochs: int = 20, lr: float = 0.01):
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()

    classifier = nn.Linear(backbone.embed_dim, num_classes).to(device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for ep in range(epochs):
        classifier.train()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.no_grad():
                feats = backbone(x)["cls_token"]
            logits = classifier(feats)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # eval
        classifier.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                feats = backbone(x)["cls_token"]
                pred = classifier(feats).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        print(f"[linear] ep{ep} top-1 acc = {100 * correct / total:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--mode", choices=["linear", "knn"], default="knn")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    train_loader, val_loader, n_classes = build_loaders(args.data, args.batch_size)
    backbone = load_backbone(args.ckpt, device)

    if args.mode == "knn":
        knn_eval(backbone, train_loader, val_loader, device)
    else:
        linear_eval(backbone, train_loader, val_loader, n_classes, device)


if __name__ == "__main__":
    main()

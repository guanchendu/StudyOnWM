"""DINOv2 训练主循环（教学版，可直接 python train.py 跑起来）。

要点:
- student / teacher 都是 DINOv2Wrapper，teacher 用 student 的 EMA 更新，不接收梯度
- student 看全部视图 (2 global + n_local)，teacher 只看 2 global
- iBOT loss 只在 global 视图上算
- 总 loss = DINO + iBOT + 0.1 * KoLeo
"""
from __future__ import annotations

import copy
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from augmentations import BlockMaskGenerator, MultiCropAugmentation
from losses import DINOLoss, KoLeoLoss, iBOTLoss
from model import DINOv2Wrapper, vit_small_patch14


def cosine_schedule(base: float, final: float, step: int, total: int) -> float:
    if step >= total:
        return final
    return final + 0.5 * (base - final) * (1 + math.cos(math.pi * step / total))


def collate(batch):
    crops = [b[0] for b in batch]
    globals_ = torch.stack([c["globals"] for c in crops])         # (B, 2, 3, 224, 224)
    locals_ = torch.stack([c["locals"] for c in crops])           # (B, n_local, 3, 96, 96)
    return globals_, locals_


@torch.no_grad()
def ema_update(student: nn.Module, teacher: nn.Module, m: float) -> None:
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(m).add_(ps.data, alpha=1 - m)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="ImageFolder root")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_local", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.04)
    parser.add_argument("--ema_base", type=float, default=0.992)
    parser.add_argument("--ema_final", type=float, default=1.0)
    parser.add_argument("--out_dim", type=int, default=65536)
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    aug = MultiCropAugmentation(global_size=224, local_size=96, n_local=args.n_local)
    dataset = ImageFolder(args.data, transform=aug)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                        pin_memory=True, drop_last=True, collate_fn=collate)

    backbone = vit_small_patch14(img_size=224)
    student = DINOv2Wrapper(backbone, dino_out_dim=args.out_dim, ibot_out_dim=args.out_dim).to(device)
    teacher = copy.deepcopy(student).to(device)
    for p in teacher.parameters():
        p.requires_grad = False

    dino_loss = DINOLoss(args.out_dim).to(device)
    ibot_loss = iBOTLoss(args.out_dim).to(device)
    koleo_loss = KoLeoLoss().to(device)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mask_gen = BlockMaskGenerator(num_patches_side=224 // 14, mask_ratio=0.4)

    total_steps = args.epochs * len(loader)
    step = 0

    for epoch in range(args.epochs):
        for globals_, locals_ in loader:
            B = globals_.size(0)
            globals_ = globals_.to(device, non_blocking=True)        # (B, 2, 3, 224, 224)
            locals_ = locals_.to(device, non_blocking=True)

            # --- 生成 mask (每张 global crop 一个) ---
            masks = torch.stack([mask_gen() for _ in range(B * 2)]).to(device)  # (2B, N_patch)

            # --- Teacher forward (无 mask, 仅 globals) ---
            with torch.no_grad():
                t_global = teacher(globals_.flatten(0, 1))           # (2B, ...)
                t_cls_logits = t_global["cls_logits"]
                t_patch_logits = t_global["patch_logits"]
                # 拆成两组，按 view 分
                t_cls_per_view = list(t_cls_logits.view(2, B, -1))   # 长度 2

            # --- Student forward ---
            # globals 走有 mask 的分支（给 iBOT loss）
            s_global = student(globals_.flatten(0, 1), masks=masks)
            s_cls_g = s_global["cls_logits"]                          # (2B, D)
            s_patch_g = s_global["patch_logits"]                      # (2B, N, D)

            # locals 不需要 mask
            s_local = student(locals_.flatten(0, 1))
            s_cls_l = s_local["cls_logits"]                           # (n_local*B, D)

            s_cls_per_view = list(s_cls_g.view(2, B, -1)) + list(s_cls_l.view(args.n_local, B, -1))

            # --- Losses ---
            l_dino = dino_loss(s_cls_per_view, t_cls_per_view)
            l_ibot = ibot_loss(s_patch_g, t_patch_logits, masks)
            l_koleo = sum(koleo_loss(s_global["cls_token"][i * B:(i + 1) * B]) for i in range(2)) / 2.0
            loss = l_dino + l_ibot + 0.1 * l_koleo

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 3.0)
            optimizer.step()

            # --- EMA teacher 更新 (动量 cosine 从 base -> 1.0) ---
            m = cosine_schedule(args.ema_base, args.ema_final, step, total_steps)
            ema_update(student, teacher, m)

            if step % 20 == 0:
                print(f"ep{epoch} step{step} loss={loss.item():.3f} "
                      f"dino={l_dino.item():.3f} ibot={l_ibot.item():.3f} koleo={l_koleo.item():.3f} m={m:.4f}")
            step += 1

        torch.save({
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "epoch": epoch,
        }, Path(args.ckpt_dir) / f"dinov2_vits14_ep{epoch}.pt")


if __name__ == "__main__":
    main()

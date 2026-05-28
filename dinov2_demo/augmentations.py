"""DINOv2 多视图增强 + iBOT 的 block-wise mask 生成。"""
from __future__ import annotations

import math
import random
import torch
from PIL import Image
from torchvision import transforms


class MultiCropAugmentation:
    """每张图产出 2 张 global crop (224) + n_local 张 local crop (96)。"""

    def __init__(self, global_size: int = 224, local_size: int = 96, n_local: int = 8):
        self.n_local = n_local
        flip_color = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        self.global_t = transforms.Compose([
            transforms.RandomResizedCrop(global_size, scale=(0.32, 1.0), interpolation=Image.BICUBIC),
            flip_color,
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            normalize,
        ])
        self.local_t = transforms.Compose([
            transforms.RandomResizedCrop(local_size, scale=(0.05, 0.32), interpolation=Image.BICUBIC),
            flip_color,
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))], p=0.5),
            normalize,
        ])

    def __call__(self, img: Image.Image) -> dict:
        globals_ = [self.global_t(img) for _ in range(2)]
        locals_ = [self.local_t(img) for _ in range(self.n_local)]
        return {"globals": torch.stack(globals_), "locals": torch.stack(locals_)}


class BlockMaskGenerator:
    """对每张 global crop 生成 block-wise mask（iBOT 的做法）。"""

    def __init__(self, num_patches_side: int = 16, mask_ratio: float = 0.4, min_block: int = 4, max_block: int = 64):
        self.h = self.w = num_patches_side
        self.n = num_patches_side ** 2
        self.target = int(self.n * mask_ratio)
        self.min_block, self.max_block = min_block, max_block

    def __call__(self) -> torch.Tensor:
        mask = torch.zeros(self.h, self.w, dtype=torch.bool)
        covered = 0
        for _ in range(20):
            if covered >= self.target:
                break
            size = random.randint(self.min_block, self.max_block)
            aspect = math.exp(random.uniform(math.log(0.3), math.log(3.3)))
            bh = int(round(math.sqrt(size * aspect)))
            bw = int(round(math.sqrt(size / aspect)))
            if bh < 1 or bw < 1 or bh > self.h or bw > self.w:
                continue
            top = random.randint(0, self.h - bh)
            left = random.randint(0, self.w - bw)
            new = (~mask[top:top + bh, left:left + bw]).sum().item()
            if new == 0:
                continue
            mask[top:top + bh, left:left + bw] = True
            covered += new
        return mask.flatten()

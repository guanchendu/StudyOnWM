"""ViT-S/14 backbone + DINO projection head.

DINOv2 在 ViT 上加了 4 个 register tokens（来自 "Vision Transformers Need Registers"），
用来吸纳全局信息，降低 attention map 中的伪影。这里实现了带 register 的简化版。
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 14, in_chans: int = 3, embed_dim: int = 384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                 # (B, C, H/P, W/P)
        return x.flatten(2).transpose(1, 2)  # (B, N, C)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 6, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """ViT-S/14 with CLS + register tokens. DINOv2 default: 4 registers."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 14,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        num_register_tokens: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        N = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
        # 位置编码只覆盖 CLS + patches；register 不加位置编码（DINOv2 的做法）
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + N, embed_dim))
        self.num_register_tokens = num_register_tokens

        self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.register_tokens, std=0.02)

    def interpolate_pos_encoding(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """支持多裁剪不同分辨率：对位置编码做双三次插值。"""
        N_orig = self.pos_embed.shape[1] - 1
        N_new = x.shape[1] - 1
        if N_new == N_orig:
            return self.pos_embed
        cls_pe = self.pos_embed[:, :1]
        patch_pe = self.pos_embed[:, 1:]
        dim = patch_pe.shape[-1]
        side = int(math.sqrt(N_orig))
        new_h = h // self.patch_embed.patch_size
        new_w = w // self.patch_embed.patch_size
        patch_pe = patch_pe.reshape(1, side, side, dim).permute(0, 3, 1, 2)
        patch_pe = F.interpolate(patch_pe, size=(new_h, new_w), mode="bicubic", align_corners=False)
        patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, new_h * new_w, dim)
        return torch.cat([cls_pe, patch_pe], dim=1)

    def prepare_tokens(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        x = self.patch_embed(x)                                    # (B, N, C)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.interpolate_pos_encoding(x, H, W)
        regs = self.register_tokens.expand(B, -1, -1)
        # 顺序: [CLS, REGS..., PATCHES...]
        return torch.cat([x[:, :1], regs, x[:, 1:]], dim=1)

    def forward(self, x: torch.Tensor, masks: torch.Tensor | None = None) -> dict:
        """
        Args:
            x: (B, 3, H, W)
            masks: (B, N) bool, True 表示该 patch 被 mask（iBOT 用）
        Returns:
            dict with cls_token (B, C) and patch_tokens (B, N, C)
        """
        tokens = self.prepare_tokens(x)
        if masks is not None:
            # 把被 mask 的 patch token 替换成 cls_token 的副本（简化版的 mask token）
            n_special = 1 + self.num_register_tokens
            patch = tokens[:, n_special:]
            mask_token = self.cls_token.expand_as(patch)
            patch = torch.where(masks.unsqueeze(-1), mask_token, patch)
            tokens = torch.cat([tokens[:, :n_special], patch], dim=1)

        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        n_special = 1 + self.num_register_tokens
        return {
            "cls_token": tokens[:, 0],
            "patch_tokens": tokens[:, n_special:],
        }


class DINOHead(nn.Module):
    """3-layer MLP + weight-normalized 最后一层，输出 prototype logits。"""

    def __init__(self, in_dim: int, out_dim: int = 65536, hidden_dim: int = 2048, bottleneck_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        last = nn.Linear(bottleneck_dim, out_dim, bias=False)
        self.last_layer = nn.utils.weight_norm(last)
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False  # 论文里冻结 g

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


class DINOv2Wrapper(nn.Module):
    """Backbone + 两个独立 head：一个给 CLS（DINO loss），一个给 patch（iBOT loss）。"""

    def __init__(self, backbone: VisionTransformer, dino_out_dim: int = 65536, ibot_out_dim: int = 65536):
        super().__init__()
        self.backbone = backbone
        self.dino_head = DINOHead(backbone.embed_dim, dino_out_dim)
        self.ibot_head = DINOHead(backbone.embed_dim, ibot_out_dim)

    def forward(self, x: torch.Tensor, masks: torch.Tensor | None = None) -> dict:
        out = self.backbone(x, masks=masks)
        return {
            "cls_logits": self.dino_head(out["cls_token"]),
            "patch_logits": self.ibot_head(out["patch_tokens"]),
            "cls_token": out["cls_token"],
        }


def vit_small_patch14(**kwargs) -> VisionTransformer:
    return VisionTransformer(patch_size=14, embed_dim=384, depth=12, num_heads=6, **kwargs)

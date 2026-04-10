import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=192):
        super().__init__()
        assert img_size % patch_size == 0

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # 用 Conv2d 模拟“切 patch + 线性映射”
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W)                # 224x224 -> 14x14 patches -> 196 tokens
        x = self.proj(x)                 # (B, D, H/P, W/P)  #(224 - 16) / 16 + 1 = 14
        x = x.flatten(2)                # (B, D, N)
        x = x.transpose(1, 2)           # (B, N, D)
        return x


class ViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=192,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        num_classes=10,
        dropout=0.1,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # 1. patch embedding
        x = self.patch_embed(x)  # (B, N, D)

        # 2. 拼 cls token
        B = x.size(0)
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls_token, x], dim=1)          # (B, N+1, D)

        # 3. 加位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 4. Transformer 编码
        x = self.encoder(x)

        # 5. 取 cls token
        x = self.norm(x[:, 0])   # (B, D)  #第二维里的第 0 个 token 单独取出来了

        # 6. 分类头
        x = self.head(x)         # (B, num_classes)
        return x


if __name__ == "__main__":
    model = ViT()
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(out.shape)  # (2, 10)

"""
V-JEPA 核心实现（单样本训练演示）
====================================
复现自: V-JEPA: Latent Video Prediction for Visual Representation Learning

本代码聚焦论文核心:
  1. 3D Patch Embedding (3D 卷积分块)
  2. Context Encoder (ViT, 只看可见 patch)
  3. Target Encoder (ViT, 看完整视频, EMA 更新)
  4. Predictor (较小的 Transformer, 预测被遮位置)
  5. 3D Multi-Block Masking (短程 + 长程)
  6. L1 loss + stop-gradient + EMA (防塌缩机制)

输入: 随机 Tensor [1, 3, 64, 224, 224] (B, C, T, H, W)
     其中 64 帧会按每 4 帧取 1 被下采样到 16 帧
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


# ============================================================
# 1. 3D Patch Embedding
# ============================================================
class PatchEmbed3D(nn.Module):
    """
    把视频 [B, C, T, H, W] 切成 3D tokens。
    论文配置: tubelet_size=2, patch_size=16
      输入 [B, 3, 16, 224, 224]
      输出 [B, 8*14*14, d] = [B, 1568, d]
    """
    def __init__(self, img_size=224, patch_size=16, tubelet_size=2,
                 in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size

        # 3D 卷积: 时间步长=tubelet_size, 空间步长=patch_size
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

        # 输出形状: (T/tubelet, H/patch, W/patch)
        self.grid_t = None  # 运行时确定

    def forward(self, x):
        # x: [B, C, T, H, W]
        x = self.proj(x)  # [B, D, T', H', W']
        B, D, T, H, W = x.shape
        self.grid_t, self.grid_h, self.grid_w = T, H, W
        x = x.flatten(2).transpose(1, 2)  # [B, T'*H'*W', D]
        return x


# ============================================================
# 2. 3D Sin-Cos Positional Embedding
# ============================================================
def get_3d_sincos_pos_embed(embed_dim, grid_t, grid_h, grid_w):
    """
    生成 3D sin-cos 位置编码, 输出 [T*H*W, embed_dim]
    把 embed_dim 分成三份: 时间 1/4, 高 3/8, 宽 3/8
    """
    assert embed_dim % 16 == 0, "embed_dim 需要能被 16 整除"
    t_dim = embed_dim // 4
    h_dim = embed_dim * 3 // 8
    w_dim = embed_dim - t_dim - h_dim

    def _1d_sincos(pos, dim):
        # pos: [N], dim: 标量
        omega = torch.arange(dim // 2, dtype=torch.float32)
        omega = 1.0 / (10000 ** (omega / (dim / 2)))
        out = pos[:, None].float() * omega[None, :]
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)  # [N, dim]

    t = torch.arange(grid_t)
    h = torch.arange(grid_h)
    w = torch.arange(grid_w)

    # meshgrid -> 每个位置的 (t, h, w)
    grid = torch.stack(torch.meshgrid(t, h, w, indexing='ij'), dim=-1)  # [T, H, W, 3]
    grid = grid.reshape(-1, 3)  # [T*H*W, 3]

    emb_t = _1d_sincos(grid[:, 0], t_dim)
    emb_h = _1d_sincos(grid[:, 1], h_dim)
    emb_w = _1d_sincos(grid[:, 2], w_dim)
    return torch.cat([emb_t, emb_h, emb_w], dim=1)  # [T*H*W, embed_dim]


# ============================================================
# 3. Transformer Block (标准 pre-norm)
# ============================================================
class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================
# 4. Vision Transformer (Context / Target Encoder)
# ============================================================
class VisionTransformer(nn.Module):
    """
    两用:
      - Context Encoder: 只接收可见 token (通过 forward(x, ids_keep))
      - Target Encoder: 接收所有 token (forward(x) 不做 gather)
    """
    def __init__(self, img_size=224, patch_size=16, tubelet_size=2,
                 num_frames=16, in_chans=3, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbed3D(
            img_size, patch_size, tubelet_size, in_chans, embed_dim
        )
        self.embed_dim = embed_dim

        # 预计算 3D 位置编码 (注册为 buffer, 不参与训练)
        grid_t = num_frames // tubelet_size
        grid_h = img_size // patch_size
        grid_w = img_size // patch_size
        self.num_patches = grid_t * grid_h * grid_w
        pos_embed = get_3d_sincos_pos_embed(embed_dim, grid_t, grid_h, grid_w)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0))  # [1, N, D]

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, ids_keep=None):
        """
        x: [B, C, T, H, W]
        ids_keep: [B, N_keep] 可见 patch 的索引; None 表示全部保留
        """
        x = self.patch_embed(x)           # [B, N, D]
        x = x + self.pos_embed            # 加位置编码

        if ids_keep is not None:
            # 只挑出可见的 token (Context Encoder 的关键效率优化)
            B, N, D = x.shape
            idx = ids_keep.unsqueeze(-1).expand(-1, -1, D)
            x = torch.gather(x, dim=1, index=idx)  # [B, N_keep, D]

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x


# ============================================================
# 5. Predictor (较小的 Transformer)
# ============================================================
class Predictor(nn.Module):
    """
    输入: context tokens (N_keep 个) + mask tokens (N_mask 个)
    输出: 对被遮位置的预测表示 (N_mask 个)

    mask token = 共享可学习向量 + 3D 位置编码
    论文里 predictor 比 encoder 浅、窄。
    """
    def __init__(self, encoder_embed_dim=768, predictor_embed_dim=384,
                 num_patches=1568, depth=6, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        # 把 encoder 维度投影到 predictor 维度
        self.embed = nn.Linear(encoder_embed_dim, predictor_embed_dim)

        # 共享的可学习 mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # 位置编码 (predictor 有自己的一份, 维度不同)
        # 简化起见, 这里用可学习的位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, predictor_embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            Block(predictor_embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(predictor_embed_dim)

        # 投影回 encoder 维度, 方便与 target 对齐
        self.proj = nn.Linear(predictor_embed_dim, encoder_embed_dim)

    def forward(self, z_ctx, ids_keep, ids_mask):
        """
        z_ctx:    [B, N_keep, D_enc]   Context Encoder 的输出
        ids_keep: [B, N_keep]          可见位置的索引
        ids_mask: [B, N_mask]          被遮位置的索引
        返回: [B, N_mask, D_enc] 对被遮位置的预测
        """
        B, N_keep, _ = z_ctx.shape
        N_mask = ids_mask.shape[1]
        D = self.pos_embed.shape[-1]

        # 1) 投影 context tokens
        x_ctx = self.embed(z_ctx)  # [B, N_keep, D]

        # 2) 给 context 加上它们自己位置的 pos embed
        pos_ctx = torch.gather(
            self.pos_embed.expand(B, -1, -1), dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )
        x_ctx = x_ctx + pos_ctx

        # 3) 构造 mask tokens: 共享向量 + 位置编码
        pos_mask = torch.gather(
            self.pos_embed.expand(B, -1, -1), dim=1,
            index=ids_mask.unsqueeze(-1).expand(-1, -1, D)
        )
        x_mask = self.mask_token.expand(B, N_mask, -1) + pos_mask  # [B, N_mask, D]

        # 4) 拼接后一起过 transformer
        x = torch.cat([x_ctx, x_mask], dim=1)  # [B, N_keep + N_mask, D]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # 5) 只取出 mask 部分作为预测输出, 投影回 encoder 维度
        x_mask_out = x[:, N_keep:, :]  # [B, N_mask, D]
        return self.proj(x_mask_out)   # [B, N_mask, D_enc]


# ============================================================
# 6. 3D Multi-Block Masking
# ============================================================
def sample_block_mask_2d(grid_h, grid_w, spatial_scale, aspect_ratio_range=(0.75, 1.5)):
    """在一帧 (grid_h x grid_w) 上随机采样一个矩形块, 返回布尔数组 [H, W]."""
    area = spatial_scale * grid_h * grid_w
    ar = torch.empty(1).uniform_(*aspect_ratio_range).item()
    h = max(1, int(round(math.sqrt(area / ar))))
    w = max(1, int(round(math.sqrt(area * ar))))
    h = min(h, grid_h)
    w = min(w, grid_w)
    top = torch.randint(0, grid_h - h + 1, (1,)).item()
    left = torch.randint(0, grid_w - w + 1, (1,)).item()
    mask = torch.zeros(grid_h, grid_w, dtype=torch.bool)
    mask[top:top + h, left:left + w] = True
    return mask


def build_3d_multiblock_mask(grid_t, grid_h, grid_w,
                             num_blocks, spatial_scale,
                             aspect_ratio_range=(0.75, 1.5)):
    """
    论文的 3D Multi-Block:
      1) 在 2D 平面上采 num_blocks 个块, 取并集
      2) 把 2D 遮罩沿时间维度重复 grid_t 次
    返回:
      ids_keep: [N_keep] 可见 patch 索引
      ids_mask: [N_mask] 被遮 patch 索引
    """
    mask_2d = torch.zeros(grid_h, grid_w, dtype=torch.bool)
    for _ in range(num_blocks):
        mask_2d |= sample_block_mask_2d(grid_h, grid_w, spatial_scale, aspect_ratio_range)

    # 沿时间维度复制 -> [T, H, W]
    mask_3d = mask_2d.unsqueeze(0).expand(grid_t, -1, -1).contiguous()
    mask_flat = mask_3d.flatten()  # [T*H*W]

    ids_mask = torch.nonzero(mask_flat, as_tuple=False).squeeze(1)
    ids_keep = torch.nonzero(~mask_flat, as_tuple=False).squeeze(1)
    return ids_keep, ids_mask


def make_batch_masks(batch_size, grid_t, grid_h, grid_w, mask_cfg):
    """为整个 batch 生成遮罩, 保持每个样本的 N_keep 一致 (简化版)."""
    ids_keep_list, ids_mask_list = [], []
    min_keep, min_mask = None, None
    for _ in range(batch_size):
        ids_keep, ids_mask = build_3d_multiblock_mask(
            grid_t, grid_h, grid_w,
            num_blocks=mask_cfg['num_blocks'],
            spatial_scale=mask_cfg['spatial_scale'],
        )
        ids_keep_list.append(ids_keep)
        ids_mask_list.append(ids_mask)
        min_keep = len(ids_keep) if min_keep is None else min(min_keep, len(ids_keep))
        min_mask = len(ids_mask) if min_mask is None else min(min_mask, len(ids_mask))

    # 截齐到最短, 方便 batch 处理
    ids_keep = torch.stack([x[:min_keep] for x in ids_keep_list], dim=0)
    ids_mask = torch.stack([x[:min_mask] for x in ids_mask_list], dim=0)
    return ids_keep, ids_mask


# ============================================================
# 7. 视频时间下采样 (64 帧 -> 16 帧)
# ============================================================
def temporal_subsample(video, target_frames=16):
    """
    video: [B, C, T, H, W], T=64
    每 T/target_frames 帧取 1 帧 -> [B, C, target_frames, H, W]
    """
    T = video.shape[2]
    stride = T // target_frames
    idx = torch.arange(0, target_frames) * stride
    return video.index_select(dim=2, index=idx.to(video.device))


# ============================================================
# 8. EMA 更新 (关键的防塌缩机制)
# ============================================================
@torch.no_grad()
def update_ema(target_model, context_model, tau):
    """
    Target Encoder 的参数 = tau * target + (1-tau) * context
    tau 越接近 1, target 更新越慢.
    论文用 tau ~ 0.998 起步, 训练中线性增大到 1.0
    """
    for p_t, p_c in zip(target_model.parameters(), context_model.parameters()):
        p_t.data.mul_(tau).add_(p_c.data, alpha=1.0 - tau)
    # BN/LayerNorm 的 buffer 也同步一下
    for b_t, b_c in zip(target_model.buffers(), context_model.buffers()):
        b_t.data.copy_(b_c.data)


# ============================================================
# 9. V-JEPA 完整封装
# ============================================================
class VJEPA(nn.Module):
    def __init__(self,
                 img_size=224, patch_size=16, tubelet_size=2, num_frames=16,
                 encoder_embed_dim=768, encoder_depth=12, encoder_heads=12,
                 predictor_embed_dim=384, predictor_depth=6, predictor_heads=12,
                 mlp_ratio=4.0):
        super().__init__()

        # Context Encoder (走梯度)
        self.context_encoder = VisionTransformer(
            img_size=img_size, patch_size=patch_size, tubelet_size=tubelet_size,
            num_frames=num_frames, embed_dim=encoder_embed_dim,
            depth=encoder_depth, num_heads=encoder_heads, mlp_ratio=mlp_ratio,
        )

        # Target Encoder (EMA, 不走梯度)
        self.target_encoder = deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)  # 冻结: 只通过 EMA 更新

        num_patches = self.context_encoder.num_patches

        # Predictor
        self.predictor = Predictor(
            encoder_embed_dim=encoder_embed_dim,
            predictor_embed_dim=predictor_embed_dim,
            num_patches=num_patches,
            depth=predictor_depth,
            num_heads=predictor_heads,
            mlp_ratio=mlp_ratio,
        )

    def forward(self, video, ids_keep, ids_mask):
        """
        video:    [B, C, T, H, W]  (已时间下采样到 16 帧)
        ids_keep: [B, N_keep]
        ids_mask: [B, N_mask]
        返回: pred [B, N_mask, D], target [B, N_mask, D]
        """
        # ---------- Context: 只处理可见 patch ----------
        z_ctx = self.context_encoder(video, ids_keep=ids_keep)

        # ---------- Predictor: 预测被遮位置 ----------
        pred = self.predictor(z_ctx, ids_keep, ids_mask)

        # ---------- Target: 处理完整视频 + stop-gradient ----------
        with torch.no_grad():  # <<<<<< STOP-GRADIENT 在这里 >>>>>>
            s_all = self.target_encoder(video, ids_keep=None)   # [B, N, D]
            # 取出被遮位置的 target
            B, _, D = s_all.shape
            target = torch.gather(
                s_all, dim=1,
                index=ids_mask.unsqueeze(-1).expand(-1, -1, D)
            )  # [B, N_mask, D]

        return pred, target


# ============================================================
# 10. 主演示: 一次完整的训练步骤
# ============================================================
def demo():
    # -------- 设备 --------
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[设备] {device}")

    torch.manual_seed(42)

    # -------- 构造输入 --------
    # 用户要求: Random 生成 [1, 3, 64, 224, 224]
    # (论文输入 64 帧后会下采样到 16 帧)
    video_raw = torch.randn(1, 3, 64, 224, 224, device=device)
    print(f"[输入] 原始视频 shape = {tuple(video_raw.shape)}  (B,C,T,H,W)")

    video = temporal_subsample(video_raw, target_frames=16)
    print(f"[输入] 下采样后 shape = {tuple(video.shape)}  (64->16 帧)")

    # -------- 构造模型 --------
    # 为了让单样本能在 MPS/M 系列上顺畅跑, 这里用比论文小得多的配置:
    #   encoder dim 384, depth 4; predictor dim 192, depth 2
    # 把参数调大一点就接近 ViT-B / ViT-L
    model = VJEPA(
        img_size=224, patch_size=16, tubelet_size=2, num_frames=16,
        encoder_embed_dim=384, encoder_depth=4,  encoder_heads=6,
        predictor_embed_dim=192, predictor_depth=2, predictor_heads=6,
    ).to(device)

    n_ctx = sum(p.numel() for p in model.context_encoder.parameters())
    n_tgt = sum(p.numel() for p in model.target_encoder.parameters())
    n_pred = sum(p.numel() for p in model.predictor.parameters())
    print(f"[模型] Context Encoder: {n_ctx/1e6:.2f}M 参数 (需梯度)")
    print(f"[模型] Target  Encoder: {n_tgt/1e6:.2f}M 参数 (冻结, EMA 更新)")
    print(f"[模型] Predictor:        {n_pred/1e6:.2f}M 参数 (需梯度)")

    # -------- 遮罩配置 (两种) --------
    grid_t = 16 // 2   # 8  tubelet_size=2 的意思:把每 2 帧连续的帧一起卷成 1 个 token,16 帧全部都用到,只是两两合并后得到 8 个时间 token。

    grid_h = 224 // 16  # 14
    grid_w = 224 // 16  # 14
    num_patches = grid_t * grid_h * grid_w  # 1568
    print(f"[Patch] 时空网格 = ({grid_t}, {grid_h}, {grid_w}), 共 {num_patches} tokens")

    short_cfg = {'num_blocks': 8, 'spatial_scale': 0.15}
    long_cfg  = {'num_blocks': 2, 'spatial_scale': 0.70}

    # -------- 优化器 --------
    # 只优化 Context Encoder 和 Predictor 的参数
    trainable = (
        list(model.context_encoder.parameters()) +
        list(model.predictor.parameters())
    )
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.05)

    ema_tau = 0.998  # EMA 动量

    # -------- 训练若干步 --------
    # 虽然只有一组数据, 但跑多步能看到 loss 下降 & EMA 追赶
    NUM_STEPS = 20
    print(f"\n开始训练 {NUM_STEPS} 步 (同一批数据, 观察 loss 与 EMA 行为)\n")
    print(f"{'step':>4} | {'loss':>10} | {'||θ_ctx - θ̄||':>14} | {'||grad(ctx)||':>14}")
    print("-" * 60)

    for step in range(NUM_STEPS):
        # 1) 每步重新采样 Multi-Mask (论文: short + long 共享同一次 target 计算)
        #    这里演示时先跑 short-range, 再跑 long-range, loss 相加
        ids_keep_s, ids_mask_s = make_batch_masks(1, grid_t, grid_h, grid_w, short_cfg)
        ids_keep_l, ids_mask_l = make_batch_masks(1, grid_t, grid_h, grid_w, long_cfg)

        ids_keep_s = ids_keep_s.to(device)
        ids_mask_s = ids_mask_s.to(device)
        ids_keep_l = ids_keep_l.to(device)
        ids_mask_l = ids_mask_l.to(device)

        # 2) 前向 + 计算 L1 loss
        #    (真正的 multi-mask 应共享 target 计算, 这里为了代码清晰分开写)
        pred_s, target_s = model(video, ids_keep_s, ids_mask_s)
        pred_l, target_l = model(video, ids_keep_l, ids_mask_l)

        loss_short = F.l1_loss(pred_s, target_s)
        loss_long  = F.l1_loss(pred_l, target_l)
        loss = 0.5 * (loss_short + loss_long)

        # 3) 反向: 只更新 context + predictor
        #    Target Encoder 因为 requires_grad_(False) 和 torch.no_grad()
        #    双重保护, 不会收到任何梯度
        optimizer.zero_grad()
        loss.backward()

        # 记录 context encoder 梯度范数 (监控学习信号)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=10.0)

        optimizer.step()

        # 4) EMA 更新 Target Encoder (在 optimizer.step() 之后)
        update_ema(model.target_encoder, model.context_encoder, tau=ema_tau)

        # 5) 监控: 计算 context vs target 参数的距离
        with torch.no_grad():
            diff_sq = 0.0
            for p_c, p_t in zip(model.context_encoder.parameters(),
                                model.target_encoder.parameters()):
                diff_sq += (p_c - p_t).pow(2).sum().item()
            param_diff = math.sqrt(diff_sq)

        print(f"{step:>4} | {loss.item():>10.6f} | "
              f"{param_diff:>14.4f} | {grad_norm.item():>14.4f}")

    print("\n[完成] 可以观察到:")
    print("  1) Loss 持续下降 -> Context + Predictor 在学习")
    print("  2) ||θ_ctx - θ̄|| 不为 0 -> EMA 让 Target 滞后跟随, 避免塌缩")
    print("  3) Target Encoder 全程 requires_grad=False, 没有梯度流")


if __name__ == "__main__":
    demo()


###sometest
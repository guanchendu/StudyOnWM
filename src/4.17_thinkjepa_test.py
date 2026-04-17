"""
4.17_thinkjepa_test.py

ThinkJEPA 复现 — tworoom 数据集上的世界模型

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
论文: ThinkJEPA: Empowering Latent World Models with Large Vision-Language Reasoning Model

核心思路:
    1. 并行使用 LLM (Qwen3-4B) 和 ViT 对长/短视频提取特征
         - 短视频分支: ViT 编码当前帧 → JEPA 上下文 token (student)
         - 长视频分支: 将视频序列转为文字描述 (动作/proprio 轨迹) → Qwen
                      → HPRE (Hierarchical Pyramid Representation Extraction)
                      → K 个金字塔层级的 thinker guidance 向量
    2. HPRE 把 Qwen 的 L 层 hidden states 均分为 K 组,每组先做层间 mean pool,
       再做 token-wise mean pool (attention_mask 加权) 得到一个固定维度向量。
    3. JEPA Predictor: 每一层 transformer block 从对应的金字塔层级生成
       FiLM 参数 (γl, βl),对块输入做 FiLM(z; γl, βl) = γl ⊙ z + βl。
    4. Target encoder 用 EMA 更新 (JEPA 标准做法)。
    5. 损失: 预测 latent 与 target latent 的 smooth_L1。

训练约束:
    - 每 epoch 仅跑一个 batch (加速调试)
    - tqdm 进度条显示
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import stable_pretraining as spt
import stable_worldmodel as swm
from transformers import AutoModelForCausalLM, AutoTokenizer

from Background.utils import get_column_normalizer, get_img_preprocessor

MODEL_DIR  = PROJECT_ROOT / "models" / "Qwen3-4B-Instruct-2507"
DATA_DIR   = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "src" / "outputs" / "thinkjepa_tworoom"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── 超参数 ────────────────────────────────────────────────────────────────────
IMG_SIZE            = 64
NUM_STEPS           = 4          # 每条序列帧数
FRAMESKIP           = 5
BATCH_SIZE          = 64          # Qwen 4B 很大,batch 设小一点
EPOCHS              = 10
LR                  = 1e-4
PATCH_SIZE          = 8
EMBED_DIM           = 192        # ViT / Predictor 特征维度
CTX_DEPTH           = 4          # ViT 上下文 encoder 深度
PRED_DEPTH          = 4          # JEPA predictor 深度 (最好 == NUM_PYRAMID)
HEADS               = 6
NUM_PYRAMID         = 4          # HPRE 金字塔层级数 K
QWEN_MAX_LEN        = 128        # 截断 Qwen 输入 token 数
EMA_MOMENTUM        = 0.996


# ── Device / dtype ───────────────────────────────────────────────────────────
def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def choose_dtype_qwen(device: torch.device):
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


# ── Dataset ──────────────────────────────────────────────────────────────────
def build_dataset(cache_dir, img_size, num_steps, frameskip):
    keys = ["pixels", "action", "proprio"]
    dataset = swm.data.HDF5Dataset(
        name="tworoom",
        keys_to_load=keys,
        keys_to_cache=["action", "proprio"],
        num_steps=num_steps,
        frameskip=frameskip,
        transform=None,
        cache_dir=str(cache_dir),
    )
    transforms = [
        get_img_preprocessor(source="pixels", target="pixels", img_size=img_size)
    ]
    for col in ("action", "proprio"):
        transforms.append(get_column_normalizer(dataset, col, col))
    dataset.transform = spt.data.transforms.Compose(*transforms)
    return dataset


# ── 小型 ViT 编码器 (短视频分支) ─────────────────────────────────────────────
class PatchEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_chans=3, embed_dim=192):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        # x: [B, C, H, W] → [B, N, D]
        return self.proj(x).flatten(2).transpose(1, 2)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=6, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    def __init__(self, img_size=64, patch_size=8, embed_dim=192, depth=4, heads=6):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.pos_embed   = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, heads) for _ in range(depth)])
        self.norm   = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)   # [B, N, D]


# ── HPRE 模块 (Hierarchical Pyramid Representation Extraction) ──────────────
class HierarchicalPyramidExtractor:
    """
    将 Qwen 的 L 层 hidden states 压缩为 K 个金字塔层级的 guidance 向量。
        浅层  → 词法/句法     (Level 1)
        中浅层 → 局部语义      (Level 2)
        中深层 → 深层推理      (Level 3)
        深层  → 高层抽象       (Level 4)
    每个 level: layer-wise mean → token-wise mean (with attention_mask) → [B, D_qwen]
    """

    def __init__(self, num_pyramid_levels: int = 4):
        self.K = num_pyramid_levels

    def extract(self, hidden_states: tuple, attention_mask: torch.Tensor | None = None):
        layer_outputs = hidden_states[1:]   # 跳过 embedding 层
        L = len(layer_outputs)
        K = self.K

        if attention_mask is not None:
            mask  = attention_mask.float().unsqueeze(-1)          # [B, seq, 1]
            denom = mask.sum(dim=1).clamp(min=1.0)                # [B, 1]

        pyramid_reps: list[torch.Tensor] = []
        for k in range(K):
            start = int(k * L / K)
            end   = int((k + 1) * L / K)
            # stack: [n_group, B, seq, D]
            stacked = torch.stack(
                [layer_outputs[i].float() for i in range(start, end)],
                dim=0,
            )
            # layer-wise mean → [B, seq, D]
            level = stacked.mean(dim=0)
            # token-wise (masked) mean → [B, D]
            if attention_mask is not None:
                level = (level * mask).sum(dim=1) / denom
            else:
                level = level.mean(dim=1)
            pyramid_reps.append(level)
        return pyramid_reps


# ── Thinker: Qwen 文本推理 + HPRE ────────────────────────────────────────────
class Thinker:
    """
    输入 batch 的 chat-formatted 文本,输出 K 个金字塔层级 guidance 向量。
    Qwen 参数全部 frozen,只做前向。
    """
    def __init__(self, model_dir: Path, device: torch.device, dtype, num_pyramid_levels=4):
        self.device = device
        self.dtype  = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Qwen 是 decoder-only,做批量前向时用左 padding 更自然;
        # HPRE 用 attention_mask 做加权平均,右 padding 也可以。这里用 left padding。
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.hpre = HierarchicalPyramidExtractor(num_pyramid_levels)
        self.hidden_size = self.model.config.hidden_size
        self.num_pyramid_levels = num_pyramid_levels

    @torch.no_grad()
    def __call__(self, chat_texts: list[str]) -> list[torch.Tensor]:
        inputs = self.tokenizer(
            chat_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=QWEN_MAX_LEN,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs, output_hidden_states=True)
        pyr = self.hpre.extract(out.hidden_states, attention_mask=inputs["attention_mask"])
        # 转回 float32 方便后续 FiLM / 梯度
        return [p.float() for p in pyr]


def build_video_prompts(raw_batch: dict, tokenizer, num_steps: int) -> list[str]:
    """把一个 batch 的 (action, proprio) 序列转成 chat-formatted 长视频描述"""
    B = raw_batch["action"].shape[0]
    chat_texts: list[str] = []
    for b in range(B):
        lines = []
        for t in range(num_steps):
            a = raw_batch["action"][b, t].detach().cpu().float().numpy()
            p = raw_batch["proprio"][b, t].detach().cpu().float().numpy()
            a_s = ", ".join(f"{x:+.2f}" for x in a)
            p_s = ", ".join(f"{x:+.2f}" for x in p)
            lines.append(f"t={t}: action=[{a_s}], proprio=[{p_s}]")
        body = "\n".join(lines)
        user = (
            "Below is a short robotic video described by its action and "
            "proprioception trajectory in a two-room navigation environment:\n"
            f"{body}\n"
            "Reason step by step about what is happening physically, and "
            "summarize what the robot's next state is likely to look like."
        )
        messages = [
            {"role": "system",
             "content": "You are a world-model reasoning assistant. "
                        "You think carefully about robot dynamics."},
            {"role": "user", "content": user},
        ]
        chat_texts.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )
    return chat_texts


# ── FiLM 生成器 ──────────────────────────────────────────────────────────────
class FiLMGenerator(nn.Module):
    """
    从 guidance vector (Qwen 金字塔表征) 生成 (γ, β),用于调制 predictor block 输入。
    初始化为零,使 γ=1, β=0 (恒等),训练早期稳定。
    """
    def __init__(self, guidance_dim: int, feat_dim: int):
        super().__init__()
        self.proj = nn.Linear(guidance_dim, feat_dim * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.feat_dim = feat_dim

    def forward(self, guidance: torch.Tensor):
        gb = self.proj(guidance)                    # [B, 2*D]
        gamma, beta = gb.chunk(2, dim=-1)           # 各 [B, D]
        return 1.0 + gamma, beta


# ── FiLM-conditioned Transformer Block (JEPA Predictor block) ───────────────
class FiLMTransformerBlock(nn.Module):
    def __init__(self, dim, heads=6, mlp_ratio=4.0, guidance_dim=None):
        super().__init__()
        self.film_gen = FiLMGenerator(guidance_dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, z: torch.Tensor, guidance: torch.Tensor):
        # FiLM 调制: FiLM(z; γl, βl) = γl ⊙ z + βl
        gamma, beta = self.film_gen(guidance)                 # [B, D]
        z = gamma.unsqueeze(1) * z + beta.unsqueeze(1)        # [B, N, D]

        # 标准 transformer
        y = self.norm1(z)
        y, _ = self.attn(y, y, y, need_weights=False)
        z = z + y
        z = z + self.mlp(self.norm2(z))
        return z


# ── JEPA Predictor ───────────────────────────────────────────────────────────
class JEPAPredictor(nn.Module):
    """
    预测下一帧 latent: 输入上下文 token + action/proprio,
    每一层 FiLM 调制来自 Qwen HPRE 的对应金字塔层。
    """
    def __init__(self, feat_dim, depth, heads, guidance_dim, action_dim, proprio_dim):
        super().__init__()
        self.action_proj = nn.Linear(action_dim + proprio_dim, feat_dim)
        self.blocks = nn.ModuleList([
            FiLMTransformerBlock(feat_dim, heads, guidance_dim=guidance_dim)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, context_tokens, action, proprio, pyramid_reps):
        """
        context_tokens : [B, N, D]   — ViT 编码的当前帧 token
        action, proprio: [B, *]
        pyramid_reps   : list of K tensors [B, D_qwen]
        """
        B, N, D = context_tokens.shape
        ap = torch.cat([action, proprio], dim=-1)
        ap_token = self.action_proj(ap).unsqueeze(1)           # [B, 1, D]
        z = torch.cat([context_tokens, ap_token], dim=1)       # [B, N+1, D]

        K = len(pyramid_reps)
        L = len(self.blocks)
        for l, blk in enumerate(self.blocks):
            # 把 L 个 block 映射到 K 个金字塔层
            k = min(int(l * K / L), K - 1)
            z = blk(z, pyramid_reps[k])
        z = self.norm(z)
        return z[:, :N].contiguous()   # 丢掉 action 辅助 token,对齐 target token 长度


# ── ThinkJEPA World Model ────────────────────────────────────────────────────
class ThinkJEPA(nn.Module):
    def __init__(
        self,
        img_size, patch_size, embed_dim,
        ctx_depth, pred_depth, heads,
        qwen_hidden_dim, action_dim, proprio_dim,
    ):
        super().__init__()
        self.context_encoder = ViTEncoder(img_size, patch_size, embed_dim, ctx_depth, heads)
        self.target_encoder  = ViTEncoder(img_size, patch_size, embed_dim, ctx_depth, heads)
        # target 初始化 = student,并冻结
        for p_t, p_s in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            p_t.data.copy_(p_s.data)
            p_t.requires_grad = False

        self.predictor = JEPAPredictor(
            feat_dim=embed_dim,
            depth=pred_depth,
            heads=heads,
            guidance_dim=qwen_hidden_dim,
            action_dim=action_dim,
            proprio_dim=proprio_dim,
        )

    @torch.no_grad()
    def ema_update(self, momentum: float = 0.996):
        for p_t, p_s in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)

    def forward(self, pixels_t, pixels_tp1, action, proprio, pyramid_reps):
        z_ctx = self.context_encoder(pixels_t)
        with torch.no_grad():
            z_tgt = self.target_encoder(pixels_tp1)
        z_pred = self.predictor(z_ctx, action, proprio, pyramid_reps)
        return z_pred, z_tgt


# ── 批次准备 ─────────────────────────────────────────────────────────────────
def make_transition_pair(raw_batch):
    """从 [B, T, ...] 的序列 batch 中取出第一个 (t=0 → t=1) 转移对"""
    pixels  = raw_batch["pixels"].float()
    action  = raw_batch["action"].float()
    proprio = raw_batch["proprio"].float()
    return {
        "pixels_t":    pixels[:, 0],
        "pixels_tp1":  pixels[:, 1],
        "action_t":    action[:, 0],
        "proprio_t":   proprio[:, 0],
    }


# ── 主流程 ───────────────────────────────────────────────────────────────────
def main():
    device     = choose_device()
    dtype_qwen = choose_dtype_qwen(device)
    print(f"[env] device={device}, qwen_dtype={dtype_qwen}")

    # 1. 数据 ----------------------------------------------------------------
    dataset = build_dataset(DATA_DIR, IMG_SIZE, NUM_STEPS, FRAMESKIP)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    sample = next(iter(loader))
    action_dim  = sample["action"].shape[-1]
    proprio_dim = sample["proprio"].shape[-1]
    print(f"[data] pixels={tuple(sample['pixels'].shape)}, "
          f"action_dim={action_dim}, proprio_dim={proprio_dim}")

    # 2. Thinker (Qwen + HPRE) ----------------------------------------------
    print(f"[model] Loading Qwen thinker from {MODEL_DIR} ...")
    thinker = Thinker(MODEL_DIR, device, dtype_qwen, NUM_PYRAMID)
    print(f"[thinker] num_hidden_layers={thinker.model.config.num_hidden_layers}, "
          f"hidden_size={thinker.hidden_size}, K={NUM_PYRAMID}")

    # 3. JEPA 世界模型 -------------------------------------------------------
    model = ThinkJEPA(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        ctx_depth=CTX_DEPTH,
        pred_depth=PRED_DEPTH,
        heads=HEADS,
        qwen_hidden_dim=thinker.hidden_size,
        action_dim=action_dim,
        proprio_dim=proprio_dim,
    ).to(device)

    trainable = list(model.context_encoder.parameters()) + list(model.predictor.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=LR)

    n_params = sum(p.numel() for p in trainable)
    print(f"[jepa] trainable params: {n_params:,}")
    print("=" * 72)
    print("ThinkJEPA training  (1 batch per epoch)")
    print("=" * 72)

    # 4. 训练 ---------------------------------------------------------------
    for epoch in range(1, EPOCHS + 1):
        model.train()
        # 每 epoch 只跑一个 batch: total=1
        pbar = tqdm(total=1, desc=f"Epoch {epoch:03d}/{EPOCHS}", leave=True)

        raw_batch = next(iter(loader))          # 一个 batch 就行
        batch = make_transition_pair(raw_batch)
        batch = {k: v.to(device) for k, v in batch.items()}

        # 4a. 构造长视频文本并过 Qwen → HPRE 金字塔 guidance
        chat_texts = build_video_prompts(raw_batch, thinker.tokenizer, NUM_STEPS)
        pyramid = thinker(chat_texts)           # list of K tensors [B, D_qwen]

        # 4b. JEPA 前向
        z_pred, z_tgt = model(
            batch["pixels_t"],
            batch["pixels_tp1"],
            batch["action_t"],
            batch["proprio_t"],
            pyramid,
        )

        # 4c. 预测 loss (token-level smooth L1,JEPA 标准做法之一)
        loss = F.smooth_l1_loss(z_pred.contiguous(), z_tgt.contiguous())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()

        # 4d. EMA target encoder 更新
        model.ema_update(EMA_MOMENTUM)

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            pred_norm=f"{z_pred.norm().item():.2f}",
            tgt_norm=f"{z_tgt.norm().item():.2f}",
        )
        pbar.update(1)
        pbar.close()

    # 5. 保存 ---------------------------------------------------------------
    save_path = OUTPUT_DIR / "thinkjepa_tworoom.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "img_size": IMG_SIZE, "patch_size": PATCH_SIZE,
                "embed_dim": EMBED_DIM, "ctx_depth": CTX_DEPTH,
                "pred_depth": PRED_DEPTH, "heads": HEADS,
                "num_pyramid": NUM_PYRAMID,
                "qwen_hidden_dim": thinker.hidden_size,
                "action_dim": action_dim, "proprio_dim": proprio_dim,
            },
        },
        save_path,
    )
    print(f"\n[save] ThinkJEPA → {save_path}")
    print("Done.")


if __name__ == "__main__":
    main()

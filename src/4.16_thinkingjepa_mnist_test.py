"""
4.16_thinkingjepa_mnist_test.py

随机从 CIFAR-10 数据集中选择一张图片，使用 Qwen3-4B-Instruct 模型生成文字回答，
并依据 ThinkJEPA 论文的 Hierarchical Pyramid Representation Extraction (HPRE) 模块
提取模型每一层的表征，构建多尺度层级金字塔特征。

参考:
    ThinkJEPA: Empowering Latent World Models with Large Vision-Language Reasoning Model

流程:
    1. 从 CIFAR-10 中随机采样一张图片（含真实类别标签）
    2. 构建 prompt: "这是一张 <class_name> 的图片，请描述其视觉特征"
    3. 前向传播 Qwen，开启 output_hidden_states=True 拿到所有层表征
    4. HPRE 模块: 将 L 层 hidden states 分为 K 个金字塔级别，
       每级别做 layer-wise mean pooling + token-wise mean pooling → 一个固定维度向量
    5. 独立调用 model.generate 生成文字回答
    6. 可视化 & 保存结果（PNG + JSON）
"""

import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torchvision import datasets, transforms
except ImportError as exc:
    raise SystemExit(
        "Missing dependencies. Please install: pip install torch transformers torchvision"
    ) from exc


# ── paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR    = PROJECT_ROOT / "models" / "Qwen3-4B-Instruct-2507"
DATA_DIR     = PROJECT_ROOT / "data"          # CIFAR-10 存放在 data/cifar-10-batches-py/
OUTPUT_DIR   = PROJECT_ROOT / "src" / "outputs" / "thinkjepa_cifar"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── device / dtype ────────────────────────────────────────────────────────────
def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def choose_dtype(device: str):
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


# ── CIFAR-10 ──────────────────────────────────────────────────────────────────
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
CIFAR10_CLASSES_ZH = [
    "飞机", "汽车", "鸟", "猫", "鹿",
    "狗", "青蛙", "马", "轮船", "卡车",
]


def load_random_cifar_sample(data_dir: Path, seed: int | None = None):
    """返回 (image_tensor [3,32,32], label int, class_name_en str, class_name_zh str, dataset_index int)"""
    if seed is not None:
        random.seed(seed)
    transform = transforms.ToTensor()
    dataset = datasets.CIFAR10(root=str(data_dir), train=False, download=True, transform=transform)
    idx = random.randint(0, len(dataset) - 1)
    image, label = dataset[idx]
    return image, int(label), CIFAR10_CLASSES[label], CIFAR10_CLASSES_ZH[label], idx


def build_prompt(class_name_zh: str, class_name_en: str) -> tuple[str, str]:
    """返回 (system_prompt, user_prompt)"""
    system = "你是一个专业的图像识别助手，擅长分析自然图像中的物体。"
    user = (
        f'我有一张 CIFAR-10 数据集中的图片，真实类别为“{class_name_zh}”（{class_name_en}）。\n'
        f'请确认这是一张{class_name_zh}的图片，并简要描述{class_name_zh}通常的外观视觉特征。'
    )
    return system, user

# ── HPRE 模块 ─────────────────────────────────────────────────────────────────
class HierarchicalPyramidRepresentationExtractor:
    """
    ThinkJEPA 中的 Hierarchical Pyramid Representation Extraction (HPRE) 模块。

    核心思想:
        LLM 的不同深度层次捕获不同粒度的特征:
            - 浅层 (Level 1): 词法 / 句法特征
            - 中浅层 (Level 2): 局部语义特征
            - 中深层 (Level 3): 深层推理特征
            - 深层 (Level 4): 高层抽象 / 任务特征

    实现:
        1. 将 L 个 transformer 层均分为 K 个 group
        2. 每个 group 内:
               a. stack hidden states → [n_group, seq_len, hidden_dim]
               b. layer-wise mean pool → [seq_len, hidden_dim]
               c. token-wise mean pool → [hidden_dim]
        3. 输出 K 个表征向量构成金字塔
    """

    def __init__(self, num_pyramid_levels: int = 4):
        self.num_pyramid_levels = num_pyramid_levels

    def extract(self, hidden_states: tuple) -> dict:
        """
        Args:
            hidden_states: tuple, 长度 = num_layers + 1
                           hidden_states[0] = embedding 层输出 [batch, seq, dim]
                           hidden_states[i] = 第 i 个 transformer block 的输出
        Returns:
            {
                "pyramid_reps"  : List[Tensor [hidden_dim]],  长度 K
                "per_layer_reps": List[Tensor [hidden_dim]],  长度 L (不含 embedding 层)
                "layer_groups"  : List[List[int]],            每 level 对应的层索引 (0-based)
            }
        """
        # 跳过 index-0 的 embedding 层，只取 transformer block 输出  拿到各层输出 因为 index 0 是Embedding 的
        layer_outputs = hidden_states[1:]   # (L, batch=1, seq_len, hidden_dim)
        L = len(layer_outputs)
        K = self.num_pyramid_levels

        # ── 每层表征: 在 token 维度做 mean pool ────────────────────────────
        '''
        Step 2：每层做 token mean pooling
        把一句话的所有 token 向量取平均，得到这一层对整个输入的"综合理解"：
        '''
        per_layer_reps: list[torch.Tensor] = []
        for h in layer_outputs:
            # h: [1, seq_len, hidden_dim]  → mean over tokens → [hidden_dim]
            rep = h[0].float().mean(dim=0).cpu()
            per_layer_reps.append(rep)

        # ── 分组: 将 L 层均分为 K 个 pyramid level ──────────────────────────
        '''
        把 L 层均匀分成 K 组，例如 L=36, K=4：
        Level 1: layers  0~8   (浅层，词法特征)
        Level 2: layers  9~17  (语义特征)
        Level 3: layers 18~26  (推理特征)
        Level 4: layers 27~35  (高层抽象)
        '''
        layer_groups: list[list[int]] = []
        for k in range(K):
            start = int(k * L / K)
            end   = int((k + 1) * L / K)
            layer_groups.append(list(range(start, end)))

        # ── 金字塔级别表征 ───────────────────────────────────────────────────
        pyramid_reps: list[torch.Tensor] = []
        for group_indices in layer_groups:
            # stack: [n_group, seq_len, hidden_dim]
            group_hidden = torch.stack(
                [layer_outputs[i][0].float() for i in group_indices], dim=0
            )
            '''
            每组做两次 pooling
            [n_group, seq_len, hidden_dim]
                ↓ mean(dim=0)  ← 把同组的层平均（层间融合）
            [seq_len, hidden_dim]
                ↓ mean(dim=0)  ← 把所有 token 平均（序列压缩）
            [hidden_dim]       ← 最终这个 level 的表征向量
            '''
            # layer-wise mean → [seq_len, hidden_dim]
            level_rep = group_hidden.mean(dim=0)
            # token-wise mean → [hidden_dim]
            level_rep = level_rep.mean(dim=0).cpu()
            pyramid_reps.append(level_rep)

        return {
            "pyramid_reps":   pyramid_reps,
            "per_layer_reps": per_layer_reps,
            "layer_groups":   layer_groups,
        }


# ── Qwen 前向 + 生成 ──────────────────────────────────────────────────────────
def qwen_forward_and_generate(
    model,
    tokenizer,
    system: str,
    user: str,
    device: str,
    max_new_tokens: int = 120,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> tuple[str, tuple]:
    """
    先做一次完整的前向传播拿 hidden_states，
    再独立调用 generate 拿文字回答。

    Returns:
        reply (str), hidden_states (tuple)
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )  #只做格式化，把 messages 拼成 Qwen 要求的对话模板字符串
    inputs = tokenizer(chat_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        # ── 1. 前向: 拿所有层 hidden states ─────────────────────────────
        fwd_out = model(**inputs, output_hidden_states=True)
        hidden_states = fwd_out.hidden_states   # tuple[num_layers+1]

        # ── 2. 生成: 拿文字回答 ──────────────────────────────────────────
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        output_ids = model.generate(**inputs, **gen_kwargs)

    new_token_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    reply = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
    return reply, hidden_states


# ── 可视化 ────────────────────────────────────────────────────────────────────
def visualize(
    image: torch.Tensor,
    label: int,
    class_name_en: str,
    class_name_zh: str,
    reply: str,
    hpre_result: dict,
    output_dir: Path,
) -> Path:
    pyramid_reps   = hpre_result["pyramid_reps"]
    per_layer_reps = hpre_result["per_layer_reps"]
    layer_groups   = hpre_result["layer_groups"]
    K = len(pyramid_reps)
    L = len(per_layer_reps)

    # 总列数 = max(K+1, 3)，两行
    ncols = max(K + 1, 3)
    fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 8))
    fig.suptitle(
        f"ThinkJEPA HPRE — CIFAR-10: {class_name_zh} ({class_name_en})  |  "
        f'Qwen3-4B reply: "{reply[:60]}..."',
        fontsize=10,
        wrap=True,
    )

    # ── Row 0, Col 0: CIFAR-10 图片 (RGB, 32×32) ──────────────────────────
    ax = axes[0, 0]
    # image: [3, 32, 32] → [32, 32, 3]，值域 [0,1]
    ax.imshow(image.permute(1, 2, 0).numpy())
    ax.set_title(f"CIFAR-10  label={label}\n({class_name_zh} / {class_name_en})", fontsize=9)
    ax.axis("off")

    # ── Row 0, Col 1~K: 每个金字塔层的 hidden rep heatmap ─────────────────
    for k, rep in enumerate(pyramid_reps):
        ax = axes[0, k + 1]
        vec = rep.numpy()
        # 将 1D 向量 reshape 成近似正方形展示
        side = int(np.ceil(np.sqrt(len(vec))))
        padded = np.zeros(side * side)
        padded[: len(vec)] = vec
        img = padded.reshape(side, side)
        vmax = np.abs(img).max() or 1.0
        im = ax.imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(
            f"Pyramid Level {k+1}\n"
            f"(Qwen layers {layer_groups[k][0]+1}–{layer_groups[k][-1]+1})",
            fontsize=8,
        )
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 隐藏 Row 0 多余的格子
    for col in range(K + 1, ncols):
        axes[0, col].axis("off")

    # ── Row 1, Col 0: 每层表征的 L2 norm ──────────────────────────────────
    ax = axes[1, 0]
    norms = [r.norm().item() for r in per_layer_reps]
    ax.plot(range(1, L + 1), norms, marker="o", markersize=3, linewidth=1.5, color="#4878CF")
    for k_idx, group in enumerate(layer_groups):
        ax.axvspan(group[0] + 1, group[-1] + 1, alpha=0.08, label=f"L{k_idx+1}")
    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("L2 Norm")
    ax.set_title("Per-Layer Hidden State L2 Norm")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)

    # ── Row 1, Col 1: 相邻层 cosine similarity ────────────────────────────
    ax = axes[1, 1]
    sims = [
        F.cosine_similarity(
            per_layer_reps[i].unsqueeze(0),
            per_layer_reps[i + 1].unsqueeze(0),
        ).item()
        for i in range(L - 1)
    ]
    ax.plot(range(1, len(sims) + 1), sims, marker="o", markersize=3, linewidth=1.5, color="#D65F5F")
    ax.set_xlabel("Layer pair (i, i+1)")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Adjacent-Layer Cosine Similarity")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # ── Row 1, Col 2: 金字塔各级 norm 柱状图 ──────────────────────────────
    ax = axes[1, 2]
    pyramid_norms = [r.norm().item() for r in pyramid_reps]
    colors = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#C4AD66", "#77BEDB"]
    ax.bar(
        [f"L{k+1}\n(layers {layer_groups[k][0]+1}–{layer_groups[k][-1]+1})" for k in range(K)],
        pyramid_norms,
        color=colors[:K],
    )
    ax.set_ylabel("L2 Norm")
    ax.set_title("Pyramid Level Representation Norms")
    ax.grid(True, alpha=0.3, axis="y")

    # 隐藏 Row 1 多余格子
    for col in range(3, ncols):
        axes[1, col].axis("off")

    plt.tight_layout()
    save_path = output_dir / "thinkjepa_cifar_result.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    device = choose_device()
    dtype  = choose_dtype(device)
    print(f"[env] device={device}, dtype={dtype}")

    # 1. 随机 CIFAR-10 样本
    image, label, class_en, class_zh, sample_idx = load_random_cifar_sample(DATA_DIR)
    print(f"[cifar] index={sample_idx}, label={label}, class={class_zh} ({class_en})")

    # 2. 构建 prompt
    system_prompt, user_prompt = build_prompt(class_zh, class_en)
    print(f"[prompt] {user_prompt}")

    # 3. 加载 Qwen
    print(f"[model] Loading from {MODEL_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device).eval()

    num_layers  = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    print(f"[model] num_hidden_layers={num_layers}, hidden_size={hidden_size}")

    # 4. 前向传播 + 生成
    print("[qwen] Running forward pass & generation ...")
    reply, hidden_states = qwen_forward_and_generate(
        model, tokenizer, system_prompt, user_prompt, device
    )
    print(f"\n{'='*60}")
    print(f"[Qwen Reply]\n{reply}")
    print(f"{'='*60}\n")
    print(f"[hidden_states] total tensors: {len(hidden_states)} "
          f"(1 embedding + {len(hidden_states)-1} transformer layers)")

    # 5. HPRE: 层级金字塔表征提取
    hpre = HierarchicalPyramidRepresentationExtractor(num_pyramid_levels=4)
    result = hpre.extract(hidden_states)

    print("\n[HPRE] Hierarchical Pyramid Representation Extraction Results:")
    for k, (rep, group) in enumerate(zip(result["pyramid_reps"], result["layer_groups"])):
        print(
            f"  Level {k+1}: Qwen layers {group[0]+1}–{group[-1]+1} "
            f"| rep shape={tuple(rep.shape)} | L2 norm={rep.norm():.4f}"
        )

    # 6. 保存 JSON（只存 norms + meta，向量太大不全存）
    save_meta = {
        "cifar_label":        label,
        "cifar_class_en":     class_en,
        "cifar_class_zh":     class_zh,
        "cifar_sample_idx":   sample_idx,
        "qwen_reply":         reply,
        "num_qwen_layers":    num_layers,
        "hidden_size":        hidden_size,
        "num_pyramid_levels": hpre.num_pyramid_levels,
        "layer_groups":       result["layer_groups"],
        "pyramid_norms":      [r.norm().item() for r in result["pyramid_reps"]],
        "per_layer_norms":    [r.norm().item() for r in result["per_layer_reps"]],
        # 存金字塔表征向量（维度大，可按需注释掉）
        "pyramid_reps": [r.tolist() for r in result["pyramid_reps"]],
    }
    json_path = OUTPUT_DIR / "thinkjepa_cifar_meta.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(save_meta, f, ensure_ascii=False, indent=2)
    print(f"\n[save] Metadata → {json_path}")

    # 7. 可视化
    vis_path = visualize(image, label, class_en, class_zh, reply, result, OUTPUT_DIR)
    print(f"[save] Visualization → {vis_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()

"""
4.17_tunning_qwen_cifar.py

使用 CIFAR-10 数据集对 Qwen3-4B-Instruct 进行简单 SFT (监督微调) Demo
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

思路：
  CIFAR-10 是图像分类数据集，Qwen3 是文本模型。
  我们将图像转化为"像素统计文本描述"作为输入，让模型生成类别标签。

  微调策略（无需 peft 库）：
    - 冻结全部层（36 个 Transformer Block + embedding）
    - 只解冻最后 TRAINABLE_BLOCKS 个 Block + lm_head
    - 用小学习率做 SFT：对 <class_label> completion 部分计算 cross-entropy loss

数据格式（Chat template）：
  <|im_start|>user
  这张图片的像素统计：红={r:.1f} 绿={g:.1f} 蓝={b:.1f} 亮度={brightness:.1f}
  图片类别是什么（从10个CIFAR-10类别中选一个）？<|im_end|>
  <|im_start|>assistant
  {class_name}<|im_end|>

输出：
  - 训练 loss 曲线 PNG
  - 微调前 / 微调后对比 JSON
"""

import json
import os
import pickle
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# ── 路径 ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
MODEL_DIR      = PROJECT_ROOT / "models" / "Qwen3-4B-Instruct-2507"
DATA_DIR       = PROJECT_ROOT / "data" / "cifar-10-batches-py"
OUTPUT_DIR     = PROJECT_ROOT / "src" / "outputs" / "tunning_qwen_cifar"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 超参数 ───────────────────────────────────────────────────────────────────
TRAINABLE_BLOCKS = 2          # 解冻最后 N 个 Transformer Block + lm_head
TRAIN_SAMPLES    = 2000         # 训练集大小（每类约 8 张）
EVAL_SAMPLES     = 200         # 验证 / 对比样本数
N_STEPS          = 100         # 训练步数（每步 batch_size=1）
LR               = 5e-5       # 学习率
MAX_NEW_TOKENS   = 8          # 推理时最大生成 token 数

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
CIFAR10_CLASSES_ZH = [
    "飞机", "汽车", "鸟", "猫", "鹿",
    "狗", "青蛙", "马", "船", "卡车",
]

# ── 设备 ─────────────────────────────────────────────────────────────────────
def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def choose_dtype(device: str):
    """
    MPS 上 float16 可能不稳定；fine-tuning 时 float32 最安全。
    CUDA 用 bfloat16 节省显存。
    """
    if device == "cuda":
        return torch.bfloat16
    return torch.float32   # MPS / CPU

# ── CIFAR-10 加载 ─────────────────────────────────────────────────────────────
def load_cifar_batch(path: Path) -> tuple:
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    images = d[b"data"].reshape(-1, 3, 32, 32).astype(np.float32)
    labels = np.array(d[b"labels"])
    return images, labels


def build_dataset(data_dir: Path, n_train: int, n_eval: int, seed: int = 42):
    """从 CIFAR-10 batch_1 构建文本格式训练 / 验证集"""
    rng = random.Random(seed)
    images, labels = load_cifar_batch(data_dir / "data_batch_1")

    samples = list(zip(images, labels))
    rng.shuffle(samples)

    def img_to_text(img: np.ndarray) -> str:
        """
        将 32×32 RGB 图转成更丰富的像素统计文本描述
        添加颜色比例、绿色比例（区分草地/自然类）、蓝色比例（天空/水面类）
        使特征对部分类别有更强区分力
        """
        r, g, b = img[0], img[1], img[2]
        r_mean, g_mean, b_mean = r.mean(), g.mean(), b.mean()
        total = r_mean + g_mean + b_mean + 1e-6
        r_ratio = r_mean / total          # 红色占比 → 动物/车辆
        g_ratio = g_mean / total          # 绿色占比 → 青蛙/鹿/鸟
        b_ratio = b_mean / total          # 蓝色占比 → 飞机/船
        brightness = total / 3
        contrast = (r.std() + g.std() + b.std()) / 3
        # 边缘密度：粗略用行间差分估计纹理量
        edge = (np.abs(np.diff(r, axis=0)).mean()
                + np.abs(np.diff(g, axis=1)).mean()) / 2
        return (
            f"红色均值={r_mean:.0f} 绿色均值={g_mean:.0f} 蓝色均值={b_mean:.0f} "
            f"红色占比={r_ratio:.2f} 绿色占比={g_ratio:.2f} 蓝色占比={b_ratio:.2f} "
            f"亮度={brightness:.0f} 对比度={contrast:.0f} 边缘密度={edge:.0f}"
        )

    train_data, eval_data = [], []
    for img, label in samples[:n_train]:
        train_data.append({"desc": img_to_text(img), "label": CIFAR10_CLASSES[label], "label_zh": CIFAR10_CLASSES_ZH[label]})
    for img, label in samples[n_train: n_train + n_eval]:
        eval_data.append({"desc": img_to_text(img), "label": CIFAR10_CLASSES[label], "label_zh": CIFAR10_CLASSES_ZH[label]})

    return train_data, eval_data


# ── Prompt 构造 ───────────────────────────────────────────────────────────────
def make_prompt(desc: str) -> str:
    classes_str = "、".join(CIFAR10_CLASSES)
    return (
        f"这张图片的像素统计信息如下：{desc}\n"
        f"请从以下10个CIFAR-10类别中选择一个作为答案（只输出英文类别名）：{classes_str}"
    )


def make_full_text(desc: str, label: str, tokenizer) -> str:
    """构造完整的 Chat 格式字符串（用于训练）"""
    messages = [
        {"role": "user",      "content": make_prompt(desc)},
        {"role": "assistant", "content": label},
    ]
    # apply_chat_template 不加 generation_prompt（训练时需要完整序列）
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def make_query_text(desc: str, tokenizer) -> str:
    """只有 user 部分 + generation_prompt（用于推理）
    Qwen3 支持 enable_thinking=False 来关闭思考模式，避免生成 <think> token 干扰类别预测
    """
    messages = [
        {"role": "user", "content": make_prompt(desc)},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


# ── 模型冻结策略 ──────────────────────────────────────────────────────────────
def freeze_model(model, n_trainable_blocks: int):
    """
    冻结除最后 n_trainable_blocks 个 Block 和 lm_head 之外的所有参数
    """
    # 全部冻结
    for p in model.parameters():
        p.requires_grad_(False)

    total_layers = model.config.num_hidden_layers   # 36
    trainable_from = total_layers - n_trainable_blocks

    # 解冻最后 n 个 block
    for i, layer in enumerate(model.model.layers):
        if i >= trainable_from:
            for p in layer.parameters():
                p.requires_grad_(True)

    # 解冻 lm_head
    for p in model.lm_head.parameters():
        p.requires_grad_(True)

    # 统计
    total  = sum(p.numel() for p in model.parameters())
    active = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[freeze] 总参数: {total/1e6:.1f}M  可训练: {active/1e6:.1f}M ({100*active/total:.2f}%)")


# ── 计算单条样本的 SFT Loss ────────────────────────────────────────────────────
def compute_sft_loss(model, tokenizer, desc: str, label: str, device: str):
    """
    正确的 SFT Loss：分别 tokenize prompt 和 completion，再拼接。

    之前的 bug：用 rfind 截断字符串后单独 tokenize，BPE 在字符串边界处
    会产生不同的合并结果（例如末尾的空格或换行会影响相邻 token 的合并），
    导致 prompt_len 偏差，mask 打在错误位置，loss 飙升甚至为负。

    正确做法：
      prompt_ids     = tokenize(query_with_generation_prompt)
      completion_ids = tokenize(label) + [<|im_end|>]
      input_ids      = concat(prompt_ids, completion_ids)
      labels         = input_ids; labels[:prompt_len] = -100
    这样两段各自独立 tokenize，拼接后不存在边界歧义。
    """
    # ── prompt 部分（以 <|im_start|>assistant\n 结尾）──────────────────────
    prompt_text = make_query_text(desc, tokenizer)
    prompt_ids  = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False
    )["input_ids"]           # (1, prompt_len)

    # ── completion 部分（label token + <|im_end|>）────────────────────────
    label_ids  = tokenizer(
        label, return_tensors="pt", add_special_tokens=False
    )["input_ids"]           # (1, n_label_tokens)

    im_end_id  = tokenizer.convert_tokens_to_ids("<|im_end|>")
    im_end_ids = torch.tensor([[im_end_id]])   # (1, 1)

    completion_ids = torch.cat([label_ids, im_end_ids], dim=1)  # (1, n+1)

    # ── 拼接，构造 labels（mask prompt）──────────────────────────────────
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1).to(device)
    prompt_len = prompt_ids.shape[1]

    labels = input_ids.clone()
    labels[:, :prompt_len] = -100    # 只对 completion 部分计算 loss

    outputs = model(input_ids=input_ids, labels=labels)
    return outputs.loss


# ── 推理（生成类别标签）──────────────────────────────────────────────────────
@torch.inference_mode()
def predict(model, tokenizer, desc: str, device: str) -> str:
    query = make_query_text(desc, tokenizer)
    enc   = tokenizer(query, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(device)

    out = model.generate(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id,
    )
    # 只解码新生成的部分
    new_ids = out[0, input_ids.shape[1]:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return text


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    device = choose_device()
    dtype  = choose_dtype(device)
    print(f"[env] device={device}  dtype={dtype}")

    # ── 1. 加载数据集 ────────────────────────────────────────────────────────
    print(f"[data] 构建数据集 train={TRAIN_SAMPLES} eval={EVAL_SAMPLES} ...")
    train_data, eval_data = build_dataset(DATA_DIR, TRAIN_SAMPLES, EVAL_SAMPLES)
    print(f"[data] 训练样本示例:")
    s = train_data[0]
    print(f"  desc : {s['desc']}")
    print(f"  label: {s['label']} ({s['label_zh']})")

    # ── 2. 加载 tokenizer & 模型 ─────────────────────────────────────────────
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        sys.exit("请在 wm conda 环境中运行：conda run -n wm python src/4.17_tunning_qwen_cifar.py")

    print(f"[model] 加载 tokenizer from {MODEL_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[model] 加载模型 (dtype={dtype}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)

    # ── 3. 冻结策略 ──────────────────────────────────────────────────────────
    freeze_model(model, TRAINABLE_BLOCKS)

    # ── 4. 微调前推理（before）──────────────────────────────────────────────
    print("\n[eval] 微调前推理样本 ...")
    model.eval()
    before_results = []
    for i, sample in enumerate(eval_data[:5]):
        pred = predict(model, tokenizer, sample["desc"], device)
        correct = sample["label"].lower() in pred.lower()
        before_results.append({
            "idx": i, "label": sample["label"],
            "pred_before": pred, "correct_before": correct
        })
        print(f"  [{i}] GT={sample['label']:12s}  Pred={pred!r:20s}  {'✓' if correct else '✗'}")

    # ── 5. 训练循环 ──────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=0.01
    )

    # 按类别分组，保证训练时各类别均匀采样，防止 mode collapse
    from collections import defaultdict
    class_buckets: dict = defaultdict(list)
    for s in train_data:
        class_buckets[s["label"]].append(s)
    classes_list = list(class_buckets.keys())

    loss_history = []
    print(f"\n[train] 开始训练 {N_STEPS} 步 (均匀类别采样) ...")

    model.train()
    rng_train = random.Random(0)
    for step in range(N_STEPS):
        # 均匀轮转类别，再在类内随机选一个样本
        cls    = classes_list[step % len(classes_list)]
        sample = rng_train.choice(class_buckets[cls])

        optimizer.zero_grad()
        loss = compute_sft_loss(model, tokenizer, sample["desc"], sample["label"], device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if (step + 1) % 10 == 0 or step == 0:
            print(f"  step {step+1:3d}/{N_STEPS}  loss={loss_val:.4f}  label={sample['label']}")

    # ── 6. 微调后推理（after）───────────────────────────────────────────────
    print("\n[eval] 微调后推理样本 ...")
    model.eval()
    after_results = []
    for i, sample in enumerate(eval_data[:5]):
        pred = predict(model, tokenizer, sample["desc"], device)
        correct = sample["label"].lower() in pred.lower()
        after_results.append({
            "idx": i, "label": sample["label"],
            "pred_after": pred, "correct_after": correct
        })
        print(f"  [{i}] GT={sample['label']:12s}  Pred={pred!r:20s}  {'✓' if correct else '✗'}")

    # ── 7. 保存结果 ──────────────────────────────────────────────────────────
    # 7a. 损失曲线
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, N_STEPS + 1), loss_history, marker="o", markersize=3)
    ax.set_xlabel("Step")
    ax.set_ylabel("SFT Loss (CE)")
    ax.set_title(f"Qwen3-4B SFT on CIFAR-10 text prompts\n"
                 f"(trainable blocks={TRAINABLE_BLOCKS}, lr={LR})")
    ax.grid(True, alpha=0.3)
    loss_png = OUTPUT_DIR / "loss_curve.png"
    fig.savefig(loss_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[save] 损失曲线 → {loss_png}")

    # 7b. before/after 对比 JSON
    comparison = []
    for b, a in zip(before_results, after_results):
        comparison.append({
            "label":          b["label"],
            "pred_before":    b["pred_before"],
            "correct_before": b["correct_before"],
            "pred_after":     a["pred_after"],
            "correct_after":  a["correct_after"],
        })
    result_json = OUTPUT_DIR / "comparison.json"
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "trainable_blocks": TRAINABLE_BLOCKS,
                "train_samples": TRAIN_SAMPLES,
                "n_steps": N_STEPS,
                "lr": LR,
            },
            "loss_history": loss_history,
            "comparison": comparison,
        }, f, ensure_ascii=False, indent=2)
    print(f"[save] 对比结果 → {result_json}")

    # 7c. 打印汇总
    acc_before = sum(r["correct_before"] for r in before_results) / len(before_results)
    acc_after  = sum(r["correct_after"]  for r in after_results)  / len(after_results)
    print(f"\n{'─'*50}")
    print(f"微调前准确率 (5 samples): {acc_before*100:.0f}%")
    print(f"微调后准确率 (5 samples): {acc_after*100:.0f}%")
    print(f"Loss: {loss_history[0]:.4f} → {loss_history[-1]:.4f}")
    print(f"{'─'*50}")
    print("Done. 输出文件位于:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

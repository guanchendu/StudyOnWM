import torch
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
from utils import get_img_preprocessor, get_column_normalizer

# ── 1. 基础配置 ──────────────────────────────────────────
DATASET_NAME  = "tworoom"       # 对应 tworoom.h5
KEYS_TO_LOAD  = ["pixels", "action", "proprio"]
KEYS_TO_CACHE = ["action", "proprio"]
NUM_STEPS     = 4               # history_size(3) + num_preds(1)
FRAMESKIP     = 5               # 每隔5帧取一帧
IMG_SIZE      = 224
BATCH_SIZE    = 8
TRAIN_SPLIT   = 0.9
SEED          = 42

# ── 2. 创建原始数据集 ─────────────────────────────────────
dataset = swm.data.HDF5Dataset(
    name=DATASET_NAME,
    keys_to_load=KEYS_TO_LOAD,
    keys_to_cache=KEYS_TO_CACHE,
    num_steps=NUM_STEPS,
    frameskip=FRAMESKIP,
    transform=None,             # 先不挂transform，后面再加
    cache_dir="/Users/guanchendu/NJUST/PhD_0/Code/le-wm-main",

)

print(f"数据集大小: {len(dataset)} 个样本")
print(f"字段列表:   {dataset.column_names}")

# ── 3. 构建预处理 Transform ──────────────────────────────
transforms = []

# 3a. 图像：Resize(224) + ImageNet 归一化
img_transform = get_img_preprocessor(
    source='pixels', target='pixels', img_size=IMG_SIZE
)
transforms.append(img_transform)

# 3b. 非图像字段：Z-Score 标准化
for col in KEYS_TO_LOAD:
    if col.startswith("pixels"):
        continue
    normalizer = get_column_normalizer(dataset, col, col)
    transforms.append(normalizer)

# 3c. 合并所有 transform
dataset.transform = spt.data.transforms.Compose(*transforms)

# ── 4. 划分训练集 / 验证集 ────────────────────────────────
rnd_gen = torch.Generator().manual_seed(SEED)
train_set, val_set = spt.data.random_split(
    dataset,
    lengths=[TRAIN_SPLIT, 1 - TRAIN_SPLIT],
    generator=rnd_gen,
)
print(f"训练集: {len(train_set)} 条  验证集: {len(val_set)} 条")

# ── 5. 创建 DataLoader ────────────────────────────────────
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    generator=rnd_gen,
)
val_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
)

# ── 6. 取一个 batch 看看形状 ──────────────────────────────
batch = next(iter(train_loader))

print("\n── batch 内容 ──")
for key, val in batch.items():
    print(f"  {key:10s}: shape={val.shape}, dtype={val.dtype}")

# 预期输出：
# pixels    : shape=(8, 4, 3, 224, 224), dtype=float32   ← (B, T, C, H, W)
# action    : shape=(8, 4, 10),          dtype=float32   ← (B, T, frameskip*2)
# proprio   : shape=(8, 4, D),           dtype=float32   ← (B, T, proprio_dim)

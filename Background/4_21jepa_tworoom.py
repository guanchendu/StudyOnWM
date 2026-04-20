r"""
Simple JEPA world model for the TwoRoom dataset.

思路:
    - Context encoder 对 o_t 编码得到 z_t
    - Target encoder 是 context encoder 的 EMA 拷贝 (stop-gradient), 对 o_{t+1}
      编码得到 \hat z_{t+1}
    - Predictor 接受 [z_t, a_t, p_t] 预测 \tilde z_{t+1}
    - 损失在 latent 空间: smooth_l1(\tilde z_{t+1}, stopgrad(\hat z_{t+1}))
    - 通过 EMA target + 轻量 VICReg 方差/协方差正则防止 representation collapse

训练 + 测试都在本脚本里:
    - python 4_21jepa_tworoom.py --mode train   : 训练并保存最佳 checkpoint
    - python 4_21jepa_tworoom.py --mode test    : 加载 checkpoint 做单步 / 多步 rollout 评估
    - python 4_21jepa_tworoom.py --mode all     : 先 train 再 test
"""

import argparse
import copy
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_column_normalizer, get_img_preprocessor


DEFAULT_CACHE_DIR = "/Users/guanchendu/Code/StudyOnWM/data"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "jepa_tworoom"
CKPT_PATH = OUTPUT_DIR / "jepa_best.pt"


# ── 数据 ──────────────────────────────────────────────────────────────────
def build_dataset(cache_dir, img_size, num_steps, frameskip):
    keys_to_load = ["pixels", "action", "proprio"]
    keys_to_cache = ["action", "proprio"]

    dataset = swm.data.HDF5Dataset(
        name="tworoom",
        keys_to_load=keys_to_load,
        keys_to_cache=keys_to_cache,
        num_steps=num_steps,
        frameskip=frameskip,
        transform=None,
        cache_dir=cache_dir,
    )

    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=img_size)]
    for col in ("action", "proprio"):
        transforms.append(get_column_normalizer(dataset, col, col))

    dataset.transform = spt.data.transforms.Compose(*transforms)
    return dataset


def build_loaders(args):
    dataset = build_dataset(args.cache_dir, args.img_size, args.num_steps, args.frameskip)
    rnd_gen = torch.Generator().manual_seed(args.seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[args.train_split, 1 - args.train_split],
        generator=rnd_gen,
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        generator=rnd_gen,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )
    return dataset, train_loader, val_loader


# ── 模型 ──────────────────────────────────────────────────────────────────
class ConvEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(128, 128, 4, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self, latent_dim, action_dim, proprio_dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim + proprio_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z, a, p):
        return self.net(torch.cat([z, a, p], dim=-1))


class JEPAWorldModel(nn.Module):
    def __init__(self, action_dim, proprio_dim, latent_dim=256, ema_tau=0.996):
        super().__init__()
        self.latent_dim = latent_dim
        self.ema_tau = ema_tau

        self.encoder = ConvEncoder(latent_dim)
        self.predictor = Predictor(latent_dim, action_dim, proprio_dim)

        # target encoder: EMA copy, no gradient
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_target(self):
        tau = self.ema_tau
        for p_tgt, p_src in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            p_tgt.data.mul_(tau).add_(p_src.data, alpha=1 - tau)
        for b_tgt, b_src in zip(self.target_encoder.buffers(), self.encoder.buffers()):
            b_tgt.data.copy_(b_src.data)

    def encode(self, pixels):
        return self.encoder(pixels)

    @torch.no_grad()
    def encode_target(self, pixels):
        return self.target_encoder(pixels)

    def predict(self, z, a, p):
        return self.predictor(z, a, p)


# ── 损失: JEPA + 轻量 VICReg 防塌缩 ──────────────────────────────────────
def vicreg_regularization(z, eps=1e-4):
    """只要求 std≥1 并惩罚协方差非对角元，鼓励表示有方差、各维度解相关。"""
    z = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0) + eps)
    std_loss = F.relu(1.0 - std).mean()

    n, d = z.shape
    cov = (z.T @ z) / max(n - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = (off_diag ** 2).sum() / d
    return std_loss, cov_loss


def flatten_pairs(batch):
    """(B, T, ...) → 展平成 (B*(T-1), ...) 的 (t, t+1) 对."""
    pixels = batch["pixels"].float()
    action = batch["action"].float()
    proprio = batch["proprio"].float()
    return {
        "pixels_t":   pixels[:, :-1].reshape(-1, *pixels.shape[2:]),
        "pixels_tp1": pixels[:, 1:].reshape(-1, *pixels.shape[2:]),
        "action_t":   action[:, :-1].reshape(-1, action.shape[-1]),
        "proprio_t":  proprio[:, :-1].reshape(-1, proprio.shape[-1]),
    }


# ── 训练 / 验证 epoch ─────────────────────────────────────────────────────
def run_epoch(model, loader, device, optimizer, std_w, cov_w):
    is_train = optimizer is not None
    model.train(is_train)

    agg = {"loss": 0.0, "pred_loss": 0.0, "std_loss": 0.0, "cov_loss": 0.0,
           "cos_pred_tgt": 0.0, "cos_id_tgt": 0.0}
    total = 0

    for raw in loader:
        batch = flatten_pairs(raw)
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.set_grad_enabled(is_train):
            z_t = model.encode(batch["pixels_t"])
            z_pred = model.predict(z_t, batch["action_t"], batch["proprio_t"])
            with torch.no_grad():
                z_tgt = model.encode_target(batch["pixels_tp1"])

            pred_loss = F.smooth_l1_loss(z_pred, z_tgt)
            std_loss, cov_loss = vicreg_regularization(z_pred)
            loss = pred_loss + std_w * std_loss + cov_w * cov_loss

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                model.update_target()

        bs = batch["pixels_t"].shape[0]
        total += bs
        agg["loss"]        += loss.item() * bs
        agg["pred_loss"]   += pred_loss.item() * bs
        agg["std_loss"]    += std_loss.item() * bs
        agg["cov_loss"]    += cov_loss.item() * bs
        # 额外: pred vs target 的余弦 相似度, 以及 identity baseline (z_t vs z_{t+1}) 的余弦
        with torch.no_grad():
            agg["cos_pred_tgt"] += F.cosine_similarity(z_pred, z_tgt).mean().item() * bs
            agg["cos_id_tgt"]   += F.cosine_similarity(z_t,    z_tgt).mean().item() * bs

    return {k: v / total for k, v in agg.items()}


# ── 多步 rollout 评估 ─────────────────────────────────────────────────────
@torch.no_grad()
def rollout_eval(model, loader, device, max_steps=3):
    """从 o_0 开始, 仅用 predictor 在 latent 空间滚动 k 步, 与 target encoder 对比。"""
    model.eval()
    per_step = {k: {"pred_loss": 0.0, "cos_pred_tgt": 0.0, "cos_id_tgt": 0.0, "n": 0}
                for k in range(1, max_steps + 1)}

    for raw in loader:
        pixels  = raw["pixels"].float().to(device)     # (B, T, C, H, W)
        action  = raw["action"].float().to(device)     # (B, T, A)
        proprio = raw["proprio"].float().to(device)    # (B, T, P)
        B, T = pixels.shape[:2]
        k = min(max_steps, T - 1)

        z_0 = model.encode(pixels[:, 0])               # 初始 context
        z_id = z_0.clone()                              # identity baseline
        z = z_0
        for step in range(1, k + 1):
            z = model.predict(z, action[:, step - 1], proprio[:, step - 1])
            z_tgt = model.encode_target(pixels[:, step])

            per_step[step]["pred_loss"]    += F.smooth_l1_loss(z, z_tgt).item() * B
            per_step[step]["cos_pred_tgt"] += F.cosine_similarity(z,   z_tgt).mean().item() * B
            per_step[step]["cos_id_tgt"]   += F.cosine_similarity(z_id, z_tgt).mean().item() * B
            per_step[step]["n"]            += B

    out = {}
    for step, s in per_step.items():
        if s["n"] == 0:
            continue
        out[step] = {
            "pred_loss":    s["pred_loss"]    / s["n"],
            "cos_pred_tgt": s["cos_pred_tgt"] / s["n"],
            "cos_id_tgt":   s["cos_id_tgt"]   / s["n"],
        }
    return out


# ── CLI ───────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Simple JEPA world model on tworoom.")
    p.add_argument("--mode", choices=["train", "test", "all"], default="all")
    p.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--num-steps", type=int, default=4)
    p.add_argument("--frameskip", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--ema-tau", type=float, default=0.996)
    p.add_argument("--std-weight", type=float, default=1.0)
    p.add_argument("--cov-weight", type=float, default=0.04)
    p.add_argument("--train-split", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--rollout-steps", type=int, default=3)
    return p.parse_args()


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model_from_sample(sample, args, device):
    action_dim = sample["action"].shape[-1]
    proprio_dim = sample["proprio"].shape[-1]
    model = JEPAWorldModel(
        action_dim=action_dim,
        proprio_dim=proprio_dim,
        latent_dim=args.latent_dim,
        ema_tau=args.ema_tau,
    ).to(device)
    return model, action_dim, proprio_dim


def train(args, device):
    dataset, train_loader, val_loader = build_loaders(args)
    sample = next(iter(train_loader))
    model, action_dim, proprio_dim = build_model_from_sample(sample, args, device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )

    print("=" * 72)
    print("JEPA World Model on TwoRoom")
    print("=" * 72)
    print(f"device       : {device}")
    print(f"dataset size : {len(dataset)}")
    print(f"pixels shape : {tuple(sample['pixels'].shape)}")
    print(f"action dim   : {action_dim}  proprio dim: {proprio_dim}")
    print(f"num params   : {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 72)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, device, optimizer, args.std_weight, args.cov_weight)
        va = run_epoch(model, val_loader,   device, None,      args.std_weight, args.cov_weight)

        print(
            f"[Epoch {epoch:03d}] "
            f"tr_loss={tr['loss']:.4f} tr_pred={tr['pred_loss']:.4f} "
            f"tr_std={tr['std_loss']:.4f} tr_cov={tr['cov_loss']:.4f} | "
            f"va_pred={va['pred_loss']:.4f} "
            f"cos(pred,tgt)={va['cos_pred_tgt']:.3f} "
            f"cos(id,tgt)={va['cos_id_tgt']:.3f}"
        )

        if va["pred_loss"] < best_val:
            best_val = va["pred_loss"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": vars(args),
                "action_dim": action_dim,
                "proprio_dim": proprio_dim,
                "best_val_pred_loss": best_val,
            }, CKPT_PATH)

    print(f"best checkpoint -> {CKPT_PATH}")


def test(args, device):
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"checkpoint not found: {CKPT_PATH}. 先运行 --mode train")

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    _, _, val_loader = build_loaders(args)

    model = JEPAWorldModel(
        action_dim=ckpt["action_dim"],
        proprio_dim=ckpt["proprio_dim"],
        latent_dim=ckpt["config"]["latent_dim"],
        ema_tau=ckpt["config"]["ema_tau"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("=" * 72)
    print("JEPA Test (单步 + 多步 latent rollout)")
    print("=" * 72)
    print(f"checkpoint   : {CKPT_PATH}")
    print(f"best val loss: {ckpt.get('best_val_pred_loss', float('nan')):.4f}")

    single = run_epoch(model, val_loader, device, None, args.std_weight, args.cov_weight)
    print(
        f"[single-step] pred_loss={single['pred_loss']:.4f} "
        f"cos(pred,tgt)={single['cos_pred_tgt']:.3f} "
        f"cos(id,tgt)={single['cos_id_tgt']:.3f} "
        f"(cos 越接近 1 越好, identity 是无预测 baseline)"
    )

    rollout = rollout_eval(model, val_loader, device, max_steps=args.rollout_steps)
    for step, m in rollout.items():
        print(
            f"[rollout k={step}] pred_loss={m['pred_loss']:.4f} "
            f"cos(pred,tgt)={m['cos_pred_tgt']:.3f} "
            f"cos(id,tgt)={m['cos_id_tgt']:.3f}"
        )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = pick_device()

    if args.mode in ("train", "all"):
        train(args, device)
    if args.mode in ("test", "all"):
        test(args, device)


if __name__ == "__main__":
    main()

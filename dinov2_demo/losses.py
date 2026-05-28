"""DINOv2 的三个损失:

1. DINO loss   —— image-level self-distillation (CLS token)
2. iBOT loss   —— patch-level masked prediction
3. KoLeo loss  —— 鼓励 batch 内 embedding 均匀分布的正则项
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    """teacher 输出做 center + sharpen，student 输出与 teacher 算交叉熵。

    多视图: student 看全部 (2 global + N local)，teacher 只看 2 global。
    Loss 在 (teacher_view, student_view) 配对上求和，跳过相同视图的配对。
    """

    def __init__(self, out_dim: int, teacher_temp: float = 0.04, student_temp: float = 0.1, center_momentum: float = 0.9):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    @torch.no_grad()
    def update_center(self, teacher_logits: torch.Tensor) -> None:
        # teacher_logits: (n_global * B, D)
        batch_center = teacher_logits.mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def forward(self, student_logits_list: list[torch.Tensor], teacher_logits_list: list[torch.Tensor]) -> torch.Tensor:
        # teacher: center + sharpen + softmax，detach
        teacher_probs = [F.softmax((t - self.center) / self.teacher_temp, dim=-1).detach() for t in teacher_logits_list]
        student_log_probs = [F.log_softmax(s / self.student_temp, dim=-1) for s in student_logits_list]

        total, n_terms = 0.0, 0
        for ti, t_p in enumerate(teacher_probs):
            for si, s_lp in enumerate(student_log_probs):
                if ti == si:  # 同一视图不算
                    continue
                total = total - (t_p * s_lp).sum(dim=-1).mean()
                n_terms += 1

        # 更新 center 用 batch 内所有 teacher 输出
        self.update_center(torch.cat(teacher_logits_list, dim=0))
        return total / max(n_terms, 1)


class iBOTLoss(nn.Module):
    """masked image modeling on patch tokens: teacher 看完整图，student 看 masked 图。

    只在被 mask 的位置上算 cross-entropy。
    """

    def __init__(self, out_dim: int, teacher_temp: float = 0.07, student_temp: float = 0.1, center_momentum: float = 0.9):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, 1, out_dim))

    @torch.no_grad()
    def update_center(self, teacher_logits: torch.Tensor) -> None:
        # teacher_logits: (B, N, D) -> 沿 B,N 平均
        batch_center = teacher_logits.mean(dim=(0, 1), keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def forward(self, student_patch_logits: torch.Tensor, teacher_patch_logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            student_patch_logits, teacher_patch_logits: (B, N, D)
            masks: (B, N) bool — True 处计算 loss
        """
        teacher_p = F.softmax((teacher_patch_logits - self.center) / self.teacher_temp, dim=-1).detach()
        student_lp = F.log_softmax(student_patch_logits / self.student_temp, dim=-1)
        per_token = -(teacher_p * student_lp).sum(dim=-1)  # (B, N)

        masks_f = masks.float()
        denom = masks_f.sum().clamp(min=1.0)
        loss = (per_token * masks_f).sum() / denom

        self.update_center(teacher_patch_logits.detach())
        return loss


class KoLeoLoss(nn.Module):
    """Kozachenko–Leonenko 熵估计的近似: 鼓励 batch 内每个样本与最近邻拉开距离。"""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (B, C)
        x = F.normalize(features, dim=-1, p=2)
        sim = x @ x.t()                                # (B, B)
        sim.fill_diagonal_(-2.0)                       # 排除自己
        nn_sim = sim.max(dim=-1).values                # 最近邻余弦相似度
        nn_dist = torch.sqrt(torch.clamp(2 - 2 * nn_sim, min=self.eps))
        return -torch.log(nn_dist + self.eps).mean()

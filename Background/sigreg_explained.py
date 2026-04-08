import torch
from torch import nn


class SIGRegExplained(nn.Module):
    """
    Sketch Isotropic Gaussian Regularizer 的可读版实现。

    这个模块的目标不是直接做预测，而是约束 latent embedding 的分布：
    希望 embedding 在整体上更接近“各向同性高斯分布”。

    直观理解：
    1. 先把高维向量随机投影到很多个一维方向上。
    2. 如果原始高维分布真的接近标准高斯，那么这些一维投影也应该接近一维高斯。
    3. 于是我们比较“投影后的经验分布”与“标准高斯的理论特征函数”之间的差距。
    4. 差距越大，regularization loss 越大。

    输入:
        proj: shape = (T, B, D)
            T: 时间维
            B: batch 维
            D: embedding 维度
    输出:
        一个标量 loss
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        """
        参数:
            knots:
                在积分区间 [0, 3] 上取多少个离散采样点。
                这些点用于近似计算 Epps-Pulley 统计量中的积分。

            num_proj:
                每次 forward 随机采样多少个投影方向。
                数值越大，对分布的刻画越稳定，但计算也更贵。
        """
        super().__init__()
        self.num_proj = num_proj

        # 在 [0, 3] 上均匀采样若干个点。
        # 这些 t 点是后面计算特征函数 phi(t) 时使用的离散采样位置。
        t = torch.linspace(0, 3, knots, dtype=torch.float32)

        # 采样点间隔，用于数值积分近似。
        dt = 3 / (knots - 1)

        # 使用梯形积分法的权重：
        # 中间点权重大约是 2*dt，首尾点权重减半为 dt。
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt

        # 高斯窗 exp(-t^2 / 2)。
        # 对标准高斯 N(0,1)，它的特征函数正好就是 exp(-t^2 / 2)。
        window = torch.exp(-t.square() / 2.0)

        # register_buffer 表示：
        # 这些张量属于模块状态，会随着 .to(device) 一起搬运，
        # 但它们不是可学习参数，不会被优化器更新。
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """
        计算 SIGReg 正则项。

        参数:
            proj:
                shape = (T, B, D)
                一般可以理解为某个时间序列 batch 的 embedding。

        返回:
            标量张量，表示当前 batch 的正则化损失。
        """
        if proj.ndim != 3:
            raise ValueError(f"Expected proj to have shape (T, B, D), got {tuple(proj.shape)}")

        _, batch_size, dim = proj.shape

        # ------------------------------------------------------------
        # 1. 采样随机投影方向 A
        # ------------------------------------------------------------
        # A 的 shape = (D, P)
        # P = num_proj，表示采样了多少个随机方向。
        #
        # 每一列都是一个 D 维随机方向。
        # 后面 proj @ A 之后，就相当于把每个 D 维 embedding
        # 投影到了这 P 个一维方向上。
        A = torch.randn(dim, self.num_proj, device=proj.device, dtype=proj.dtype)

        # 把每个投影方向归一化成单位向量，避免不同方向长度不同。
        A = A / A.norm(p=2, dim=0, keepdim=True).clamp_min(1e-12)

        # ------------------------------------------------------------
        # 2. 做随机投影
        # ------------------------------------------------------------
        # proj: (T, B, D)
        # A:    (D, P)
        # 结果: (T, B, P)
        #
        # 含义：
        # 每个时间步、每个样本、每个随机方向，都会得到一个标量投影值。
        projected = proj @ A

        # ------------------------------------------------------------
        # 3. 在每个 t 采样点上估计经验特征函数
        # ------------------------------------------------------------
        # projected.unsqueeze(-1): (T, B, P, 1)
        # self.t:                  (K,)
        # 广播后得到:              (T, B, P, K)
        #
        # 这里的 x_t 对应投影值乘以不同的 t。
        x_t = projected.unsqueeze(-1) * self.t

        # 对经验特征函数 E[e^{itX}] 做实部/虚部分解：
        # E[cos(tX)] + i E[sin(tX)]
        #
        # mean(dim=1) 是对 batch 维做平均，近似经验期望。
        empirical_cos = x_t.cos().mean(dim=1)  # (T, P, K)
        empirical_sin = x_t.sin().mean(dim=1)  # (T, P, K)

        # ------------------------------------------------------------
        # 4. 与标准高斯的理论特征函数比较
        # ------------------------------------------------------------
        # 标准高斯的一维特征函数:
        #   phi(t) = exp(-t^2 / 2)
        #
        # 它是纯实数，所以目标虚部应接近 0。
        # 因此误差分成两部分：
        #   (经验实部 - 理论实部)^2 + (经验虚部 - 0)^2
        err = (empirical_cos - self.phi).square() + empirical_sin.square()

        # ------------------------------------------------------------
        # 5. 对 t 做加权积分近似
        # ------------------------------------------------------------
        # err:      (T, P, K)
        # weights:  (K,)
        # 输出:     (T, P)
        #
        # 这一步把离散采样点上的误差压缩成每个时间步、每个投影方向上的统计量。
        statistic = err @ self.weights

        # 原始实现中再乘以 batch_size，保持与经验统计量尺度一致。
        statistic = statistic * batch_size

        # 最后对所有时间步和所有投影方向取平均，得到一个标量损失。
        return statistic.mean()


if __name__ == "__main__":
    # 一个简单的可运行示例，便于你单独理解这个模块的输入输出。
    torch.manual_seed(0)

    reg = SIGRegExplained(knots=17, num_proj=128)

    # 假设有:
    # T = 4 个时间步
    # B = 8 个样本
    # D = 192 维 embedding
    dummy_proj = torch.randn(4, 8, 192)

    loss = reg(dummy_proj)
    print("SIGReg loss:", float(loss))

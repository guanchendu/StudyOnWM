### LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels



这篇论文针对的是 **JEPA（Joint-Embedding Predictive Architecture）做 world model 时的表征坍缩问题**。

**DINO-WM**（冻结预训练编码器）：用 DINOv2 这种大模型提供表征，不训练编码器所以不会坍缩。但每帧要编码成大量 token，规划非常慢（47 秒）

**PLDM**（端到端，但很复杂）：用 7 个 loss 项、6 个超参数来防止坍缩，调参困难，训练不稳定

**Dreamer / TD-MPC**（任务相关）：需要奖励信号或特权状态信息，不是通用的 world model



## LeWM 提出了什么方法

**核心贡献：只用两个 loss 就实现端到端稳定训练**，把超参数从 6 个减到 1 个。

两个 loss 分别是：

1. **预测损失 L_pred**：标准的 MSE，让 predictor 预测的下一帧隐表征接近编码器编码的真实下一帧
2. **SIGReg 正则项**：强制隐空间的分布接近各向同性高斯分布，从数学上保证不会坍缩
3. SIGReg 的直觉：如果所有表征都坍缩到一个点，它们在任意方向上的投影都不会是高斯分布（会是一个尖峰）。所以强制"投影后必须像高斯"就等于禁止坍缩。



原始像素 o_t → [Encoder (ViT-Tiny, ~5M参数)] → 隐表征 z_t (192维, 单个token)
                                                        ↓
                                            z_t + 动作 a_t
                                                        ↓
                                         [Predictor (ViT-Small, ~10M参数)]
                                                        ↓
                                            预测的下一状态 ẑ_{t+1}
                                                        ↓
                              L = MSE(ẑ_{t+1}, z_{t+1}) + λ·SIGReg(Z)

规划阶段（测试时）

1. 编码当前观测：z_start = Encoder(当前图像)
2. 编码目标观测：z_goal = Encoder(目标图像)
3. CEM 优化：
   - 随机采样 300 条动作序列
   - 每条在 predictor 里 rollout：z_0 → z_1 → ... → z_H
   - 评估终点 z_H 与 z_goal 的距离
   - 筛选 top-K 精英，更新采样分布
   - 重复 ~30 轮
4. MPC：只执行前几步动作，拿到新观测后重新规划


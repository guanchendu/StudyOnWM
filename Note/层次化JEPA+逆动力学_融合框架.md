# 层次化 JEPA + 逆动力学 融合框架

## 1. 动机

| 问题 | 现有 LeWM 的局限 | 我们的解决方案 |
|------|----------------|--------------|
| 长 horizon | 只能 5 步，自回归误差累积 | 粗细粒度层次化预测 |
| 动作标注依赖 | 需要完整的 (o_t, a_t, o_{t+1}) | 逆动力学自监督学 latent action |

两个问题合在一起解决，形成一个统一框架：**大量无标注视频预训练 + 少量有标注数据对齐 + 层次化规划**。

---

## 2. 总体架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│   阶段一：自监督预训练（大量无标注视频，不需要任何动作标签）               │
│                                                                      │
│   ┌──────────┐      ┌──────────────────┐      ┌──────────┐          │
│   │ Encoder  │      │ Inverse Dynamics │      │ Target   │          │
│   │          │─z_t─→│    Model         │←z_{t+1}│ Encoder  │          │
│   │ (online) │      │  (z_t,z_{t+1})→â_t     │  (EMA)   │          │
│   └──────────┘      └────────┬─────────┘      └──────────┘          │
│        │                     │ â_t                    │              │
│        │                     ↓                        │              │
│        │         ┌───────────────────────┐            │              │
│        │         │   Coarse Predictor    │            │              │
│        z_0──────→│ z_0 + â_{0:h} → ẑ_h  │   z_h^target ← 监督      │
│        │         └───────────┬───────────┘            │              │
│        │                     │ ẑ_h (condition)        │              │
│        │                     ↓                        │              │
│        │         ┌───────────────────────┐            │              │
│        │         │    Fine Predictor     │            │              │
│        z_0──────→│ z_0+â_0+cond(ẑ_h)    │   z_1^target ← 监督      │
│        │         │      → ẑ_1            │            │              │
│        │         └───────────────────────┘            │              │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
                                ↓
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│   阶段二：动作空间对齐（少量有标注数据）                                │
│                                                                      │
│   冻结阶段一全部参数，只训练：                                         │
│                                                                      │
│   ┌────────────────┐        ┌────────────────┐                      │
│   │ Action Encoder │        │ Action Decoder │                      │
│   │  a_t → ã_t     │        │  â_t → a_t     │                      │
│   └────────────────┘        └────────────────┘                      │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
                                ↓
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│   规划（Inference）                                                   │
│                                                                      │
│   CEM采样真实动作 → Action Encoder → latent动作                      │
│   → Coarse + Fine Rollout → cost                                    │
│   → CEM更新 → Action Decoder → 执行真实动作                          │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. 阶段一：自监督预训练

### 3.1 数据要求

纯视频序列，不需要动作标签：

```
数据: {o_0, o_1, o_2, ..., o_T}   ← 只有观测帧
不需要: {a_0, a_1, ...}            ← 不需要动作
```

### 3.2 各模块详细设计

#### Encoder + Target Encoder

和层次化文档一致：

```
Encoder:        o_t → z_t  (B, num_tokens, D)    ← 参与梯度更新
Target Encoder: o_t → z_t^{target}                ← EMA，不参与梯度，提供监督
```

#### Inverse Dynamics Model（核心新增模块）

```
输入: z_t (B, D), z_{t+1} (B, D)          ← 两个相邻帧的 latent（池化后的向量）
输出: â_t (B, latent_action_dim)            ← 学出来的 latent action

网络结构:
  concat(z_t, z_{t+1})  →  MLP  →  bottleneck  →  â_t
  (B, 2D)                         (B, latent_action_dim)

bottleneck 是关键：
  - latent_action_dim << latent_dim（比如 D=256, latent_action_dim=32）
  - 防止 â_t 把 z_{t+1} 的全部信息都编码进去
```

#### 防止 Collapse 的正则化

逆动力学最大的风险是 â_t 坍缩（所有 â_t 都一样，predictor 直接忽略它）。三种防护手段：

```
方法 1: 信息瓶颈（推荐，最简单）
  - latent_action_dim 设得小（32 或 64）
  - 加 L2 正则: L_reg = ||â_t||²
  - 这样 â_t 被迫只编码最关键的变化信息

方法 2: VQ 离散化
  - â_t 先过一个 VQ codebook 量化为离散 token
  - 类似 VQ-VAE，天然防止 collapse
  - 额外好处：离散 action space 更容易做规划采样
  
  â_t_continuous = MLP(z_t, z_{t+1})
  â_t = VQ(â_t_continuous)   ← 从 codebook 中找最近的向量

方法 3: 方差正则化
  - 强制 â_t 的每个维度在 batch 内有足够的方差
  - L_var = max(0, threshold - Var(â_t))
  - 防止所有样本的 â_t 坍缩到同一个点
```

#### Coarse Predictor

和之前一样，只是把真实 action 换成 latent action：

```
之前: z_t + actions[0:h] + proprios[0:h] → ẑ_{t+h}     ← 用真实动作
现在: z_t + [â_0, â_1, ..., â_{h-1}]     → ẑ_{t+h}     ← 用 latent action

â_t 的计算:
  â_0 = inverse_model(z_0, z_1)
  â_1 = inverse_model(z_1, z_2)
  ...
  â_{h-1} = inverse_model(z_{h-1}, z_h)
```

#### Fine Predictor

Self-Attention + Cross-Attention，action 输入换成 latent action：

```
之前: z_t + a_t + proprio_t + cond(ẑ_{t+h}) → ẑ_{t+1}   ← 真实动作
现在: z_t + â_t             + cond(ẑ_{t+h}) → ẑ_{t+1}   ← latent action

Fine Predictor 内部:
  action_token = action_proj(â_t)             ← â_t 投影到 token
  x = concat([z_t tokens, action_token])      ← (B, num_tokens+1, D)
  for block in transformer_blocks:
      x = self_attention(x)
      x = cross_attention(Q=x, K/V=coarse_cond)
      x = ffn(x)
  ẑ_{t+1} = output_proj(mean_pool(x))
```

### 3.3 阶段一 Loss

```
L_coarse = SmoothL1(ẑ_{t+h}, z_{t+h}^{target})           # 粗粒度预测
L_fine   = SmoothL1(ẑ_{t+1}, z_{t+1}^{target})           # 细粒度预测
L_reg    = ||â_t||²                                        # latent action 正则化
                                                           # （或 VQ commitment loss）

L_total  = L_fine + λ_coarse * L_coarse + λ_reg * L_reg
```

注意：**没有对 â_t 本身的直接监督**。â_t 完全通过正向预测的 loss 间接学习——inverse model 输出的 â_t 必须让 predictor 做出好的预测，否则 loss 下不去。

### 3.4 训练数据流（单个样本）

```
输入: 视频帧 [o_0, o_1, o_2, ..., o_h]

Step 1: 编码所有帧
  z_0 = encoder(o_0)                    ← online encoder
  z_1^{target} = target_encoder(o_1)    ← EMA encoder，用于监督
  z_h^{target} = target_encoder(o_h)

  z_1 = encoder(o_1)                    ← 用于逆动力学输入（需要梯度）

Step 2: 逆动力学计算 latent actions
  â_0 = inverse_model(pool(z_0), pool(z_1))
  â_1 = inverse_model(pool(z_1), pool(z_2))
  ...
  â_{h-1} = inverse_model(pool(z_{h-1}), pool(z_h))

  注意：这里 z_1, z_2, ... 也需要用 online encoder 编码
  因为逆动力学需要梯度回传到 encoder

Step 3: 粗粒度预测
  ẑ_h = coarse_predictor(z_0, [â_0, â_1, ..., â_{h-1}])
  L_coarse = SmoothL1(ẑ_h, pool(z_h^{target}))

Step 4: 细粒度预测（带 curriculum）
  coarse_cond = curriculum_blend(ẑ_h, pool(z_h^{target}))
  ẑ_1 = fine_predictor(z_0, â_0, coarse_cond)
  L_fine = SmoothL1(ẑ_1, pool(z_1^{target}))

Step 5: 总 loss
  L = L_fine + λ_coarse * L_coarse + λ_reg * mean(||â_t||²)

Step 6: 更新
  optimizer.step()
  target_encoder.ema_update(encoder)
```

---

## 4. 阶段二：动作空间对齐

### 4.1 数据要求

少量有动作标签的数据：

```
数据: {(o_t, a_t, o_{t+1})} × N    ← N 可以很小，比如总数据的 5%~10%
```

### 4.2 训练什么

**冻结阶段一所有参数**（Encoder, Target Encoder, Inverse Model, Coarse/Fine Predictor），只训练两个轻量网络：

```
Action Encoder:  a_t (真实动作, dim=action_dim) → ã_t (latent action, dim=latent_action_dim)
Action Decoder:  â_t (latent action) → ã_t^{real} (重建的真实动作)

两个都是小 MLP（2~3 层），参数量很小
```

### 4.3 对齐 Loss

```
# 先用阶段一的模型算出 latent action
z_t = encoder(o_t)           ← 冻结
z_{t+1} = encoder(o_{t+1})   ← 冻结
â_t = inverse_model(z_t, z_{t+1})  ← 冻结，这是阶段一学出的 latent action

# Action Encoder: 真实动作映射到 latent space
ã_t = action_encoder(a_t)

# 对齐: 两条路径得到的 latent action 应该一致
L_align = ||ã_t - â_t.detach()||²

# Action Decoder: latent action 解码回真实动作
a_t_recon = action_decoder(â_t)
L_decode = ||a_t_recon - a_t||²

# 总 loss
L_phase2 = L_align + L_decode
```

### 4.4 为什么要两个网络

```
Action Encoder (a_t → ã_t):
  规划时用。CEM 在真实动作空间采样 → 转到 latent → 喂给 predictor

Action Decoder (â_t → a_t):
  执行时用。规划完拿到最优 latent action → 转回真实动作 → 发给机器人

两个方向都要打通。
```

---

## 5. 规划流程

### 5.1 完整流程

```
已知: o_0（当前观测）, z_goal（目标状态的 latent）

═══ Step 1: CEM 在真实动作空间采样 ═══

  candidates: N 条动作序列，每条 H 步
  [a_0^{(i)}, a_1^{(i)}, ..., a_{H-1}^{(i)}]   i = 1..N

═══ Step 2: 真实动作 → Latent 动作 ═══

  对每条候选的每个时间步:
  ã_t^{(i)} = action_encoder(a_t^{(i)})

═══ Step 3: Hierarchical Rollout ═══

  z_0 = encoder(o_0)

  对候选 i:

    # ---- 粗粒度 waypoints ----
    ẑ_h   = coarse_predictor(z_0, [ã_0, ..., ã_{h-1}])
    ẑ_{2h} = coarse_predictor(ẑ_h, [ã_h, ..., ã_{2h-1}])
    ...

    # ---- 细粒度填充 ----
    # 第一段 (condition = ẑ_h)
    ẑ_1 = fine_predictor(z_0,  ã_0, cond=ẑ_h)
    ẑ_2 = fine_predictor(ẑ_1, ã_1, cond=ẑ_h)
    ...

    # 第二段 (condition = ẑ_{2h})
    ẑ_{h+1} = fine_predictor(ẑ_h, ã_h, cond=ẑ_{2h})
    ...

    # 计算 cost
    cost_i = ||ẑ_H - z_goal||²

═══ Step 4: CEM 更新 ═══

  取 top-k 低 cost 的候选
  更新采样分布 (μ, σ)
  重复 Step 1-4 共 M 轮

═══ Step 5: 执行 ═══

  最优候选的第一个动作 a_0*  ← 已经是真实动作，直接执行
  （或者取最优 latent action，用 Action Decoder 解码）
  
  拿到新观测 o_1，回到 Step 1（receding horizon）
```

### 5.2 规划时不需要 Inverse Model

注意：规划时**完全不需要** Inverse Dynamics Model。它只在阶段一训练时使用。

```
训练时: Inverse Model 从 (z_t, z_{t+1}) 生成 â_t  ← 因为没有真实 a_t
规划时: Action Encoder 从 a_t 生成 ã_t              ← 因为 CEM 采样的就是真实 a_t
```

两者在同一个 latent action space 里（阶段二的对齐保证了这一点），所以 predictor 可以无缝使用。

---

## 6. 训练阶段对比

| | 阶段一（预训练） | 阶段二（对齐） | 规划 |
|---|---|---|---|
| **数据** | 纯视频，无动作标签 | 少量有动作标签 | 无 |
| **Encoder** | ✅ 训练 | ❄️ 冻结 | ❄️ 推理 |
| **Target Encoder** | ✅ EMA 更新 | ❄️ 冻结 | 不需要 |
| **Inverse Model** | ✅ 训练 | ❄️ 冻结（生成 â_t） | 不需要 |
| **Coarse Predictor** | ✅ 训练 | ❄️ 冻结 | ❄️ 推理 |
| **Fine Predictor** | ✅ 训练 | ❄️ 冻结 | ❄️ 推理 |
| **Action Encoder** | 不存在 | ✅ 训练 | ❄️ 推理 |
| **Action Decoder** | 不存在 | ✅ 训练 | ❄️ 推理 |
| **动作表示** | â_t（逆动力学） | â_t ↔ a_t 对齐 | ã_t = AE(a_t) |

---

## 7. 关键设计决策

### 7.1 Latent Action 维度

```
latent_action_dim 的选择至关重要:

太大（= latent_dim）:
  → â_t 能编码 z_{t+1} 的全部信息
  → predictor 直接把 â_t 复制到输出，不学任何动态
  → 退化为 autoencoder

太小（= 1 或 2）:
  → 表达力不够，无法区分不同类型的动作
  → predictor 没有足够信息做预测

推荐: latent_action_dim = latent_dim / 4 ~ latent_dim / 8
       例如 latent_dim=256 → latent_action_dim=32~64
```

### 7.2 逆动力学的输入用 online encoder 还是 target encoder

```
推荐: 两个输入都用 online encoder

  â_t = inverse_model(pool(z_t), pool(z_{t+1}))
  其中 z_t = encoder(o_t), z_{t+1} = encoder(o_{t+1})

原因: 需要梯度从 inverse model 回传到 encoder
      如果用 target encoder，梯度被 stop gradient 截断
      encoder 就无法从逆动力学 loss 中受益
```

### 7.3 Curriculum Learning 不变

```
和层次化文档一样:
  Fine Predictor 的 coarse condition 逐步从 GT 切换到 predicted

  curriculum_ratio = min(1.0, epoch / (total_epochs × 0.7))
  coarse_cond = ratio × ẑ_h + (1 - ratio) × z_h^{target}
```

---

## 8. 实验计划

### 8.1 验证阶段一：自监督预训练有效性

```
实验 A: 表征质量
  - 在 tworoom 上只跑阶段一（无动作标签）
  - 对学出的 z_t 做 probing（线性探测能否预测位置、速度等）
  - 对比原版 LeWM（有动作标签）的 z_t 质量

实验 B: Latent Action 质量
  - 可视化 â_t 的分布（t-SNE）
  - 看不同真实动作对应的 â_t 是否可区分
  - 验证没有 collapse
```

### 8.2 验证阶段二：少量标注够不够

```
实验 C: 标注比例消融
  - 用 1%, 5%, 10%, 50%, 100% 的有标注数据做阶段二
  - 看 action encoder/decoder 的对齐质量
  - 看最终规划 success rate

  预期: 5%~10% 就能达到接近 100% 标注的效果
  这是论文的核心卖点之一
```

### 8.3 验证完整框架

```
实验 D: 最终规划 success rate
  - 完整两阶段训练 + CEM 规划
  - 对比:
    1. LeWM 原版（有标注，单步 predictor）
    2. 只有层次化（有标注，粗细粒度，无逆动力学）
    3. 只有逆动力学（无标注，单步 predictor）
    4. 完整框架（无标注预训练 + 少量标注对齐 + 层次化）
  - 在 tworoom, DMControl, 更多环境上测试

实验 E: 长 horizon 规划
  - 固定模型，增加规划 horizon（5, 10, 20, 50 步）
  - 对比单步自回归 vs 层次化的误差累积
  - 预期: 层次化在长 horizon 上优势明显
```

### 8.4 消融实验

```
实验 F: 各组件消融
  1. 有/无逆动力学（直接用真实动作 vs latent action）
  2. 有/无层次化（单步 vs 粗细粒度）
  3. 不同融合方式（Cross-Attention vs FiLM vs Concat vs AdaLN）
  4. 不同 latent_action_dim（16, 32, 64, 128）
  5. 不同 horizon_h（3, 5, 10）
  6. 不同 collapse 防护（bottleneck vs VQ vs variance reg）
```

---

## 9. 论文故事线

```
Title (暂定):
  Hierarchical JEPA World Models with Self-Supervised Action Discovery

故事:
  1. 现有 JEPA 世界模型（LeWM）有两个核心局限:
     - 短 horizon 规划（误差累积）
     - 依赖动作标注（获取成本高，限制了 scalability）

  2. 我们提出 XXX 框架:
     - 层次化粗细粒度预测，解决长 horizon
     - 逆动力学自监督学习 latent action，去掉标注依赖
     - 两阶段训练: 大量无标注预训练 + 少量标注对齐

  3. 实验表明:
     - 仅用 5% 标注数据达到全标注 XX% 的性能
     - 长 horizon 规划 success rate 从 XX% 提升到 XX%
     - 表征质量显著优于 baseline

投稿目标: ACM MM 2026 / CVPR 2027
```

---

## 10. 代码规划

```
src/
├── hierarchical_jepa_worldmodel.py      ← 已完成（层次化部分）
├── inverse_dynamics_model.py            ← 待写（逆动力学模块）
├── hierarchical_jepa_with_invdyn.py     ← 待写（完整融合框架，阶段一）
├── action_alignment.py                  ← 待写（阶段二：动作对齐）
├── plan_hierarchical_invdyn.py          ← 待写（规划 + CEM）
└── eval_hierarchical_invdyn.py          ← 待写（评估脚本）
```

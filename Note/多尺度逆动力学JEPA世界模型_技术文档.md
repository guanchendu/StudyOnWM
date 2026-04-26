# 多尺度逆动力学 JEPA 世界模型 — 技术文档

## 1. 核心思想

#### 在不同时间尺度上做逆动力学，让模型**自动发现动作的层次化抽象**：

- **粗粒度逆动力学**：(z_t, z_{t+h}) → â^{coarse} — 用一个向量回答"这 h 步整体做了什么"
- **细粒度逆动力学**：(z_t, z_{t+1}) → â^{fine} — 用一个向量回答"这一步具体做了什么"

模型不需要人为定义什么是"高层动作"vs"低层动作"，多尺度逆动力学自然会学出这种抽象层次。

### 与 src_wm（v1）的关键区别

| | v1 (src_wm) | v2 (src_wm2) |
|---|---|---|
| 逆动力学 | 只有单步 inv(z_t, z_{t+1}) | 双尺度：fine + coarse |
| 粗粒度 predictor 输入 | z_t + [â_0,...,â_{h-1}] (拼接 h 个细粒度动作) | z_t + â^{coarse} (一个粗粒度动作抽象) |
| 核心 novelty | 技术组合 | 自动发现多尺度动作抽象 |
| 论文故事 | "层次化 + 逆动力学" | "Emergent action abstraction via multi-scale inverse dynamics" |

---

## 2. 总体架构

```
┌──────────────────────────────────────────────────────────────────────────┐
│  阶段一：自监督预训练（大量无标注视频，不需要动作标签）                       │
│                                                                          │
│  o_t ──→ [Encoder] ──→ z_t    (B, num_tokens, D)                        │
│  o_{t+1} → [Encoder] → z_{t+1}                                          │
│  o_{t+h} → [Encoder] → z_{t+h}                                          │
│                                                                          │
│  ┌─────────────────────────────┐  ┌──────────────────────────────┐      │
│  │   Fine Inverse Dynamics     │  │  Coarse Inverse Dynamics     │      │
│  │   (z_t, z_{t+1}) → â^fine  │  │  (z_t, z_{t+h}) → â^coarse  │      │
│  └──────────────┬──────────────┘  └──────────────┬───────────────┘      │
│                 │                                 │                      │
│                 ▼                                 ▼                      │
│          [Fine Predictor]                 [Coarse Predictor]            │
│          z_t + â^fine                     z_t + â^coarse                │
│          + cond(ẑ_{t+h})                        │                      │
│                 │                                 │                      │
│                 ▼                                 ▼                      │
│              ẑ_{t+1} ──→ L_fine            ẑ_{t+h} ──→ L_coarse        │
│                                                                          │
│  + L_reg_fine + L_reg_coarse (防 collapse 正则化)                        │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────┐
│  阶段二：动作空间对齐（少量有标注数据）                                     │
│                                                                          │
│  冻结阶段一全部参数，训练两对翻译器：                                       │
│                                                                          │
│  细粒度对齐:                                                              │
│    FineActionEncoder:    a_t → ã^fine                                    │
│    FineActionDecoder:    â^fine → a_t                                    │
│    L_align_fine = ||ã^fine - â^fine||²                                   │
│                                                                          │
│  粗粒度对齐:                                                              │
│    CoarseActionEncoder:  [a_t,...,a_{t+h-1}] → ã^coarse                 │
│    CoarseActionDecoder:  â^coarse → [a_t,...,a_{t+h-1}]                 │
│    L_align_coarse = ||ã^coarse - â^coarse||²                            │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌──────────────────────────────────────────────────────────────────────────┐
│  两级规划                                                                 │
│                                                                          │
│  第一级: CEM 在粗粒度动作空间搜索                                         │
│    采样 â^coarse → CoarsePredictor → waypoints → 选最优粗路径             │
│                                                                          │
│  第二级: 在每对 waypoints 之间，CEM 在细粒度动作空间搜索                    │
│    采样 â^fine → FinePredictor → 细粒度轨迹 → 选最优细动作                │
│                                                                          │
│  执行: FineActionDecoder 解码 → 真实动作 → 发给机器人                     │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 各模块详细设计

### 3.1 Encoder + Target Encoder

与 v1 完全一致：

```
TokenEncoder: image (B, 3, H, W) → tokens (B, num_tokens, D)
EMATargetEncoder: momentum=0.996, 不参与梯度, 提供监督目标
```

### 3.2 Fine Inverse Dynamics（单步）

```
输入: z_t (B, D), z_{t+1} (B, D)     ← 相邻帧的 pooled latent
输出: â^fine (B, fine_action_dim)      ← 细粒度 latent action

网络: concat(z_t, z_{t+1}) → MLP → bottleneck → â^fine
      (B, 2D)               (B, hidden)   (B, fine_action_dim)

fine_action_dim 推荐: 32 (远小于 latent_dim=256)
```

### 3.3 Coarse Inverse Dynamics（跨 h 步）—— 核心新增

```
输入: z_t (B, D), z_{t+h} (B, D)     ← 跨 h 步的两帧
输出: â^coarse (B, coarse_action_dim)  ← 粗粒度 latent action（动作抽象）

网络: concat(z_t, z_{t+h}) → MLP → bottleneck → â^coarse
      (B, 2D)                (B, hidden)    (B, coarse_action_dim)

coarse_action_dim 推荐: 64 (比 fine 大，因为要编码 h 步的抽象信息)
```

**为什么 coarse 和 fine 是独立的网络而不共享？**

因为它们学的东西语义不同：
- fine 学的是 "向右移动 0.5"（具体操作）
- coarse 学的是 "从房间 A 走到房间 B"（抽象策略）

共享参数会互相干扰。

### 3.4 Coarse Predictor

```
v1: 输入 = z_t + [â_0^fine, ..., â_{h-1}^fine]    ← h 个细粒度动作拼接
    input_dim = latent_dim + h * fine_action_dim    ← 随 h 增大

v2: 输入 = z_t + â^coarse                           ← 1 个粗粒度动作
    input_dim = latent_dim + coarse_action_dim       ← 固定大小，不受 h 影响

网络: concat(z_pooled, â^coarse) → MLP → ẑ_{t+h}
      (B, D + coarse_action_dim)        (B, D)
```

**好处**：input_dim 不再依赖 h，可以灵活调整 h 而不改架构。

### 3.5 Fine Predictor（Self-Attention + Cross-Attention）

与 v1 结构一致，只是 action 输入换成 â^fine：

```
输入:
  z_t:       (B, num_tokens, D)     ← 当前 latent tokens
  â^fine:    (B, fine_action_dim)   ← 细粒度 latent action
  cond:      (B, D)                 ← 粗粒度预测的 waypoint ẑ_{t+h}

内部:
  action_token = proj(â^fine)                      ← (B, 1, D)
  x = concat([z_t tokens, action_token])           ← (B, N+1, D)
  coarse_kv = proj(cond)                           ← (B, 1, D)
  
  for block in transformer_blocks:
      x = self_attention(x)                        ← tokens 互相 attend
      x = cross_attention(Q=x, K/V=coarse_kv)     ← 从粗粒度获取方向
      x = ffn(x)
  
  ẑ_{t+1} = output_proj(mean_pool(x))             ← (B, D)
```

---

## 4. 阶段一训练细节

### 4.1 数据流（单个样本）

```
输入: 视频帧 [o_0, o_1, ..., o_h]    ← 不需要动作标签

Step 1: 用 online encoder 编码所有帧（需要梯度）
  z_0 = encoder(o_0)
  z_1 = encoder(o_1)
  ...
  z_h = encoder(o_h)

Step 2: 用 target encoder 编码监督目标（无梯度）
  z_1^tgt = target_encoder(o_1).mean(dim=1)    ← (B, D)
  z_h^tgt = target_encoder(o_h).mean(dim=1)    ← (B, D)

Step 3: 细粒度逆动力学
  â_0^fine = fine_inv(pool(z_0), pool(z_1))
  â_1^fine = fine_inv(pool(z_1), pool(z_2))
  ...
  â_{h-1}^fine = fine_inv(pool(z_{h-1}), pool(z_h))

Step 4: 粗粒度逆动力学
  â^coarse = coarse_inv(pool(z_0), pool(z_h))   ← 只算一次，跨 h 步

Step 5: 粗粒度预测
  ẑ_h = coarse_predictor(z_0, â^coarse)
  L_coarse = SmoothL1(ẑ_h, z_h^tgt)

Step 6: 细粒度预测（带 curriculum）
  coarse_cond = blend(ẑ_h, z_h^tgt, curriculum_ratio)
  ẑ_1 = fine_predictor(z_0, â_0^fine, coarse_cond)
  L_fine = SmoothL1(ẑ_1, z_1^tgt)

Step 7: 正则化
  L_reg_fine   = mean(||â^fine||²) + variance_reg(â^fine)
  L_reg_coarse = mean(||â^coarse||²) + variance_reg(â^coarse)

Step 8: 总 loss
  L = L_fine + λ_c * L_coarse + λ_rf * L_reg_fine + λ_rc * L_reg_coarse
```

### 4.2 多步细粒度训练（可选增强）

除了只预测 z_1，还可以 rollout 多步细粒度预测来训练：

```
ẑ_1 = fine_predictor(z_0,       â_0^fine, coarse_cond)  → L_fine_0
ẑ_2 = fine_predictor(ẑ_1_token, â_1^fine, coarse_cond)  → L_fine_1
...

L_fine = mean(L_fine_0, L_fine_1, ..., L_fine_{h-1})
```

这样 fine predictor 也会学到多步 rollout 的鲁棒性。但实现上更复杂，可以作为后续改进。

### 4.3 Curriculum Learning

与 v1 一致：

```
curriculum_ratio = min(1.0, epoch / (total_epochs × 0.7))

coarse_cond = ratio × ẑ_h.detach() + (1 - ratio) × z_h^tgt

训练前期: 用 GT，让 fine predictor 先学会基本预测
训练后期: 用 coarse predictor 预测值，贴近推理场景
```

---

## 5. 阶段二：动作空间对齐

### 5.1 两对翻译器

```
细粒度翻译器（和 v1 一样）:
  FineActionEncoder:  a_t (action_dim) → ã^fine (fine_action_dim)
  FineActionDecoder:  â^fine (fine_action_dim) → a_t (action_dim)

粗粒度翻译器（新增）:
  CoarseActionEncoder: [a_t, ..., a_{t+h-1}] (h × action_dim) → ã^coarse (coarse_action_dim)
  CoarseActionDecoder: â^coarse (coarse_action_dim) → [a_t, ..., a_{t+h-1}] (h × action_dim)
```

**粗粒度翻译器的含义**：
- Encoder: 把 h 个具体动作"压缩"成一个抽象描述（和 coarse inverse dynamics 学出的对齐）
- Decoder: 把一个抽象描述"展开"成 h 个具体动作

### 5.2 Loss

```
# 冻结阶段一，用有标注数据 (o_t, a_t, ..., a_{t+h-1}, o_{t+1}, ..., o_{t+h})

# 从阶段一获取 latent actions（冻结）
â^fine = fine_inv(encoder(o_t), encoder(o_{t+1}))
â^coarse = coarse_inv(encoder(o_t), encoder(o_{t+h}))

# 细粒度对齐
ã^fine = fine_action_encoder(a_t)
L_align_fine = ||ã^fine - â^fine.detach()||²
L_decode_fine = ||fine_action_decoder(â^fine) - a_t||²
L_cycle_fine = ||fine_action_decoder(ã^fine) - a_t||²

# 粗粒度对齐
actions_seq = concat([a_t, ..., a_{t+h-1}])   ← (B, h * action_dim)
ã^coarse = coarse_action_encoder(actions_seq)
L_align_coarse = ||ã^coarse - â^coarse.detach()||²
L_decode_coarse = ||coarse_action_decoder(â^coarse) - actions_seq||²
L_cycle_coarse = ||coarse_action_decoder(ã^coarse) - actions_seq||²

# 总 loss
L = (L_align_fine + L_decode_fine + 0.5 * L_cycle_fine)
  + (L_align_coarse + L_decode_coarse + 0.5 * L_cycle_coarse)
```

---

## 6. 两级规划

### 6.1 为什么比 v1 更高效

```
v1: CEM 在细粒度动作空间搜索完整的 H 步序列
    搜索空间 = H × action_dim  ← 随 horizon 线性增长，维度灾难

v2: 分两级搜索
    第一级: CEM 在粗粒度动作空间搜索 ← 搜索空间 = (H/h) × coarse_action_dim
    第二级: 在每个 segment 内，CEM 在细粒度搜索 ← 搜索空间 = h × fine_action_dim

    总搜索量更小，且粗粒度先确定方向，细粒度在局部优化
```

### 6.2 逆动力学 Warm-Start（核心改进）

之前的问题：逆动力学只在阶段一训练时使用，规划时闲置。CEM 从零开始盲目搜索。

改进：**用逆动力学为 CEM 提供初始化**，让搜索从一个合理的起点开始。

```
逆动力学的能力：给定 (z_current, z_target)，直接输出"应该做什么动作"

规划时直接利用：
  â^c_init = coarse_inv_dyn(z_0, z_goal)   ← "整体应该往哪个方向"
  â^f_init = fine_inv_dyn(z_t, z_target)    ← "下一步具体怎么走"

不替代 CEM，而是给 CEM 一个 warm start:
  μ = invdyn_estimate     ← 不是零向量，而是逆动力学的估计
  σ = 0.5                 ← 方差更小，在估计附近搜索即可
```

### 6.3 完整流程（带 Warm-Start）

```
已知: o_0（当前观测），z_goal（目标 latent）
设 H = 总 horizon, h = 粗粒度步长, K = H/h 段

═══ 第一级：粗粒度规划（逆动力学初始化 + CEM 精调）═══

  z_0 = encoder(o_0)

  # ---- Warm-Start: 逆动力学给出初始路径 ----
  z = z_0
  coarse_inits = []
  for k in range(K):
    â^c = coarse_inv_dyn(z, z_goal)       ← 逆动力学估计
    coarse_inits.append(â^c)
    z = coarse_predictor(z, â^c)          ← 用估计往前推，为下一段提供起点

  # ---- CEM 以逆动力学估计为初始化 ----
  μ_coarse = stack(coarse_inits)          ← 不是零向量！
  σ_coarse = 0.5                          ← 比随机初始化 σ=1 小

  for cem_iter in range(M1):
    candidates = sample(μ_coarse, σ_coarse)   ← 在逆动力学估计附近采样
    
    for each candidate:
      rollout with coarse_predictor → cost = ||ẑ_H - z_goal||²

    top-k → update μ_coarse, σ_coarse

  waypoints = rollout(z_0, best_coarse_actions)

═══ 第二级：细粒度规划（同样带 Warm-Start）═══

  for k in range(K):
    z_start = waypoints[k]
    z_target = waypoints[k+1]

    # ---- Warm-Start: 细粒度逆动力学逐步估计 ----
    z = z_start
    fine_inits = []
    for t in range(h):
      â^f = fine_inv_dyn(z, z_target)     ← 逆动力学估计
      fine_inits.append(â^f)
      z = fine_predictor(z, â^f, cond=z_target)

    # ---- CEM 精调 ----
    μ_fine = stack(fine_inits)
    σ_fine = 0.5

    for cem_iter in range(M2):
      candidates = sample(μ_fine, σ_fine)
      rollout with fine_predictor → cost = ||ẑ - z_target||²
      top-k → update μ_fine, σ_fine

    all_fine_actions.extend(best_fine_for_segment_k)

═══ 执行 ═══

  a_0_real = fine_action_decoder(all_fine_actions[0])
  执行 → 新观测 o_1 → 回到第一级 (receding horizon)
```

### 6.4 逆动力学在训练 vs 规划中的角色

```
训练时 (阶段一):
  逆动力学: (z_t, z_{t+1}) → â^fine
            (z_t, z_{t+h}) → â^coarse
  作用: 为 predictor 生成 latent action 输入
        同时塑造了有意义的 latent action space

规划时:
  逆动力学: (z_current, z_goal) → â^init
  作用: 给 CEM 提供 warm-start 初始化
        利用"给定起点和终点，推断动作"的能力
        CEM 不再从零搜索，而是在合理估计附近精调

两个阶段都有贡献，逆动力学不再是只训练时用的辅助模块。
```

### 6.5 消融实验设计

```
这个改进本身就是一个重要的消融实验：

  A. CEM 随机初始化 (μ=0, σ=1)        ← baseline
  B. CEM 逆动力学初始化 (μ=invdyn, σ=0.5)  ← 我们

  对比指标:
  - CEM 收敛速度 (cost vs iteration 曲线)
  - 最终规划 success rate
  - 相同 CEM 步数下的性能差异

  预期: B 在更少 CEM 步数下就能达到 A 的最终性能
        → 逆动力学对规划有直接贡献，不只是训练辅助
```

### 6.6 规划效率对比

### 6.3 规划效率对比

```
假设: H=20, h=5, action_dim=4

v1 单级 CEM:
  每条候选: 20 × 4 = 80 维
  CEM 需要大量样本覆盖 80 维空间

v2 两级 CEM:
  第一级: 每条候选 4 × coarse_action_dim = 4 × 64 = 256 维
          但只有 4 个 coarse step，且 coarse predictor 速度快
  第二级: 每段只有 5 × fine_action_dim = 5 × 32 = 160 维
          且有 waypoint 作为目标，搜索空间大幅缩小

  总搜索效率远优于 v1
```

---

## 7. 关键设计决策

### 7.1 维度选择

| 参数 | 推荐值 | 理由 |
|------|--------|------|
| latent_dim | 256 | 状态表示维度 |
| fine_action_dim | 32 | 远小于 latent_dim，防止编码过多信息 |
| coarse_action_dim | 64 | 比 fine 大，需要编码 h 步的抽象信息 |
| num_tokens | 4 | Encoder 输出 token 数 |
| horizon_h | 5 | 粗粒度跳步数 |

### 7.2 防 Collapse

两个逆动力学模型都需要防 collapse，但策略略有不同：

```
Fine Inverse Dynamics:
  - 信息瓶颈: fine_action_dim=32 << latent_dim=256
  - L2 正则: λ * ||â^fine||²
  - 方差正则: 确保 batch 内 â^fine 有方差

Coarse Inverse Dynamics:
  - 信息瓶颈: coarse_action_dim=64
  - L2 正则: λ * ||â^coarse||²
  - 方差正则: 同上
  - 额外: 粗粒度更容易 collapse（因为跳步大，很多不同轨迹终点可能相似）
    → 可以考虑对 â^coarse 加 VQ 离散化，强制使用 codebook 中的离散 token
```

### 7.3 Coarse 和 Fine 逆动力学的关系

它们是独立的网络，但学到的表示应该有语义一致性：

```
â^coarse 应该 ≈ 某种"聚合"的 [â^fine_0, ..., â^fine_{h-1}]

可选的一致性 loss（实验性质，不一定需要）:
  聚合后的 fine actions:
    â^fine_agg = MLP([â^fine_0, ..., â^fine_{h-1}])
  一致性:
    L_consist = ||â^fine_agg - â^coarse.detach()||²

这个 loss 鼓励 coarse action 确实是 fine actions 的"摘要"
但不是必须的，可以作为消融实验的一个变量
```

---

## 8. 实验计划

### 8.1 核心实验

```
实验 A: 自动发现动作抽象的验证（核心贡献）
  - t-SNE 可视化 â^coarse 和 â^fine
  - 预期: â^coarse 按"高层策略"聚类（去房间A、去房间B、穿过门）
          â^fine 按"具体操作"聚类（左转、右转、前进）
  - 与真实动作标签做对比，看聚类是否有语义

实验 B: 规划性能
  - 完整框架 vs baselines
    1. LeWM 原版（有标注，单步）
    2. v1（层次化 + 单步逆动力学）
    3. v2（层次化 + 多尺度逆动力学）← 我们
    4. Dreamer-v3
    5. TD-MPC2
  - 在 tworoom, DMControl 等环境上对比

实验 C: 数据效率
  - 固定阶段一（预训练），改变阶段二的标注比例: 1%, 5%, 10%, 50%, 100%
  - 证明少量标注就够

实验 D: 长 horizon 规划
  - 增加规划 horizon (5, 10, 20, 50)
  - v2 的两级规划应该在长 horizon 上优势明显
```

### 8.2 消融实验

```
实验 E: 组件消融
  1. 有/无粗粒度逆动力学（用 v1 的拼接 fine actions 替代）
  2. 有/无细粒度逆动力学（只用 coarse）
  3. 单级 vs 两级 CEM 规划
  4. 不同融合方式: Cross-Attention vs FiLM vs Concat vs AdaLN
  5. 不同 fine_action_dim: 16, 32, 64, 128
  6. 不同 coarse_action_dim: 32, 64, 128
  7. 不同 horizon_h: 3, 5, 10
  8. 有/无一致性 loss (L_consist)
```

---

## 9. 论文故事线

```
Title:
  Multi-Scale Inverse Dynamics for Emergent Action Abstraction
  in JEPA World Models

Abstract 核心 claim:
  1. 我们提出在 JEPA 世界模型中引入多尺度逆动力学
  2. 模型在不同时间尺度上自动发现动作的层次化抽象:
     粗粒度学到高层策略，细粒度学到低层操作
  3. 这种 emergent abstraction 不需要人为定义动作层次
  4. 结合两阶段训练（大量无标注预训练 + 少量标注对齐）
     和两级规划，实现高效的长 horizon 规划
  5. 仅用 X% 标注数据达到全标注 Y% 的性能

Related Work 定位:
  - JEPA World Models (V-JEPA, LeWM, Thinking-JEPA) — 我们的基础
  - Hierarchical World Models (Dreamer, Director, TD-MPC) — 我们也做层次化，但动作抽象是学出来的
  - Inverse Dynamics (ICM, APV, LAPO) — 我们扩展到多尺度
  - Option/Skill Discovery (Option-Critic, DADS) — 我们的粗粒度逆动力学自然发现 skill

核心区别: 不是 "发现 skill 再用 skill 规划" 这种两步走
         而是 "多尺度逆动力学在训练中同时学出 skill 和对应的 predictor"
         两者是 end-to-end 联合优化的

投稿目标: NeurIPS 2027 / ICLR 2027 / CVPR 2027
```

---

## 10. 代码结构

```
src/src_wm2/
├── __init__.py
├── models.py              — 全部模型组件
│   ├── TokenEncoder
│   ├── EMATargetEncoder
│   ├── FineInverseDynamics      ← (z_t, z_{t+1}) → â^fine
│   ├── CoarseInverseDynamics    ← (z_t, z_{t+h}) → â^coarse  [NEW]
│   ├── CoarsePredictor          ← z_t + â^coarse → ẑ_{t+h}  [MODIFIED]
│   ├── FineTransformerBlock
│   ├── FinePredictor            ← z_t + â^fine + cond → ẑ_{t+1}
│   ├── FineActionEncoder/Decoder
│   ├── CoarseActionEncoder/Decoder  [NEW]
│   └── MultiScaleInvDynWorldModel   ← 整合所有组件
├── train_phase1.py        — 阶段一: 双尺度自监督预训练
├── train_phase2.py        — 阶段二: 双尺度动作对齐
├── planning.py            — 两级 CEM 规划
└── evaluate.py            — 评估脚本

运行:
  cd /Users/guanchendu/Code/StudyOnWM/src
  
  # 阶段一
  conda run -n wm python -m src_wm2.train_phase1 --epochs 100
  
  # 阶段二
  conda run -n wm python -m src_wm2.train_phase2 \
    --phase1-ckpt outputs/multiscale_invdyn/best_phase1.pt \
    --label-fraction 0.1
  
  # 评估
  conda run -n wm python -m src_wm2.evaluate \
    --phase1-ckpt outputs/multiscale_invdyn/best_phase1.pt \
    --phase2-ckpt outputs/multiscale_invdyn/best_phase2.pt
```

---

## 11. 与 v1 的迁移关系

从 v1 迁移到 v2 改动量不大:

| 文件 | 改动 |
|------|------|
| models.py | +CoarseInverseDynamics, 改 CoarsePredictor 输入, +CoarseActionEncoder/Decoder |
| train_phase1.py | +计算 â^coarse, +L_reg_coarse |
| train_phase2.py | +粗粒度对齐 loss |
| planning.py | 重写为两级 CEM |
| evaluate.py | 适配新的 planning 接口 |

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

网络: concat(z_pooled, â^coarse) → MLP → ẑ_{t+h}^tokens
      (B, D + coarse_action_dim)         (B, num_tokens, D)
```

**关键升级（vs v1）**：输出从 pooled `(B, D)` 改为 token 序列 `(B, num_tokens, D)`，
让 multi-step rollout 时下一步输入有真实的 token 多样性，而不是 pooled→expand 的假 tokens。

**好处**：
- input_dim 不再依赖 h，灵活调整 h
- 多步链式调用时与训练分布一致

### 3.5 Fine Predictor（Self-Attention + Cross-Attention）

```
输入:
  z_t:       (B, num_tokens, D)              ← 当前 latent tokens
  â^fine:    (B, fine_action_dim)            ← 细粒度 latent action
  cond:      (B, num_tokens, D) 或 (B, D)    ← 粗粒度 waypoint ẑ_{t+h}

内部:
  action_token = proj(â^fine)                      ← (B, 1, D)
  x = concat([z_t tokens, action_token])           ← (B, N+1, D)
  coarse_kv = proj(cond)                           ← (B, num_tokens, D) — 多 token KV

  for block in transformer_blocks:
      x = self_attention(x)                        ← tokens 互相 attend
      x = cross_attention(Q=x, K/V=coarse_kv)     ← 从粗粒度多个 token 获取上下文
      x = ffn(x)

  ẑ_{t+1}^tokens = output_proj(x[:, :N, :])       ← (B, num_tokens, D)
                                                     丢弃 action_token 位置
```

**关键升级（vs v1）**：
- 输出 token 序列 `(B, num_tokens, D)` 而非 pooled `(B, D)`，支持多步 rollout
- coarse_cond 也接受 token 序列，cross-attention 有更丰富的 KV 上下文

---

## 4. 阶段一训练细节

### 4.1 数据流（单个样本，K 段 coarse + 多步链式 rollout）

**关键设计**：训练时的 rollout 分布要尽量与规划时一致。
- 规划时 coarse predictor 链式 K 段 → 训练时也 K 段
- 规划时 fine predictor 链式 h 步 → 训练时也 h 步
- 规划时 fine 段 k≥1 起点是 coarse 输出 → 训练时也如此

```python
输入: 视频帧 [o_0, o_1, ..., o_{Kh}]    ← 共 Kh+1 帧（默认 K=2 → 11 帧）

Step 1: online encoder 编码所有帧
  all_z[t] = encoder(o_t)                      ← (B, num_tokens, D), t∈[0, Kh]

Step 2: target encoder 编码所有监督目标（无梯度）
  z^tgt[t] = target_encoder(o_t)              ← (B, num_tokens, D), t∈[1, Kh]

Step 3: 细粒度逆动力学（GT pair → fine action）  # 预测的是 从当前状态走到下一帧，最关键的那一步动作变化是什么
  â_t^fine = fine_inv(pool(all_z[t]), pool(all_z[t+1]))    ← t∈[0, Kh-1]

Step 4: 粗粒度逆动力学（每段 GT pair → coarse action）
  â_k^coarse = coarse_inv(pool(all_z[kh]), pool(all_z[(k+1)h]))    ← k∈[0, K-1]

Step 5: 多步 Coarse rollout（链式，梯度全程贯穿）
'''
从真实起点出发，连续使用粗粒度 latent action，链式地产生一串 h 步间隔的 waypoint，并让每个 waypoint 都对齐真实未来状态。
'''
  z = all_z[0]
  for k in range(K):
      z = coarse_predictor(z, â_k^coarse)     ← 输出 (B, num_tokens, D)
      z_coarse_pred[k] = z                       #### 这是把每一段预测出来的 waypoint 保存起来
      L_coarse_k = SmoothL1(z, z^tgt[(k+1)h-1])    ← token-level

  L_coarse = mean(L_coarse_0, ..., L_coarse_{K-1})

Step 6: 多步 Fine rollout（每段独立 + scheduled sampling）
  teacher_forcing_ratio = max(0, 1 - curriculum_ratio)

  for k in range(K):
      coarse_cond = curriculum × z_coarse_pred[k] + (1-curriculum) × z^tgt[(k+1)h-1]

      # 段起点：k=0 用真实 encoder，k>0 用 coarse 输出（与规划一致）
      z_pred = all_z[0] if k == 0 else z_coarse_pred[k-1]

      for t in range(h):
          global_t = k*h + t
          z_pred = fine_predictor(z_pred, â_{global_t}^fine, coarse_cond)
          L_fine_{global_t} = SmoothL1(z_pred, z^tgt[global_t])    ← token-level

          # Scheduled sampling: 用 GT 替换下一步输入（教学减弱）
          if t < h-1 and rand() < teacher_forcing_ratio:
              z_pred = all_z[global_t + 1]

  L_fine = mean(L_fine_0, ..., L_fine_{Kh-1})

Step 7: 正则化（聚合所有段）
  L_reg_fine   = ||all_â^fine||² + variance_reg(all_â^fine)
  L_reg_coarse = ||all_â^coarse||² + variance_reg(all_â^coarse)

Step 8: 总 loss
  L = L_fine + λ_c × L_coarse + λ_rf × L_reg_fine + λ_rc × L_reg_coarse
```

### 4.2 多步 rollout 设计要点

#### 为什么要多步链式？

单步训练 → 多步规划 = compounding error。Predictor 从未见过自己的输出作为输入，
规划时累积误差快速发散。链式训练让 predictor 学会容忍自己的预测误差。

#### 为什么 coarse 也要多步？（v2 关键升级）

规划阶段 coarse predictor 链式 K 次（生成 K+1 个 waypoint），但 v1/早期 v2 只单步训练。
现在 K_train ≥ 2，coarse predictor 在第二段时输入是自己的输出，与规划匹配。

#### 为什么 fine 段 k>0 起点用 coarse 输出？

规划时 `waypoints[k]`（k≥1）= coarse_predictor 的链式输出。Fine predictor 必须
学会从这种输入起步预测。训练时段 1 起点 z_pred = z_coarse_pred[0] 就是这个分布。

#### 梯度策略

- coarse rollout：**不 detach**，让 segment 1 的 loss 反向传到 segment 0 的 coarse_predictor，
  使其学会预测对后续步骤友好的状态
- fine rollout 内部：**不 detach**，h 步链都贯穿梯度
- coarse_cond 喂给 fine：**detach**（保持各 scale 训练信号纯净）
- 段间 fine 起点 z_coarse_pred[k-1]：**不 detach**（让 fine loss 也帮 coarse 学习）

### 4.3 Curriculum Learning + Teacher Forcing

两个 schedule 协同：

```
curriculum_ratio       = min(1, epoch / (total_epochs × 0.7))     ← 0 → 1
teacher_forcing_ratio  = max(0, 1 - curriculum_ratio)             ← 1 → 0

训练前期 (cur=0):
  coarse_cond = z^tgt          ← 用真实 EMA 目标作 cond，让 fine 先学
  teacher_forcing = 1.0        ← fine 链式时基本都用 GT 替换，避免噪声雪崩

训练后期 (cur=1):
  coarse_cond = ẑ_coarse       ← 用 coarse predictor 输出（贴近规划）
  teacher_forcing = 0.0        ← fine 链式全用自己的输出（贴近规划）
```

**协同的逻辑**：训练前期 predictor 还没学会，链式输出全是噪声；如果不 teacher force，
loss 信号被噪声掩盖。后期模型成熟，让链式贯通才能学到鲁棒性。

---

## 5. 阶段二：动作空间对齐 + Proprio 解码

### 5.1 三组翻译器

```
细粒度翻译器（和 v1 一样）:
  FineActionEncoder:  a_t (action_dim) → ã^fine (fine_action_dim)
  FineActionDecoder:  â^fine (fine_action_dim) → a_t (action_dim)

粗粒度翻译器（v2 新增）:
  CoarseActionEncoder: [a_t, ..., a_{t+h-1}] (h × action_dim) → ã^coarse (coarse_action_dim)
  CoarseActionDecoder: â^coarse (coarse_action_dim) → [a_t, ..., a_{t+h-1}] (h × action_dim)

Proprio 解码器（v2 新增，关键）:
  ProprioDecoder: pool(z) (latent_dim) → proprio (proprio_dim)
```

**粗粒度翻译器的含义**：
- Encoder: 把 h 个具体动作"压缩"成一个抽象描述（和 coarse inverse dynamics 学出的对齐）
- Decoder: 把一个抽象描述"展开"成 h 个具体动作

**Proprio 解码器为什么必要**：

旧版 `TwoLevelCostModel.get_cost` 直接拿潜空间向量的前几维和 proprio 比较：
```python
cost = (z_pooled[:, :proprio_dim] - goal_proprio)²    ← 假设 latent 前几维是 proprio
```
但 JEPA-style 训练没有任何信号让 latent 前几维对齐 proprio，这个假设错误。

新设计用 `ProprioDecoder` 显式地学一个 latent → proprio 的映射：
```python
cost = (proprio_dec(pool(z)) - goal_proprio)²        ← 学过的映射
```

**关键训练细节**：proprio_dec 需要在两种 latent 分布上训练：
- `encode(pixels[:, t])` 输出（encoder 空间）
- `coarse_predictor(z, a)` / `fine_predictor(z, a, cond)` 输出（predictor 空间）

规划时 cost 是在 rollout 后的 predictor 输出上计算，所以必须涵盖这种分布。

### 5.2 Loss

```
# 冻结阶段一，用有标注数据 (o_t, a_t, ..., a_{t+h-1}, proprio_t, ..., o_{t+h})

# 从阶段一获取 latent states + actions + predictor 输出（全部冻结）
z_0 = encoder(o_0); z_1 = encoder(o_1); z_h = encoder(o_h)
â^fine = fine_inv(pool(z_0), pool(z_1))
â^coarse = coarse_inv(pool(z_0), pool(z_h))

# Predictor 输出（关键：proprio_dec 必须在这个分布上训练）
ẑ_h = coarse_predictor(z_0, â^coarse)
ẑ_1 = fine_predictor(z_0, â^fine, ẑ_h)

# === 细粒度动作对齐 ===
ã^fine = fine_action_encoder(a_t)
L_align_fine = ||ã^fine - â^fine.detach()||²
L_decode_fine = ||fine_action_decoder(â^fine) - a_t||²
L_cycle_fine = ||fine_action_decoder(ã^fine) - a_t||²

# === 粗粒度动作对齐 ===
actions_seq = concat([a_t, ..., a_{t+h-1}])
ã^coarse = coarse_action_encoder(actions_seq)
L_align_coarse = ||ã^coarse - â^coarse.detach()||²
L_decode_coarse = ||coarse_action_decoder(â^coarse) - actions_seq||²
L_cycle_coarse = ||coarse_action_decoder(ã^coarse) - actions_seq||²

# === Proprio 解码（v2 新增）===
# 在 encoder 输出上训练
L_p_enc = mean(
  ||proprio_dec(z_0) - proprio_0||²,
  ||proprio_dec(z_1) - proprio_1||²,
  ||proprio_dec(z_h) - proprio_h||²,
)
# 在 predictor 输出上训练（与规划 cost 路径对齐）
L_p_pred = mean(
  ||proprio_dec(ẑ_h) - proprio_h||²,
  ||proprio_dec(ẑ_1) - proprio_1||²,
)
L_proprio = L_p_enc + L_p_pred

# 总 loss
L = (L_align_fine + L_decode_fine + 0.5 * L_cycle_fine)
  + (L_align_coarse + L_decode_coarse + 0.5 * L_cycle_coarse)
  + λ_p × L_proprio
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

### 6.3 完整流程（带 Warm-Start，与训练完全对齐）

```
已知: o_0（当前观测），o_goal（目标观测）
设 H = 总 horizon, h = 粗粒度步长, K = H/h 段

  # 关键: goal 用 target encoder（而非 online encoder）
  # 原因: 训练时监督目标由 EMA target encoder 产生，predictor 输出分布
  #       与 target encoder 对齐而非 online encoder
  z_0 = encoder(o_0)                       ← (1, num_tokens, D) tokens
  z_goal = target_encoder(o_goal)          ← (1, num_tokens, D) tokens

═══ 第一级：粗粒度规划（逆动力学初始化 + CEM 精调）═══

  # ---- Warm-Start: 逆动力学给出初始路径 ----
  z = z_0
  coarse_inits = []
  for k in range(K):
    â^c = coarse_inv_dyn(pool(z), pool(z_goal))   ← 逆动力学估计
    coarse_inits.append(â^c)
    z = coarse_predictor(z, â^c)                  ← 输出 tokens，用于下一步

  # ---- CEM 以逆动力学估计为初始化 ----
  μ_coarse = stack(coarse_inits)          ← 不是零向量！
  σ_coarse = 0.5                          ← 比随机初始化 σ=1 小

  for cem_iter in range(M1):
    candidates = sample(μ_coarse, σ_coarse)
    z = z_0.expand(N, ...)                ← (N, num_tokens, D)
    for k in range(K):
      z = coarse_predictor(z, candidates[:, k])    ← 链式 K 步，全程 tokens

    cost = sum_token_dim((z - z_goal)²)            ← Token-level cost
    top-k → update μ_coarse, σ_coarse

  waypoints = rollout(z_0, best_coarse_actions)    ← K+1 个 token 序列

═══ 第二级：细粒度规划（同样带 Warm-Start）═══

  for k in range(K):
    z_start = waypoints[k]                ← (num_tokens, D), k=0 是 encoder, k>0 是 coarse 输出
    z_target = waypoints[k+1]             ← (num_tokens, D)

    # ---- Warm-Start: 细粒度逆动力学逐步估计 ----
    z = z_start
    fine_inits = []
    for t in range(h):
      â^f = fine_inv_dyn(pool(z), pool(z_target))
      fine_inits.append(â^f)
      z = fine_predictor(z, â^f, z_target)         ← coarse_cond 也是 tokens

    # ---- CEM 精调 ----
    μ_fine = stack(fine_inits)
    σ_fine = 0.5

    for cem_iter in range(M2):
      candidates = sample(μ_fine, σ_fine)
      z = z_start.expand(N, ...)
      cond = z_target.expand(N, ...)
      for t in range(h):
        z = fine_predictor(z, candidates[:, t], cond)    ← 链式 h 步

      cost = sum_token_dim((z - z_target)²)              ← Token-level cost
      top-k → update μ_fine, σ_fine

    all_fine_actions.extend(best_fine_for_segment_k)

═══ 执行 ═══

  a_0_real = fine_action_decoder(all_fine_actions[0])
  执行 → 新观测 o_1 → 回到第一级 (receding horizon)
```

**与训练分布的对齐点**（v2 关键升级）：

| 维度 | 训练 | 规划 | 一致性 |
|------|------|------|--------|
| Coarse predictor 链式步数 | K_train=2 | K=4 | 都 ≥2，分布相近 |
| Fine predictor 链式步数 | h | h | ✓ |
| Fine 段 k=0 起点 | encoder(o_0) | encoder(o_0) | ✓ |
| Fine 段 k>0 起点 | z_coarse_pred[k-1] | waypoints[k]=coarse 输出 | ✓ |
| Coarse_cond 来源 | curriculum→coarse 输出 | waypoints[k+1]=coarse 输出 | ✓ |
| 监督目标分布 | target_encoder | target_encoder(goal) | ✓ |
| Loss / cost 维度 | token-level | token-level | ✓ |

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

实验 F: 训练-规划对齐（v2 升级专项）
  9. Predictor 输出: pooled (B, D) vs tokens (B, num_tokens, D)
       → 验证 token 序列对多步 rollout 的必要性
  10. Coarse_segments K_train = 1 vs 2 vs 4
       → 验证 coarse 多步训练对长 horizon 规划的影响
  11. Teacher forcing schedule:
       (a) 全程 GT (tf=1)         → 训练快但与规划脱节
       (b) 全程链式 (tf=0)         → 早期训练发散
       (c) tf=1−curriculum (我们)   → 协同 schedule
  12. Cost 函数: pooled vs token-level
       → 验证 token-level cost 的判别力
  13. Goal encoder: online vs target encoder
       → 验证与 EMA 监督源对齐的影响
  14. 段间 fine 起点: encoder(o_h) vs coarse_predictor(z_0, â^c)
       → 验证训练分布与规划分布对齐
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

---

## 12. Methodology(公式化描述)

本节用统一的数学符号给出方法的完整定义,涵盖**模型组件**、**两阶段训练**与**两级规划推理**。

### 12.1 符号约定

| 符号 | 含义 |
|------|------|
| $o_t \in \mathbb{R}^{3 \times H \times W}$ $$a_t$$ | 第 $t$ 帧 RGB 观测 |
| $a_t \in \mathbb{R}^{d_a}$ | 第 $t$ 步真实动作(仅阶段二可见) |
| $p_t \in \mathbb{R}^{d_p}$ | 第 $t$ 步 proprioception |
| $h$ | 粗粒度跳步数(segment 长度) |
| $K$ | segment 数,总 horizon $H = Kh$ |
| $N$ | encoder 输出 token 数 |
| $D$ | latent 维度 |
| $d_f, d_c$ | fine / coarse latent action 维度 |
| $z_t \in \mathbb{R}^{N \times D}$ | online encoder 输出 |
| $\bar z_t \in \mathbb{R}^{N \times D}$ | EMA target encoder 输出(无梯度) |
| $\hat a^{f}_t \in \mathbb{R}^{d_f}$ | fine latent action |
| $\hat a^{c}_k \in \mathbb{R}^{d_c}$ | 第 $k$ 段 coarse latent action |
| $\theta, \bar\theta$ | online / target encoder 参数 |
| $\mathrm{pool}(\cdot)$ | token 平均池化:$\mathbb{R}^{N\times D} \to \mathbb{R}^{D}$ |
| $\mathrm{sg}[\cdot]$ | stop-gradient 算子 |

### 12.2 模型组件

**Encoder(online + EMA target)**

$$
z_t = E_\theta(o_t), \qquad \bar z_t = E_{\bar\theta}(o_t), \qquad \bar\theta \leftarrow \tau\,\bar\theta + (1-\tau)\,\theta
$$

其中 $\tau = 0.996$ 为动量系数。

**双尺度逆动力学**

$$
\hat a^{f}_t   = \mathcal{I}_f\bigl(\mathrm{pool}(z_t),\, \mathrm{pool}(z_{t+1})\bigr)
\quad \in \mathbb{R}^{d_f}
$$

$$
\hat a^{c}_k   = \mathcal{I}_c\bigl(\mathrm{pool}(z_{kh}),\, \mathrm{pool}(z_{(k+1)h})\bigr)
\quad \in \mathbb{R}^{d_c}
$$

**双尺度 Predictor**

$$
\hat z_{(k+1)h} = \mathcal{P}_c\bigl(z_{kh},\, \hat a^{c}_k\bigr)
\quad \in \mathbb{R}^{N \times D}
$$

$$
\hat z_{t+1} = \mathcal{P}_f\bigl(z_t,\, \hat a^{f}_t,\, c\bigr),
\qquad c \in \mathbb{R}^{N \times D} \text{ 为 coarse condition}
$$

$\mathcal{P}_f$ 内部为 self-attention + cross-attention(Q 来自 $z_t$ 与 action token,KV 来自 $c$)。

**Phase 2 翻译器与 proprio 解码器**

$$
\tilde a^{f}_t = \Phi_f(a_t),\qquad
\hat a^{f}_t \xrightarrow{\Phi_f^{-1}} a_t
$$

$$
\tilde a^{c}_k = \Phi_c(a_{kh:(k+1)h}),\qquad
\hat a^{c}_k \xrightarrow{\Phi_c^{-1}} a_{kh:(k+1)h}
$$

$$
\hat p_t = \Psi\bigl(\mathrm{pool}(z_t)\bigr) \in \mathbb{R}^{d_p}
$$

---

### 12.3 阶段一:自监督预训练

输入一段 $Kh+1$ 帧无标注视频 $\{o_0, o_1, \ldots, o_{Kh}\}$。

#### 步骤 1 — 编码

$$
z_t = E_\theta(o_t),\ t \in [0, Kh]
\qquad
\bar z_t = E_{\bar\theta}(o_t),\ t \in [1, Kh]
$$

#### 步骤 2 — 提取 latent actions

$$
\hat a^{f}_t = \mathcal{I}_f\bigl(\mathrm{pool}(z_t), \mathrm{pool}(z_{t+1})\bigr),\quad t \in [0, Kh-1]
$$

$$
\hat a^{c}_k = \mathcal{I}_c\bigl(\mathrm{pool}(z_{kh}), \mathrm{pool}(z_{(k+1)h})\bigr),\quad k \in [0, K-1]
$$

#### 步骤 3 — Coarse 链式 rollout

令 $u_0 = z_0$,递推

$$
u_{k+1} = \mathcal{P}_c(u_k, \hat a^{c}_k),\quad k = 0,\ldots,K-1
$$

定义 $z^{c}_k := u_{k+1}$ 为第 $k$ 段 coarse 预测输出。Coarse loss:

$$
\mathcal{L}_{\text{coarse}} = \frac{1}{K}\sum_{k=0}^{K-1}\, \mathcal{S}\bigl(z^{c}_k,\ \bar z_{(k+1)h}\bigr)
$$

其中 $\mathcal{S}(\cdot,\cdot)$ 为 token-level SmoothL1:

$$
\mathcal{S}(x, y) = \frac{1}{ND}\sum_{i=1}^{N}\sum_{j=1}^{D} \mathrm{smooth}_{L_1}\bigl(x_{ij} - y_{ij}\bigr)
$$

#### 步骤 4 — Fine 链式 rollout(段独立 + scheduled sampling)

设当前 epoch 比率

$$
\beta = \min\!\Bigl(1,\ \frac{\text{epoch}}{0.7\cdot \text{epochs}_{\max}}\Bigr) \in [0,1]
\qquad
\rho = 1 - \beta
$$

$\beta$ 为 curriculum 权重,$\rho$ 为 teacher-forcing 概率。

第 $k$ 段的 coarse condition:

$$
c_k = \beta \cdot \mathrm{sg}[z^{c}_k] + (1-\beta)\cdot \bar z_{(k+1)h-1}
$$

段起点(与规划对齐):

$$
v^{(k)}_0 =
\begin{cases}
z_0, & k = 0 \\
z^{c}_{k-1}, & k \geq 1
\end{cases}
$$

段内 $h$ 步链式($t = 0,\ldots,h-1$,$\,g = kh + t$):

$$
v^{(k)}_{t+1} = \mathcal{P}_f\bigl(v^{(k)}_t,\ \hat a^{f}_g,\ c_k\bigr)
$$

每步 loss

$$
\ell_g = \mathcal{S}\bigl(v^{(k)}_{t+1},\ \bar z_{g+1}\bigr)
$$

Scheduled sampling(下一步输入随机替换为 GT):

$$
v^{(k)}_{t+1} \leftarrow
\begin{cases}
z_{g+1}, & \text{w.p. } \rho \text{ (only if } t < h-1) \\
v^{(k)}_{t+1}, & \text{otherwise}
\end{cases}
$$

Fine loss:

$$
\mathcal{L}_{\text{fine}} = \frac{1}{Kh}\sum_{g=0}^{Kh-1}\ell_g
$$

#### 步骤 5 — 防 collapse 正则

设 batch 内所有 fine actions 集合 $\mathcal{A}_f = \{\hat a^{f}_g\}_{g}$,coarse 同理:

$$
\mathcal{L}_{\text{reg}}^{f} = \underbrace{\mathbb{E}_g\|\hat a^{f}_g\|_2^2}_{\text{magnitude}}
+ \lambda_v \underbrace{\max\!\bigl(0,\ \gamma - \sqrt{\mathrm{Var}(\mathcal{A}_f)+\epsilon}\bigr)}_{\text{variance}}
$$

$\mathcal{L}_{\text{reg}}^{c}$ 形式相同。

#### 步骤 6 — 总目标

$$
\boxed{\;
\mathcal{L}_{\text{phase1}}
= \mathcal{L}_{\text{fine}}
+ \lambda_c\,\mathcal{L}_{\text{coarse}}
+ \lambda_{rf}\,\mathcal{L}_{\text{reg}}^{f}
+ \lambda_{rc}\,\mathcal{L}_{\text{reg}}^{c}
\;}
$$

参数更新:

$$
\theta \leftarrow \theta - \eta\,\nabla_\theta \mathcal{L}_{\text{phase1}}
\qquad
\bar\theta \leftarrow \tau\bar\theta + (1-\tau)\theta
$$

**梯度通路总结**

| 信号路径 | 是否反传 |
|---|---|
| coarse rollout 段间 $u_k \to u_{k+1}$ | ✓ 不 detach |
| fine rollout 段内 $v_t \to v_{t+1}$ | ✓ 不 detach |
| 段间起点 $z^{c}_{k-1} \to v^{(k)}_0$ | ✓ 不 detach(fine loss 也帮 coarse 学) |
| coarse condition $z^{c}_k \to c_k$ | ✗ detach(stop-grad) |
| 监督目标 $\bar z_t$ | ✗ EMA encoder 永远无梯度 |

---

### 12.4 阶段二:动作对齐 + Proprio 解码

冻结 $\theta, \bar\theta, \mathcal{I}_f, \mathcal{I}_c, \mathcal{P}_f, \mathcal{P}_c$。在标注子集 $\mathcal{D}_{\text{lab}} = \{(o_t, a_{t:t+h}, p_{t:t+h})\}$ 上训练 $\Phi_f, \Phi_f^{-1}, \Phi_c, \Phi_c^{-1}, \Psi$。

设

$$
z_0 = E_\theta(o_0),\quad z_1 = E_\theta(o_1),\quad z_h = E_\theta(o_h)
$$

$$
\hat a^{f} = \mathcal{I}_f(\mathrm{pool}(z_0), \mathrm{pool}(z_1)),
\quad
\hat a^{c} = \mathcal{I}_c(\mathrm{pool}(z_0), \mathrm{pool}(z_h))
$$

$$
\hat z_h = \mathcal{P}_c(z_0, \hat a^{c}),
\quad
\hat z_1 = \mathcal{P}_f(z_0, \hat a^{f}, \hat z_h)
$$

#### Fine 翻译器

$$
\mathcal{L}_{\text{align}}^{f} = \|\Phi_f(a_0) - \mathrm{sg}[\hat a^{f}]\|_2^2
$$

$$
\mathcal{L}_{\text{dec}}^{f} = \|\Phi_f^{-1}(\hat a^{f}) - a_0\|_2^2,
\quad
\mathcal{L}_{\text{cyc}}^{f} = \|\Phi_f^{-1}(\Phi_f(a_0)) - a_0\|_2^2
$$

#### Coarse 翻译器(令 $\mathbf{a}_{0:h} = [a_0,\ldots,a_{h-1}]$)

$$
\mathcal{L}_{\text{align}}^{c} = \|\Phi_c(\mathbf{a}_{0:h}) - \mathrm{sg}[\hat a^{c}]\|_2^2
$$

$$
\mathcal{L}_{\text{dec}}^{c} = \|\Phi_c^{-1}(\hat a^{c}) - \mathbf{a}_{0:h}\|_2^2,
\quad
\mathcal{L}_{\text{cyc}}^{c} = \|\Phi_c^{-1}(\Phi_c(\mathbf{a}_{0:h})) - \mathbf{a}_{0:h}\|_2^2
$$

#### Proprio 解码器

在 encoder 与 predictor **两种 latent 分布**上训练:

$$
\mathcal{L}_{\text{prop}}
= \frac{1}{3}\sum_{t \in \{0,1,h\}} \|\Psi(\mathrm{pool}(z_t)) - p_t\|_2^2
+ \frac{1}{2}\sum_{\hat z \in \{\hat z_1, \hat z_h\}} \|\Psi(\mathrm{pool}(\hat z)) - p_\bullet\|_2^2
$$

#### 总目标

$$
\boxed{\;
\mathcal{L}_{\text{phase2}}
= \bigl(\mathcal{L}_{\text{align}}^{f} + \mathcal{L}_{\text{dec}}^{f} + 0.5\,\mathcal{L}_{\text{cyc}}^{f}\bigr)
+ \bigl(\mathcal{L}_{\text{align}}^{c} + \mathcal{L}_{\text{dec}}^{c} + 0.5\,\mathcal{L}_{\text{cyc}}^{c}\bigr)
+ \lambda_p\,\mathcal{L}_{\text{prop}}
\;}
$$

---

### 12.5 推理:逆动力学 Warm-Start 的两级 CEM 规划

输入:当前观测 $o_{\text{cur}}$、目标观测 $o_{\text{goal}}$、horizon $H = Kh$。

#### 编码(目标用 EMA encoder)

$$
z^{(0)} = E_\theta(o_{\text{cur}}),
\qquad
z^{\star} = E_{\bar\theta}(o_{\text{goal}})
$$

#### 第一级:Coarse CEM

**Step A — 逆动力学 warm-start**
$$
u_0 = z^{(0)};\quad
\hat a^{c,\text{init}}_k = \mathcal{I}_c(\mathrm{pool}(u_k), \mathrm{pool}(z^{\star})),
\quad u_{k+1} = \mathcal{P}_c(u_k, \hat a^{c,\text{init}}_k)
$$

初始化 CEM 分布:

$$
\boldsymbol{\mu}^{c,(0)} = [\hat a^{c,\text{init}}_0, \ldots, \hat a^{c,\text{init}}_{K-1}],
\qquad
\boldsymbol{\sigma}^{c,(0)} = 0.5 \cdot \mathbf{1}
$$

**Step B — CEM 迭代**($i = 1,\ldots,M_1$)

采样 $S$ 条候选:

$$
A^{(s)} = \boldsymbol{\mu}^{c,(i-1)} + \boldsymbol{\sigma}^{c,(i-1)} \odot \boldsymbol{\epsilon}^{(s)},
\quad \boldsymbol{\epsilon}^{(s)} \sim \mathcal{N}(0, I)
$$

链式 rollout:

$$
z^{(s)}_0 = z^{(0)},\quad
z^{(s)}_{k+1} = \mathcal{P}_c(z^{(s)}_k, A^{(s)}_k),\ k=0,\ldots,K-1
$$

**Token-level cost**:

$$
J^{(s)} = \|z^{(s)}_K - z^{\star}\|_F^2 = \sum_{n=1}^{N}\sum_{j=1}^{D} \bigl(z^{(s)}_{K,nj} - z^{\star}_{nj}\bigr)^2
$$

取 top-$L$ 精英集合 $\mathcal{E}^{(i)} = \{A^{(s)} : J^{(s)} \in \text{top-}L\}$:

$$
\boldsymbol{\mu}^{c,(i)} = \frac{1}{L}\sum_{A \in \mathcal{E}^{(i)}} A,
\qquad
\boldsymbol{\sigma}^{c,(i)} = \mathrm{std}(\mathcal{E}^{(i)})
$$

收敛后取 $A^{c\star} = \boldsymbol{\mu}^{c,(M_1)}$,生成 **waypoints** $\{w_0,\ldots,w_K\}$:

$$
w_0 = z^{(0)},\qquad w_{k+1} = \mathcal{P}_c(w_k, A^{c\star}_k)
$$

#### 第二级:Fine CEM(对每段 $k$ 独立执行)

记 $z^{\dagger} = w_k$,$\,z^{\ddagger} = w_{k+1}$。

**Step A — Fine 逆动力学 warm-start**

$$
v_0 = z^{\dagger};\quad
\hat a^{f,\text{init}}_t = \mathcal{I}_f(\mathrm{pool}(v_t), \mathrm{pool}(z^{\ddagger})),
\quad v_{t+1} = \mathcal{P}_f(v_t, \hat a^{f,\text{init}}_t, z^{\ddagger})
$$

$$
\boldsymbol{\mu}^{f,(0)} = [\hat a^{f,\text{init}}_0, \ldots, \hat a^{f,\text{init}}_{h-1}],\quad
\boldsymbol{\sigma}^{f,(0)} = 0.5 \cdot \mathbf{1}
$$

**Step B — CEM 迭代**($j = 1,\ldots,M_2$)

$$
B^{(s)} = \boldsymbol{\mu}^{f,(j-1)} + \boldsymbol{\sigma}^{f,(j-1)} \odot \boldsymbol{\epsilon}^{(s)}
$$

$$
v^{(s)}_0 = z^{\dagger},\quad
v^{(s)}_{t+1} = \mathcal{P}_f(v^{(s)}_t, B^{(s)}_t, z^{\ddagger})
$$

$$
J^{(s)} = \|v^{(s)}_h - z^{\ddagger}\|_F^2
$$

更新 $\boldsymbol{\mu}^{f}, \boldsymbol{\sigma}^{f}$ 同前。最终 $A^{f\star}_k = \boldsymbol{\mu}^{f,(M_2)}$。

#### 解码与执行

拼接所有段:$A^{f\star} = [A^{f\star}_0, \ldots, A^{f\star}_{K-1}] \in \mathbb{R}^{Kh \times d_f}$。

逐步解码为真实动作:

$$
a^{\text{real}}_g = \Phi_f^{-1}(A^{f\star}_g),\quad g = 0,\ldots,Kh-1
$$

执行 $a^{\text{real}}_0$,接收新观测 $o^\prime$,**receding horizon** 重新规划。

---

### 12.6 训练 ↔ 推理一致性表

| 维度 | 训练定义 | 推理定义 | 对齐机制 |
|------|---------|---------|----------|
| Coarse 起点 | $u_0 = z_0$ | $u_0 = z^{(0)} = E_\theta(o_{\text{cur}})$ | 同源 |
| Coarse 段间起点 | $u_k$(predictor 输出) | $u_k = \mathcal{P}_c(u_{k-1},\cdot)$ | 同分布 |
| Fine 段 0 起点 | $z_0$ | $w_0 = z^{(0)}$ | 同源 |
| Fine 段 $k\geq 1$ 起点 | $z^{c}_{k-1}$ | $w_k = \mathcal{P}_c(\cdots)$ | 同分布 |
| Coarse cond 来源 | $\beta z^{c}_k + (1-\beta)\bar z_\bullet \xrightarrow{\beta\to 1} z^{c}_k$ | $w_{k+1} = \mathcal{P}_c$ 输出 | curriculum 收敛 |
| 监督/cost 目标 | $\bar z_t$(EMA) | $z^{\star} = E_{\bar\theta}(o_{\text{goal}})$ | 同 encoder |
| Loss/cost 维度 | token-level $\sum_{n,j}$ | token-level $\|\cdot\|_F^2$ | 同度量 |
| Predictor 输出形态 | $(N,D)$ tokens | $(N,D)$ tokens | 同形状 |

这张对齐表是 v2 相对 v1 的核心改进体现:**所有训练时 predictor 看到的输入分布,在推理时都能被精确复现**,从根本上缓解 compounding error。


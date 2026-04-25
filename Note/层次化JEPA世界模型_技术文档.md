# 层次化 JEPA 世界模型 — 粗细粒度融合技术文档

## 1. 核心思想

LeWM 当前只能做短 horizon 规划（5步），自回归 rollout 误差累积严重。我们引入**两级 predictor**：

- **粗粒度 Predictor**：从 z_t 直接跳 h 步预测 z_{t+h}，负责长程规划骨架
- **细粒度 Predictor**：在粗粒度 waypoint 的指导下，逐步预测 z_{t+1}，负责具体动作

粗粒度信息通过 **Cross-Attention** 注入到细粒度预测过程中，不改变细粒度的输出目标。

---

## 2. 模型架构

### 2.1 总体结构

```
                          ┌─────────────────────────┐
                          │    Target Encoder (EMA)  │
                          │  提供 z_1^target, z_h^target │
                          └────────────┬────────────┘
                                       │ (supervision)
                                       ↓
pixels_0 ──→ [Encoder] ──→ z_0 (B, num_tokens, D)
                             │
              ┌──────────────┼──────────────────┐
              ↓                                  ↓
     [Coarse Predictor]                  [Fine Predictor]
     z_0 + actions[0:h]                 z_0 + a_0 + cond(ẑ_h)
              ↓                                  ↓
            ẑ_h ──→ L_coarse              ẑ_1 ──→ L_fine
              │                                  ↑
              └──── (curriculum blend) ──────────┘
                    粗粒度输出作为细粒度的条件
```

### 2.2 Encoder

输出 token 序列而非单一向量，为 Fine Predictor 的 Self-Attention 提供基础：

```
TokenEncoder: image (B, 3, H, W) → tokens (B, num_tokens, D)

Conv layers → AdaptiveAvgPool → Linear → Reshape → LayerNorm
```

Target Encoder 是 Encoder 的 EMA 副本（momentum=0.996），参数不参与梯度，仅用于生成监督目标。

### 2.3 Coarse Predictor

简单 MLP，输入池化后的 z_t + 拼接 h 步的 action/proprio：

```
输入: z_t.mean(dim=1) ∥ flatten(actions[0:h]) ∥ flatten(proprios[0:h])
     (B, D)             (B, h*action_dim)        (B, h*proprio_dim)

网络: Linear → ReLU → Linear → ReLU → Linear → LayerNorm

输出: ẑ_{t+h}  (B, D)
```

### 2.4 Fine Predictor（核心：Self-Attention + Cross-Attention）

#### 输入构造

```
z_t tokens: (B, num_tokens, D)    ← Encoder 输出的 token 序列
action_token: (B, 1, D)           ← action+proprio 经 Linear 投影

拼接 → x: (B, num_tokens + 1, D)
```

#### Transformer Block 内部（共 num_layers 层）

```
┌────────────────────────────────────────────────────┐
│                                                    │
│   x ──→ LayerNorm ──→ Self-Attention ──→ + ──→ x  │
│   │                                         ↑     │
│   └─────────────────────────────────────────┘     │
│                                                    │
│   x ──→ LayerNorm ──→ Cross-Attention ──→ + ──→ x │
│   │                    Q=x  K/V=coarse      ↑     │
│   └─────────────────────────────────────────┘     │
│                                                    │
│   x ──→ LayerNorm ──→ FFN ──→ + ──→ x             │
│   │                            ↑                   │
│   └────────────────────────────┘                   │
│                                                    │
└────────────────────────────────────────────────────┘
```

**Self-Attention**：tokens 之间互相 attend，捕捉 z_t 各 token 之间的关系以及和 action 的交互。

**Cross-Attention**：
- Q 来自当前 fine tokens（细粒度）
- K, V 来自 coarse_proj(ẑ_{t+h})（粗粒度方向）
- 细粒度主动查询粗粒度中的方向信息
- coarse_kv 的 shape 是 (B, 1, D)，即只有一个 token 作为 K/V

**FFN**：Linear → GELU → Linear

#### 输出

```
x: (B, num_tokens+1, D) → mean pool → LayerNorm → Linear → ẑ_{t+1}: (B, D)
```

---

## 3. 监督与 Loss

两个 predictor 各自有独立的监督目标，都来自 Target Encoder：

```
L_coarse = SmoothL1(ẑ_{t+h}, z_{t+h}^{target})     # z_{t+h}^{target} = target_encoder(o_{t+h}).mean(dim=1)
L_fine   = SmoothL1(ẑ_{t+1}, z_{t+1}^{target})      # z_{t+1}^{target} = target_encoder(o_{t+1}).mean(dim=1)

L_total  = L_fine + λ * L_coarse
```

**关键点**：粗粒度信息只在 Fine Predictor 的计算过程中通过 Cross-Attention 注入，不改变输出的监督目标。细粒度的目标始终是 z_{t+1}^{target}。

---

## 4. Curriculum Learning（解决 train-test mismatch）

### 问题

训练时 Fine Predictor 的 coarse condition 如果总是用 Target Encoder 的 GT 值 z_h^{target}（很准），推理时只有 Coarse Predictor 的预测值 ẑ_h（有误差），会导致 mismatch。

### 解决

线性 curriculum，逐步把 condition 从 GT 切换到预测值：

```
curriculum_ratio = min(1.0, epoch / (total_epochs × 0.7))

coarse_cond = ratio × ẑ_h^{coarse}  +  (1 - ratio) × z_h^{target}
```

| 训练阶段 | curriculum_ratio | condition 来源 |
|---------|-----------------|--------------|
| 前期 (epoch 1~30%) | 0.0 ~ 0.4 | 主要是 GT，让模型先学会基本预测 |
| 中期 (30%~70%) | 0.4 ~ 1.0 | 逐渐过渡到预测值 |
| 后期 (70%~100%) | 1.0 | 完全用预测值，和推理一致 |

注意 coarse_cond 要 **detach**，不让 fine loss 的梯度回传到 coarse predictor（两个 predictor 各自用各自的 loss 训练）。

---

## 5. 规划流程（Inference）

### 已知

- o_0：当前观测
- z_goal：目标状态

### Step 1: CEM 采样候选 action 序列

```
candidates = CEM.sample(N 条, 每条长 H 步)
```

### Step 2: 对每条候选做 hierarchical rollout

```
对候选 i 的 action 序列 [a_0, a_1, ..., a_{H-1}]:

  z_0 = encoder(o_0)

  # ---- 粗粒度：生成 waypoints ----
  ẑ_h  = coarse_predictor(z_0, actions[0:h])
  ẑ_{2h} = coarse_predictor(ẑ_h, actions[h:2h])
  ...

  # ---- 细粒度：在 waypoints 之间逐步展开 ----
  # 第一段：z_0 → ẑ_h，condition = ẑ_h
  ẑ_1 = fine_predictor(z_0,  a_0, cond=ẑ_h)
  ẑ_2 = fine_predictor(ẑ_1, a_1, cond=ẑ_h)
  ...
  ẑ_h = fine_predictor(ẑ_{h-1}, a_{h-1}, cond=ẑ_h)

  # 第二段：ẑ_h → ẑ_{2h}，condition = ẑ_{2h}
  ẑ_{h+1} = fine_predictor(ẑ_h, a_h, cond=ẑ_{2h})
  ...

  # 计算 cost
  cost_i = ||ẑ_H - z_goal||²
```

### Step 3: CEM 更新

```
取 cost 最低的 top-k 条候选
更新采样分布 (mean, var)
重复 Step 1-3 共 M 轮
```

### Step 4: 执行

```
取最终最优 action 序列的第一个 a_0，执行
拿到新观测 o_1，重新从 Step 1 开始（receding horizon）
```

---

## 6. 训练 vs 规划 对比

| | z_0 来源 | actions 来源 | coarse condition | 监督目标 |
|---|---|---|---|---|
| **训练** | encoder(数据集 o_0) | 数据集 GT actions | curriculum blend | z_{t+1}^{target}, z_{t+h}^{target} |
| **规划** | encoder(当前观测) | CEM 采样优化 | coarse predictor 预测 | 无（用 cost 评估） |

---

## 7. 超参数参考

| 超参数 | 值 | 说明 |
|-------|-----|------|
| latent_dim | 256 | latent 向量维度 |
| num_tokens | 4 | Encoder 输出 token 数量 |
| horizon_h | 5 | 粗粒度跳步数 |
| nhead | 4 | attention head 数量 |
| num_fine_layers | 3 | Fine Predictor 的 Transformer block 数量 |
| dim_ff | 512 | FFN 隐层维度 |
| ema_momentum | 0.996 | Target Encoder EMA 动量 |
| lambda_coarse | 1.0 | 粗粒度 loss 权重 |
| curriculum_warmup | 0.7 | 训练前 70% 线性增加 curriculum ratio |
| lr | 3e-4 | 学习率 |
| optimizer | AdamW | 带 weight decay |
| scheduler | CosineAnnealing | 余弦退火 |

---

## 8. 代码位置

完整实现: `src/hierarchical_jepa_worldmodel.py`

运行命令:
```bash
cd /Users/guanchendu/Code/StudyOnWM/src
python hierarchical_jepa_worldmodel.py --horizon-h 5 --num-steps 8 --epochs 100
```

---

## 9. 下一步：结合逆动力学

当前框架依赖数据集中的动作标签。下一步计划：

1. 在粗粒度层去掉动作输入，用逆动力学模型自监督学习 latent action
2. 细粒度层保留少量动作标签做规划
3. 形成完整的 "层次化 JEPA + 逆动力学" 框架

这个组合可以同时解决 **长 horizon 规划** 和 **动作标注依赖** 两个问题，是投稿的核心故事。

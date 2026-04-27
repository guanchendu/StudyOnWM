# src_wm: Hierarchical JEPA + 单尺度逆动力学世界模型

在 TwoRoom 环境上做 goal-conditioned 规划的世界模型。架构是**单尺度细粒度逆动力学 + 粗细粒度层次化预测器**。逆动力学只在 Phase 1 训练时当作 latent action 的监督信号，**规划时不调用**，避免分布外问题。

---

## 1. 整体方法

```
                    Phase 1（无标签像素）
       ┌──────────────────────────────────────────────────┐
       │  encoder ─┐                                      │
       │           │                                      │
       │           ↓        ┌── coarse_predictor ─┐       │
       │     latent z  ────►│   z_t + h fine acts │       │
       │           │        └─────────┬───────────┘       │
       │           ↓                  ↓ ẑ_{t+h}           │
       │     inverse_dyn ──► â_t                          │
       │           ↓                  ↓                   │
       │           └────────► fine_predictor ──► ẑ_{t+1}  │
       └──────────────────────────────────────────────────┘
                            │
                            ▼ 冻结
                    Phase 2（少量带标）
       ┌──────────────────────────────────────────────────┐
       │  action_encoder:  a_t (real)   →  ã_t (latent)   │
       │  action_decoder:  â_t (latent) →  a_t (real)     │
       │  proprio_decoder: z (latent)   →  proprio        │
       └──────────────────────────────────────────────────┘
                            │
                            ▼
                    Evaluate（CEM 规划）
       ┌──────────────────────────────────────────────────┐
       │  CEM 在真实 action 空间采样                      │
       │     ↓ action_encoder                             │
       │  latent_actions                                  │
       │     ↓ hierarchical_rollout (coarse + fine)       │
       │  z_final                                         │
       │     ↓ proprio_decoder                            │
       │  cost = ‖proprio_pred - goal_proprio‖²           │
       └──────────────────────────────────────────────────┘
```

**核心设计选择**

- **单尺度 IDM**：只学 `(z_t, z_{t+1}) → â_t`，不学 coarse IDM。避免"`(z_current, z_goal_far)`"这种训练分布外的查询。
- **层次化预测**：coarse_predictor 一次走 h 步给 waypoint；fine_predictor 通过 cross-attention 把 waypoint 当作条件，做单步细化。
- **planning 不调 IDM**：候选动作来自 CEM，经 action_encoder 映射到 latent action 空间后喂给 predictor。IDM 仅训练时使用。
- **真实 cost**：`proprio_decoder` 把 latent 状态翻译到 proprio 空间，与 `goal_proprio` 计算 MSE。

---

## 2. 文件结构

| 文件 | 职责 |
|------|-----|
| [models.py](models.py) | 所有 nn.Module（encoder / IDM / 两个 predictor / action enc-dec / proprio decoder） |
| [train_phase1.py](train_phase1.py) | 自监督预训练，无 action 标签，多步 rollout + scheduled sampling + VICReg 状态正则 |
| [train_phase2.py](train_phase2.py) | 冻结 Phase 1，训 action_enc/dec + proprio_decoder，少量带标数据 |
| [planning.py](planning.py) | `HierarchicalCostModel`：兼容 `swm.CEMSolver` 的 cost 模型 |
| [evaluate.py](evaluate.py) | 在 swm.World 里跑 MPC + CEM，统计 success_rate |

---

## 3. 模型组件（models.py）

### 3.1 TokenEncoder
4 层卷积 + AdaptiveAvgPool2d(2,2) → Linear → 重排为 `(B, num_tokens, D)`，每个 token 过 LayerNorm。
**保留 token 结构**是关键：fine_predictor 的 self-attention 要在 token 之间做交互，而不是单一向量。

### 3.2 EMATargetEncoder
`encoder` 的 EMA 副本（默认 momentum=0.996），无梯度，提供 `smooth_l1_loss(z_pred, z_target)` 的稳定 target。

### 3.3 InverseDynamicsModel（单尺度）
3 层 MLP：`pool(z_t) || pool(z_{t+1}) → â_t`。Bottleneck（`latent_action_dim ≪ latent_dim`）防止把 z_{t+1} 全部塞进 â_t。

### 3.4 CoarsePredictor
3 层 MLP，输入 `pool(z_t) || flatten(â_0..â_{h-1})`，输出 `num_tokens * latent_dim` 后 reshape + LayerNorm，得到 `(B, num_tokens, D)`。
**输出是 token 序列**（不是 (B, D)），rollout 时不会丢掉 token 结构。

### 3.5 FinePredictor
Pre-norm Transformer，每层 SelfAttn → CrossAttn → FFN：
- 输入 token：`[z_t 的 N 个 token, action_token]`
- cross-attention 的 K/V 来自 `coarse_proj(coarse_cond)`（waypoint）
- 输出截取前 N 个 token，过 LayerNorm + Linear，得到 `(B, num_tokens, D)`

### 3.6 ActionEncoder / ActionDecoder（Phase 2）
两个对称的 3 层 MLP，把真实 action 与 latent action 互相映射。Phase 2 同时训 align loss + decode loss + cycle loss。

### 3.7 ProprioDecoder（Phase 2）
3 层 MLP：`pool(z) → proprio`。Phase 2 在 **encoder 输出 + predictor 输出** 两套分布上同时监督，保证 planning 时 cost 在分布内。

### 3.8 HierarchicalInvDynWorldModel
组装上述组件，提供 `encode / encode_target / compute_latent_action / update_target_encoder / pool`。

---

## 4. Phase 1 训练（train_phase1.py）

### 4.1 数据
- `swm.HDF5Dataset(name="tworoom")`，`num_steps = K*h + 1`（默认 K=2, h=5 → 11 帧）
- `frameskip=5`：每个 latent 时间步对应 5 帧环境步
- 不使用 action / proprio 标签

### 4.2 损失（统一 chained rollout，与 planning 完全对齐）

设 K = `coarse_segments`，h = `horizon_h`。

> **更新（P2 修复）**：Phase 1 的 rollout 结构现已与 [planning.py](planning.py) `hierarchical_rollout` 严格一致：
> 每段 coarse 和 fine 的起点都用**上一段 fine rollout 的最后一步输出**（不是上一段 coarse 输出）。
> 训练分布与规划分布完全对齐，避免多段规划（horizon > h）时的分布偏移。

1. **编码**：online 编码全部 K*h+1 帧，target encoder 编码 1..Kh 帧
2. **逆动力学**：算 K*h 个 fine action `â_t = inv(z_t, z_{t+1})`
3. **统一 chained rollout**（与 planning 同款）：
   ```
   z_current = z_0  # 真实 encoder 输出
   for k in 0..K-1:
       z_waypoint = coarse_predictor(z_current, [â_{kh}..â_{(k+1)h-1}])
       coarse_loss += smooth_l1(z_waypoint, z_target_{(k+1)h-1})

       z_t = z_current
       for t in 0..h-1:
           z_t = fine_predictor(z_t, â_{kh+t}, coarse_cond)
           fine_loss += smooth_l1(z_t, z_target_{kh+t})
           # 段内 scheduled sampling: 偶尔把 z_t 换成 GT
       z_current = z_t   # ← 关键: 下一段的 coarse AND fine 都从 fine 输出开始
   ```
   - 段间**没有** teacher forcing（planning 没有 GT 可换，所以 train 也不换）
   - `coarse_cond` 在 curriculum 下从 EMA target 渐变到 coarse 预测
4. **action 正则**：L2 + per-dim variance threshold（防 collapse）
5. **state 正则**（VICReg 风格）：
   - variance：`F.relu(1.0 - std(z, dim=batch))` per-dim
   - covariance：off-diag(z 的协方差矩阵) 的 Frobenius 范数 / D
   - 防止 encoder/EMA 一起坍塌成常数

### 4.3 关键监控
- `state_std_min` —— 应明显 > 0（建议 > 0.5）。接近 0 说明 dimensional collapse
- `coarse_loss` / `fine_loss` —— 在 curriculum 完全进入自回归后还能保持 → multi-step rollout 真的学到了
- `action_var` —— 防止 latent action 退化为常数

### 4.4 命令
```bash
cd /Users/guanchendu/Code/StudyOnWM
conda run -n wm python -m src.src_wm.train_phase1 \
  --epochs 100 --coarse-segments 2 --num-steps 11
```

输出：`src/outputs/hierarchical_invdyn/best_phase1.pt`

---

## 5. Phase 2 训练（train_phase2.py）

> **更新（P1 修复）**：Phase 2 的 proprio decoder 现在在**完整 chained rollout 的全部中间状态**上监督，
> 而不是只看单步 `z_h_pred` / `z_1_pred`。这样 planning 时 CEM 真正打分的 latent 分布
> 就在 `proprio_dec` 的训练分布之内。

### 5.1 数据
- 序列长度自动设为 `K*h+1`（默认 K=2, h=5 → 11 帧）
- K 默认从 Phase 1 checkpoint 的 `coarse_segments` 字段读取，可用 `--coarse-segments` 覆盖
- 用 `--label-fraction 0.1` 模拟少量带标
- 用 action + proprio 标签

### 5.2 损失

**Action 对齐**（不变）：
```
L_align    = MSE(action_enc(a_t), inv_dyn(z_t, z_{t+1}))
L_decode   = MSE(action_dec(â_t), a_t)
L_cycle    = MSE(action_dec(action_enc(a_t)), a_t)
```

**Proprio 监督**（核心更新）—— 三条监督路径：
```
(1) Encoder 输出（覆盖广）：
    L_p_enc  = mean_t MSE(proprio_dec(encoder(pixels_t)), proprio_t)   # 所有 K*h+1 帧

(2) IDM-action 路径的 chained rollout（匹配 Phase 1 训练分布）：
    z_current = encoder(pixels_0)
    for k in 0..K-1:
        z_wp = coarse_predictor(z_current, idm_actions[kh:(k+1)h])
        L_p_wp_idm += MSE(proprio_dec(z_wp), proprio_{(k+1)h})
        z_t = z_current
        for t in 0..h-1:
            z_t = fine_predictor(z_t, idm_actions[kh+t], z_wp)
            L_p_fine_idm += MSE(proprio_dec(z_t), proprio_{kh+t+1})
        z_current = z_t

(3) Real-action 路径的 chained rollout（匹配 planning 真实分布）：
    同 (2)，但 latent action 来自 action_encoder(real_action)
    → 让 action_encoder 也参与梯度，学到能让 proprio_dec 出准确预测的映射
```

**总损失**：
```
L_proprio  = L_p_enc
           + 0.5 * (L_p_fine_idm  + L_p_wp_idm)
           + λ_real * 0.5 * (L_p_fine_real + L_p_wp_real)
L_total    = (L_align + L_decode + 0.5*L_cycle) + λ_proprio * L_proprio
```

为什么需要两条 rollout 路径：
- **(2) IDM 路径**：Phase 1 的 predictor 是用 IDM latent action 训出来的，所以 (2) 是 predictor 的"原生"输入分布
- **(3) Real-action 路径**：planning 时 latent action 来自 `action_encoder(real_action)`，分布与 IDM 输出可能不同 → 必须直接监督这条路径，否则 proprio_dec 在 planning 时仍然分布外
- 两者结合让 proprio_dec 在两套分布上都准确

### 5.3 命令
```bash
cd /Users/guanchendu/Code/StudyOnWM
conda run -n wm python -m src.src_wm.train_phase2 \
  --phase1-ckpt src/outputs/hierarchical_invdyn/best_phase1.pt \
  --label-fraction 0.1 --epochs 50
# 可选：覆盖 K 或调整 real-path 权重
#   --coarse-segments 2 --lambda-real-path 1.0
```

输出：`src/outputs/hierarchical_invdyn/best_phase2.pt`

### 5.4 关键监控
- `proprio_idm_fine` / `proprio_real_fine`：两条路径在 fine rollout 输出上的 MSE。**两者都应该收敛到与 `proprio_enc` 同量级**。
- 如果 `proprio_real_fine >> proprio_idm_fine` → action_encoder 还没学好；调大 `--lambda-real-path` 或加训
- 如果 `proprio_idm_fine >> proprio_enc` → predictor 输出分布太偏离 encoder 分布；可能 Phase 1 训得不够（multi-step rollout 还没收敛）

---

## 6. 规划与评估（planning.py + evaluate.py）

### 6.1 HierarchicalCostModel.get_cost

```python
# input: candidates (B, S, horizon, action_block * action_dim)
# 1. 编码当前像素 → z_start (B*S, num_tokens, D)
# 2. real action → action_encoder → latent_actions (B*S, total_steps, latent_action_dim)
# 3. hierarchical rollout：
#    for each coarse segment of h fine actions:
#       z_waypoint = coarse_predictor(z_current, h actions)
#       for t in range(h):
#           z_current = fine_predictor(z_current, action_t, z_waypoint)
# 4. proprio_pred = proprio_dec(z_final)
# 5. cost = ||proprio_pred - goal_proprio||²
```

### 6.2 evaluate.py（MPC）
- `swm.World` × `swm.CEMSolver` × `WorldModelPolicy`
- 每 `receding_horizon` 步重规划一次
- `action_block` = 5：每个规划动作在环境里重复 5 帧（与训练 frameskip 对齐）
- `goal_offset_steps` = 25：起点和目标在数据集上相隔多少步
- `eval_budget` = 50：每个 episode 最多多少环境步
- 启动时自动检查两个 checkpoint 是否存在，找不到给出友好提示
- `--save-video` / `--no-save-video`（用 `argparse.BooleanOptionalAction`，避免老版本 `type=bool` 的陷阱）

### 6.3 命令
```bash
cd /Users/guanchendu/Code/StudyOnWM
conda run -n wm python -m src.src_wm.evaluate \
  --phase1-ckpt src/outputs/hierarchical_invdyn/best_phase1.pt \
  --phase2-ckpt src/outputs/hierarchical_invdyn/best_phase2.pt \
  --num-eval 50
```

输出：`src/outputs/hierarchical_invdyn/eval_src_wm.json` + `eval_videos/`

---

## 7. 关键设计决策与对比

### 7.1 为什么单尺度 IDM
- 主流（VPT / LAPO / Diffuser / Director）**都不在 plan 时直接调 IDM 处理 (current, far_goal)**
- 训练 IDM 见过的输入都是相邻帧；planning 时如果让它处理相隔很远的 (z_current, z_goal)，分布外
- 本实现遵循 VPT 风格：**IDM 只在训练阶段提供 latent action 监督**，planning 时不调用

### 7.2 为什么 predictor 输出 token 而不是 (B, D)
旧版本两个 predictor 都用了 `mean(dim=1)` 把 token 池化掉，rollout 时只能 `unsqueeze(1).expand(N, ...)` 把同一个向量复制 N 份 —— self-attention 退化成恒等映射，多 token 设计失效。
新版本 predictor 直接输出 `(B, num_tokens, D)`，token 之间的差异被保留。

### 7.3 为什么 cost 必须用 proprio_dec
旧版本 `cost = z_final.norm()` 是 placeholder，CEM 在最小化 latent 范数，与目标无关。
现在的 `cost = ‖proprio_dec(z) - goal_proprio‖²` 才是真正"朝目标走"的代价函数。

### 7.4 为什么需要 multi-step chained rollout 训练（且 train/planning 完全对齐）
旧版本只训 z_0 → z_1 一步，但 planning 时 fine 要连续滚 h 步、coarse 要连续滚 K 段，分布外 → compounding error。
**当前版本**：Phase 1 训练用与 [planning.py](planning.py) `hierarchical_rollout` **逐字一致**的链式结构 —— 每段 coarse/fine 起点都是上一段 fine rollout 的最终输出。这样无论规划多少段，分布都不偏移。

### 7.5 为什么 Phase 2 需要 chained rollout supervision（不只是单步）
旧版本只在 `z_h_pred` 和 `z_1_pred`（单步 predictor 输出）上训 `proprio_dec`，但 planning 真正打分的 latent 是**完整 chained rollout 后的 z_final**，可能经过 K 段 + h*K 步 fine。
**当前版本**：Phase 2 完整复刻 planning 的 chained rollout，把每段 coarse waypoint 和每步 fine 输出都加进 `proprio_dec` 损失。
而且跑**两条 rollout 路径**：(a) IDM-action 路径（匹配 Phase 1 训练分布）+ (b) real-action via `action_encoder` 路径（匹配 planning 真实分布）。这样 `proprio_dec` 在两套分布上都准确，CEM 评分才可信。

### 7.6 为什么需要 state variance reg
只用 `smooth_l1(z_pred, z_target)` 时，encoder 和 EMA 一起坍塌成常数也能让 loss → 0。LayerNorm 能挡掉最 trivial 的常数坍塌但挡不住 dimensional collapse。VICReg 的 variance + covariance 项在每个 batch 维度上强制 z 有信息量。

---

## 8. 默认超参速查

| 超参 | 默认 | 含义 |
|------|------|------|
| `latent_dim` | 256 | 状态 latent 维度 |
| `latent_action_dim` | 32 | latent action 维度（bottleneck） |
| `num_tokens` | 4 | 每帧 token 数 |
| `horizon_h` | 5 | 一段 coarse 包含的 fine 步数 |
| `coarse_segments` | 2 | 训练时 K 段链式 coarse rollout |
| `num_steps` | 11 | 一个样本的帧数（≥ K*h+1） |
| `frameskip` | 5 | 每 latent 步在环境中跳几帧 |
| `ema_momentum` | 0.996 | target encoder EMA |
| `lambda_coarse` | 1.0 | coarse loss 权重 |
| `lambda_reg_action` | 0.01 | action 正则权重 |
| `lambda_state_var` | 1.0 | state variance 正则权重 |
| `lambda_state_cov` | 0.04 | state covariance 正则权重 |
| `curriculum_warmup` | 0.7 | 训练前 70% 走 GT 监督 → 后续切到自回归 |
| `--label-fraction` (P2) | 0.1 | 用多少比例的带标数据 |
| `--coarse-segments` (P2) | 跟 P1 | Phase 2 K 段链式 rollout，默认从 P1 ckpt 读 |
| `--lambda-proprio` (P2) | 1.0 | proprio 总损失权重 |
| `--lambda-real-path` (P2) | 1.0 | real-action rollout proprio 损失权重（0 = 只用 IDM 路径） |
| Phase 2 `num_steps` | K*h+1 | 自动设置（≥ chained rollout 所需长度） |
| Eval `horizon` | 5 | 一次规划多少 latent 动作 |
| Eval `receding_horizon` | 5 | 每多少步重规划 |
| Eval `action_block` | 5 | 每个 latent 动作重复几帧 |
| Eval `num_samples` | 300 | CEM 每步采样数 |
| Eval `cem_steps` | 30 | CEM 迭代步数 |
| Eval `topk` | 30 | CEM 精英数 |

---

## 9. Checkpoint 字段

### best_phase1.pt
```python
{
  "model_state_dict": ...,   # HierarchicalInvDynWorldModel 完整权重
  "config": vars(args),      # 包含 latent_dim, horizon_h, num_tokens, ...
  "best_val_loss": float,
  "epoch": int,
}
```

### best_phase2.pt
```python
{
  "action_encoder_state_dict": ...,
  "action_decoder_state_dict": ...,
  "proprio_dec_state_dict": ...,
  "config": vars(args),             # 含 hidden_dim, proprio_hidden_dim, lambda_real_path
  "phase1_config": p1_cfg,
  "coarse_segments": int,           # K used at training time
  "action_dim": int,
  "proprio_dim": int,
  "best_val_loss": float,
  "epoch": int,
}
```

---

## 10. 已知限制与潜在改进

- **逆动力学的可辨识性**：在像素信息不足（缺速度等）时 IDM 学到的可能是"transition code"而不是真实控制量。可考虑用多帧输入。
- **VICReg 阈值是固定的**：`variance_threshold=1.0` 是经验值，不一定最优。可以试 0.5 / 2.0 看 `state_std_min` 的稳定性。
- **CEM 默认参数偏保守**：`num_samples=300, cem_steps=30` 适用于 Mac MPS；GPU 上可放大到 512+/15 步。
- **没有 inverse-dynamics warm-start**：和 src_wm2 不同，本版本完全靠 CEM 的随机初始化搜索动作。如果想试 warm-start，需要单独训一个 `goal_conditioned_policy(z_current, z_goal) → first action`，而不是直接复用 IDM。
- **Phase 2 计算量随 K 线性增长**：chained rollout supervision 让 Phase 2 一个 batch 要跑 2*K*h 次 fine_predictor + 2*K 次 coarse_predictor。K=2、h=5 时大约是旧版的 10 倍（旧版只跑 1 次 fine + 1 次 coarse）。`--batch-size` 已从 256 降到 128 适配；GPU 上可调回。

---

## 11. 改动历史（按时间倒序）

### v3 — Train/Planning 分布完全对齐（最新）
- **P1 修复**：Phase 2 的 `proprio_dec` 现在在完整 chained rollout 的所有中间状态上监督，而不是只看单步 `z_h_pred` / `z_1_pred`。新增两条 rollout 路径（IDM + real-action），保证 planning 时 CEM 评分的 latent 在 `proprio_dec` 训练分布内。
- **P2 修复**：Phase 1 的 chained rollout 改为与 [planning.py](planning.py) `hierarchical_rollout` 逐字一致 —— 每段 coarse/fine 起点都用上一段 fine 输出（不再用上一段 coarse 输出）。多段规划时不再有训练/规划分布偏移。
- **操作坑修复**：`--save-video` 改用 `argparse.BooleanOptionalAction`（`--no-save-video` 关闭，避免 `type=bool "False" → True` 陷阱）；`evaluate.py` 启动时检查两个 checkpoint 是否存在，缺失给出友好错误并提示训练命令；`train_phase2.py` 同样检查 Phase 1 checkpoint 存在。
- 文件：[train_phase1.py:175-225](train_phase1.py:175)、[train_phase2.py:106-216](train_phase2.py:106)、[evaluate.py:177-186](evaluate.py:177)。

### v2 — 修复 5 个评估闭环问题
- ① cost 从 `z_final.norm()` placeholder 换成 `‖proprio_dec(z) - goal_proprio‖²`
- ② 新增 `ProprioDecoder` 类
- ③ `CoarsePredictor` / `FinePredictor` 输出从 `(B, D)` 改为 `(B, num_tokens, D)`，保留 token 结构
- ④ Phase 1 fine 训练从单步 z_0→z_1 改为 K*h 步 multi-step rollout + scheduled sampling
- ⑤ 新增 VICReg-style state variance + covariance reg，加入 `state_std_min` 监控
- 删除 `plan_single_step` 孤儿代码

### v1 — 初始版本
- 单尺度细粒度 IDM + coarse/fine 层次化 predictor
- Phase 1 自监督 + Phase 2 action align（无 proprio_dec）
- cost 是 placeholder，evaluate 路径未闭环

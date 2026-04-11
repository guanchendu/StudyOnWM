# leworld model 可以做的方向

## 1. 长horizon规划

论文自己也承认，当前只能做短horizon规划（5步），自回归rollout会累积误差。可以引入层次化世界模型，在不同时间尺度上建模，实现粗到细的规划。

思路

1.Memory机制

当前predictor只看最近N帧（N=3）的context，长horizon rollout时早期信息会丢失。

**具体做法**：可以给predictor加一个外部memory，比如：

- **压缩memory**：每隔K步把当前latent状态压缩写入一个memory bank，predictor在预测时可以通过cross-attention读取历史关键状态。这样即使rollout很长，也能回顾起点和中间关键节点。
- **类似Transformer-XL的recurrence**：把之前segment的hidden states缓存下来，扩展有效感受野而不增加计算复杂度。

2.层次化粗细粒度对齐

**两层结构**：

- **粗粒度predictor**：输入 z_t，直接预测 z_{t+k}（跳K步），负责长程规划骨架
- **细粒度predictor**：就是现有的单步predictor，负责在粗粒度waypoints之间填充具体动作

**规划流程**：

1. 先用粗粒度predictor规划出稀疏的waypoint序列：z_0 → z_K → z_{2K} → ... → z_goal
2. 再用细粒度predictor在相邻waypoints之间做短horizon规划

**训练时的对齐**：这是核心难点。两层predictor需要在同一个latent space里工作，对齐方式可以是：

- **共享encoder**：两个predictor都作用于同一个encoder的输出，粗粒度predictor的target就是encoder(o_{t+K})，天然对齐
- **一致性loss**：细粒度predictor连续rollout K步的结果 ẑ_{t+K}，应该和粗粒度predictor一步预测的结果接近，加一个对齐损失

**两个思路可以结合**：粗粒度predictor产生的waypoints本身就可以作为memory，供细粒度predictor在做局部规划时参考目标方向，避免局部规划偏离全局路径。

---------



## 2. 随机建模

在predictor是确定性的，无法捕捉环境的随机性。可以让predictor输出分布（如高斯混合或用flow-based方法），这样在随机环境中更鲁棒，

思路：

我想到的最简单的方法

```python
ẑ_{t+1} = predictor(z_t, a_t) + σ · ε,   ε ~ N(0, I)
```



σ 可以是：

- **固定常数**：最简单，当作超参数调
- **可学习标量**：让网络自己学一个全局σ
- **依赖状态的**：σ(z_t, a_t)，用一个小MLP输出，这样不确定性大的状态（比如碰撞瞬间）σ自动变大

## 在规划时怎么用

原来CEM只rollout一条确定性轨迹来评估每个action序列。加了随机性后，可以对同一个action序列**采样多条轨迹**，用平均cost来评估：

```
对每个候选action序列:
    for i in 1..N:
        rollout一条带随机扰动的轨迹，得到 cost_i
    最终cost = mean(cost_1, ..., cost_N)
```

这样规划天然就考虑了不确定性，更鲁棒。



训练时也需要适配。如果predictor输出的是分布，loss从MSE变成负对数似然：

```
L = -log p(z_{t+1} | z_t, a_t)
  = ||z_{t+1} - ẑ_{t+1}||² / (2σ²) + log σ
```

这样σ不会无限大（因为有log σ惩罚），也不会塌缩到0（因为要容纳预测误差）。这其实就是学一个异方差高斯，但实现上只比现在多输出一个σ，非常轻量。

claude给出的几种打法

1.Diffusion-based Predictor

把下一步预测建模为去噪过程：

```
训练：对 z_{t+1} 加噪得到 z_{t+1}^{noisy}，让predictor学去噪
推理：从纯噪声出发，条件在 (z_t, a_t) 上逐步去噪得到 ẑ_{t+1}
```

表达力最强，能建模任意复杂的多模态分布，但推理需要多步去噪，速度慢。

2.把连续 latent 离散化为有限个 token，predictor 输出下一个 token 的**分类概率**：

```
p(z_{t+1} = k | z_t, a_t),  k = 1, ..., K
```

天然就是分布，可以直接采样。类似 IRIS、VQ-VAE 的思路。

----

## 3.SIGReg在低维环境的局限

论文在Two-Room上表现不佳，因为强制高维latent匹配各向同性高斯与环境的低内在维度冲突。一个改进是自适应地学习latent空间的有效维度，或用更灵活的先验分布替代固定的各向同性高斯。

**这个应该是数学系那边搞的我不太懂**

-----

## 4. 逆动力学模型自监督地学习动作表示

不需要人告诉模型"执行了什么动作"，让模型自己从前后两帧的变化中**反推**出动作应该是什么。

当前LeWM需要数据集里有**动作标签** a_t（比如"机械臂往右移动0.5"），这些标签采集成本高。

逆动力学模型做的事情是**反过来**：

```
正向：知道状态z_t和动作a_t → 预测下一个状态z_{t+1}  （这就是现在的predictor）
逆向：知道状态z_t和下一个状态z_{t+1} → 推断中间发生了什么动作â_t
```



你不知道机器人执行了什么动作，但逆动力学模型可以学习：

```
â_t = inverse_model(z_t, z_{t+1})
```

它输出的 â_t 不是真实的物理动作，而是一个**学出来的"动作表示"**，表达的是"从这个状态到那个状态需要发生什么变化"。

然后正向predictor用这个学出来的动作表示来预测：

```
ẑ_{t+1} = predictor(z_t, â_t)
```

两个模型互相监督，不需要真实动作标签。

看上去很有趣到时候试一下



## 5.online 

LeWM完全离线训练，部署时模型固定。加入在线微调或test-time adaptation机制，让模型在新环境中快速适应分布偏移。

**多任务联合训练**：一开始就在多个环境上同时训练，而非先训后调

**adapter微调**：冻住主体参数，只微调一个小的adapter模块来适应新环境，这样不会遗忘

## 6. 规划算法 CEM

CEM是零阶方法，受维度灾难限制。可以利用latent dynamics可微的特性，用基于梯度的轨迹优化（如iLQR或直接反向传播through dynamics），大幅提升规划效率和质量。

LeWM的predictor是全可微的，完全可以直接反向传播梯度来优化动作序列：

```
loss = ||predictor(z_t, a_{1:H}) - z_goal||²
grad = ∂loss / ∂a_{1:H}
a_{1:H} = a_{1:H} - lr * grad
```
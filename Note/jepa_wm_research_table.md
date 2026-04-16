# JEPA 世界模型 Related Work 梳理表

> 覆盖 2024年底 - 2026年3月 的主要 JEPA / JEPA-WM 论文。每篇按:**前人 Gap → Contribution → 如何解决 Gap → Future Work** 的结构梳理,方便你写 related work 和找 research idea。

---

## 快速速查表

| 编号 | 模型 | arXiv | 时间 | 机构 | 核心贡献(一句话) |
|---|---|---|---|---|---|
| 1 | **V-JEPA 2 / 2-AC** | 2506.09985 | 2025.6 | Meta FAIR | 1B 参数视频世界模型 + 零样本机器人规划 |
| 2 | **V-JEPA 2.1** | (GitHub) | 2026.3 | Meta FAIR | Dense Predictive Loss,时间一致的 dense features |
| 3 | **LeJEPA** | 2511.08544 | 2025.11 | Brown/Meta | JEPA 第一个严格理论,SIGReg 取代启发式 |
| 4 | **LeWorldModel (LeWM)** | 2603.19312 | 2026.3 | Mila/NYU/SAIL | 第一个端到端从像素稳定训的 JEPA 世界模型 |
| 5 | **VL-JEPA** | 2512.10942 | 2025.12 | Meta FAIR/HKUST | 视觉-语言版 JEPA,潜空间预测文本 embedding |
| 6 | **DINO-WM** | 2411.04983 | 2024.11 | NYU | 冻结 DINOv2 + predictor,开启 JEPA-WM 路线 |
| 7 | **DINO-World** | 2507.19468 | 2025.7 | Meta FAIR | 大规模视频预训练的通用 JEPA-WM |
| 8 | **JEPA-WMs / Drive-JEPA** | 2512.24497 | 2025.12 | Meta FAIR | 系统消融找到 JEPA-WM 最优配方 |
| 9 | **Value-guided JEPA** | 2601.00844 | 2025.12 | NYU/ENS | 潜空间距离 ≈ 负值函数,改善 MPC 规划 |
| 10 | **Variational JEPA (VJEPA/BJEPA)** | 2601.14354 | 2026.1 | - | JEPA 的概率化/贝叶斯化推广 |
| 11 | **EB-JEPA library** | 2602.03604 | 2026.2 | Meta FAIR | 能量视角的统一 JEPA 开源库 |
| 12 | **BiJEPA** | 2603.00049 | 2026.2 | - | 双向 JEPA + 临界范数正则化 |
| 13 | **IE-JEPA** | 2602.12245 | 2026.2 | - | 把 JEPA 能量连接到 QRL quasimetric 理论 |

---

## 详细对照表

### 1. V-JEPA 2 / V-JEPA 2-AC (Assran et al., Meta FAIR, 2025.6)

**arXiv:** 2506.09985

| 维度 | 内容 |
|---|---|
| **前人 Gap** | ① 已有 world model 多从 state-action 交互数据学习,依赖奖励反馈(Dreamer, TD-MPC2);② 基于视频生成的 world model(Cosmos)预测像素,浪费容量在不可预测细节;③ 缺乏能利用互联网级别无动作标注视频的范式 |
| **Contribution** | ① V-JEPA 2:1.2B 参数,在 1M+ 小时视频上 action-free 自监督预训练;② V-JEPA 2-AC:用 <62h Droid 机器人视频做 action-conditioned 后训练;③ 双 Franka 机械臂零样本部署(两个实验室 80% pick-and-place);④ Scaling 配方:progressive resolution、data curation、3D-RoPE、block-causal attention;⑤ 规划 16s/动作 vs Cosmos 的 4min/动作 |
| **如何解决 Gap** | 两阶段分离"通用世界理解(视频自监督)"和"动作条件化(少量交互数据)";潜空间预测避免像素生成的低效;L1 distance + CEM 在平滑凸的 energy landscape 上规划 |
| **Future Work(原文 §9)** | ① 世界模型与 LLM 对齐(用语言指定目标而非图像);② **Hierarchical JEPA 做长程规划**(目前 autoregressive rollout 误差累积);③ Camera-invariant 表示(当前对相机位置敏感);④ 扩展到非机械臂场景 |

---

### 2. V-JEPA 2.1 (Mur-Labadia et al., Meta FAIR, 2026.3)

**发布于 GitHub `facebookresearch/vjepa2`,配套论文在 pipeline 中**

| 维度 | 内容 |
|---|---|
| **前人 Gap** | V-JEPA 2 学到的 features 时间一致性差,缺乏 dense(逐 patch)特征质量;只在 global(整段视频)任务上强,dense prediction 任务一般 |
| **Contribution** | ① **Dense Predictive Loss**:所有 token(可见+mask)都参与自监督损失;② **Deep Self-Supervision**:在 encoder 多个中间层都施加自监督损失;③ Multi-Modal Tokenizers(图像+视频共用);④ 模型和数据 scaling 并行 |
| **如何解决 Gap** | Dense 损失迫使每个 patch 都学到可预测表示 → 时间一致性 dense features;deep supervision 防止只有最后层学好 |
| **Future Work** | (论文未正式发布完整 future work 章节)隐含方向:更大规模 scaling、与 AC 预测器结合、更多下游 dense prediction 任务 |

---

### 3. LeJEPA (Balestriero & LeCun, 2025.11)

**arXiv:** 2511.08544

| 维度 | 内容 |
|---|---|
| **前人 Gap** | ① JEPA 领域缺乏理论指导,"启发式驱动 R&D";② 已有 JEPA 依赖 stop-gradient、teacher-student、EMA、schedulers 等 ad-hoc trick 防塌缩;③ 已有理论(MI bounds)是 post-hoc 解释,没有 principled guidance;④ 训练 loss 与下游性能不相关,无法做 model selection |
| **Contribution** | ① **理论结果**:证明 isotropic Gaussian 是 JEPA embedding 的最优分布(最小化 worst-case 下游预测风险,linear/nonlinear probe 都成立);② **SIGReg**(Sketched Isotropic Gaussian Regularization):基于 Cramér-Wold 定理的 1D 投影正则,线性时间/内存复杂度;③ 单一 trade-off 超参,无 stop-gradient、无 teacher-student、无 schedulers、~50 行代码;④ 训练 loss 与下游性能相关性高达 99%,首次可用 loss 做 model selection;⑤ 在 10+ 数据集、60+ 架构上验证;⑥ 实证:in-domain SSL 击败 DINOv2/v3 transfer(Galaxy10) |
| **如何解决 Gap** | 理论上证明目标分布(Gaussian)→ 设计高效正则器(SIGReg)直接强制该分布 → 所有防塌缩的启发式都变得不必要 |
| **Future Work** | ① 将 LeJEPA 扩展到世界模型/动作条件设置(直接催生了 LeWM);② 进一步 scaling(1.8B+ 已验证稳定);③ 更多 domain-specific 预训练应用;④ 探索最优分布在 multimodal 设置下的推广 |

---

### 4. LeWorldModel (LeWM) (Maes, Le Lidec, Scieur, LeCun, Balestriero, 2026.3)

**arXiv:** 2603.19312

| 维度 | 内容 |
|---|---|
| **前人 Gap** | 三类前人方法各有缺陷:① **端到端方法(PLDM)**:7 项 VICReg 损失,超参多,训练不稳;② **Foundation-based 方法(DINO-WM)**:冻结 DINOv2 encoder,放弃端到端,token 数多(规划慢);③ **任务特定方法(Dreamer、TD-MPC)**:需要奖励信号或特权 state |
| **Contribution** | ① 第一个**端到端从原始像素稳定训练的 JEPA 世界模型**;② 只用 2 个 loss 项(pred + SIGReg,从 LeJEPA 继承);③ ~15M 参数,单 GPU 几小时训完;④ 每帧编码为单个 192 维 token(比 DINO-WM 少 ~200 倍);⑤ 规划快 48 倍(0.98s vs 47s);⑥ Probing 实验证明潜空间编码物理量(agent 位置、块速度等);⑦ Violation-of-Expectation:能检测物理上不可能的事件(瞬移)但忽略纯视觉扰动(颜色);⑧ Emergent **Temporal Latent Path Straightening**(潜轨迹随训练变直,无显式监督) |
| **如何解决 Gap** | SIGReg(from LeJEPA)消除防塌缩启发式 → 端到端可行;小 encoder + 极少 token → 快速规划;reward-free + reconstruction-free → 任务无关 |
| **Future Work(原文 §Limitations & Future Work)** | ① **规划仍限于短 horizon**,需要长程规划机制(hierarchy、sub-goals);② 在低 intrinsic dimensionality 任务(Two-Room)上表现差,Gaussian 正则可能不适合;③ DINO-WM 在复杂 3D(OGBench-Cube)仍有优势 → 需要更强视觉 priors;④ 扩展到真实机器人;⑤ 结合 pretrained features 与端到端训练的折中方案 |

---

### 5. VL-JEPA (Chen, Shukor, Moutakanni, ..., LeCun, Fung, 2025.12)

**arXiv:** 2512.10942

| 维度 | 内容 |
|---|---|
| **前人 Gap** | ① 传统 VLM 自回归 token 生成慢、计算贵;② token 生成把语义建模和表层语言变化(同义句、措辞)混在一起;③ 不同任务(分类、检索、VQA)需要不同架构;④ JEPA 之前限于窄域(mazes、pick-and-place),未用于通用视觉-语言 |
| **Contribution** | ① 第一个通用视觉-语言 JEPA:预测**连续文本 embedding** 而非 token;② X-Encoder 用 V-JEPA 2,Predictor 用 Llama 3 Transformer 层初始化;③ Selective decoding:推理时只在需要时调用文本解码器,2.85× 解码操作减少;④ 比 token-space VLM 性能更强,参数少 50%;⑤ 统一 embedding 空间原生支持分类/检索/判别式 VQA/生成,无需架构修改;⑥ 只用 1.6B 参数达到 InstructBLIP/QwenVL 水平 |
| **如何解决 Gap** | 在连续 embedding 空间预测 → 解耦"语义推理"和"token 生成" → 参数效率 ↑、streaming 友好、多任务统一 |
| **Future Work** | ① 进一步 scaling(当前 1.6B 还小);② 更多模态加入(音频、3D);③ 更强的 text decoder;④ 应用到机器人 language-conditioned planning(与 V-JEPA 2-AC 结合);⑤ "潜空间 reasoning"范式的更深探索 |

---

### 6. DINO-WM (Zhou et al., NYU, 2024.11)

**arXiv:** 2411.04983

| 维度 | 内容 |
|---|---|
| **前人 Gap** | ① 已有 world model 要么任务特定(需奖励),要么基于像素重建(高维低效);② Model-based RL 方法(Dreamer、TD-MPC2)在无奖励设定下退化;③ 在线策略学习数据成本高 |
| **Contribution** | ① 用**冻结 DINOv2** patch features 作为 state 表示,额外训练 predictor 预测未来 patch features;② 支持纯离线轨迹训练(behavior data,无奖励);③ 支持 test-time 行为优化(MPC);④ Goal 以图像指定,目标就是 feature-space 的预测目标 → task-agnostic 规划;⑤ 在无奖励设定下超越 DreamerV3 和 TD-MPC2 |
| **如何解决 Gap** | 预训练 encoder 自然防塌缩,无需特殊 trick;patch features 紧凑;offline + reward-free 设定 → 通用规划 |
| **Future Work** | ① 从不同动作空间迁移(多机器人、多任务);② 融合更多模态(proprioception);③ 更复杂的视觉域(本文用了 PushT、PointMaze 等简单域);④ 整合 language goals |

---

### 7. DINO-World (Baldassarre et al., Meta FAIR, 2025.7)

**arXiv:** 2507.19468

| 维度 | 内容 |
|---|---|
| **前人 Gap** | ① 视频生成类 world model(COSMOS)预测像素,在密集预测/直觉物理任务上效率低;② DINO-WM 规模小,只在窄域验证;③ 缺乏"通用"JEPA 视频世界模型 |
| **Contribution** | ① **DINO-world**:在 DINOv2 潜空间训练 future predictor,大规模非精选视频数据集;② 覆盖驾驶、室内、仿真多类场景 → 通用 world model;③ 在视频分割预测、深度预测、直觉物理多个 benchmark 上超越 Cosmos;④ 支持 action-conditioned fine-tuning → 通过潜空间 rollout 做规划 |
| **如何解决 Gap** | DINOv2 冻结 encoder 提供强视觉先验 + 大规模视频 scaling predictor → 同时获得通用性和效率 |
| **Future Work** | ① 更深入的 action-conditioning;② 端到端训练 encoder(后来由 LeWM 部分实现);③ 实机器人部署;④ 多步 rollout 的稳定性 |

---

### 8. JEPA-WMs / Drive-JEPA (Terver, Yang, Ponce, Bardes, LeCun, 2025.12)

**arXiv:** 2512.24497

| 维度 | 内容 |
|---|---|
| **前人 Gap** | JEPA-WM 家族(PLDM、DINO-WM、DINO-World、V-JEPA 2-AC)设计选择五花八门:encoder 类型、context 长度、rollout horizon、predictor 条件化方式、训练目标、规划算法。但**没有系统研究这些选择各自多重要** → 研究者不知道改哪个 |
| **Contribution** | ① 系统消融 encoder、context length、rollout horizon、predictor conditioning、模型大小、planner 的每一个;② 关键发现:**训练和规划 regime 对齐**(context 和 rollout 匹配时 latent dynamics 更可优化);③ 动态模型 scaling(仿真用小模型够、真实数据用大模型);④ 全零奖励、目标条件、自监督训练足够;⑤ 提出 Drive-JEPA,在导航和操作任务上同时超越 DINO-WM 和 V-JEPA-2-AC;⑥ 开源代码、数据、权重 |
| **如何解决 Gap** | 控制变量的系统实验提供"什么 matters"的实证指南 |
| **Future Work** | ① Language conditioning 扩展;② Diffusion-based planner(当前用 CEM);③ 更多真实机器人数据;④ 把 value function / QRL 引入(连接 Destrade et al.);⑤ Hierarchical 结构 |

---

### 9. Value-guided JEPA Planning (Destrade, Bounou, Le Lidec, Ponce, LeCun, 2025.12)

**arXiv:** 2601.00844

| 维度 | 内容 |
|---|---|
| **前人 Gap** | ① 标准 JEPA 只用 prediction loss 训练,潜空间几何结构对规划不友好;② MPC 在这种潜空间里容易陷入**局部极小值**;③ 规划 loss(pred 到 goal 的距离)和真正的 value function 不对齐 |
| **Contribution** | ① 引入**新损失项**,让 latent 空间中的欧氏距离(或 quasi-distance)近似**负的目标条件 value function**(reaching cost);② 灵感来自 QRL(Quasimetric RL);③ 两种训练策略:"Sep"(先单独用 L_VF 训 state encoder,再训 predictor)和联合;④ 在 Wall、Two Rooms 等环境上 MPC 规划性能显著提升 |
| **如何解决 Gap** | value-aware 的 loss 塑造潜空间 → 距离 = cost-to-go → 优化 landscape 更平滑 → MPC 少掉进局部极小 |
| **Future Work** | ① 扩展到更复杂的环境(本文实验集中在 2D 导航);② 与不确定性估计结合(→ 连接 VJEPA);③ 引入轨迹级监督;④ 真实机器人验证;⑤ 结合 intrinsic energy 理论(→ IE-JEPA) |

---

### 10. Variational JEPA / VJEPA & BJEPA (Huang, 2026.1)

**arXiv:** 2601.14354

| 维度 | 内容 |
|---|---|
| **前人 Gap** | ① 已有 JEPA 都是**确定性**的,用回归目标训练,掩盖了概率语义;② 没有对未来潜状态**不确定性**的建模;③ 没有形式化证明 JEPA 表征是最优控制的**sufficient information state**;④ 在有噪声/干扰的环境中易塌缩 |
| **Contribution** | ① **VJEPA**:概率化 JEPA,用变分目标学习未来 latent 的预测分布;② 理论统一 JEPA 与 Predictive State Representations (PSRs) 和贝叶斯滤波;③ 证明 VJEPA 表征可作为最优控制的 sufficient information state,无需像素重建;④ 形式化的防塌缩保证;⑤ **BJEPA**:把预测 belief 分解为 dynamics expert + prior expert(Product of Experts),支持零样本任务迁移和约束满足;⑥ 实证:在噪声环境中过滤高方差干扰,优于生成式 baseline |
| **如何解决 Gap** | 变分目标 + 贝叶斯视角把概率严格性引入 JEPA,并给出最优控制的理论保证 |
| **Future Work** | ① Scaling 到大模型和真实机器人;② 与 LLM priors 结合(BJEPA 的 prior expert 可以是 LLM);③ 不确定性引导的探索(exploration);④ 可微分规划 |

---

### 11. EB-JEPA Library (Terver, Balestriero, Dervishi, ..., LeCun, Bar, 2026.2)

**arXiv:** 2602.03604

| 维度 | 内容 |
|---|---|
| **前人 Gap** | JEPA 生态碎片化,每个 paper 有自己的实现;研究者想做消融或新 idea,要从头造轮子;单 GPU 规模的教学/研究示例缺失 |
| **Contribution** | ① 轻量级开源库,每个 example 几乎 self-contained,单 GPU 几小时可训;② 三类示例:image JEPA(CIFAR-10)、video JEPA、action-conditioned video JEPA(Two Rooms 环境)的世界模型+规划;③ 统一"能量视角"下的 JEPA 训练框架;④ 与 stable-worldmodel 和 stable-pretraining 配合 |
| **如何解决 Gap** | 提供社区基础设施,降低 JEPA 研究门槛 |
| **Future Work** | ① 更多 example(3D、multimodal);② 更大规模的版本;③ 社区贡献的新架构 |

---

### 12. BiJEPA (2026.2)

**arXiv:** 2603.00049

| 维度 | 内容 |
|---|---|
| **前人 Gap** | ① 标准 JEPA(如 I-JEPA)是**单向**的:Context → Target;② 忽略了 Target → Context 的逆向信息 → 信号浪费、性能下降;③ 直接做对称预测会引起 **representation explosion**(新的不稳定失败模式) |
| **Contribution** | ① **BiJEPA**:双向 JEPA,强制 Context 和 Target 之间的**循环一致预测**(cycle-consistent);② 引入**临界范数正则化**(critical norm regularization)防止表征爆炸;③ 在三个模态上验证:周期信号、Lorenz 混沌吸引子、MNIST |
| **如何解决 Gap** | 双向预测增加学习信号,防爆炸正则稳定新架构 |
| **Future Work** | ① 扩展到高维视频/图像;② 应用到 world modeling(当前只在合成/小数据验证);③ 与 LeJEPA 的 SIGReg 结合 |

---

### 13. IE-JEPA (Intrinsic-Energy JEPA, 2026.2)

**arXiv:** 2602.12245

| 维度 | 内容 |
|---|---|
| **前人 Gap** | ① JEPA 训练产生的 energy function(latent 预测误差)和 RL 里的 cost-to-go / value function 什么关系?不清楚;② Value-guided JEPA(Destrade)经验上让两者对齐,但没有理论;③ 缺少 JEPA 和 QRL(Quasimetric RL)的桥梁 |
| **Contribution** | ① **理论结果**:在温和假设下(intrinsic energy = 轨迹上累积 local effort 的 infimum),任何 IE-JEPA 的 energy 必然满足 quasimetric 不等式;② 在 goal-reaching 问题里,最优 cost-to-go 正好是 intrinsic energy → IE-JEPA energy ∈ QRL 的 quasimetric value class;③ 提出对称 JEPA 能量在非对称动态下的"基本障碍"(asymmetry 不是美学选择) |
| **如何解决 Gap** | 纯理论连接,不提出新算法,但为 JEPA-for-planning 提供几何结构指导 |
| **Future Work** | ① 基于 intrinsic energy 设计具体的训练算法;② 实证验证 quasimetric 结构对 planning 的帮助;③ 与 Destrade 的 value-guided JEPA 合并成完整方法 |

---

## 给你的 Research Idea 提炼

读完这些论文后,我看到几个**未被充分解决、且你可以攻的方向**:

**A. 长程规划(Long-horizon planning)** — 这是 V-JEPA 2 / LeWM 都明确指出的 limitation。可行路线:

- H-JEPA(LeCun 2022 提过但没真正做好)+ LeJEPA 理论
- Sub-goal generation + 层级潜空间
- Diffusion-based planner 在 latent 空间(Terver et al. 未来工作提到过)

**B. 不确定性+鲁棒性** — VJEPA 开了头,但还很早期。你可以:
- 把 VJEPA/BJEPA 和 V-JEPA 2-AC 合并,做带不确定性的 action-conditioned WM
- 这个方向直接对应真实机器人 deployment 的关键痛点

**C. Value-aware latent geometry** — Destrade 和 IE-JEPA 提供了切入点:
- 结合 LeJEPA 的 Gaussian 结构 + value 结构,设计新 regularizer
- 在 Drive-JEPA 的实验平台上消融

**D. 端到端 vs foundation encoder 的折中** — LeWM 未来工作明确提到:
- Encoder 的 task-aware adaptation(预训练起步 + 端到端微调)
- 混合架构(global features 用 DINO,local dynamics 用 LeWM)

**E. Language-conditioned planning** — V-JEPA 2 future work 核心问题:
- VL-JEPA + V-JEPA 2-AC 的自然合并,但没人系统做过
- BJEPA 的 prior expert 可以是语言模型

---

## 建议的读法

1. **今明两天**:把 Terver et al.(JEPA-WMs 消融)精读一遍,这是整个 landscape 最系统的总结,等于一篇免费的 survey。
2. **下周**:精读 LeJEPA + LeWM(你已经看过 LeWM,把 LeJEPA 理论补上);同时精读 V-JEPA 2 原文的 Limitation 和 Future Work 章节。
3. **再下周**:从上面 ABCDE 中选 1-2 个你最感兴趣的,挑对应的 paper 精读(比如选 B 就读 VJEPA + Destrade + V-JEPA 2-AC)。
4. **同时进行**:把 EB-JEPA library 跑起来,在 `examples/ac_video_jepa` 上做一个小修改,做为你进入 coding 阶段的起点。

需要我针对上面 A-E 里哪个方向做更深入的 idea 拆解(包括可能的 baseline、评测基准、算力估算)?

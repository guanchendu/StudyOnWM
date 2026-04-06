## How what is world model

"""
World Model Demo — 什么是 World Model？
========================================
World Model 的核心思想：
  给定当前状态 s_t 和动作 a_t，预测下一个状态 s_{t+1}
  即学习环境的动力学函数：s_{t+1} = f(s_t, a_t)

本 demo 用一个 6x6 网格世界 + 一个小型 MLP 来演示这个过程。
"""
#


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# ============================================================
# 1. 定义一个简单的网格世界（Ground-Truth Environment）
# ============================================================
# 网格中的元素：0=空地, 1=墙, 2=agent, 3=目标
# 动作空间：0=上, 1=下, 2=左, 3=右

GRID_H, GRID_W = 6, 6
NUM_ACTIONS = 4
CELL_TYPES = 4  # empty, wall, agent, goal

# 固定的墙和目标位置
WALLS = [(1, 1), (1, 2), (3, 3), (3, 4), (4, 1)]
GOAL = (5, 5)

ACTION_NAMES = ["上(↑)", "下(↓)", "左(←)", "右(→)"]
ACTION_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def make_base_grid():
    """创建基础网格（墙 + 目标，不含 agent）"""
    grid = np.zeros((GRID_H, GRID_W), dtype=np.int64)
    for r, c in WALLS:
        grid[r, c] = 1
    grid[GOAL[0], GOAL[1]] = 3
    return grid


def place_agent(grid, row, col):
    """在网格上放置 agent"""
    g = grid.copy()
    g[row, col] = 2
    return g


def step_env(grid, action):
    """
    真实环境的状态转移函数（ground truth）
    这就是 world model 要学习的目标函数
    """
    # 找到 agent 位置
    agent_pos = np.argwhere(grid == 2)
    if len(agent_pos) == 0:
        return grid.copy()
    ar, ac = agent_pos[0]

    # 计算新位置
    dr, dc = ACTION_DELTAS[action]
    nr, nc = ar + dr, ac + dc

    # 碰壁检测：越界或撞墙则不动
    if nr < 0 or nr >= GRID_H or nc < 0 or nc >= GRID_W:
        nr, nc = ar, ac
    elif grid[nr, nc] == 1:  # 墙
        nr, nc = ar, ac

    # 生成新状态
    new_grid = make_base_grid()
    new_grid[nr, nc] = 2
    return new_grid


# ============================================================
# 2. 数据集生成：从真实环境采集 (s, a, s') 三元组
# ============================================================

def generate_dataset(num_samples=2000):
    """
    随机采样 (state, action, next_state) 转移对
    这模拟了机器人在真实世界中的探索数据采集
    """
    states, actions, next_states = [], [], []
    base = make_base_grid()

    # 找出所有可以放置 agent 的空位
    empty_cells = [(r, c) for r in range(GRID_H) for c in range(GRID_W)
                   if base[r, c] == 0]

    for _ in range(num_samples):
        # 随机放置 agent
        ar, ac = random.choice(empty_cells)
        state = place_agent(base, ar, ac)

        # 随机选择动作
        action = random.randint(0, NUM_ACTIONS - 1)

        # 执行动作，得到真实的下一状态
        next_state = step_env(state, action)

        states.append(state)
        actions.append(action)
        next_states.append(next_state)

    return (
        torch.tensor(np.array(states), dtype=torch.long),
        torch.tensor(actions, dtype=torch.long),
        torch.tensor(np.array(next_states), dtype=torch.long),
    )


# ============================================================
# 3. World Model：一个小型 MLP
# ============================================================
# 输入：当前状态的 one-hot 编码 + 动作的 one-hot 编码
# 输出：预测的下一状态（每个格子的类别概率）

class WorldModel(nn.Module):
    """
    最简单的 World Model：
    f(s_t, a_t) -> s_{t+1}

    输入: one_hot(state) [6*6*4=144维] + one_hot(action) [4维] = 148维
    输出: 每个格子的类别预测 [6*6 个格子，每个格子 4 类] = 144维

    这就是 world model 的本质——
    学习 "如果在状态 s 下执行动作 a，世界会变成什么样"
    """

    def __init__(self, hidden_dim=256):
        super().__init__()
        state_dim = GRID_H * GRID_W * CELL_TYPES  # 144
        action_dim = NUM_ACTIONS                    # 4
        output_dim = GRID_H * GRID_W * CELL_TYPES  # 144

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, state, action):
        """
        state:  (batch, H, W) long tensor, 值为 0-3
        action: (batch,) long tensor, 值为 0-3
        return: (batch, H*W, CELL_TYPES) 每个格子的类别 logits
        """
        batch = state.shape[0]

        # 状态 → one-hot
        s_flat = state.view(batch, -1)  # (batch, 36)
        s_onehot = F.one_hot(s_flat, CELL_TYPES).float().view(batch, -1)  # (batch, 144)

        # 动作 → one-hot
        a_onehot = F.one_hot(action, NUM_ACTIONS).float()  # (batch, 4)

        # 拼接输入
        x = torch.cat([s_onehot, a_onehot], dim=-1)  # (batch, 148)

        # 前向传播
        logits = self.net(x)  # (batch, 144)
        logits = logits.view(batch, GRID_H * GRID_W, CELL_TYPES)
        return logits

    def predict(self, state, action):
        """推理：返回预测的网格"""
        with torch.no_grad():
            logits = self.forward(state.unsqueeze(0), action.unsqueeze(0))
            pred = logits.argmax(dim=-1).squeeze(0)
            return pred.view(GRID_H, GRID_W)


# ============================================================
# 4. 训练 World Model
# ============================================================

def train_world_model():
    print("=" * 60)
    print("  World Model 训练 Demo")
    print("  核心公式: s_{t+1} = f_θ(s_t, a_t)")
    print("=" * 60)

    # 生成数据集
    print("\n[1] 生成训练数据：从真实环境采集 (s, a, s') 转移对...")
    states, actions, next_states = generate_dataset(3000)
    print(f"    采集了 {len(states)} 条转移数据")

    # 划分训练集和测试集
    split = 2500
    train_s, train_a, train_ns = states[:split], actions[:split], next_states[:split]
    test_s, test_a, test_ns = states[split:], actions[split:], next_states[split:]

    # 初始化模型
    model = WorldModel(hidden_dim=256)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"\n[2] World Model 结构:")
    print(f"    输入: state one-hot ({GRID_H}×{GRID_W}×{CELL_TYPES}=144维)")
    print(f"         + action one-hot ({NUM_ACTIONS}维)")
    print(f"         = 148维")
    print(f"    隐藏层: 148 → 256 → 256")
    print(f"    输出: 每个格子的类别预测 ({GRID_H}×{GRID_W}=36个格子 × {CELL_TYPES}类)")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    总参数量: {total_params:,}")

    # 训练循环
    print(f"\n[3] 开始训练...")
    print(f"    {'Epoch':>6}  {'Train Loss':>11}  {'Test Acc':>9}  {'状态'}")
    print(f"    {'─'*6}  {'─'*11}  {'─'*9}  {'─'*20}")

    batch_size = 128
    num_epochs = 80

    for epoch in range(1, num_epochs + 1):
        model.train()
        # Mini-batch 训练
        perm = torch.randperm(split)
        epoch_loss = 0
        n_batches = 0

        for i in range(0, split, batch_size):
            idx = perm[i:i + batch_size]
            s_batch = train_s[idx]
            a_batch = train_a[idx]
            ns_batch = train_ns[idx]

            logits = model(s_batch, a_batch)  # (batch, 36, 4)
            target = ns_batch.view(-1, GRID_H * GRID_W)  # (batch, 36)

            loss = criterion(logits.view(-1, CELL_TYPES), target.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches

        # 测试集评估
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                test_logits = model(test_s, test_a)
                test_pred = test_logits.argmax(dim=-1)  # (500, 36)
                test_target = test_ns.view(-1, GRID_H * GRID_W)
                acc = (test_pred == test_target).float().mean().item()

            if acc < 0.5:
                status = "🔴 还在瞎猜"
            elif acc < 0.8:
                status = "🟡 开始学到规律"
            elif acc < 0.95:
                status = "🟢 基本掌握物理规则"
            else:
                status = "✅ 完全理解世界动力学"

            print(f"    {epoch:>6}  {avg_loss:>11.4f}  {acc:>8.1%}  {status}")

    # ============================================================
    # 5. 可视化预测结果
    # ============================================================
    print(f"\n[4] 预测演示：模型 '想象' 执行动作后的世界状态")
    print("=" * 60)

    CELL_CHARS = {0: "· ", 1: "██", 2: "🤖", 3: "⭐"}

    def print_grid(grid, title=""):
        if title:
            print(f"  {title}")
        arr = grid.numpy() if isinstance(grid, torch.Tensor) else grid
        for r in range(GRID_H):
            row_str = "    "
            for c in range(GRID_W):
                row_str += CELL_CHARS[int(arr[r, c])]
            print(row_str)

    model.eval()
    base = make_base_grid()
    empty_cells = [(r, c) for r in range(GRID_H) for c in range(GRID_W)
                   if base[r, c] == 0]

    for demo_i in range(3):
        print(f"\n--- 演示 {demo_i + 1} ---")
        ar, ac = random.choice(empty_cells)
        state = place_agent(base, ar, ac)
        action = random.randint(0, 3)

        state_t = torch.tensor(state, dtype=torch.long)
        action_t = torch.tensor(action, dtype=torch.long)

        true_next = step_env(state, action)
        pred_next = model.predict(state_t, action_t).numpy()

        match = np.array_equal(pred_next, true_next)

        print_grid(state, f"当前状态 s_t (agent 在 [{ar},{ac}]):")
        print(f"\n  执行动作 a_t = {ACTION_NAMES[action]}\n")
        print_grid(true_next, "真实下一状态 s_{t+1} (ground truth):")
        print()
        print_grid(pred_next, "模型预测 f_θ(s_t, a_t):")
        print(f"\n  预测{'正确 ✅' if match else '错误 ❌'}")

    # ============================================================
    # 6. 多步 Rollout：用 world model "做梦"
    # ============================================================
    print(f"\n\n{'=' * 60}")
    print("  多步 Rollout：World Model '做梦'")
    print("  不与真实环境交互，完全在 '想象' 中推演未来")
    print("=" * 60)

    ar, ac = 0, 0
    state = place_agent(base, ar, ac)
    state_t = torch.tensor(state, dtype=torch.long)

    # 预定义一组动作序列
    action_seq = [1, 1, 3, 3, 1, 1, 3, 3]  # 下下右右下下右右 → 走向目标
    print(f"\n  动作序列: {' → '.join(ACTION_NAMES[a] for a in action_seq)}")
    print(f"\n  Step 0 (初始状态):")
    print_grid(state)

    # 用真实环境做对比
    true_state = state.copy()
    imagined_state_t = state_t.clone()

    all_match = True
    for step, action in enumerate(action_seq):
        # 真实环境
        true_state = step_env(true_state, action)

        # World model 想象 (autoregressive: 用上一步的预测作为输入)
        action_t = torch.tensor(action, dtype=torch.long)
        imagined_next = model.predict(imagined_state_t, action_t)
        imagined_state_t = imagined_next.long()

        match = np.array_equal(imagined_next.numpy(), true_state)
        if not match:
            all_match = False

        print(f"\n  Step {step + 1}: 执行 {ACTION_NAMES[action]}  "
              f"{'✅' if match else '❌ 想象偏离现实!'}")
        if not match:
            print(f"    真实:")
            print_grid(true_state)
            print(f"    想象:")
            print_grid(imagined_next)

    if all_match:
        print(f"\n  🎉 World Model 的 {len(action_seq)} 步想象完全正确！")
        print(f"     它已经学会了这个世界的物理规则。")
    else:
        print(f"\n  ⚠️  长步 rollout 中出现误差累积（这是 world model 的核心挑战）")

    # ============================================================
    # 7. 总结
    # ============================================================
    print(f"""
{'=' * 60}
  总结：World Model 的核心概念
{'=' * 60}

  1. World Model 学习的是环境的状态转移函数:
     s_{{t+1}} = f_θ(s_t, a_t)

  2. 训练数据来自真实环境的交互:
     收集 (state, action, next_state) 三元组

  3. 训练好后，World Model 可以:
     - 在 '想象' 中推演未来（不需要真实环境）
     - 为策略学习提供 '虚拟环境'
     - 辅助规划：比较不同动作序列的后果

  4. 核心挑战:
     - 误差累积：多步 rollout 时预测误差会逐步放大
     - 泛化能力：对未见过的状态的预测可能不准
     → 这就是为什么视频扩散模型被引入:
       大规模视频预训练让 world model 见过更多 '世界的样子'

  5. 在机器人领域 (如 LingBot-VA):
     - state = 视觉观测 (图像帧)
     - action = 机器人关节动作
     - next_state = 下一帧图像
     - World Model = 视频预测模型
""")


if __name__ == "__main__":
    train_world_model()
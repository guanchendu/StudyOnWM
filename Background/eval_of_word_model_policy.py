"""
A tiny, self-contained evaluation walkthrough.

This file mirrors the structure of the project's eval.py, but it does not rely on
stable_worldmodel, Hydra, HDF5 files, or any project-specific dataset classes.

It is meant for debugging and learning:
1. Build a toy dataset with multiple episodes.
2. Fit simple normalizers for numeric columns.
3. Find valid evaluation start points.
4. Sample a few starts reproducibly.
5. Evaluate a toy policy inside a toy world.

Run:
    python eval_walkthrough.py

Suggested debugging checkpoints:
- inside `build_fake_dataset`
- after `ep_indices` is computed
- inside the scaler loop
- after `valid_indices` is computed
- inside `ToyWorld.evaluate_from_dataset`
"""
#
# 先建环境
# 设定 world.max_episode_steps，也就是环境一次最多跑多少步。
#
# 读入并整理数据集
# 拿到所有数据行，每一行都带着：
#
# 属于哪个 episode_idx
# 是这一条轨迹里的第几步 step_idx
# 当前的 state/action/pixels
# 给数值列做归一化准备
# 对 state、action 这类数值特征拟合 scaler，后面模型/策略用的时候尺度会更稳定。
#
# 统计每条 episode 的长度
# 先知道每条轨迹有多长，才能判断哪些位置还能当起点。
#
# 计算每条 episode 的最大合法起点
# 根据 goal_offset_steps 算出：
# “这条轨迹最晚能从第几步开始，不会越界”。
#
# 把这个限制映射到每一行数据
# 也就是判断数据集里的每一行，能不能作为合法起点。
#
# 统计所有合法起点
# 得到 valid_indices，它表示：
# “整个数据集中，哪些行可以拿来做评估起点”。
#
# 从这些合法起点里随机采样
# 抽出这次真正要评估的几个起点。
#
# 反查这些起点对应的 episode 和 start step
# 也就是拿到：
#
# 来自哪条轨迹 eval_episodes
# 从第几步开始 eval_start_idx
# 把这些起点送进环境做评估
# 让 policy 从这些起点出发，朝对应 goal 去执行，然后统计 metrics。

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

import numpy as np


class SimpleStandardScaler:
    """Small replacement for sklearn StandardScaler."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "SimpleStandardScaler":
        x = np.asarray(x, dtype=np.float64)
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Call fit() before transform().")
        return (np.asarray(x, dtype=np.float64) - self.mean_) / self.scale_


class ToyDataset:
    """A minimal dataset API compatible with the original eval.py flow."""

    def __init__(self, columns: dict[str, np.ndarray]) -> None:
        self.columns = columns
        self.column_names = list(columns.keys())

    def get_col_data(self, name: str) -> np.ndarray:
        return self.columns[name]

    def get_row_data(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        return {name: values[indices] for name, values in self.columns.items()}


def build_fake_dataset(
        num_episodes: int = 6,
        min_len: int = 6,
        max_len: int = 11,
        seed: int = 7,
) -> ToyDataset:
    """
    Create a tiny 2D navigation-style dataset.

    Each row is one timestep with:
    - episode_idx
    - step_idx
    - state: 2D position
    - action: 2D move delta
    - pixels: fake image tensor placeholder
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for ep in range(num_episodes):
        length = int(rng.integers(min_len, max_len + 1))
        position = rng.normal(loc=0.0, scale=1.0, size=2)

        for step in range(length):
            action = rng.normal(loc=0.0, scale=0.35, size=2)
            next_position = position + action

            rows.append(
                {
                    "episode_idx": ep,
                    "step_idx": step,
                    "state": position.copy(),
                    "action": action.copy(),
                    "pixels": np.full((4, 4, 3), fill_value=ep * 10 + step, dtype=np.uint8),
                }
            )
            position = next_position

    columns = {
        "episode_idx": np.array([row["episode_idx"] for row in rows], dtype=np.int64),
        "step_idx": np.array([row["step_idx"] for row in rows], dtype=np.int64),
        "state": np.array([row["state"] for row in rows], dtype=np.float64),
        "action": np.array([row["action"] for row in rows], dtype=np.float64),
        "pixels": np.array([row["pixels"] for row in rows], dtype=np.uint8),
    }

    # Inject one NaN row so you can see why the scaler code removes bad rows.
    columns["state"][3, 1] = np.nan
    return ToyDataset(columns)


def get_episodes_length(dataset: ToyDataset, episodes: np.ndarray) -> np.ndarray:
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")

    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths, dtype=np.int64)


@dataclass
class DatasetConfig:
    keys_to_cache: list[str] = field(default_factory=lambda: ["state", "action", "pixels"])


@dataclass
class EvalConfig:
    dataset_name: str = "fake_twod_room"
    eval_budget: int = 4
    goal_offset_steps: int = 2
    num_eval: int = 5


@dataclass
class WorldConfig:
    max_episode_steps: int = 0


@dataclass
class PlanConfig:
    horizon: int = 2
    action_block: int = 2


@dataclass
class Config:
    seed: int = 42
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    world: WorldConfig = field(default_factory=WorldConfig)
    plan_config: PlanConfig = field(default_factory=PlanConfig)


class ToyPolicy:
    """Move directly toward the goal with a capped step size."""

    def act(self, state: np.ndarray, goal_state: np.ndarray) -> np.ndarray:
        delta = goal_state - state
        norm = np.linalg.norm(delta)
        if norm < 1e-8:
            return np.zeros_like(delta)
        step = min(0.6, norm)
        return delta / norm * step


class RandomPolicy:
    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, state: np.ndarray, goal_state: np.ndarray) -> np.ndarray:
        del state, goal_state
        return self.rng.normal(loc=0.0, scale=0.4, size=2)


class ToyWorld:
    def __init__(self, max_episode_steps: int, image_shape: tuple[int, int]) -> None:
        self.max_episode_steps = max_episode_steps
        self.image_shape = image_shape
        self.policy: ToyPolicy | RandomPolicy | None = None

    def set_policy(self, policy: ToyPolicy | RandomPolicy) -> None:
        self.policy = policy

    def evaluate_from_dataset(
            self,
            dataset: ToyDataset,
            start_steps: list[int],
            goal_offset_steps: int,
            eval_budget: int,
            episodes_idx: list[int],
            callables: dict[str, Any] | None = None,
            video_path: str | None = None,
    ) -> dict[str, Any]:
        del callables, video_path
        if self.policy is None:
            raise RuntimeError("Call set_policy() before evaluate_from_dataset().")

        episode_col = dataset.get_col_data("episode_idx")
        step_col = dataset.get_col_data("step_idx")
        state_col = dataset.get_col_data("state")

        successes = []
        final_distances = []
        rollout_lengths = []

        for eval_id, (ep_id, start_step) in enumerate(zip(episodes_idx, start_steps)):
            start_mask = (episode_col == ep_id) & (step_col == start_step)
            goal_mask = (episode_col == ep_id) & (step_col == start_step + goal_offset_steps)

            start_state = state_col[start_mask][0].copy()
            goal_state = state_col[goal_mask][0].copy()

            current = np.nan_to_num(start_state, nan=0.0)
            goal = np.nan_to_num(goal_state, nan=0.0)

            for _ in range(eval_budget):
                action = self.policy.act(current, goal)
                current = current + action

            dist = float(np.linalg.norm(current - goal))
            success = dist < 0.35

            successes.append(success)
            final_distances.append(dist)
            rollout_lengths.append(eval_budget)

            print(
                f"[eval {eval_id}] episode={ep_id}, start={start_step}, "
                f"goal_step={start_step + goal_offset_steps}, final_dist={dist:.4f}, success={success}"
            )

        return {
            "num_evals": len(episodes_idx),
            "success_rate": float(np.mean(successes)),
            "mean_final_distance": float(np.mean(final_distances)),
            "mean_rollout_length": float(np.mean(rollout_lengths)),
        }


def main() -> None:
    cfg = Config()

    assert (
            cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    print("1) Build toy environment")
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = ToyWorld(max_episode_steps=cfg.world.max_episode_steps, image_shape=(224, 224))
    print("   world.max_episode_steps =", cfg.world.max_episode_steps)
    # 意思是创建了一个假的环境 world，并设置每个 episode 最多允许跑 8 步。
    print("\n2) Load toy dataset")
    dataset = build_fake_dataset(seed=cfg.seed)
    stats_dataset = dataset
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)
    print("   unique episodes =", ep_indices.tolist())
    """
    这份测试数据里一共有 6 个不同的 episode，它们的 ID 分别是 0, 1, 2, 3, 4, 5。

    也就是说，这个 toy dataset 不是一条长轨迹，而是由 6 条小轨迹拼起来的。

    比如原始 episode_idx 这一列可能是：

    [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, ...]
    去重以后就得到：

    [0, 1, 2, 3, 4, 5]
    所以你可以把 unique episodes 理解成：

    “这次评估数据里有哪些轨迹编号可用。”

    后面会基于这些 episode 去做：

    统计每条 episode 长度
    计算每条 episode 的合法起点
    从这些 episode 里抽样评估
    如果你想更贴近原版 eval.py 的理解，那这里的 ep_indices 基本就是：
    “评估时可遍历的所有 episode 列表”。

    """
    print("\n3) Fit per-column processors")
    process: dict[str, SimpleStandardScaler] = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            print(f"   skip {col}: image-like column")
            continue

        processor = SimpleStandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

        print(
            f"   fitted {col}: mean={np.round(processor.mean_, 3)}, "
            f"std={np.round(processor.scale_, 3)}"
        )  # 计算state和action 的标准化

    print("\n4) Compute valid starting points")
    episode_len = get_episodes_length(dataset, ep_indices)  # 意思是每条 episode 的长度分别是：[6, 10, 10, 8, 11, 9]
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1  # 意思是每条 episode 里，最晚可以从哪一步开始评估。max start per episode = {0: 3, 1: 7, 2: 7, 3: 5, 4: 8, 5: 6}
    max_start_idx_dict = {
        int(ep_id): int(max_start_idx[i]) for i, ep_id in enumerate(ep_indices)
    }
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )  # “给数据集里的每一行，都配上它所在 episode 的最大合法起点。”

    valid_mask = dataset.get_col_data(
        "step_idx") <= max_start_per_row  # 拿每一行自己的 step_idx 去比较：这一行的 step 有没有超过“它所属 episode 允许的最大起点”？
    valid_indices = np.nonzero(valid_mask)[0]
    print("   episode lengths =", episode_len.tolist())
    print("   max start per episode =", max_start_idx_dict)
    print("   valid starting points found =", int(valid_mask.sum()))  # valid starting points found = 42

    print("\n5) Sample evaluation starts")
    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices), size=cfg.eval.num_eval, replace=False
    )
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    eval_rows = dataset.get_row_data(random_episode_indices)
    eval_episodes = eval_rows[col_name]
    eval_start_idx = eval_rows["step_idx"]

    print("   sampled dataset rows =",
          random_episode_indices.tolist())  # 意思是从那 42 个合法起点里，随机抽了 5 行数据来做评估。sampled dataset rows = [3, 21, 22, 34, 38]
    print("   sampled episodes =",
          eval_episodes.tolist())  # 说明这 5 个样本分别来自哪些 episode。 sampled episodes = [0, 2, 2, 4, 4]第一个样本来自 episode 0第二、第三个来自 episode 2第四、第五个来自 episode 4

    print("   sampled start steps =",
          eval_start_idx.tolist())  ##sampled start steps = [3, 5, 6, 0, 4]  说明这 5 个样本在各自 episode 里是从第几步开始的。

    print("\n6) Evaluate policy")
    """
    [eval 0] episode=0, start=3, goal_step=5, final_dist=0.0000, success=True

    意思是第 0 次评估里：
    选中了 episode=0
    起点是 start=3
    目标步是 goal_step=5
    policy 从起点开始执行动作
    最后离目标的距离是 0.0000
    所以判定成功 success=True
    """
    policy = ToyPolicy()
    world.set_policy(policy)

    start_time = time.time()
    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=None,
        video_path=None,
    )
    end_time = time.time()

    print("\n7) Metrics")
    print(metrics)
    print(f"evaluation_time = {end_time - start_time:.6f} seconds")

    print("\n8) Optional processor demo")
    first_state = dataset.get_col_data("state")[0:1]
    scaled_state = process["state"].transform(np.nan_to_num(first_state, nan=0.0))
    print("   raw state[0]   =", np.round(first_state, 3))
    print("   scaled state[0]=", np.round(scaled_state, 3))
    """
    这是在演示标准化器的作用：

    原始 state 是 [-1.04, 0.75]
    标准化后变成了 [-0.25, 0.62]
    也就是把原始数值按前面学到的均值和标准差做了缩放。
    """


if __name__ == "__main__":
    main()

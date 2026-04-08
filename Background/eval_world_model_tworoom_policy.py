

"""
Minimal eval walkthrough built on the real `tworoom.h5` dataset.

This script is intentionally simple and self-contained:
- no Hydra
- no stable_worldmodel dependency
- uses the real dataset fields from tworoom.h5
- preserves the same high-level eval flow as eval.py

It is useful for stepping through the evaluation logic in a debugger.

Run:
    python eval_tworoom_walkthrough.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any

import h5py
import numpy as np


class SimpleStandardScaler:
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


class TwoRoomDataset:
    """
    Tiny dataset wrapper with the same shape as the toy walkthrough.

    We only load the columns we need into memory so the script stays simple.
    """

    def __init__(self, h5_path: str | Path, keys_to_load: list[str]) -> None:
        h5_path = "/Users/guanchendu/Code/StudyOnWM/data/tworoom.h5"
        self.h5_path = Path(h5_path)
        self.columns: dict[str, np.ndarray] = {}

        with h5py.File(self.h5_path, "r") as f:
            for key in keys_to_load:
                self.columns[key] = f[key][:]

        self.column_names = list(self.columns.keys())

    def get_col_data(self, name: str) -> np.ndarray:
        return self.columns[name]

    def get_row_data(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        return {name: values[indices] for name, values in self.columns.items()}


def get_episodes_length(dataset: TwoRoomDataset, episodes: np.ndarray) -> np.ndarray:
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")

    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths, dtype=np.int64)


@dataclass
class DatasetConfig:
    path: str = "tworoom.h5"
    keys_to_load: list[str] = field(
        default_factory=lambda: [
            "ep_idx",
            "step_idx",
            "action",
            "proprio",
            "pos_agent",
            "pos_target",
            "distance_to_target",
            "terminated",
            "truncated",
        ]
    )
    keys_to_cache: list[str] = field(default_factory=lambda: ["action", "proprio"])


@dataclass
class EvalConfig:
    eval_budget: int = 6
    goal_offset_steps: int = 5
    num_eval: int = 8
    success_threshold: float = 6.0


@dataclass
class WorldConfig:
    max_episode_steps: int = 0
    action_scale: float = 5.0


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


class GoalToFutureStatePolicy:
    """
    Simple hand-crafted policy.

    It moves from current position toward the future goal position and clips the
    action to the same range as the dataset, roughly [-1, 1].
    """

    def __init__(self, action_scale: float = 5.0) -> None:
        self.action_scale = action_scale

    def act(self, state: np.ndarray, goal_state: np.ndarray) -> np.ndarray:
        desired_delta = goal_state - state
        action = desired_delta / self.action_scale
        return np.clip(action, -1.0, 1.0)


class RandomPolicy:
    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def act(self, state: np.ndarray, goal_state: np.ndarray) -> np.ndarray:
        del state, goal_state
        return self.rng.uniform(low=-1.0, high=1.0, size=2)


class TwoRoomSimWorld:
    """
    Tiny simulator that approximates the dataset dynamics.

    From inspection of tworoom.h5:
        next_pos - current_pos ~= action * 5
    so we use that as our environment transition rule.
    """

    def __init__(self, max_episode_steps: int, action_scale: float) -> None:
        self.max_episode_steps = max_episode_steps
        self.action_scale = action_scale
        self.policy: GoalToFutureStatePolicy | RandomPolicy | None = None

    def set_policy(self, policy: GoalToFutureStatePolicy | RandomPolicy) -> None:
        self.policy = policy

    def evaluate_from_dataset(
        self,
        dataset: TwoRoomDataset,
        start_steps: list[int],
        goal_offset_steps: int,
        eval_budget: int,
        episodes_idx: list[int],
        success_threshold: float,
    ) -> dict[str, Any]:
        if self.policy is None:
            raise RuntimeError("Call set_policy() before evaluation.")

        ep_idx = dataset.get_col_data("ep_idx")
        step_idx = dataset.get_col_data("step_idx")
        pos_agent = dataset.get_col_data("pos_agent")
        pos_target = dataset.get_col_data("pos_target")

        final_distances = []
        successes = []
        oracle_goal_distances = []

        for eval_id, (episode_id, start_step) in enumerate(zip(episodes_idx, start_steps)):
            start_mask = (ep_idx == episode_id) & (step_idx == start_step)
            goal_mask = (ep_idx == episode_id) & (step_idx == start_step + goal_offset_steps)

            start_state = pos_agent[start_mask][0].astype(np.float64)
            future_goal_state = pos_agent[goal_mask][0].astype(np.float64)
            true_target = pos_target[start_mask][0].astype(np.float64)

            current_state = start_state.copy()

            for _ in range(eval_budget):
                action = self.policy.act(current_state, future_goal_state)
                current_state = current_state + action * self.action_scale

            final_distance = float(np.linalg.norm(current_state - future_goal_state))
            oracle_goal_distance = float(np.linalg.norm(current_state - true_target))
            success = final_distance <= success_threshold

            final_distances.append(final_distance)
            oracle_goal_distances.append(oracle_goal_distance)
            successes.append(success)

            print(
                f"[eval {eval_id}] ep={episode_id}, start={start_step}, "
                f"goal_step={start_step + goal_offset_steps}, "
                f"future_goal_dist={final_distance:.3f}, true_target_dist={oracle_goal_distance:.3f}, "
                f"success={success}"
            )

        return {
            "num_evals": len(episodes_idx),
            "success_rate": float(np.mean(successes)),
            "mean_future_goal_distance": float(np.mean(final_distances)),
            "mean_true_target_distance": float(np.mean(oracle_goal_distances)),
            "eval_budget": eval_budget,
        }


def main() -> None:
    cfg = Config()

    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    print("1) Build simulated world")
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = TwoRoomSimWorld(
        max_episode_steps=cfg.world.max_episode_steps,
        action_scale=cfg.world.action_scale,
    )
    print("   world.max_episode_steps =", cfg.world.max_episode_steps)
    print("   world.action_scale =", cfg.world.action_scale)

    print("\n2) Load tworoom dataset")
    dataset = TwoRoomDataset(cfg.dataset.path, cfg.dataset.keys_to_load)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)
    print("   loaded rows =", len(dataset.get_col_data(col_name)))
    print("   unique episodes =", len(ep_indices))
    print("   first 10 episodes =", ep_indices[:10].tolist())

    print("\n3) Fit numeric processors")
    process: dict[str, SimpleStandardScaler] = {}
    for col in cfg.dataset.keys_to_cache:
        processor = SimpleStandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

        print(
            f"   fitted {col}: mean={np.round(processor.mean_, 3)}, "
            f"std={np.round(processor.scale_, 3)}"
        )

    print("\n4) Compute valid starting points")
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {
        int(ep_id): int(max_start_idx[i]) for i, ep_id in enumerate(ep_indices)
    }
    max_start_per_row = np.array(
        [max_start_idx_dict[int(ep_id)] for ep_id in dataset.get_col_data(col_name)]
    )

    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]

    print("   first 10 episode lengths =", episode_len[:10].tolist())
    print(
        "   first 10 max start values =",
        {int(ep): max_start_idx_dict[int(ep)] for ep in ep_indices[:10]},
    )
    print("   valid starting points found =", int(valid_mask.sum()))

    print("\n5) Sample evaluation starts")
    rng = np.random.default_rng(cfg.seed)
    sampled_positions = rng.choice(len(valid_indices), size=cfg.eval.num_eval, replace=False)
    sampled_rows = np.sort(valid_indices[sampled_positions])

    row_data = dataset.get_row_data(sampled_rows)
    eval_episodes = row_data[col_name]
    eval_start_idx = row_data["step_idx"]

    print("   sampled dataset rows =", sampled_rows.tolist())
    print("   sampled episodes =", eval_episodes.tolist())
    print("   sampled start steps =", eval_start_idx.tolist())

    print("\n6) Evaluate policy")
    policy = GoalToFutureStatePolicy(action_scale=cfg.world.action_scale)
    world.set_policy(policy)

    start_time = time.time()
    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        success_threshold=cfg.eval.success_threshold,
    )
    end_time = time.time()

    print("\n7) Metrics")
    print(metrics)
    print(f"evaluation_time = {end_time - start_time:.6f} seconds")

    print("\n8) Optional normalization demo")
    first_proprio = dataset.get_col_data("proprio")[0:1]
    scaled = process["proprio"].transform(first_proprio)
    print("   raw proprio[0]   =", np.round(first_proprio, 3))
    print("   scaled proprio[0]=", np.round(scaled, 3))


if __name__ == "__main__":
    main()

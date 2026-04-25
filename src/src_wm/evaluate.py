"""
Evaluate the full Hierarchical JEPA + InvDyn framework on TwoRoom.

Loads Phase 1 + Phase 2 checkpoints and runs CEM-based planning.

Usage:
  cd /Users/guanchendu/Code/StudyOnWM/src
  python -m src_wm.evaluate \
    --phase1-ckpt outputs/hierarchical_invdyn/best_phase1.pt \
    --phase2-ckpt outputs/hierarchical_invdyn/best_phase2.pt
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from stable_worldmodel.solver import CEMSolver
from torchvision.transforms import v2 as transforms

from src_wm.planning import HierarchicalCostModel


DEFAULT_CACHE_DIR = "/Users/guanchendu/Code/StudyOnWM/data"


# ============================================================
# Utilities (reused from existing eval code)
# ============================================================


def img_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
        ]
    )


class TorchStyleStandardScaler:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "TorchStyleStandardScaler":
        x_t = torch.as_tensor(x, dtype=torch.float32)
        if x_t.ndim == 1:
            x_t = x_t.unsqueeze(-1)
        mask = ~torch.isnan(x_t).any(dim=1)
        x_t = x_t[mask]
        self.mean_ = x_t.mean(dim=0).cpu().numpy()
        self.scale_ = x_t.std(dim=0, unbiased=True).cpu().numpy()
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float32) - self.mean_) / self.scale_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float32) * self.scale_ + self.mean_


def get_dataset(cache_dir: str | Path):
    return swm.data.HDF5Dataset(
        "tworoom",
        keys_to_cache=["action", "proprio"],
        cache_dir=Path(cache_dir),
    )


def build_processors(dataset) -> dict[str, TorchStyleStandardScaler]:
    process = {}
    for col in ("action", "proprio"):
        scaler = TorchStyleStandardScaler()
        scaler.fit(dataset.get_col_data(col))
        process[col] = scaler
        if col != "action":
            process[f"goal_{col}"] = scaler
    return process


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def sample_eval_starts(dataset, num_eval: int, goal_offset_steps: int, seed: int):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - goal_offset_steps - 1
    max_start_idx_dict = {
        ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)
    }
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]

    if len(valid_indices) < num_eval:
        raise ValueError(
            f"Not enough valid starting points: need {num_eval}, got {len(valid_indices)}."
        )

    rng = np.random.default_rng(seed)
    chosen = rng.choice(valid_indices, size=num_eval, replace=False)
    chosen = np.sort(chosen)

    row_data = dataset.get_row_data(chosen)
    return row_data[col_name], row_data["step_idx"], chosen


def pick_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================
# Main
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Hierarchical JEPA + InvDyn on TwoRoom"
    )
    parser.add_argument("--phase1-ckpt", type=str, required=True)
    parser.add_argument("--phase2-ckpt", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--goal-offset-steps", type=int, default=25)
    parser.add_argument("--eval-budget", type=int, default=50)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--receding-horizon", type=int, default=5)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--cem-steps", type=int, default=30)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--var-scale", type=float, default=1.0)
    parser.add_argument("--solver-batch-size", type=int, default=25)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-video", type=bool, default=True)
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    device = pick_device(args.device)

    # ---- Load cost model ----
    cost_model = HierarchicalCostModel(
        phase1_ckpt_path=args.phase1_ckpt,
        phase2_ckpt_path=args.phase2_ckpt,
        device=device,
    )

    # ---- Dataset & processors ----
    dataset = get_dataset(args.cache_dir)
    process = build_processors(dataset)
    transform = {
        "pixels": img_transform(args.img_size),
        "goal": img_transform(args.img_size),
    }

    # ---- Environment ----
    world = swm.World(
        env_name="swm/TwoRoom-v1",
        num_envs=args.num_eval,
        max_episode_steps=2 * args.eval_budget,
        history_size=1,
        frame_skip=1,
        image_shape=(224, 224),
    )

    # ---- Solver & Policy ----
    solver = CEMSolver(
        model=cost_model,
        batch_size=args.solver_batch_size,
        num_samples=args.num_samples,
        var_scale=args.var_scale,
        n_steps=args.cem_steps,
        topk=args.topk,
        device=device,
        seed=args.seed,
    )
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=swm.PlanConfig(
            horizon=args.horizon,
            receding_horizon=args.receding_horizon,
            action_block=args.action_block,
        ),
        process=process,
        transform=transform,
    )

    # ---- Eval starts ----
    eval_episodes, eval_start_idx, sampled_rows = sample_eval_starts(
        dataset=dataset,
        num_eval=args.num_eval,
        goal_offset_steps=args.goal_offset_steps,
        seed=args.seed,
    )

    world.set_policy(policy)

    callables = [
        {"method": "_set_state", "args": {"state": {"value": "proprio"}}},
        {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_proprio"}}},
    ]

    video_path = Path(args.phase1_ckpt).parent / "eval_videos"

    # ---- Run evaluation ----
    start_time = time.time()
    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=args.goal_offset_steps,
        eval_budget=args.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=callables,
        save_video=args.save_video,
        video_path=video_path,
    )
    elapsed = time.time() - start_time

    # ---- Save results ----
    results = {
        "phase1_ckpt": str(args.phase1_ckpt),
        "phase2_ckpt": str(args.phase2_ckpt),
        "device": str(device),
        "num_eval": args.num_eval,
        "goal_offset_steps": args.goal_offset_steps,
        "eval_budget": args.eval_budget,
        "solver": {
            "horizon": args.horizon,
            "receding_horizon": args.receding_horizon,
            "action_block": args.action_block,
            "num_samples": args.num_samples,
            "cem_steps": args.cem_steps,
            "topk": args.topk,
        },
        "metrics": {
            "success_rate": float(metrics["success_rate"]),
            "episode_successes": np.asarray(
                metrics["episode_successes"]
            ).astype(int).tolist(),
        },
        "evaluation_time_sec": elapsed,
    }

    output_json = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else Path(args.phase1_ckpt).parent / "eval_hierarchical_invdyn.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")

    print("=" * 72)
    print("Evaluation: Hierarchical JEPA + Inverse Dynamics")
    print("=" * 72)
    print(f"success_rate: {metrics['success_rate']:.2f}")
    print(f"episode_successes: {metrics['episode_successes']}")
    print(f"evaluation time: {elapsed:.1f}s")
    print(f"results saved to: {output_json}")
    print("=" * 72)


if __name__ == "__main__":
    main()

"""
Evaluate Multi-Scale InvDyn World Model on TwoRoom.

Two planner modes:
  --planner-mode two_level   (default) — uses TwoLevelPlanner.plan_batch end-to-end.
                                Coarse CEM + fine CEM + inverse-dynamics warm-start
                                in latent space. Real actions come from fine_dec.
  --planner-mode coarse_cem  — legacy path. swm.CEMSolver + TwoLevelCostModel.
                                Searches in real-action space, only uses coarse
                                predictor + proprio_dec for cost. Useful as ablation.

Usage:
  cd /Users/guanchendu/Code/StudyOnWM/src
  conda run -n wm python -m src_wm2.evaluate \
    --phase1-ckpt outputs/multiscale_invdyn/best_phase1.pt \
    --phase2-ckpt outputs/multiscale_invdyn/best_phase2.pt
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

from src_wm2.planning import (
    TwoLevelCostModel,
    TwoLevelPlanner,
    TwoLevelPlannerPolicy,
)


DEFAULT_CACHE_DIR = "/Users/guanchendu/Code/StudyOnWM/data"


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
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, x):
        x_t = torch.as_tensor(x, dtype=torch.float32)
        if x_t.ndim == 1:
            x_t = x_t.unsqueeze(-1)
        mask = ~torch.isnan(x_t).any(dim=1)
        x_t = x_t[mask]
        self.mean_ = x_t.mean(dim=0).cpu().numpy()
        self.scale_ = x_t.std(dim=0, unbiased=True).cpu().numpy()
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, x):
        return (np.asarray(x, dtype=np.float32) - self.mean_) / self.scale_

    def inverse_transform(self, x):
        return np.asarray(x, dtype=np.float32) * self.scale_ + self.mean_


def get_dataset(cache_dir):
    return swm.data.HDF5Dataset(
        "tworoom", keys_to_cache=["action", "proprio"], cache_dir=Path(cache_dir),
    )


def build_processors(dataset):
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
    return np.array([np.max(step_idx[episode_idx == ep]) + 1 for ep in episodes])


def sample_eval_starts(dataset, num_eval, goal_offset_steps, seed):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - goal_offset_steps - 1
    max_start_idx_dict = {ep: max_start_idx[i] for i, ep in enumerate(ep_indices)}
    max_start_per_row = np.array(
        [max_start_idx_dict[ep] for ep in dataset.get_col_data(col_name)]
    )

    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]

    if len(valid_indices) < num_eval:
        raise ValueError(f"Need {num_eval} starts, got {len(valid_indices)}.")

    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(valid_indices, size=num_eval, replace=False))
    row_data = dataset.get_row_data(chosen)
    return row_data[col_name], row_data["step_idx"], chosen


def pick_device(s):
    if s != "auto":
        return torch.device(s)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Multi-Scale InvDyn on TwoRoom")
    p.add_argument("--phase1-ckpt", type=str, required=True)
    p.add_argument("--phase2-ckpt", type=str, required=True)
    p.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-eval", type=int, default=50)
    p.add_argument("--goal-offset-steps", type=int, default=25)
    p.add_argument("--eval-budget", type=int, default=50)
    p.add_argument("--img-size", type=int, default=64)

    # Planner selection
    p.add_argument("--planner-mode", type=str, default="two_level",
                   choices=["two_level", "coarse_cem"],
                   help="two_level uses TwoLevelPlanner end-to-end; coarse_cem uses legacy CEMSolver+TwoLevelCostModel")

    # MPC config (shared)
    p.add_argument("--horizon", type=int, default=5,
                   help="Number of distinct planned actions per replan. "
                        "For two_level mode, must equal num_coarse_segments * horizon_h.")
    p.add_argument("--receding-horizon", type=int, default=5)
    p.add_argument("--action-block", type=int, default=5,
                   help="Each planned action is repeated this many env steps (frameskip).")

    # Two-level planner CEM knobs
    p.add_argument("--num-coarse-segments", type=int, default=None,
                   help="K. Defaults to horizon // horizon_h.")
    p.add_argument("--coarse-cem-samples", type=int, default=256)
    p.add_argument("--coarse-cem-steps", type=int, default=10)
    p.add_argument("--coarse-topk", type=int, default=32)
    p.add_argument("--fine-cem-samples", type=int, default=256)
    p.add_argument("--fine-cem-steps", type=int, default=10)
    p.add_argument("--fine-topk", type=int, default=32)
    p.add_argument("--invdyn-init", type=int, default=1,
                   help="1 = warm-start CEM with inverse dynamics; 0 = random init (ablation).")
    p.add_argument("--init-sigma", type=float, default=0.5)

    # Legacy CEMSolver knobs (only used when --planner-mode coarse_cem)
    p.add_argument("--num-samples", type=int, default=300)
    p.add_argument("--cem-steps", type=int, default=30)
    p.add_argument("--topk", type=int, default=30)
    p.add_argument("--var-scale", type=float, default=1.0)
    p.add_argument("--solver-batch-size", type=int, default=25)

    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--save-video", type=bool, default=True)
    p.add_argument("--output-json", type=str, default=None)
    return p.parse_args()


def build_policy(args, device, process, transform):
    """Build either the TwoLevelPlannerPolicy (default) or the legacy
    WorldModelPolicy + CEMSolver + TwoLevelCostModel stack."""
    plan_cfg = swm.PlanConfig(
        horizon=args.horizon,
        receding_horizon=args.receding_horizon,
        action_block=args.action_block,
    )

    if args.planner_mode == "two_level":
        planner = TwoLevelPlanner(args.phase1_ckpt, args.phase2_ckpt, device)
        policy = TwoLevelPlannerPolicy(
            planner=planner,
            config=plan_cfg,
            num_coarse_segments=args.num_coarse_segments,
            coarse_cem_samples=args.coarse_cem_samples,
            coarse_cem_steps=args.coarse_cem_steps,
            coarse_topk=args.coarse_topk,
            fine_cem_samples=args.fine_cem_samples,
            fine_cem_steps=args.fine_cem_steps,
            fine_topk=args.fine_topk,
            invdyn_init=bool(args.invdyn_init),
            init_sigma=args.init_sigma,
            process=process,
            transform=transform,
        )
        return policy

    # coarse_cem (legacy)
    cost_model = TwoLevelCostModel(args.phase1_ckpt, args.phase2_ckpt, device)
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
    return swm.policy.WorldModelPolicy(
        solver=solver,
        config=plan_cfg,
        process=process,
        transform=transform,
    )


def main():
    args = parse_args()
    device = pick_device(args.device)

    dataset = get_dataset(args.cache_dir)
    process = build_processors(dataset)
    transform = {
        "pixels": img_transform(args.img_size),
        "goal": img_transform(args.img_size),
    }

    world = swm.World(
        env_name="swm/TwoRoom-v1",
        num_envs=args.num_eval,
        max_episode_steps=2 * args.eval_budget,
        history_size=1,
        frame_skip=1,
        image_shape=(224, 224),
    )

    policy = build_policy(args, device, process, transform)

    eval_episodes, eval_start_idx, sampled_rows = sample_eval_starts(
        dataset, args.num_eval, args.goal_offset_steps, args.seed,
    )

    world.set_policy(policy)

    callables = [
        {"method": "_set_state", "args": {"state": {"value": "proprio"}}},
        {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_proprio"}}},
    ]

    video_path = Path(args.phase1_ckpt).parent / f"eval_videos_{args.planner_mode}"

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

    results = {
        "planner_mode": args.planner_mode,
        "phase1_ckpt": str(args.phase1_ckpt),
        "phase2_ckpt": str(args.phase2_ckpt),
        "device": str(device),
        "num_eval": args.num_eval,
        "metrics": {
            "success_rate": float(metrics["success_rate"]),
            "episode_successes": np.asarray(metrics["episode_successes"]).astype(int).tolist(),
        },
        "evaluation_time_sec": elapsed,
        "config": {
            "horizon": args.horizon,
            "receding_horizon": args.receding_horizon,
            "action_block": args.action_block,
            "invdyn_init": bool(args.invdyn_init),
            "num_coarse_segments": args.num_coarse_segments,
        },
    }

    output_json = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else Path(args.phase1_ckpt).parent / f"eval_{args.planner_mode}.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")

    print("=" * 72)
    print(f"Evaluation: Multi-Scale InvDyn ({args.planner_mode})")
    print("=" * 72)
    print(f"success_rate: {metrics['success_rate']:.2f}")
    print(f"time: {elapsed:.1f}s")
    print(f"saved: {output_json}")
    print("=" * 72)


if __name__ == "__main__":
    main()

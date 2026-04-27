"""
Two-level CEM planning with inverse dynamics warm-start.

Key insight: inverse dynamics models are not just for training —
they directly provide initial action estimates for CEM at both scales.

All rollouts pass full token sequences (B, num_tokens, D) between steps,
matching the multi-step rollout used during training.

Level 1 (Coarse):
  1. Coarse inverse dynamics gives initial estimate: â^c = inv_coarse(z_t, z_goal)
  2. CEM refines around this estimate → waypoints

Level 2 (Fine):
  1. Fine inverse dynamics gives initial estimates: â^f = inv_fine(z_t, z_target)
  2. CEM refines around these estimates → fine actions
"""

import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from stable_worldmodel.policy import BasePolicy

from src_wm2.models import (
    CoarseActionDecoder,
    CoarseActionEncoder,
    FineActionDecoder,
    FineActionEncoder,
    MultiScaleInvDynWorldModel,
    ProprioDecoder,
)


def _load_phase1(ckpt_path: str | Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["config"]
    model = MultiScaleInvDynWorldModel(
        latent_dim=cfg["latent_dim"],
        fine_action_dim=cfg["fine_action_dim"],
        coarse_action_dim=cfg["coarse_action_dim"],
        num_tokens=cfg["num_tokens"],
        horizon_h=cfg["horizon_h"],
        nhead=cfg["nhead"],
        num_fine_layers=cfg["num_fine_layers"],
        dim_ff=cfg["dim_ff"],
        ema_momentum=cfg["ema_momentum"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().requires_grad_(False)
    return model.to(device), cfg


class TwoLevelPlanner(nn.Module):
    """Two-level CEM planner with inverse dynamics warm-start.

    Instead of CEM starting from random (μ=0, σ=1), inverse dynamics
    provides an informed initial estimate. CEM then refines locally.

    This makes inverse dynamics useful at BOTH training and planning time.
    """

    def __init__(
        self,
        phase1_ckpt: str | Path,
        phase2_ckpt: str | Path,
        device: torch.device,
    ):
        super().__init__()

        # ---- Phase 1: world model + inverse dynamics ----
        self.world_model, p1_cfg = _load_phase1(phase1_ckpt, device)
        self.horizon_h = p1_cfg["horizon_h"]
        self.fine_action_dim = p1_cfg["fine_action_dim"]
        self.coarse_action_dim = p1_cfg["coarse_action_dim"]

        # ---- Phase 2: action encoders/decoders ----
        p2_ckpt = torch.load(phase2_ckpt, map_location="cpu")
        p2_cfg = p2_ckpt["config"]
        self.action_dim = p2_ckpt["action_dim"]

        self.fine_enc = FineActionEncoder(
            self.action_dim, self.fine_action_dim, p2_cfg["hidden_dim_fine"]
        )
        self.fine_enc.load_state_dict(p2_ckpt["fine_enc_state"])

        self.fine_dec = FineActionDecoder(
            self.fine_action_dim, self.action_dim, p2_cfg["hidden_dim_fine"]
        )
        self.fine_dec.load_state_dict(p2_ckpt["fine_dec_state"])

        self.coarse_enc = CoarseActionEncoder(
            self.action_dim, self.horizon_h,
            self.coarse_action_dim, p2_cfg["hidden_dim_coarse"],
        )
        self.coarse_enc.load_state_dict(p2_ckpt["coarse_enc_state"])

        self.coarse_dec = CoarseActionDecoder(
            self.coarse_action_dim, self.action_dim,
            self.horizon_h, p2_cfg["hidden_dim_coarse"],
        )
        self.coarse_dec.load_state_dict(p2_ckpt["coarse_dec_state"])

        self.eval().requires_grad_(False)
        self.to(device)

    @property
    def device(self) -> torch.device:
        return next(self.world_model.encoder.parameters()).device

    # ================================================================
    # Inverse dynamics warm-start (batched)
    # ================================================================

    @torch.inference_mode()
    def _coarse_invdyn_init_batch(
        self,
        z_start: torch.Tensor,
        z_goal: torch.Tensor,
        num_segments: int,
    ) -> torch.Tensor:
        """Coarse inverse dynamics warm-start, batched over envs.

        Args:
            z_start: (B, num_tokens, D)
            z_goal:  (B, num_tokens, D)

        Returns:
            (B, num_segments, coarse_action_dim)
        """
        pool = self.world_model.pool
        inits = []
        z = z_start
        z_goal_pool = pool(z_goal)

        for _ in range(num_segments):
            a_c = self.world_model.coarse_inv_dyn(pool(z), z_goal_pool)  # (B, coarse_dim)
            inits.append(a_c)
            z = self.world_model.coarse_predictor(z, a_c)

        return torch.stack(inits, dim=1)

    @torch.inference_mode()
    def _fine_invdyn_init_batch(
        self,
        z_start: torch.Tensor,
        z_target: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        """Fine inverse dynamics warm-start, batched over envs.

        Args:
            z_start:  (B, num_tokens, D)
            z_target: (B, num_tokens, D)

        Returns:
            (B, num_steps, fine_action_dim)
        """
        pool = self.world_model.pool
        inits = []
        z = z_start
        z_target_pool = pool(z_target)

        for _ in range(num_steps):
            a_f = self.world_model.fine_inv_dyn(pool(z), z_target_pool)  # (B, fine_dim)
            inits.append(a_f)
            z = self.world_model.fine_predictor(z, a_f, z_target)

        return torch.stack(inits, dim=1)

    # ================================================================
    # Two-level planning (batched over envs)
    # ================================================================

    @torch.inference_mode()
    def plan_batch(
        self,
        current_pixels: torch.Tensor,
        goal_pixels: torch.Tensor,
        num_coarse_segments: int = 1,
        coarse_cem_samples: int = 256,
        coarse_cem_steps: int = 10,
        coarse_topk: int = 32,
        fine_cem_samples: int = 256,
        fine_cem_steps: int = 10,
        fine_topk: int = 32,
        invdyn_init: bool = True,
        init_sigma: float = 0.5,
    ) -> torch.Tensor:
        """Batched two-level CEM planning.

        Args:
            current_pixels: (B, 3, H, W)
            goal_pixels:    (B, 3, H, W)
            num_coarse_segments: K. Total fine actions returned = K * horizon_h.
            invdyn_init: if True, warm-start CEM with inverse dynamics estimate.

        Returns:
            real_actions: (B, K * horizon_h, action_dim) — decoded fine actions.
        """
        K = num_coarse_segments
        h = self.horizon_h
        device = self.device

        # Use target (EMA) encoder for goal — supervision targets in training
        # are produced by the EMA encoder, so the predictor's output distribution
        # is aligned with the target encoder, not the online encoder.
        z_start = self.world_model.encode(current_pixels.to(device))     # (B, N_tok, D)
        z_goal = self.world_model.encode_target(goal_pixels.to(device))  # (B, N_tok, D)

        B, N_tok, D = z_start.shape

        # ============================================================
        # Level 1: Coarse CEM with inverse dynamics warm-start
        # ============================================================

        if invdyn_init:
            mu_c = self._coarse_invdyn_init_batch(z_start, z_goal, K)  # (B, K, cd)
            sigma_c = torch.full_like(mu_c, init_sigma)
        else:
            mu_c = torch.zeros(B, K, self.coarse_action_dim, device=device)
            sigma_c = torch.ones(B, K, self.coarse_action_dim, device=device)

        S = coarse_cem_samples
        z_start_exp = z_start.unsqueeze(1).expand(B, S, N_tok, D).reshape(B * S, N_tok, D)
        z_goal_exp = z_goal.unsqueeze(1).expand(B, S, N_tok, D).reshape(B * S, N_tok, D)
        batch_idx_c = torch.arange(B, device=device).unsqueeze(1).expand(B, coarse_topk)

        for _ in range(coarse_cem_steps):
            noise = torch.randn(B, S, K, self.coarse_action_dim, device=device)
            candidates = (mu_c.unsqueeze(1) + sigma_c.unsqueeze(1) * noise).clamp(-3, 3)

            flat_cand = candidates.reshape(B * S, K, self.coarse_action_dim)
            z = z_start_exp
            for k in range(K):
                z = self.world_model.coarse_predictor(z, flat_cand[:, k])

            costs = (z - z_goal_exp).pow(2).sum(dim=(-1, -2)).view(B, S)

            top_idx = costs.topk(coarse_topk, largest=False, dim=1).indices  # (B, topk)
            top = candidates[batch_idx_c, top_idx]  # (B, topk, K, cd)
            mu_c = top.mean(dim=1)
            sigma_c = top.std(dim=1).clamp(min=0.01)

        best_coarse = mu_c  # (B, K, cd)

        # Compute waypoints (token-level), batched
        waypoints = [z_start]  # each (B, N_tok, D)
        z = z_start
        for k in range(K):
            z = self.world_model.coarse_predictor(z, best_coarse[:, k])
            waypoints.append(z)

        # ============================================================
        # Level 2: Fine CEM per segment, batched over envs
        # ============================================================

        S_f = fine_cem_samples
        batch_idx_f = torch.arange(B, device=device).unsqueeze(1).expand(B, fine_topk)

        all_fine_real = []  # list of (B, h, action_dim)

        for k in range(K):
            z_seg_start = waypoints[k]      # (B, N_tok, D)
            z_seg_target = waypoints[k + 1] # (B, N_tok, D)

            if invdyn_init:
                mu_f = self._fine_invdyn_init_batch(z_seg_start, z_seg_target, h)
                sigma_f = torch.full_like(mu_f, init_sigma)
            else:
                mu_f = torch.zeros(B, h, self.fine_action_dim, device=device)
                sigma_f = torch.ones(B, h, self.fine_action_dim, device=device)

            z_seg_start_exp = z_seg_start.unsqueeze(1).expand(B, S_f, N_tok, D).reshape(B * S_f, N_tok, D)
            z_seg_target_exp = z_seg_target.unsqueeze(1).expand(B, S_f, N_tok, D).reshape(B * S_f, N_tok, D)

            for _ in range(fine_cem_steps):
                noise = torch.randn(B, S_f, h, self.fine_action_dim, device=device)
                candidates = (mu_f.unsqueeze(1) + sigma_f.unsqueeze(1) * noise).clamp(-3, 3)

                flat_cand = candidates.reshape(B * S_f, h, self.fine_action_dim)
                z = z_seg_start_exp
                for t in range(h):
                    z = self.world_model.fine_predictor(
                        z, flat_cand[:, t], z_seg_target_exp
                    )

                costs = (z - z_seg_target_exp).pow(2).sum(dim=(-1, -2)).view(B, S_f)

                top_idx = costs.topk(fine_topk, largest=False, dim=1).indices
                top = candidates[batch_idx_f, top_idx]  # (B, topk, h, fd)
                mu_f = top.mean(dim=1)
                sigma_f = top.std(dim=1).clamp(min=0.01)

            # Decode mu_f (B, h, fine_dim) → real actions (B, h, action_dim)
            real = self.fine_dec(mu_f.reshape(B * h, -1)).view(B, h, -1)
            all_fine_real.append(real)

        return torch.cat(all_fine_real, dim=1)  # (B, K * h, action_dim)

    @torch.inference_mode()
    def plan(
        self,
        current_pixels: torch.Tensor,
        goal_pixels: torch.Tensor,
        num_coarse_segments: int = 4,
        **kwargs,
    ) -> list[torch.Tensor]:
        """Single-instance planning. Wraps plan_batch for B=1.

        Returns: list of real actions [a_0, ..., a_{K*h-1}], each (action_dim,)
        """
        actions = self.plan_batch(
            current_pixels, goal_pixels,
            num_coarse_segments=num_coarse_segments,
            **kwargs,
        )  # (1, K*h, action_dim)
        return list(actions.squeeze(0))


# ================================================================
# Policy wrapper for swm.World — uses TwoLevelPlanner directly
# ================================================================


class TwoLevelPlannerPolicy(BasePolicy):
    """Policy that calls TwoLevelPlanner.plan_batch on each replan.

    Mirrors WorldModelPolicy's MPC loop (action buffer, receding horizon)
    but bypasses swm's CEMSolver and TwoLevelCostModel: planning happens
    end-to-end in latent space using the coarse+fine predictor hierarchy
    with inverse-dynamics warm-start.

    Each fine action produced by the planner corresponds to `action_block`
    environment steps (matching the frameskip used at training time), so
    fine actions are repeat_interleaved before being pushed to the buffer.
    """

    def __init__(
        self,
        planner: TwoLevelPlanner,
        config,
        num_coarse_segments: int | None = None,
        coarse_cem_samples: int = 256,
        coarse_cem_steps: int = 10,
        coarse_topk: int = 32,
        fine_cem_samples: int = 256,
        fine_cem_steps: int = 10,
        fine_topk: int = 32,
        invdyn_init: bool = True,
        init_sigma: float = 0.5,
        process: dict | None = None,
        transform: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.type = "two_level_planner"
        self.cfg = config
        self.planner = planner

        h = planner.horizon_h
        K = num_coarse_segments if num_coarse_segments is not None else max(1, config.horizon // h)
        assert K * h == config.horizon, (
            f"PlanConfig.horizon ({config.horizon}) must equal "
            f"num_coarse_segments ({K}) * horizon_h ({h}). "
            f"Either set num_coarse_segments explicitly or choose horizon = K * h."
        )
        self.num_coarse_segments = K

        self.coarse_cem_samples = coarse_cem_samples
        self.coarse_cem_steps = coarse_cem_steps
        self.coarse_topk = coarse_topk
        self.fine_cem_samples = fine_cem_samples
        self.fine_cem_steps = fine_cem_steps
        self.fine_topk = fine_topk
        self.invdyn_init = invdyn_init
        self.init_sigma = init_sigma

        self.process = process or {}
        self.transform = transform or {}
        self._action_buffer: deque | None = None

    @property
    def flatten_receding_horizon(self) -> int:
        return self.cfg.receding_horizon * self.cfg.action_block

    def set_env(self, env):
        self.env = env
        self._action_buffer = deque(maxlen=self.flatten_receding_horizon)

    def get_action(self, info_dict: dict, **kwargs) -> np.ndarray:
        assert hasattr(self, "env"), "Environment not set for the policy"
        assert "pixels" in info_dict, "'pixels' must be provided in info_dict"
        assert "goal" in info_dict, "'goal' must be provided in info_dict"

        info_dict = self._prepare_info(info_dict)

        if len(self._action_buffer) == 0:
            # info_dict['pixels'], info_dict['goal']: (num_envs, history, C, H, W)
            current = info_dict["pixels"][:, -1].to(
                self.planner.device, dtype=torch.float32
            )
            goal = info_dict["goal"][:, -1].to(
                self.planner.device, dtype=torch.float32
            )

            actions = self.planner.plan_batch(
                current, goal,
                num_coarse_segments=self.num_coarse_segments,
                coarse_cem_samples=self.coarse_cem_samples,
                coarse_cem_steps=self.coarse_cem_steps,
                coarse_topk=self.coarse_topk,
                fine_cem_samples=self.fine_cem_samples,
                fine_cem_steps=self.fine_cem_steps,
                fine_topk=self.fine_topk,
                invdyn_init=self.invdyn_init,
                init_sigma=self.init_sigma,
            )  # (num_envs, K*h, action_dim)

            actions = actions.detach().cpu()

            # Each fine action corresponds to action_block env steps.
            keep_actions = self.cfg.receding_horizon
            block = self.cfg.action_block
            plan = actions[:, :keep_actions]                     # (num_envs, R, action_dim)
            plan = plan.repeat_interleave(block, dim=1)           # (num_envs, R*block, action_dim)

            # extend buffer with R*block items, each (num_envs, action_dim)
            self._action_buffer.extend(plan.permute(1, 0, 2))

        action = self._action_buffer.popleft()                    # (num_envs, action_dim)
        action = action.reshape(*self.env.action_space.shape).numpy()

        if "action" in self.process:
            action = self.process["action"].inverse_transform(action)

        return action


# ================================================================
# Cost model for swm.CEMSolver compatibility (legacy / ablation path)
# ================================================================


class TwoLevelCostModel(nn.Module):
    """Cost model compatible with swm.CEMSolver interface.

    Uses coarse-level rollout for fast evaluation of candidate action sequences.
    Rollouts pass full token sequences between steps. The ProprioDecoder
    (trained in Phase 2) maps the rolled-out latent state to proprio space
    for cost computation against the goal_proprio target.
    """

    def __init__(
        self,
        phase1_ckpt: str | Path,
        phase2_ckpt: str | Path,
        device: torch.device,
    ):
        super().__init__()

        self.world_model, p1_cfg = _load_phase1(phase1_ckpt, device)
        self.horizon_h = p1_cfg["horizon_h"]
        self.coarse_action_dim = p1_cfg["coarse_action_dim"]

        p2_ckpt = torch.load(phase2_ckpt, map_location="cpu")
        self.action_dim = p2_ckpt["action_dim"]
        self.proprio_dim = p2_ckpt["proprio_dim"]
        p2_cfg = p2_ckpt["config"]

        self.coarse_enc = CoarseActionEncoder(
            self.action_dim, self.horizon_h,
            self.coarse_action_dim, p2_cfg["hidden_dim_coarse"],
        )
        self.coarse_enc.load_state_dict(p2_ckpt["coarse_enc_state"])

        self.fine_enc = FineActionEncoder(
            self.action_dim, p1_cfg["fine_action_dim"], p2_cfg["hidden_dim_fine"],
        )
        self.fine_enc.load_state_dict(p2_ckpt["fine_enc_state"])

        self.proprio_dec = ProprioDecoder(
            p1_cfg["latent_dim"], self.proprio_dim, p2_cfg["hidden_dim_proprio"],
        )
        self.proprio_dec.load_state_dict(p2_ckpt["proprio_dec_state"])

        self.eval().requires_grad_(False)
        self.to(device)

    @property
    def device(self) -> torch.device:
        return next(self.world_model.encoder.parameters()).device

    @torch.inference_mode()
    def get_cost(
        self,
        info_dict: dict,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate candidates using hierarchical rollout + proprio decoding."""
        pixels = info_dict["pixels"][:, :, -1].to(self.device, dtype=torch.float32)

        if "goal_proprio" in info_dict:
            goal_proprio = info_dict["goal_proprio"][:, :, -1].to(
                self.device, dtype=torch.float32
            )
        else:
            raise KeyError("Expected 'goal_proprio' in info_dict")

        batch_size, num_samples = pixels.shape[:2]
        h = self.horizon_h

        flat_pixels = pixels.reshape(batch_size * num_samples, *pixels.shape[2:])
        z = self.world_model.encode(flat_pixels)  # (BxS, num_tokens, D)

        _, _, horizon, flat_dim = candidates.shape
        action_block = flat_dim // self.action_dim

        actions = candidates.view(
            batch_size, num_samples, horizon, action_block, self.action_dim
        ).to(self.device, dtype=torch.float32)

        total_steps = horizon * action_block
        actions_flat = actions.reshape(
            batch_size * num_samples, total_steps, self.action_dim
        )

        BxS = batch_size * num_samples
        num_coarse_steps = total_steps // h

        for seg in range(num_coarse_steps):
            seg_actions = actions_flat[:, seg * h : (seg + 1) * h]
            a_coarse = self.coarse_enc(seg_actions.reshape(BxS, -1))
            z = self.world_model.coarse_predictor(z, a_coarse)

        # Decode latent to proprio space for proper cost computation
        proprio_pred = self.proprio_dec(z)               # (BxS, proprio_dim)
        goal_flat = goal_proprio.reshape(BxS, -1)        # (BxS, proprio_dim)
        cost = (proprio_pred - goal_flat).pow(2).sum(dim=-1)

        return cost.view(batch_size, num_samples)

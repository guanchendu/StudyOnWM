"""
Two-level CEM planning with inverse dynamics warm-start.

Key insight: inverse dynamics models are not just for training —
they directly provide initial action estimates for CEM at both scales.

Level 1 (Coarse):
  1. Coarse inverse dynamics gives initial estimate: â^c = inv_coarse(z_t, z_goal)
  2. CEM refines around this estimate → waypoints

Level 2 (Fine):
  1. Fine inverse dynamics gives initial estimates: â^f = inv_fine(z_t, z_target)
  2. CEM refines around these estimates → fine actions
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src_wm2.models import (
    CoarseActionDecoder,
    CoarseActionEncoder,
    FineActionDecoder,
    FineActionEncoder,
    MultiScaleInvDynWorldModel,
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
    # Inverse dynamics warm-start
    # ================================================================

    @torch.inference_mode()
    def _coarse_invdyn_init(
        self,
        z_start_pooled: torch.Tensor,
        z_goal: torch.Tensor,
        num_segments: int,
    ) -> torch.Tensor:
        """Use coarse inverse dynamics to generate initial coarse action sequence.

        Iteratively: predict â^c, step forward with coarse predictor, repeat.

        Args:
            z_start_pooled: (1, D)
            z_goal:         (1, D)
            num_segments:   K

        Returns:
            coarse_actions_init: (K, coarse_action_dim)
        """
        num_tokens = self.world_model.encoder.num_tokens
        inits = []
        z = z_start_pooled  # (1, D)

        for _ in range(num_segments):
            a_c = self.world_model.coarse_inv_dyn(z, z_goal)  # (1, coarse_action_dim)
            inits.append(a_c.squeeze(0))
            z_tokens = z.unsqueeze(1).expand(-1, num_tokens, -1)
            z = self.world_model.coarse_predictor(z_tokens, a_c)

        return torch.stack(inits, dim=0)  # (K, coarse_action_dim)

    @torch.inference_mode()
    def _fine_invdyn_init(
        self,
        z_start_pooled: torch.Tensor,
        z_target: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        """Use fine inverse dynamics to generate initial fine action sequence.

        At each step, fine inv dyn estimates the action toward z_target,
        then fine predictor steps forward.

        Args:
            z_start_pooled: (1, D)
            z_target:       (1, D)
            num_steps:      h

        Returns:
            fine_actions_init: (h, fine_action_dim)
        """
        num_tokens = self.world_model.encoder.num_tokens
        inits = []
        z = z_start_pooled  # (1, D)

        for _ in range(num_steps):
            a_f = self.world_model.fine_inv_dyn(z, z_target)  # (1, fine_action_dim)
            inits.append(a_f.squeeze(0))
            z_tokens = z.unsqueeze(1).expand(-1, num_tokens, -1)
            z = self.world_model.fine_predictor(z_tokens, a_f, z_target)

        return torch.stack(inits, dim=0)  # (h, fine_action_dim)

    # ================================================================
    # Two-level planning
    # ================================================================

    @torch.inference_mode()
    def plan(
        self,
        current_pixels: torch.Tensor,
        goal_pixels: torch.Tensor,
        num_coarse_segments: int = 4,
        coarse_cem_samples: int = 256,
        coarse_cem_steps: int = 20,
        coarse_topk: int = 32,
        fine_cem_samples: int = 256,
        fine_cem_steps: int = 20,
        fine_topk: int = 32,
        invdyn_init: bool = True,
        init_sigma: float = 0.5,
    ) -> list[torch.Tensor]:
        """
        Two-level CEM planning with optional inverse dynamics warm-start.

        Args:
            current_pixels:      (1, 3, H, W)
            goal_pixels:         (1, 3, H, W)
            num_coarse_segments: K segments of h steps each
            invdyn_init:         if True, use inverse dynamics for CEM initialization
                                 if False, fall back to random init (for ablation)
            init_sigma:          initial σ when using invdyn init (smaller = trust init more)

        Returns:
            list of real actions [a_0, a_1, ..., a_{K*h-1}]
        """
        K = num_coarse_segments
        h = self.horizon_h
        num_tokens = self.world_model.encoder.num_tokens
        device = self.device

        z_start = self.world_model.encode(current_pixels.to(device))
        z_start_pooled = self.world_model.pool(z_start)  # (1, D)
        z_goal = self.world_model.pool(
            self.world_model.encode(goal_pixels.to(device))
        )  # (1, D)

        # ============================================================
        # Level 1: Coarse CEM with inverse dynamics warm-start
        # ============================================================

        if invdyn_init:
            mu_c = self._coarse_invdyn_init(z_start_pooled, z_goal, K)
            sigma_c = torch.full_like(mu_c, init_sigma)
        else:
            mu_c = torch.zeros(K, self.coarse_action_dim, device=device)
            sigma_c = torch.ones(K, self.coarse_action_dim, device=device)

        for _ in range(coarse_cem_steps):
            noise = torch.randn(
                coarse_cem_samples, K, self.coarse_action_dim, device=device
            )
            candidates = (mu_c.unsqueeze(0) + sigma_c.unsqueeze(0) * noise).clamp(-3, 3)

            z = z_start_pooled.expand(coarse_cem_samples, -1)
            for k in range(K):
                z_tokens = z.unsqueeze(1).expand(-1, num_tokens, -1)
                z = self.world_model.coarse_predictor(z_tokens, candidates[:, k])

            costs = (z - z_goal).pow(2).sum(dim=-1)

            _, top_idx = costs.topk(coarse_topk, largest=False)
            top = candidates[top_idx]
            mu_c = top.mean(dim=0)
            sigma_c = top.std(dim=0).clamp(min=0.01)

        best_coarse_actions = mu_c  # (K, coarse_action_dim)

        # Compute waypoints
        waypoints = [z_start_pooled.squeeze(0)]  # (D,)
        z = z_start_pooled
        for k in range(K):
            z_tokens = z.unsqueeze(1).expand(-1, num_tokens, -1)
            z = self.world_model.coarse_predictor(
                z_tokens, best_coarse_actions[k].unsqueeze(0)
            )
            waypoints.append(z.squeeze(0))

        # ============================================================
        # Level 2: Fine CEM with inverse dynamics warm-start
        # ============================================================

        all_fine_actions = []

        for k in range(K):
            z_seg_start = waypoints[k].unsqueeze(0)       # (1, D)
            z_seg_target = waypoints[k + 1].unsqueeze(0)  # (1, D)

            if invdyn_init:
                mu_f = self._fine_invdyn_init(z_seg_start, z_seg_target, h)
                sigma_f = torch.full_like(mu_f, init_sigma)
            else:
                mu_f = torch.zeros(h, self.fine_action_dim, device=device)
                sigma_f = torch.ones(h, self.fine_action_dim, device=device)

            for _ in range(fine_cem_steps):
                noise = torch.randn(
                    fine_cem_samples, h, self.fine_action_dim, device=device
                )
                candidates = (mu_f.unsqueeze(0) + sigma_f.unsqueeze(0) * noise).clamp(-3, 3)

                z = z_seg_start.expand(fine_cem_samples, -1)
                cond = z_seg_target.expand(fine_cem_samples, -1)

                for t in range(h):
                    z_tokens = z.unsqueeze(1).expand(-1, num_tokens, -1)
                    z = self.world_model.fine_predictor(
                        z_tokens, candidates[:, t], cond
                    )

                costs = (z - z_seg_target).pow(2).sum(dim=-1)

                _, top_idx = costs.topk(fine_topk, largest=False)
                top = candidates[top_idx]
                mu_f = top.mean(dim=0)
                sigma_f = top.std(dim=0).clamp(min=0.01)

            for t in range(h):
                real_action = self.fine_dec(mu_f[t].unsqueeze(0)).squeeze(0)
                all_fine_actions.append(real_action)

        return all_fine_actions


# ================================================================
# Cost model for swm.CEMSolver compatibility
# ================================================================


class TwoLevelCostModel(nn.Module):
    """Cost model compatible with swm.CEMSolver interface.

    Uses coarse-level rollout for fast evaluation of candidate action sequences.
    Also uses inverse dynamics for warm-start when available.
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
        """Evaluate candidates using hierarchical rollout."""
        pixels = info_dict["pixels"][:, :, -1].to(self.device, dtype=torch.float32)

        if "goal_proprio" in info_dict:
            goal_proprio = info_dict["goal_proprio"][:, :, -1].to(
                self.device, dtype=torch.float32
            )
        else:
            raise KeyError("Expected 'goal_proprio' in info_dict")

        batch_size, num_samples = pixels.shape[:2]
        h = self.horizon_h
        num_tokens = self.world_model.encoder.num_tokens

        flat_pixels = pixels.reshape(batch_size * num_samples, *pixels.shape[2:])
        z_start = self.world_model.encode(flat_pixels)

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

        z = self.world_model.pool(z_start)

        for seg in range(num_coarse_steps):
            seg_actions = actions_flat[:, seg * h : (seg + 1) * h]
            a_coarse = self.coarse_enc(seg_actions.reshape(BxS, -1))
            z_tokens = z.unsqueeze(1).expand(-1, num_tokens, -1)
            z = self.world_model.coarse_predictor(z_tokens, a_coarse)

        goal_flat = goal_proprio.reshape(BxS, -1)
        cost = (z[:, :goal_flat.shape[-1]] - goal_flat).pow(2).sum(dim=-1)

        return cost.view(batch_size, num_samples)

"""
Hierarchical planning with CEM.

Full pipeline:
  1. CEM samples candidate action sequences in real action space
  2. ActionEncoder maps real → latent actions
  3. Hierarchical rollout: Coarse waypoints → Fine fill-in
  4. Compute cost vs goal
  5. CEM updates sampling distribution

Usage:
  This module provides HierarchicalCostModel compatible with swm.CEMSolver.
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

from src_wm.models import (
    ActionDecoder,
    ActionEncoder,
    HierarchicalInvDynWorldModel,
)


class HierarchicalCostModel(nn.Module):
    """Cost model for CEM-based planning.

    Wraps the full Phase 1 + Phase 2 model to evaluate candidate
    action sequences via hierarchical rollout.

    Compatible with swm.solver.CEMSolver's expected interface.
    """

    def __init__(
        self,
        phase1_ckpt_path: str | Path,
        phase2_ckpt_path: str | Path,
        device: torch.device,
    ):
        super().__init__()

        # ---- Load Phase 1 ----
        p1_ckpt = torch.load(phase1_ckpt_path, map_location="cpu")
        p1_cfg = p1_ckpt["config"]

        self.world_model = HierarchicalInvDynWorldModel(
            latent_dim=p1_cfg["latent_dim"],
            latent_action_dim=p1_cfg["latent_action_dim"],
            num_tokens=p1_cfg["num_tokens"],
            horizon_h=p1_cfg["horizon_h"],
            nhead=p1_cfg["nhead"],
            num_fine_layers=p1_cfg["num_fine_layers"],
            dim_ff=p1_cfg["dim_ff"],
            ema_momentum=p1_cfg["ema_momentum"],
        )
        self.world_model.load_state_dict(p1_ckpt["model_state_dict"])
        self.world_model.eval()
        self.world_model.requires_grad_(False)

        # ---- Load Phase 2 ----
        p2_ckpt = torch.load(phase2_ckpt_path, map_location="cpu")

        self.action_dim = p2_ckpt["action_dim"]
        latent_action_dim = p1_cfg["latent_action_dim"]
        hidden_dim = p2_ckpt["config"]["hidden_dim"]

        self.action_encoder = ActionEncoder(
            self.action_dim, latent_action_dim, hidden_dim
        )
        self.action_encoder.load_state_dict(p2_ckpt["action_encoder_state_dict"])
        self.action_encoder.eval()
        self.action_encoder.requires_grad_(False)

        self.action_decoder = ActionDecoder(
            latent_action_dim, self.action_dim, hidden_dim
        )
        self.action_decoder.load_state_dict(p2_ckpt["action_decoder_state_dict"])
        self.action_decoder.eval()
        self.action_decoder.requires_grad_(False)

        self.horizon_h = p1_cfg["horizon_h"]
        self.to(device)

    @property
    def device(self) -> torch.device:
        return next(self.world_model.encoder.parameters()).device

    @torch.inference_mode()
    def hierarchical_rollout(
        self,
        z_start: torch.Tensor,
        latent_actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run hierarchical rollout: coarse waypoints then fine fill-in.

        Args:
            z_start:        (B, num_tokens, D) — initial latent state
            latent_actions:  (B, H, latent_action_dim) — full action sequence

        Returns:
            z_final: (B, D) — predicted final latent state
        """
        h = self.horizon_h
        B, H, _ = latent_actions.shape
        num_tokens = z_start.shape[1]

        num_coarse_steps = H // h
        remainder = H % h

        z_current_tokens = z_start  # (B, num_tokens, D)
        z_final = None

        for i in range(num_coarse_steps):
            seg_start = i * h
            seg_end = seg_start + h

            # ---- Coarse: generate waypoint ----
            a_block = latent_actions[:, seg_start:seg_end]  # (B, h, la_dim)
            z_waypoint = self.world_model.coarse_predictor(
                z_current_tokens, a_block
            )  # (B, D)

            # ---- Fine: fill in each step, conditioned on waypoint ----
            z_t = z_current_tokens
            for step in range(h):
                t = seg_start + step
                z_next = self.world_model.fine_predictor(
                    z_t, latent_actions[:, t], z_waypoint
                )  # (B, D)
                z_t = z_next.unsqueeze(1).expand(-1, num_tokens, -1)

            z_current_tokens = z_t
            z_final = z_next

        # Handle remainder steps (< h) with last waypoint as condition
        if remainder > 0 and z_final is not None:
            for t in range(num_coarse_steps * h, H):
                z_next = self.world_model.fine_predictor(
                    z_current_tokens, latent_actions[:, t], z_final
                )
                z_current_tokens = z_next.unsqueeze(1).expand(-1, num_tokens, -1)
            z_final = z_next

        return z_final

    @torch.inference_mode()
    def get_cost(
        self,
        info_dict: dict,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate candidate action sequences for CEM.

        Args:
            info_dict: dict with:
                "pixels":       (batch_size, num_samples, C, H, W) or with history dim
                "goal_proprio": (batch_size, num_samples, proprio_dim)
                "proprio":      (batch_size, num_samples, proprio_dim)
            candidates: (batch_size, num_samples, horizon, flat_dim)
                flat_dim = action_block * action_dim

        Returns:
            costs: (batch_size, num_samples)
        """
        pixels = info_dict["pixels"][:, :, -1].to(self.device, dtype=torch.float32)
        goal_proprio = info_dict["goal_proprio"][:, :, -1].to(
            self.device, dtype=torch.float32
        )

        batch_size, num_samples = pixels.shape[:2]

        # Flatten batch and sample dims
        flat_pixels = pixels.reshape(
            batch_size * num_samples, *pixels.shape[2:]
        )

        # Encode start state
        z_start = self.world_model.encode(flat_pixels)  # (B*S, num_tokens, D)

        # Parse candidates → real actions → latent actions
        _, _, horizon, flat_dim = candidates.shape
        action_block = flat_dim // self.action_dim
        total_steps = horizon * action_block

        # (B, S, horizon, action_block, action_dim)
        actions = candidates.view(
            batch_size, num_samples, horizon, action_block, self.action_dim
        ).to(self.device, dtype=torch.float32)

        # (B*S, total_steps, action_dim)
        actions_flat = actions.reshape(batch_size * num_samples, total_steps, self.action_dim)

        # Map to latent action space
        BxS, T, A = actions_flat.shape
        latent_actions = self.action_encoder(
            actions_flat.reshape(BxS * T, A)
        ).reshape(BxS, T, -1)  # (B*S, T, latent_action_dim)

        # Hierarchical rollout
        z_final = self.hierarchical_rollout(z_start, latent_actions)  # (B*S, D)

        # We need to decode z_final to proprio space for cost computation.
        # For now, use L2 distance in latent space to goal.
        # If a proprio decoder is available, use it instead.
        goal_flat = goal_proprio.reshape(batch_size * num_samples, -1)

        # Encode goal observation if available, otherwise use proprio distance
        # For tworoom: cost = distance between predicted proprio and goal proprio
        # This requires a proprio decoder — we'll add a simple one
        # For now, use latent space distance as proxy
        cost = z_final.norm(dim=-1)  # placeholder

        # Reshape back
        return cost.view(batch_size, num_samples)


@torch.inference_mode()
def plan_single_step(
    cost_model: HierarchicalCostModel,
    current_pixels: torch.Tensor,
    goal_pixels: torch.Tensor,
    horizon: int = 10,
    num_samples: int = 300,
    cem_steps: int = 30,
    topk: int = 30,
) -> torch.Tensor:
    """
    Simplified planning for a single environment step.

    Returns the first real action to execute.
    """
    device = cost_model.device
    action_dim = cost_model.action_dim
    h = cost_model.horizon_h
    num_tokens = cost_model.world_model.encoder.num_tokens

    # Encode current and goal states
    z_current = cost_model.world_model.encode(
        current_pixels.unsqueeze(0).to(device)
    )  # (1, num_tokens, D)
    z_goal = cost_model.world_model.encode(
        goal_pixels.unsqueeze(0).to(device)
    ).mean(dim=1)  # (1, D)

    # CEM optimization
    mu = torch.zeros(horizon, action_dim, device=device)
    sigma = torch.ones(horizon, action_dim, device=device)

    for step in range(cem_steps):
        # Sample candidates
        noise = torch.randn(num_samples, horizon, action_dim, device=device)
        candidates = mu.unsqueeze(0) + sigma.unsqueeze(0) * noise
        candidates = candidates.clamp(-1, 1)

        # Map to latent actions
        flat_cands = candidates.reshape(num_samples * horizon, action_dim)
        latent_cands = cost_model.action_encoder(flat_cands)
        latent_cands = latent_cands.reshape(num_samples, horizon, -1)

        # Rollout each candidate
        z_starts = z_current.expand(num_samples, -1, -1)
        z_finals = cost_model.hierarchical_rollout(
            z_starts, latent_cands
        )  # (num_samples, D)

        # Cost = distance to goal in latent space
        costs = (z_finals - z_goal).pow(2).sum(dim=-1)  # (num_samples,)

        # Select top-k
        _, top_idx = costs.topk(topk, largest=False)
        top_candidates = candidates[top_idx]  # (topk, horizon, action_dim)

        # Update distribution
        mu = top_candidates.mean(dim=0)
        sigma = top_candidates.std(dim=0).clamp(min=0.01)

    # Return first action from best sequence
    best_action = mu[0]  # (action_dim,)
    return best_action

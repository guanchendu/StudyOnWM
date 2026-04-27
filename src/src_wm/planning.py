"""
Hierarchical planning with CEM.

Single fine inverse dynamics + hierarchical (coarse + fine) predictors.
Inverse dynamics is NOT used at planning time — it only supplies latent
action targets during Phase 1 training. Planning maps real actions →
latent actions via the Phase 2 ActionEncoder, runs hierarchical rollout,
and scores via ProprioDecoder.

Cost (now correct):
  cost = ||proprio_dec(z_final) - goal_proprio||²

Compatible with swm.CEMSolver (get_cost interface).
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
    ProprioDecoder,
)


class HierarchicalCostModel(nn.Module):
    """Cost model for CEM-based planning.

    Loads frozen Phase 1 (encoder + predictors) and Phase 2
    (action_enc, action_dec, proprio_dec). Evaluates candidate
    real-action sequences via:
        real action → action_enc → latent action
        z_start → coarse rollout → fine rollout → z_final
        cost = ||proprio_dec(z_final) - goal_proprio||²
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
        self.world_model.eval().requires_grad_(False)

        # ---- Load Phase 2 ----
        p2_ckpt = torch.load(phase2_ckpt_path, map_location="cpu")
        self.action_dim = p2_ckpt["action_dim"]
        self.proprio_dim = p2_ckpt["proprio_dim"]
        latent_action_dim = p1_cfg["latent_action_dim"]
        latent_dim = p1_cfg["latent_dim"]
        p2_cfg = p2_ckpt["config"]

        self.action_encoder = ActionEncoder(
            self.action_dim, latent_action_dim, p2_cfg["hidden_dim"]
        )
        self.action_encoder.load_state_dict(p2_ckpt["action_encoder_state_dict"])
        self.action_encoder.eval().requires_grad_(False)

        self.action_decoder = ActionDecoder(
            latent_action_dim, self.action_dim, p2_cfg["hidden_dim"]
        )
        self.action_decoder.load_state_dict(p2_ckpt["action_decoder_state_dict"])
        self.action_decoder.eval().requires_grad_(False)

        self.proprio_dec = ProprioDecoder(
            latent_dim, self.proprio_dim, p2_cfg["proprio_hidden_dim"]
        )
        self.proprio_dec.load_state_dict(p2_ckpt["proprio_dec_state_dict"])
        self.proprio_dec.eval().requires_grad_(False)

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
        """Run hierarchical rollout: coarse waypoints + fine fill-in.

        Predictor outputs are full token sequences (B, num_tokens, D),
        so the rollout preserves token structure across steps.

        Args:
            z_start:        (B, num_tokens, D)
            latent_actions: (B, H, latent_action_dim)

        Returns:
            z_final: (B, num_tokens, D)
        """
        h = self.horizon_h
        B, H, _ = latent_actions.shape

        num_coarse_steps = H // h
        remainder = H % h

        z_current = z_start
        z_waypoint = z_current  # default for remainder if no coarse step ran

        for i in range(num_coarse_steps):
            seg_start = i * h
            seg_end = seg_start + h

            # Coarse: waypoint over the segment (token output)
            a_block = latent_actions[:, seg_start:seg_end]  # (B, h, la_dim)
            z_waypoint = self.world_model.coarse_predictor(z_current, a_block)

            # Fine: chained rollout inside the segment, conditioned on waypoint
            z_t = z_current
            for step in range(h):
                t = seg_start + step
                z_t = self.world_model.fine_predictor(
                    z_t, latent_actions[:, t], z_waypoint
                )

            z_current = z_t

        # Remainder steps (< h) — condition on last waypoint
        if remainder > 0:
            for t in range(num_coarse_steps * h, H):
                z_current = self.world_model.fine_predictor(
                    z_current, latent_actions[:, t], z_waypoint
                )

        return z_current

    @torch.inference_mode()
    def get_cost(
        self,
        info_dict: dict,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate candidate action sequences for CEM.

        Args:
            info_dict:
                "pixels":       (B, S, history, C, H, W) — current observation
                "goal_proprio": (B, S, history, proprio_dim) — goal proprio
            candidates: (B, S, horizon, flat_dim) where flat_dim = action_block * action_dim

        Returns:
            costs: (B, S)
        """
        pixels = info_dict["pixels"][:, :, -1].to(self.device, dtype=torch.float32)

        if "goal_proprio" not in info_dict:
            raise KeyError(
                "HierarchicalCostModel.get_cost requires 'goal_proprio' in info_dict"
            )
        goal_proprio = info_dict["goal_proprio"][:, :, -1].to(
            self.device, dtype=torch.float32
        )

        batch_size, num_samples = pixels.shape[:2]

        flat_pixels = pixels.reshape(batch_size * num_samples, *pixels.shape[2:])
        z_start = self.world_model.encode(flat_pixels)  # (BxS, num_tokens, D)

        _, _, horizon, flat_dim = candidates.shape
        action_block = flat_dim // self.action_dim
        total_steps = horizon * action_block

        actions = candidates.view(
            batch_size, num_samples, horizon, action_block, self.action_dim
        ).to(self.device, dtype=torch.float32)

        BxS = batch_size * num_samples
        actions_flat = actions.reshape(BxS, total_steps, self.action_dim)

        # Real actions → latent actions via Phase 2 encoder
        latent_actions = self.action_encoder(
            actions_flat.reshape(BxS * total_steps, self.action_dim)
        ).reshape(BxS, total_steps, -1)

        z_final = self.hierarchical_rollout(z_start, latent_actions)  # (BxS, N_tok, D)

        proprio_pred = self.proprio_dec(z_final)             # (BxS, proprio_dim)
        goal_flat = goal_proprio.reshape(BxS, -1)             # (BxS, proprio_dim)
        cost = (proprio_pred - goal_flat).pow(2).sum(dim=-1)

        return cost.view(batch_size, num_samples)

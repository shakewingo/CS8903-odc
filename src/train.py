#!/usr/bin/env python3
"""Train a MaskablePPO agent for land-use allocation.

Action space: Discrete(H·W·N_MOD·N_MOD). Each action index decodes to
(r, c, src, tgt) via np.unravel_index — transfer pixel mass from class `src`
to class `tgt` at cell (r, c). The mask is a flat boolean over all tuples and
encodes per-cell validity (src present, tgt has room, src != tgt) plus the
optional riparian-buffer exclusion (see --riparian-mask).

Usage examples:
    # Default config
    python src/train.py

    # Quick smoke test
    python src/train.py --total-timesteps 10000 --max-steps 100

    # Long run with LR / ent_coef tail-decay and w_buf curriculum
    python src/train.py \\
        --total-timesteps 1500000 \\
        --learning-rate 5e-5 --lr-end 5e-6 --lr-decay-frac 0.33 \\
        --ent-coef 0.005 --ent-coef-end 0.001 --ent-coef-anneal-frac 0.33 \\
        --w-buf 5.0 --w-buf-start 1.0 --w-buf-anneal-frac 0.6 \\
        --min-init-total-value 1.0 --max-grad-norm 0.25 \\
        --run-name my_run

    # Disable wandb
    python src/train.py --no-wandb
"""

import argparse
import sys
from pathlib import Path
from datetime import date
from typing import Literal

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box, Discrete
from sb3_contrib import MaskablePPO
from scipy.signal import convolve2d
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Disable torch's strict Simplex validation; the 2500-dim softmax accumulates
# fp32 error that exceeds the default tolerance but is numerically fine.
torch.distributions.Distribution.set_default_validate_args(False)

# ── Project imports ─────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import (
    PROTECTED_CLASSES, ECO_VALUES, ET_VALUES,
    N_CLASSES, N_PIXELS_PER_CELL, SEED, data_dir, log_dir,
)
from src.utils import minmax_normalize, get_logger


# ── Parse CLI arguments ────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Train MaskablePPO (joint Discrete) for Lake Malawi land-use allocation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # PPO hyper-parameters
    ppo = p.add_argument_group("PPO")
    ppo.add_argument("--total-timesteps", type=int, default=500_000)
    ppo.add_argument("--n-steps", type=int, default=2048)
    ppo.add_argument("--batch-size", type=int, default=128)
    ppo.add_argument("--n-epochs", type=int, default=10)
    ppo.add_argument("--learning-rate", type=float, default=1e-4)
    ppo.add_argument("--lr-end", type=float, default=None,
                     help="If set, linearly decay LR from --learning-rate to this "
                          "value over the last --lr-decay-frac of training")
    ppo.add_argument("--lr-decay-frac", type=float, default=0.33,
                     help="Fraction of training (from the end) over which LR decays")
    ppo.add_argument("--gamma", type=float, default=0.99)
    ppo.add_argument("--gae-lambda", type=float, default=0.95)
    ppo.add_argument("--clip-range", type=float, default=0.2)
    ppo.add_argument("--ent-coef", type=float, default=0.02)
    ppo.add_argument("--ent-coef-end", type=float, default=None,
                     help="If set, linearly anneal ent_coef from --ent-coef to this "
                          "value over the last --ent-coef-anneal-frac of training")
    ppo.add_argument("--ent-coef-anneal-frac", type=float, default=0.33,
                     help="Fraction of training (from the end) over which ent_coef anneals")
    ppo.add_argument("--vf-coef", type=float, default=0.5)
    ppo.add_argument("--max-grad-norm", type=float, default=0.5)

    # Environment
    env_g = p.add_argument_group("Environment")
    env_g.add_argument("--max-steps", type=int, default=500,
                       help="Max steps per episode")
    env_g.add_argument("--n-augment", type=int, default=5,
                       help="Number of data augmentation rounds")
    env_g.add_argument("--pixels-per-transfer", type=int, default=5,
                       help="Pixels transferred per action")
    env_g.add_argument("--max-consecutive-noops", type=int, default=10)
    env_g.add_argument("--et-dcs-tolerance", type=float, default=1.0,
                       help="ET decrease tolerance for early stop")
    env_g.add_argument("--min-mod-frac", type=float, default=0.1,
                       help="Minimum modifiable fraction to accept a sample")
    env_g.add_argument("--min-init-total-value", type=float, default=0.0,
                       help="Curriculum filter — reject training samples whose initial "
                            "total_value is below this threshold (0 = disabled). "
                            "Set to 1.0 to drop trivial grids fully covered by "
                            "protected classes.")

    # Reward shaping — spatial rewards normalized by grid max contiguity
    rwd = p.add_argument_group("Reward")
    rwd.add_argument("--spatial-scale", type=float, default=1.0,
                     help="Overall spatial reward scale (0 = disable all spatial)")
    rwd.add_argument("--w-tree", type=float, default=1.0,
                     help="Priority weight for tree contiguity bonus (high)")
    rwd.add_argument("--w-crop", type=float, default=3.0,
                     help="Priority weight for crop contiguity bonus (medium)")
    rwd.add_argument("--w-built", type=float, default=3.0,
                     help="Priority weight for built-area contiguity bonus (medium)")
    rwd.add_argument("--w-buf", type=float, default=5.0,
                     help="Priority weight for water-buffer penalty (high) — final value")
    rwd.add_argument("--w-buf-start", type=float, default=None,
                     help="If set, linearly anneal w_buf from this value to --w-buf "
                          "over the first --w-buf-anneal-frac of training. Lets PPO "
                          "learn crop placement without buffer-paranoia early on.")
    rwd.add_argument("--w-buf-anneal-frac", type=float, default=0.6,
                     help="Fraction of training (from the start) over which w_buf anneals")
    rwd.add_argument("--w-rip", type=float, default=0.0,
                     help="Weight for riparian-trees bonus: +w_rip·log1p(Σ trees·water_nb). "
                          "Rewards placing trees at water-adjacent cells.")
    rwd.add_argument("--riparian-mask", action="store_true",
                     help="Hard-forbid crop/built targets at water-adjacent cells in "
                          "the action mask (structural riparian buffer).")
    rwd.add_argument("--lambda-et", type=float, default=1.0,
                     help="Weight for ET penalty (currently unused)")
    rwd.add_argument("--reward-scale", type=float, default=1.0)

    # Network architecture
    net = p.add_argument_group("Network")
    net.add_argument("--features-out", type=int, default=128,
                     help="Feature extractor output dim")
    net.add_argument("--conv1-out", type=int, default=32)
    net.add_argument("--conv2-out", type=int, default=64)
    net.add_argument("--policy-type", type=str, default="MlpPolicy")

    # Logging / WandB
    log_g = p.add_argument_group("Logging")
    log_g.add_argument("--no-wandb", action="store_true",
                       help="Disable Weights & Biases logging")
    log_g.add_argument("--wandb-project", type=str, default="CS8903-odc")
    log_g.add_argument("--run-name", type=str, default=None,
                       help="Custom W&B run name (auto-generated if omitted)")
    log_g.add_argument("--seed", type=int, default=SEED)
    log_g.add_argument("--log-level", type=str, default="DEBUG",
                       choices=["DEBUG", "INFO", "WARNING"])

    # ET override (set all to 0 by default, matching notebook)
    p.add_argument("--use-et", action="store_true",
                   help="Use real ET values from config instead of zeros")

    return p.parse_args()


# ── Data augmentation (unchanged from train.py) ────────────────────────
def augment_data(data, size, seed, split="train_indices", n_augment=10):
    if split == "no_split":
        indices = np.concatenate((data["train_indices"], data["test_indices"]), axis=0)
    else:
        indices = data[split]

    pixel_counts = data["pixel_counts"]
    samples = np.stack([
        pixel_counts[i[0]:i[0]+size, i[1]:i[1]+size]
        for i in indices
    ])
    rng = np.random.default_rng(seed)

    if n_augment > 0:
        n_rows, n_cols = pixel_counts.shape[:2]
        for _ in range(n_augment):
            augmented = []
            for i in indices:
                dr = rng.integers(-2, 3)
                dc = rng.integers(-2, 3)
                r0 = np.clip(i[0] + dr, 0, n_rows - size)
                c0 = np.clip(i[1] + dc, 0, n_cols - size)
                augmented.append(pixel_counts[r0:r0+size, c0:c0+size])
            samples = np.concatenate([samples, np.stack(augmented)])
    return samples


# ── Gym Environment (joint Discrete action space) ─────────────────────
MODIFIABLE_CLASSES = sorted(set(range(N_CLASSES)) - PROTECTED_CLASSES)
N_MOD = len(MODIFIABLE_CLASSES)
ECO_MOD = None  # set in main()
ET_MOD = None
_KERNEL = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
_EYE_K = np.eye(N_MOD, dtype=bool)  # precomputed (K, K) identity for src != tgt


class LandUseEnv(gym.Env):
    """RL environment for land-use allocation with joint Discrete(H·W·K²) actions.

    Each action decodes as (r, c, src, tgt) and transfers a fixed fraction of
    pixel mass from class `src` to class `tgt` at cell (r, c). Reward is the
    per-step change in `total_value = Σ_cells eco(cell) + spatial_scale · (
        w_tree·log1p(tree_cont) + w_crop·log1p(crop_cont) + w_built·log1p(built_cont)
        − w_buf·log1p(buf_pen) + w_rip·log1p(trees_riparian))`.
    """
    metadata = {"render_modes": []}

    def __init__(self, size=10, max_steps=500,
                 split: Literal["train_indices", "test_indices", "no_split"] = "train_indices",
                 spatial_scale=1.0, w_tree=3.0, w_crop=1.0, w_built=1.0, w_buf=2.0,
                 w_rip=0.0, riparian_mask=False,
                 lambda_et=1.0, reward_scale=1.0, n_augment=10,
                 et_dcs_tolerance=0.1, min_mod_frac=0.1,
                 pixels_per_transfer=5, max_consecutive_noops=10,
                 min_init_total_value=0.0):
        super().__init__()
        self.size = size
        self.max_steps = max_steps
        self.lambda_et = lambda_et
        self.reward_scale = reward_scale
        self.data = np.load(Path(data_dir, "processed", "rl_dataset.npz"))
        self.split = split
        self.initial_total_et = 0.0
        self.spatial_scale = spatial_scale
        self.w_tree = w_tree
        self.w_crop = w_crop
        self.w_built = w_built
        self.w_buf = w_buf
        self.w_rip = w_rip
        self.riparian_mask = riparian_mask
        self.n_augment = n_augment
        self.et_dcs_tolerance = et_dcs_tolerance
        self.min_mod_frac = min_mod_frac
        self.pixels_per_transfer = pixels_per_transfer
        self.max_consecutive_noops = max_consecutive_noops
        self.min_init_total_value = min_init_total_value

        self._mod_idx = {cls: i for i, cls in enumerate(MODIFIABLE_CLASSES)}

        self.observation_space = Box(
            low=0.0, high=1.0,
            shape=(size, size, N_MOD),
            dtype=np.float32,
        )
        # Single joint action — every legal (r, c, src, tgt) combination
        self.action_dim = size * size * N_MOD * N_MOD
        self.action_space = Discrete(self.action_dim)

        self.state = None
        self.step_count = 0
        self.consecutive_noops = 0
        self.prev_total_value = 0.0

    # ── masks  ──────────────────────────────────────────
    def action_masks(self) -> np.ndarray:
        """Return the joint boolean mask (shape H·W·K·K,).

        (r, c, src, tgt) is legal iff:
            state[r, c, src] > 0    and    state[r, c, tgt] < 1    and    src != tgt

        With `riparian_mask=True`, additionally forbids tgt ∈ {crops, built} at
        water-adjacent cells. Target-only: moving crops/built OUT of water-adj
        cells (restoration) remains legal.

        Falls back to forcing action 0 legal if no action is valid — lets
        MaskableCategorical form a distribution and the env terminates via the
        consecutive-noop counter.
        """
        state = self.state
        src_avail = state > 0.0           # (H, W, K)
        tgt_room = state < 1.0            # (H, W, K)
        # (H, W, K_src, K_tgt)
        mask = src_avail[:, :, :, None] & tgt_room[:, :, None, :]
        mask &= ~_EYE_K[None, None, :, :]  # forbid src == tgt

        if self.riparian_mask:
            water_adj = self._water_nb > 0.0                # (H, W)
            high_impact_tgt = np.zeros(N_MOD, dtype=bool)
            high_impact_tgt[self._mod_idx[5]] = True        # crops
            high_impact_tgt[self._mod_idx[7]] = True        # built
            forbidden = water_adj[:, :, None] & high_impact_tgt[None, None, :]  # (H, W, K_tgt)
            mask &= ~forbidden[:, :, None, :]               # broadcast over K_src

        flat_mask = mask.reshape(-1)
        if not flat_mask.any():
            flat_mask = flat_mask.copy()
            flat_mask[0] = True
        return flat_mask

    # ── reward helpers (unchanged from train.py) ────────────────────
    def _compute_total_value(self):
        net_values = self.state * (ECO_MOD + ET_MOD)
        tree_n = crop_n = built_n = buf_n = rip_n = 0.0
        spatial_value = 0.0

        if self.spatial_scale > 0:
            tree_n, crop_n, built_n, buf_n, rip_n = self._compute_spatial_normalized()
            spatial_value = self.spatial_scale * (
                + self.w_tree  * tree_n
                + self.w_crop  * crop_n
                + self.w_built * built_n
                - self.w_buf   * buf_n
                + self.w_rip   * rip_n
            )

        total_values = float(net_values.sum()) + spatial_value
        return total_values, spatial_value, tree_n, crop_n, built_n, buf_n, rip_n

    def _compute_total_et(self) -> float:
        if ET_MOD is None or not ET_MOD.any():
            return 0.0
        return float((self.state.reshape(-1, N_MOD) @ ET_MOD).sum())

    def _compute_spatial_normalized(self):
        fractions = self.state

        trees = fractions[:, :, self._mod_idx[2]]
        tree_cont = float(np.sum(trees * convolve2d(trees, _KERNEL, mode="same", boundary="fill", fillvalue=0)))

        crops = fractions[:, :, self._mod_idx[5]]
        crop_cont = float(np.sum(crops * convolve2d(crops, _KERNEL, mode="same", boundary="fill", fillvalue=0)))

        built = fractions[:, :, self._mod_idx[7]]
        built_cont = float(np.sum(built * convolve2d(built, _KERNEL, mode="same", boundary="fill", fillvalue=0)))

        high_impact = crops + built
        water_nb = self._water_nb
        buffer_pen = float(np.sum(high_impact * water_nb))
        trees_riparian = float(np.sum(trees * water_nb))

        return (np.log1p(tree_cont), np.log1p(crop_cont),
                np.log1p(built_cont), np.log1p(buffer_pen),
                np.log1p(trees_riparian))

    # ── observation (unchanged) ────────────────────────────────────
    def _get_obs(self, idx) -> np.ndarray:
        """Load sample `idx` into env state and cache the water-neighbourhood
        indicator (reused by the buf_pen/rip reward terms and the mask)."""
        raw = self.samples[idx]
        mod_counts = raw[:, :, MODIFIABLE_CLASSES]
        self.state = mod_counts.astype(np.float32) / N_PIXELS_PER_CELL
        self._protected_water = (raw[:, :, 1] + raw[:, :, 4]).astype(np.float32) / N_PIXELS_PER_CELL
        self._water_nb = convolve2d(self._protected_water, _KERNEL,
                                    mode="same", boundary="fill", fillvalue=0)
        return self.state

    # ── action (decode flat index) ─────────────────────────────────
    def _apply_action(self, action):
        a = int(action)
        r, c, source_idx, target_idx = np.unravel_index(
            a, (self.size, self.size, N_MOD, N_MOD)
        )
        logger.debug(f"Attempted action a={a}: ({r}, {c}) transfer from class "
                     f"{MODIFIABLE_CLASSES[source_idx]} to {MODIFIABLE_CLASSES[target_idx]}")

        # Defensive check for unmasked callers; mask normally guarantees src != tgt.
        if source_idx == target_idx:
            logger.debug(f"No-op: source_idx {source_idx} == target_idx {target_idx}")
            return False

        cell = self.state[r, c, :]
        transfer = self.pixels_per_transfer / N_PIXELS_PER_CELL
        actual = min(transfer, float(cell[source_idx]), 1.0 - float(cell[target_idx]))
        if actual <= 0:
            logger.debug(f"No-op: actual transfer {actual} <= 0")
            return False

        cell[source_idx] -= actual
        cell[target_idx] += actual
        logger.debug(f"Applied: ({r}, {c}) transfer {actual:.4f} from "
                     f"{MODIFIABLE_CLASSES[source_idx]} to {MODIFIABLE_CLASSES[target_idx]}")
        return True

    def _simulate_delta(self, action) -> float:
        """Return Δ total_value for a hypothetical action without mutating env
        counters. Used by the greedy baseline to rank candidate actions.

        Mutates and reverts `self.state` in-place; not safe under concurrent
        step() calls.
        """
        a = int(action)
        r, c, src, tgt = np.unravel_index(a, (self.size, self.size, N_MOD, N_MOD))
        if src == tgt:
            return 0.0
        orig_src = float(self.state[r, c, src])
        orig_tgt = float(self.state[r, c, tgt])
        actual = min(self.pixels_per_transfer / N_PIXELS_PER_CELL,
                     orig_src, 1.0 - orig_tgt)
        if actual <= 0:
            return 0.0
        self.state[r, c, src] = orig_src - actual
        self.state[r, c, tgt] = orig_tgt + actual
        new_total, *_ = self._compute_total_value()
        self.state[r, c, src] = orig_src
        self.state[r, c, tgt] = orig_tgt
        return new_total - self.prev_total_value

    def get_state_dict(self) -> dict:
        """Snapshot of mutable env state (everything step() can change).

        Used by GA to restore state between chromosome evaluations. `samples`
        and `data` are shared across episodes and are intentionally excluded.
        """
        return {
            "state": self.state.copy(),
            "_protected_water": self._protected_water.copy(),
            "step_count": self.step_count,
            "consecutive_noops": self.consecutive_noops,
            "prev_total_value": self.prev_total_value,
            "prev_spatial_value": self.prev_spatial_value,
            "prev_spatial": self.prev_spatial,
            "initial_total_et": self.initial_total_et,
            "max_consecutive_noops": self.max_consecutive_noops,
        }

    def set_state_dict(self, snapshot: dict) -> None:
        self.state = snapshot["state"].copy()
        self._protected_water = snapshot["_protected_water"].copy()
        self.step_count = snapshot["step_count"]
        self.consecutive_noops = snapshot["consecutive_noops"]
        self.prev_total_value = snapshot["prev_total_value"]
        self.prev_spatial_value = snapshot["prev_spatial_value"]
        self.prev_spatial = snapshot["prev_spatial"]
        self.initial_total_et = snapshot["initial_total_et"]
        self.max_consecutive_noops = snapshot["max_consecutive_noops"]

    # ── reset / step ────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.samples = augment_data(self.data, size=self.size, seed=SEED,
                                    split=self.split, n_augment=self.n_augment)
        n_cells = self.size * self.size
        tv = None
        for _ in range(100):
            idx = self.np_random.integers(self.samples.shape[0])
            self.state = self._get_obs(idx).copy()
            mod_frac = np.count_nonzero(self.state.sum(axis=2)) / n_cells
            if mod_frac < self.min_mod_frac:
                continue
            tv = self._compute_total_value()
            if tv[0] >= self.min_init_total_value:
                break

        self.step_count = 0
        self.consecutive_noops = 0
        if tv is None:
            tv = self._compute_total_value()
        self.prev_total_value = tv[0]
        self.prev_spatial_value = tv[1]
        self.prev_spatial = (tv[2], tv[3], tv[4], tv[5], tv[6])
        self.initial_total_et = self._compute_total_et()
        info = {"total_value": self.prev_total_value, "et_decrease": 0.0, "step": self.step_count}
        logger.info(f"Reset env: mod_frac={mod_frac:.2f}, {info}")
        return self.state, info

    def step(self, action):
        logger.debug(f"Step {self.step_count}")
        applied = self._apply_action(action)

        if not applied:
            self.consecutive_noops += 1
        else:
            self.consecutive_noops = 0

        (cur_total_value, spatial_value,
         tree_cont, crop_cont, built_cont, buffer_pen,
         trees_riparian) = self._compute_total_value()
        reward = cur_total_value - self.prev_total_value
        self.prev_total_value = cur_total_value

        tree_cont_rwd = tree_cont - self.prev_spatial[0]
        crop_cont_rwd = crop_cont - self.prev_spatial[1]
        built_cont_rwd = built_cont - self.prev_spatial[2]
        buffer_pen_rwd = buffer_pen - self.prev_spatial[3]
        trees_riparian_rwd = trees_riparian - self.prev_spatial[4]
        spatial_reward = spatial_value - self.prev_spatial_value
        self.prev_spatial = (tree_cont, crop_cont, built_cont, buffer_pen, trees_riparian)
        self.prev_spatial_value = spatial_value

        self.step_count += 1

        et_decrease = (self.initial_total_et - self._compute_total_et()) / (self.initial_total_et + 1e-9)
        reward /= self.reward_scale

        # Safety: if no valid action remains in the new state, end cleanly.
        no_actions_left = not self.action_masks().any()

        termination_reason = None
        if no_actions_left:
            termination_reason = "no_actions_left"
        elif self.consecutive_noops >= self.max_consecutive_noops:
            termination_reason = "consecutive_noops"
        elif et_decrease >= self.et_dcs_tolerance:
            termination_reason = "et_decrease"
        elif self.step_count >= self.max_steps:
            termination_reason = "max_steps"
        terminated = termination_reason is not None
        truncated = False
        info = {
            "total_value": cur_total_value,
            "et_decrease": et_decrease,
            "step": self.step_count,
            "tree_contiguity": tree_cont,
            "crop_contiguity": crop_cont,
            "built_contiguity": built_cont,
            "buffer_penalty": buffer_pen,
            "trees_riparian": trees_riparian,
            "spatial_value": spatial_value,
            "spatial_reward": spatial_reward,
            "tree_cont_reward": tree_cont_rwd,
            "crop_cont_reward": crop_cont_rwd,
            "built_cont_reward": built_cont_rwd,
            "buffer_pen_reward": buffer_pen_rwd,
            "trees_riparian_reward": trees_riparian_rwd,
            "no_actions_left": bool(no_actions_left),
            "termination_reason": termination_reason,
        }
        if no_actions_left:
            logger.info(f"Early termination: no valid actions left at step {self.step_count}")
        if self.consecutive_noops >= self.max_consecutive_noops:
            logger.info(f"Early termination: {self.max_consecutive_noops} consecutive no-ops "
                        f"at step {self.step_count}")
        logger.info(f"Step {self.step_count} result: reward={reward:.4f}, "
                    f"terminated={terminated}, info={info}")
        return self.state, reward, terminated, truncated, info


# ── CNN Feature Extractor (unchanged) ──────────────────────────────────
class GridCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, conv1_features_out=32,
                 conv2_features_out=64, features_out=128):
        super().__init__(observation_space, features_out)
        n_channels = observation_space.shape[-1]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, conv1_features_out, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(conv1_features_out, conv2_features_out, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels,
                                observation_space.shape[0], observation_space.shape[1])
            n_flatten = self.cnn(dummy).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_out),
            nn.ReLU(),
        )

    def forward(self, observations):
        x = observations.permute(0, 3, 1, 2).float()
        return self.linear(self.cnn(x))


# ── WandB Callback (unchanged) ─────────────────────────────────────────
class SpatialMetricsCallback(BaseCallback):
    SPATIAL_KEYS = ("tree_contiguity", "crop_contiguity", "built_contiguity",
                    "buffer_penalty", "trees_riparian",
                    "spatial_value", "spatial_reward",
                    "tree_cont_reward", "crop_cont_reward",
                    "built_cont_reward", "buffer_pen_reward",
                    "trees_riparian_reward", "et_decrease")

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._buffers = {k: [] for k in self.SPATIAL_KEYS}

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            for k in self.SPATIAL_KEYS:
                if k in info:
                    self._buffers[k].append(info[k])
        return True

    def _on_rollout_end(self) -> None:
        for k, buf in self._buffers.items():
            if buf:
                self.logger.record(f"spatial/{k}_mean", np.mean(buf))
                buf.clear()


# ── Schedules / annealing callbacks ────────────────────────────────────
def linear_schedule_tail(initial_value: float, final_value: float,
                         decay_frac: float = 0.33):
    """Build a SB3 `progress_remaining -> value` schedule that holds
    `initial_value` for `1 - decay_frac` of training, then linearly decays to
    `final_value` over the tail `decay_frac`.
    """
    decay_frac = max(1e-6, min(1.0, decay_frac))

    def schedule(progress_remaining: float) -> float:
        progress_done = 1.0 - progress_remaining
        decay_start = 1.0 - decay_frac
        if progress_done <= decay_start:
            return initial_value
        frac = (progress_done - decay_start) / decay_frac
        return initial_value + (final_value - initial_value) * frac

    return schedule


class EntCoefAnnealCallback(BaseCallback):
    """Linearly anneal `model.ent_coef` from `start` to `end` over the last
    `anneal_frac` of training; held at `start` before the tail."""

    def __init__(self, total_timesteps: int, start: float, end: float,
                 anneal_frac: float = 0.33, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.start = start
        self.end = end
        self.anneal_frac = max(1e-6, min(1.0, anneal_frac))
        self.anneal_start_step = total_timesteps * (1.0 - self.anneal_frac)

    def _on_step(self) -> bool:
        if self.num_timesteps < self.anneal_start_step:
            new_val = self.start
        else:
            frac = (self.num_timesteps - self.anneal_start_step) / (
                self.total_timesteps - self.anneal_start_step + 1e-9
            )
            frac = max(0.0, min(1.0, frac))
            new_val = self.start + (self.end - self.start) * frac
        self.model.ent_coef = float(new_val)
        return True

    def _on_rollout_end(self) -> None:
        self.logger.record("train/ent_coef", float(self.model.ent_coef))


class WBufAnnealCallback(BaseCallback):
    """Linearly anneal env.w_buf from `start` to `end` over the first
    `anneal_frac` of training, then hold at `end`. Writes via VecEnv.set_attr
    so all wrapped envs stay in sync."""

    def __init__(self, total_timesteps: int, start: float, end: float,
                 anneal_frac: float = 0.6, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.start = start
        self.end = end
        self.anneal_frac = max(1e-6, min(1.0, anneal_frac))
        self.anneal_end_step = total_timesteps * self.anneal_frac

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.anneal_end_step:
            new_val = self.end
        else:
            frac = self.num_timesteps / (self.anneal_end_step + 1e-9)
            frac = max(0.0, min(1.0, frac))
            new_val = self.start + (self.end - self.start) * frac
        self.training_env.set_attr("w_buf", float(new_val))
        return True

    def _on_rollout_end(self) -> None:
        current = self.training_env.get_attr("w_buf")[0]
        self.logger.record("train/w_buf", float(current))


def build_value_vecs(
    eco_variant: Literal["standard", "regen"] = "standard",
    use_et: bool = False,
    assign_globals: bool = True,
):
    """Build normalised per-class eco and ET value vectors.

    `eco_variant="regen"` multiplies the crops eco value by 1.35 before
    normalisation. When `assign_globals=True`, writes module-level `ECO_MOD`
    / `ET_MOD` so env instances read consistent values.
    """
    eco_values = dict(ECO_VALUES)
    if eco_variant == "regen":
        eco_values[5] = ECO_VALUES[5] * 1.35
    et_values = ET_VALUES if use_et else {k: 0 for k in ET_VALUES}

    norm_eco = minmax_normalize(eco_values)
    norm_et = minmax_normalize(et_values)

    eco_per_class = np.zeros(N_CLASSES, dtype=np.float32)
    et_per_class = np.zeros(N_CLASSES, dtype=np.float32)
    for cls, val in norm_eco.items():
        if cls < N_CLASSES:
            eco_per_class[cls] = val
    for cls, val in norm_et.items():
        if cls < N_CLASSES:
            et_per_class[cls] = val

    if assign_globals:
        global ECO_MOD, ET_MOD
        ECO_MOD = eco_per_class[MODIFIABLE_CLASSES]
        ET_MOD = et_per_class[MODIFIABLE_CLASSES]

    return eco_per_class, et_per_class


# ── Main ───────────────────────────────────────────────────────────────
def main():
    global logger

    args = parse_args()

    logger = get_logger(
        "train",
        log_file=Path(log_dir, f"{date.today():%m%d}-joint.log"),
        stream=False,
        level=args.log_level,
    )

    # ── Value vectors ───────────────────────────────────────────────
    eco_per_class, et_per_class = build_value_vecs(
        eco_variant="standard", use_et=args.use_et, assign_globals=True,
    )

    print(f"Modifiable classes: {MODIFIABLE_CLASSES}  (n={N_MOD})")
    print(f"Joint action space: Discrete({10*10*N_MOD*N_MOD})")
    print(f"ECO_MOD: {ECO_MOD}")
    print(f"ET_MOD:  {ET_MOD}")

    # ── Build config dict (for wandb) ──────────────────────────────
    config = vars(args).copy()
    config["env_name"] = "LakeMalawiLandUse-joint"
    config["features_extractor"] = "GridCNN"
    config["action_space"] = f"Discrete({10*10*N_MOD*N_MOD})"

    # ── WandB ──────────────────────────────────────────────────────
    run_id = "local"
    if not args.no_wandb:
        import wandb
        from wandb.integration.sb3 import WandbCallback as _WandbCB
        run = wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=config,
            sync_tensorboard=True,
            save_code=True,
        )
        run_id = run.id

    # ── Environment ────────────────────────────────────────────────
    # If w_buf is being annealed, start at args.w_buf_start and let the
    # callback ramp it up to args.w_buf over the first --w-buf-anneal-frac.
    initial_w_buf = args.w_buf_start if args.w_buf_start is not None else args.w_buf
    env = LandUseEnv(
        split="train_indices",
        max_steps=args.max_steps,
        spatial_scale=args.spatial_scale,
        w_tree=args.w_tree,
        w_crop=args.w_crop,
        w_built=args.w_built,
        w_buf=initial_w_buf,
        w_rip=args.w_rip,
        riparian_mask=args.riparian_mask,
        lambda_et=args.lambda_et,
        reward_scale=args.reward_scale,
        n_augment=args.n_augment,
        et_dcs_tolerance=args.et_dcs_tolerance,
        min_mod_frac=args.min_mod_frac,
        pixels_per_transfer=args.pixels_per_transfer,
        max_consecutive_noops=args.max_consecutive_noops,
        min_init_total_value=args.min_init_total_value,
    )

    # ── Model ──────────────────────────────────────────────────────
    policy_kwargs = dict(
        features_extractor_class=GridCNN,
        features_extractor_kwargs=dict(
            features_out=args.features_out,
            conv1_features_out=args.conv1_out,
            conv2_features_out=args.conv2_out,
        ),
        share_features_extractor=True,
    )

    # LR: constant unless --lr-end is provided → linear tail decay.
    if args.lr_end is not None:
        lr_arg = linear_schedule_tail(args.learning_rate, args.lr_end,
                                      decay_frac=args.lr_decay_frac)
        print(f"[schedule] LR decays {args.learning_rate:g} -> {args.lr_end:g} "
              f"over last {args.lr_decay_frac:.0%} of training")
    else:
        lr_arg = args.learning_rate

    model = MaskablePPO(
        args.policy_type,
        env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        learning_rate=lr_arg,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        tensorboard_log=f"runs/{run_id}",
    )
    print(model.policy)

    # ── Callbacks ──────────────────────────────────────────────────
    callbacks = [SpatialMetricsCallback()]
    if args.ent_coef_end is not None:
        callbacks.append(EntCoefAnnealCallback(
            total_timesteps=args.total_timesteps,
            start=args.ent_coef,
            end=args.ent_coef_end,
            anneal_frac=args.ent_coef_anneal_frac,
        ))
        print(f"[schedule] ent_coef anneals {args.ent_coef:g} -> "
              f"{args.ent_coef_end:g} over last {args.ent_coef_anneal_frac:.0%}")
    if args.w_buf_start is not None:
        callbacks.append(WBufAnnealCallback(
            total_timesteps=args.total_timesteps,
            start=args.w_buf_start,
            end=args.w_buf,
            anneal_frac=args.w_buf_anneal_frac,
        ))
        print(f"[schedule] w_buf anneals {args.w_buf_start:g} -> "
              f"{args.w_buf:g} over first {args.w_buf_anneal_frac:.0%}")
    if not args.no_wandb:
        callbacks.append(_WandbCB(
            gradient_save_freq=100,
            model_save_path=f"models/{run_id}",
            verbose=2,
        ))

    # ── Train ──────────────────────────────────────────────────────
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks)

    if not args.no_wandb:
        run.finish()

    print("Training complete.")


if __name__ == "__main__":
    main()
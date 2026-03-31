#!/usr/bin/env python3
"""
Train a MaskablePPO agent for land-use allocation optimization.

Usage examples:
    # Default config (all spatial rewards active)
    python src/train.py

    # Quick test run
    python src/train.py --total-timesteps 10000 --max-steps 100

    # Adjust spatial vs eco balance
    python src/train.py --spatial-scale 0.2

    # Tree contiguity only (ablation)
    python src/train.py --w-tree 3.0 --w-crop 0 --w-built 0 --w-buf 0

    # Disable all spatial rewards
    python src/train.py --spatial-scale 0

    # Larger CNN
    python src/train.py --features-out 256 --conv1-out 64 --conv2-out 128

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
from gymnasium.spaces import Box, MultiDiscrete
from sb3_contrib import MaskablePPO
from scipy.signal import convolve2d
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

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
        description="Train MaskablePPO for Lake Malawi land-use allocation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # PPO hyper-parameters
    ppo = p.add_argument_group("PPO")
    ppo.add_argument("--total-timesteps", type=int, default=500_000)
    ppo.add_argument("--n-steps", type=int, default=2048)
    ppo.add_argument("--batch-size", type=int, default=128)
    ppo.add_argument("--n-epochs", type=int, default=10)
    ppo.add_argument("--learning-rate", type=float, default=1e-4)
    ppo.add_argument("--gamma", type=float, default=0.99)
    ppo.add_argument("--gae-lambda", type=float, default=0.95)
    ppo.add_argument("--clip-range", type=float, default=0.2)
    ppo.add_argument("--ent-coef", type=float, default=0.02)
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
                     help="Priority weight for water-buffer penalty (high)")
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


# ── Data augmentation ──────────────────────────────────────────────────
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


# ── Gym Environment ────────────────────────────────────────────────────
# Module-level arrays filled in main() after args are parsed
MODIFIABLE_CLASSES = sorted(set(range(N_CLASSES)) - PROTECTED_CLASSES)
N_MOD = len(MODIFIABLE_CLASSES)
ECO_MOD = None  # set in main()
ET_MOD = None
_KERNEL = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])


class LandUseEnv(gym.Env):
    """RL environment for land-use allocation optimization."""
    metadata = {"render_modes": []}

    def __init__(self, size=10, max_steps=500,
                 split: Literal["train_indices", "test_indices", "no_split"] = "train_indices",
                 spatial_scale=1.0, w_tree=3.0, w_crop=1.0, w_built=1.0, w_buf=2.0,
                 lambda_et=1.0, reward_scale=1.0, n_augment=10,
                 et_dcs_tolerance=0.1, min_mod_frac=0.1,
                 pixels_per_transfer=5, max_consecutive_noops=10):
        super().__init__()
        self.size = size
        self.max_steps = max_steps
        self.lambda_et = lambda_et
        self.reward_scale = reward_scale
        self.data = np.load(Path(data_dir, "processed", "rl_dataset.npz"))
        self.split = split
        self.initial_total_et = 0.0
        self.spatial_scale = spatial_scale
        self.w_tree = w_tree # priority level: 10, 1, 0.1
        self.w_crop = w_crop
        self.w_built = w_built
        self.w_buf = w_buf
        self.n_augment = n_augment
        self.et_dcs_tolerance = et_dcs_tolerance
        self.min_mod_frac = min_mod_frac
        self.pixels_per_transfer = pixels_per_transfer
        self.max_consecutive_noops = max_consecutive_noops

        self._mod_idx = {cls: i for i, cls in enumerate(MODIFIABLE_CLASSES)}

        self.observation_space = Box(
            low=0.0, high=1.0,
            shape=(size, size, N_MOD),
            dtype=np.float32,
        )
        self.action_space = MultiDiscrete([size, size, N_MOD, N_MOD])

        self.state = None
        self.step_count = 0
        self.consecutive_noops = 0
        self.prev_total_value = 0.0

    # ── masks ───────────────────────────────────────────────────────
    def action_masks(self) -> np.ndarray:
        row_mask = np.ones(self.size, dtype=bool)
        col_mask = np.ones(self.size, dtype=bool)
        source_mask = np.array([np.any(self.state[:, :, k] > 0) for k in range(N_MOD)])
        target_mask = np.array([np.any(self.state[:, :, k] < 1.0) for k in range(N_MOD)])
        if not source_mask.any():
            source_mask[:] = True
        if not target_mask.any():
            target_mask[:] = True
        return np.concatenate([row_mask, col_mask, source_mask, target_mask])

    # ── reward helpers ──────────────────────────────────────────────
    def _compute_total_value(self):
        net_values = self.state * (ECO_MOD + ET_MOD)
        tree_n = crop_n = built_n = buf_n = 0.0
        spatial_value = 0.0

        if self.spatial_scale > 0:
            tree_n, crop_n, built_n, buf_n = self._compute_spatial_normalized()
            spatial_value = self.spatial_scale * (
                + self.w_tree  * tree_n
                + self.w_crop  * crop_n
                + self.w_built * built_n
                - self.w_buf   * buf_n
            )

        total_values = float(net_values.sum()) + spatial_value
        return total_values, spatial_value, tree_n, crop_n, built_n, buf_n

    def _compute_total_et(self) -> float:
        return float((self.state.reshape(-1, N_MOD) @ ET_MOD).sum())

    def _compute_spatial_normalized(self):
        """Return 4 log1p-transformed spatial metrics."""
        fractions = self.state

        trees = fractions[:, :, self._mod_idx[2]]
        tree_cont = float(np.sum(trees * convolve2d(trees, _KERNEL, mode="same", boundary="fill", fillvalue=0)))

        crops = fractions[:, :, self._mod_idx[5]]
        crop_cont = float(np.sum(crops * convolve2d(crops, _KERNEL, mode="same", boundary="fill", fillvalue=0)))

        built = fractions[:, :, self._mod_idx[7]]
        built_cont = float(np.sum(built * convolve2d(built, _KERNEL, mode="same", boundary="fill", fillvalue=0)))

        water = self._protected_water
        high_impact = crops + built
        water_nb = convolve2d(water, _KERNEL, mode="same", boundary="fill", fillvalue=0)
        buffer_pen = float(np.sum(high_impact * water_nb))

        return (np.log1p(tree_cont), np.log1p(crop_cont),
                np.log1p(built_cont), np.log1p(buffer_pen))

    # ── observation ─────────────────────────────────────────────────
    def _get_obs(self, idx) -> np.ndarray:
        raw = self.samples[idx]
        mod_counts = raw[:, :, MODIFIABLE_CLASSES]
        self.state = mod_counts.astype(np.float32) / N_PIXELS_PER_CELL
        self._protected_water = (raw[:, :, 1] + raw[:, :, 4]).astype(np.float32) / N_PIXELS_PER_CELL
        return self.state

    # ── action ──────────────────────────────────────────────────────
    def _apply_action(self, action):
        r, c, source_idx, target_idx = action
        logger.debug(f"Attempted action: ({r}, {c}) transfer from class "
                     f"{MODIFIABLE_CLASSES[source_idx]} to {MODIFIABLE_CLASSES[target_idx]}")
        if source_idx == target_idx:
            logger.debug(f"No-op action due to: source_idx {source_idx} == target_idx {target_idx}")
            return False

        cell = self.state[r, c, :]
        transfer = self.pixels_per_transfer / N_PIXELS_PER_CELL
        actual = min(transfer, float(cell[source_idx]), 1.0 - float(cell[target_idx]))
        if actual <= 0:
            logger.debug(f"No-op action due to: actual transfer {actual} <= 0")
            return False

        cell[source_idx] -= actual
        cell[target_idx] += actual
        logger.debug(f"Applied action: ({r}, {c}) transfer {actual:.4f} from class "
                     f"{MODIFIABLE_CLASSES[source_idx]} to {MODIFIABLE_CLASSES[target_idx]}")
        return True

    # ── reset / step ────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.samples = augment_data(self.data, size=self.size, seed=SEED,
                                    split=self.split, n_augment=self.n_augment)
        n_cells = self.size * self.size
        for _ in range(100):
            idx = self.np_random.integers(self.samples.shape[0] - 1)
            self.state = self._get_obs(idx).copy()
            mod_frac = np.count_nonzero(self.state.sum(axis=2)) / n_cells
            if mod_frac >= self.min_mod_frac:
                break

        self.step_count = 0
        self.consecutive_noops = 0
        tv = self._compute_total_value()
        self.prev_total_value = tv[0]
        self.prev_spatial_value = tv[1]
        self.prev_spatial = (tv[2], tv[3], tv[4], tv[5])
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
         tree_cont, crop_cont, built_cont, buffer_pen) = self._compute_total_value()
        reward = cur_total_value - self.prev_total_value
        self.prev_total_value = cur_total_value

        # Per-metric deltas
        tree_cont_rwd = tree_cont - self.prev_spatial[0]
        crop_cont_rwd = crop_cont - self.prev_spatial[1]
        built_cont_rwd = built_cont - self.prev_spatial[2]
        buffer_pen_rwd = buffer_pen - self.prev_spatial[3]
        spatial_reward = spatial_value - self.prev_spatial_value
        self.prev_spatial = (tree_cont, crop_cont, built_cont, buffer_pen)
        self.prev_spatial_value = spatial_value

        self.step_count += 1

        et_decrease = (self.initial_total_et - self._compute_total_et()) / (self.initial_total_et + 1e-9)
        reward /= self.reward_scale

        terminated = (self.step_count >= self.max_steps
                      or et_decrease >= self.et_dcs_tolerance
                      or self.consecutive_noops >= self.max_consecutive_noops)
        truncated = False
        info = {
            "total_value": cur_total_value,
            "et_decrease": et_decrease,
            "step": self.step_count,
            "tree_contiguity": tree_cont,
            "crop_contiguity": crop_cont,
            "built_contiguity": built_cont,
            "buffer_penalty": buffer_pen,
            "spatial_value": spatial_value,
            "spatial_reward": spatial_reward,
            "tree_cont_reward": tree_cont_rwd,
            "crop_cont_reward": crop_cont_rwd,
            "built_cont_reward": built_cont_rwd,
            "buffer_pen_reward": buffer_pen_rwd,
        }
        if self.consecutive_noops >= self.max_consecutive_noops:
            logger.info(f"Early termination: {self.max_consecutive_noops} consecutive no-ops "
                        f"at step {self.step_count}")
        logger.info(f"Step {self.step_count} result: reward={reward:.4f}, "
                    f"terminated={terminated}, info={info}")
        return self.state, reward, terminated, truncated, info


# ── CNN Feature Extractor ──────────────────────────────────────────────
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


# ── WandB Callback ─────────────────────────────────────────────────────
class SpatialMetricsCallback(BaseCallback):
    SPATIAL_KEYS = ("tree_contiguity", "crop_contiguity", "built_contiguity",
                    "buffer_penalty", "spatial_value", "spatial_reward",
                    "tree_cont_reward", "crop_cont_reward",
                    "built_cont_reward", "buffer_pen_reward", "et_decrease")

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


# ── Main ───────────────────────────────────────────────────────────────
def main():
    global ECO_MOD, ET_MOD, logger

    args = parse_args()

    logger = get_logger(
        "train_v2",
        log_file=Path(log_dir, f"{date.today():%m%d}-v2.log"),
        stream=False,
        level=args.log_level,
    )

    # ── Value vectors ───────────────────────────────────────────────
    if args.use_et:
        et_values = ET_VALUES
    else:
        et_values = {k: 0 for k in ET_VALUES}

    norm_et = minmax_normalize(et_values)
    norm_eco = minmax_normalize(ECO_VALUES)

    eco_per_class = np.zeros(N_CLASSES, dtype=np.float32)
    et_per_class = np.zeros(N_CLASSES, dtype=np.float32)
    for cls, val in norm_eco.items():
        if cls < N_CLASSES:
            eco_per_class[cls] = val
    for cls, val in norm_et.items():
        if cls < N_CLASSES:
            et_per_class[cls] = val

    ECO_MOD = eco_per_class[MODIFIABLE_CLASSES]
    ET_MOD = et_per_class[MODIFIABLE_CLASSES]

    print(f"Modifiable classes: {MODIFIABLE_CLASSES}  (n={N_MOD})")
    print(f"ECO_MOD: {ECO_MOD}")
    print(f"ET_MOD:  {ET_MOD}")

    # ── Build config dict (for wandb) ──────────────────────────────
    config = vars(args).copy()
    config["env_name"] = "LakeMalawiLandUse-v0"
    config["features_extractor"] = "GridCNN"

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
    env = LandUseEnv(
        split="train_indices",
        max_steps=args.max_steps,
        spatial_scale=args.spatial_scale,
        w_tree=args.w_tree,
        w_crop=args.w_crop,
        w_built=args.w_built,
        w_buf=args.w_buf,
        lambda_et=args.lambda_et,
        reward_scale=args.reward_scale,
        n_augment=args.n_augment,
        et_dcs_tolerance=args.et_dcs_tolerance,
        min_mod_frac=args.min_mod_frac,
        pixels_per_transfer=args.pixels_per_transfer,
        max_consecutive_noops=args.max_consecutive_noops,
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

    model = MaskablePPO(
        args.policy_type,
        env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        learning_rate=args.learning_rate,
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

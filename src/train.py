"""train.py – PPO-based RL training for land-use optimisation.

Environment
-----------
Each episode samples one 10×10 cell window from the dataset.

State  : the current cell's land-cover fractions  →  (12,).
Action : Box(N_FREE,) logits  →  softmax  →  quantised to integer pixel
         counts (/25) for the free (non-protected) land-cover classes.
Reward : total_grid_reward_t − total_grid_reward_{t−1}.
Done   : all cells traversed  OR  total ET drops % > threshold.
"""

import argparse
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from config import *
from utils import get_logger

logger = get_logger(__name__)

# Params Setup
N_PIXELS  = GRID_KWARGS['grid_size'] ** 2
FREE_CLASSES = sorted(set(range(N_CLASSES)) - PROTECTED_CLASSES)
N_FREE = len(FREE_CLASSES)

ECO_PER_CLASS = np.zeros(N_CLASSES, dtype=np.float32)
for _c, _v in ECONOMIC_VALUES.items():
    if _c < N_CLASSES:
        ECO_PER_CLASS[_c] = _v


class LandUseEnv(gym.Env):
    """Gymnasium env for land-use optimisation.

    Each step the env traverses cells in row-major order; the agent proposes
    new land-cover fractions (continuous logits → softmax → quantised to
    1/25 multiples). Protected classes keep their original pixel counts.
    An episode ends after all SAMPLE_SIZE×SAMPLE_SIZE cells are visited
    or when total ET drops beyond a threshold.
    """

    metadata = {"render_modes": []}

    def __init__(self, split="train", et_decrease_pct=5.0):
        super().__init__()

        data = np.load(Path(data_dir, "processed", "rl_dataset.npz"))
        self.fractions_full = data["fractions"]        # (n_rows, n_cols, n_classes)
        self.norm_stats     = data["norm_stats"]       # [μ_eco, σ_eco, μ_et, σ_et]

        splits = {"train": "train_indices", "test": "test_indices"}
        self.indices = data[splits[split]]

        self.n_cells         = SAMPLE_SIZE * SAMPLE_SIZE  # 100
        self.et_decrease_pct = et_decrease_pct
        self.et_per_class    = self._fit_et(data)

        # Spaces: observe one cell's fractions; act with logits for free classes
        self.observation_space = spaces.Box(0, 1, shape=(N_CLASSES,), dtype=np.float32)
        self.action_space      = spaces.Box(-5, 5, shape=(N_FREE,),  dtype=np.float32)

        # Episode state
        self.state = None
        self.current_cell = (0, 0)
        self.current_step = 0
        self.initial_total_et  = 0.0
        self.prev_total_reward = 0.0

    ##### Helper Funcs #####
    @staticmethod
    def _fit_et(data) -> np.ndarray:
        """Least-squares fit of per-class ET from valid cells."""
        v = data["valid_mask"].ravel()
        X = data["fractions"].reshape(-1, N_CLASSES)[v]
        y = data["et_values"].ravel()[v]
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        return np.maximum(c, 0).astype(np.float32)

    def _total_reward(self) -> float:
        """Sum of normalised (eco + et) rewards across all 100 cells."""
        flat = self.state.reshape(-1, N_CLASSES)
        mu_e, s_e, mu_t, s_t = self.norm_stats
        return float(
            ((flat @ ECO_PER_CLASS - mu_e) / s_e
             + (flat @ self.et_per_class - mu_t) / s_t).sum()
        )

    def _total_et(self) -> float:
        return float((self.state.reshape(-1, N_CLASSES) @ self.et_per_class).sum())

    def _get_obs(self) -> np.ndarray:
        r, c = self.current_cell
        return self.state[r, c].copy()

    def _apply_action(self, frac: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Convert actor logits into quantised fractions (pixel-exact).

        1. Freeze protected-class pixel counts.
        2. Softmax the N_FREE logits → probabilities.
        3. Distribute remaining pixels via largest-remainder rounding.
        """
        # Protected pixels stay fixed
        prot_pixels = np.zeros(N_CLASSES, dtype=np.int32)
        for c in PROTECTED_CLASSES:
            prot_pixels[c] = round(N_PIXELS * frac[c])
        remaining = N_PIXELS - int(prot_pixels.sum())

        if remaining <= 0:
            return frac.copy()

        # Softmax (numerically stable)
        logits = np.asarray(action, dtype=np.float64)
        logits -= logits.max()
        p = np.exp(logits)
        p /= p.sum()

        # Largest-remainder quantisation
        raw    = p * remaining
        counts = np.floor(raw).astype(np.int32)
        deficit = remaining - int(counts.sum())
        if deficit > 0:
            top = np.argsort(-(raw - counts))[:deficit]
            counts[top] += 1

        # Assemble new fraction vector
        new_frac = np.zeros(N_CLASSES, dtype=np.float32)
        for c in PROTECTED_CLASSES:
            new_frac[c] = prot_pixels[c] / N_PIXELS
        for i, c in enumerate(FREE_CLASSES):
            new_frac[c] = counts[i] / N_PIXELS
        return new_frac

    
    ##### Gym API #####
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        idx = self.np_random.integers(len(self.indices))
        r0, c0 = self.indices[idx]
        self.state = self.fractions_full[
            r0 : r0 + SAMPLE_SIZE, c0 : c0 + SAMPLE_SIZE
        ].copy()

        self.initial_total_et  = self._total_et()
        self.prev_total_reward = self._total_reward()
        self.current_step = 0

        # Start traversal from top-left cell (row-major order)
        self.current_cell = (0, 0)
        return self._get_obs(), {}

    def step(self, action):
        # Apply action to current cell
        r, c = self.current_cell
        self.state[r, c] = self._apply_action(self.state[r, c], action)

        # Reward = change in total grid reward
        total_rwd = self._total_reward()
        reward = total_rwd - self.prev_total_reward
        self.prev_total_reward = total_rwd
        self.current_step += 1

        # Termination: all cells traversed or ET dropped too far
        et_drop = 100.0 * (self.initial_total_et - self._total_et()) / (
            self.initial_total_et + 1e-9
        )
        terminated = self.current_step >= self.n_cells or et_drop > self.et_decrease_pct

        # Advance to next cell in row-major order
        if not terminated:
            next_r = self.current_step // SAMPLE_SIZE
            next_c = self.current_step % SAMPLE_SIZE
            self.current_cell = (next_r, next_c)

        info = {"et_drop_pct": et_drop, "step": self.current_step}
        return self._get_obs(), float(reward), terminated, False, info


def train(total_timesteps: int = 100000):
    logger.info("Creating environment …")
    env = LandUseEnv(split="train")
    check_env(env, warn=True)
    logger.info("Environment check passed.")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=100,
        batch_size=64,
        n_epochs=400,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log=str(log_dir / "ppo_landuse"),
    )

    logger.info(f"Training PPO for {total_timesteps:,} timesteps …")
    model.learn(total_timesteps=total_timesteps)

    save_path = model_dir / "ppo_land_use"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    logger.info(f"Model saved → {save_path}")
    return model


def evaluate(n_episodes: int = 5):
    model = PPO.load(str(model_dir / "ppo_land_use"))
    env   = LandUseEnv(split="test")

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward, done = 0.0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        logger.info(
            f"Episode {ep + 1}: reward={total_reward:.4f}, "
            f"ET drop={info['et_drop_pct']:.2f} %, steps={info['step']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO land-use optimisation")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--timesteps", type=int, default=100000)
    args = parser.parse_args()

    if args.mode == "train":
        train(total_timesteps=args.timesteps)
    else:
        evaluate()
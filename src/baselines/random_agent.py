"""Random baseline: uniformly sample a legal action at each step."""
from __future__ import annotations

import numpy as np


class RandomAgent:
    """Sample uniformly from `env.action_masks()` legal indices."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, env) -> int:
        mask = env.action_masks()
        valid = np.flatnonzero(mask)
        return int(self.rng.choice(valid))

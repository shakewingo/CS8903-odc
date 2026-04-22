"""Random baseline: uniform-random legal action at each step."""
from __future__ import annotations

import numpy as np


class RandomAgent:
    """Callable `(env) -> action` that samples uniformly from the legal set
    given by `env.action_masks()`. Uses a seeded numpy RNG."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, env) -> int:
        mask = env.action_masks()
        valid = np.flatnonzero(mask)
        return int(self.rng.choice(valid))

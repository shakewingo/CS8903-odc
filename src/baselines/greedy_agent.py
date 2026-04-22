"""Greedy-myopic baseline: at each step, pick the legal action with the
largest immediate `Δ total_value`. Ties broken deterministically by
`np.argmax` (first-index-wins).
"""
from __future__ import annotations

import numpy as np


class GreedyAgent:
    """Callable `(env) -> action` returning the 1-step-optimal legal action."""

    def __call__(self, env) -> int:
        mask = env.action_masks()
        valid = np.flatnonzero(mask)
        deltas = np.array([env._simulate_delta(a) for a in valid], dtype=np.float64)
        return int(valid[int(np.argmax(deltas))])

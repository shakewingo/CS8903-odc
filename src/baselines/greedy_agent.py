"""Greedy-myopic baseline: pick the legal action with the largest immediate
`Δ total_value` at each step. Ties broken by np.argmax (first-index-wins)
for determinism.
"""
from __future__ import annotations

import numpy as np


class GreedyAgent:
    def __call__(self, env) -> int:
        mask = env.action_masks()
        valid = np.flatnonzero(mask)
        deltas = np.array([env._simulate_delta(a) for a in valid], dtype=np.float64)
        return int(valid[int(np.argmax(deltas))])

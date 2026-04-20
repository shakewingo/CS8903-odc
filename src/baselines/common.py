"""Shared infrastructure for baseline comparison against MaskablePPO V2.

Provides:
- `make_eval_env(...)` — builds a LandUseEnvV2 configured for Exp III and sets
  the train_v2 module globals (ECO_MOD, ET_MOD, logger) that the env reads
  during reward computation.
- `reset_to_index(env, idx, ...)` — deterministically positions the env at a
  specific augmented sample, zeroing counters (fixing a stale-prev_spatial
  issue present in eval_v2.run_inference).
- `run_episode(act_fn, env, idx, ...)` — runs a single episode with the given
  action-selection callable and returns a metrics dict.
- `aggregate(records)` — turns a list of per-episode records into mean±std
  summary stats.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np

from src.config import (
    ECO_VALUES, ET_VALUES,
    N_CLASSES, SEED, data_dir, log_dir,
)
from src.utils import get_logger, minmax_normalize


# ── Env factory ────────────────────────────────────────────────────────
def make_eval_env(
    split: Literal["train_indices", "test_indices", "no_split"] = "test_indices",
    eco_variant: Literal["standard", "regen"] = "regen",
    use_et: bool = False,
    spatial_scale: float = 1.0,
    w_tree: float = 1.0,
    w_crop: float = 3.0,
    w_built: float = 3.0,
    w_buf: float = 5.0,
    max_steps: int = 500,
    pixels_per_transfer: int = 5,
    max_consecutive_noops: int = 10,
    et_dcs_tolerance: float = 1.0,
    min_mod_frac: float = 0.1,
    n_augment: int = 5,
    reward_scale: float = 1.0,
    lambda_et: float = 1.0,
    log_name: str = "baselines",
):
    """Build a V2 env matching Exp III (`gbdpkgiq`) defaults.

    Side effects:
    - Sets `src.train_v2.ECO_MOD`, `ET_MOD`, `logger` module globals. These
      must be initialized before any env method that computes rewards runs,
      which is why we can't just construct the env without this helper.
    """
    import src.train_v2 as tv2

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

    tv2.ECO_MOD = eco_per_class[tv2.MODIFIABLE_CLASSES]
    tv2.ET_MOD = et_per_class[tv2.MODIFIABLE_CLASSES]
    tv2.logger = get_logger(
        log_name,
        log_file=str(Path(log_dir, f"{log_name}.log")),
        stream=False,
        level="WARNING",
    )

    env = tv2.LandUseEnvV2(
        split=split,
        max_steps=max_steps,
        spatial_scale=spatial_scale,
        w_tree=w_tree,
        w_crop=w_crop,
        w_built=w_built,
        w_buf=w_buf,
        lambda_et=lambda_et,
        reward_scale=reward_scale,
        n_augment=n_augment,
        et_dcs_tolerance=et_dcs_tolerance,
        min_mod_frac=min_mod_frac,
        pixels_per_transfer=pixels_per_transfer,
        max_consecutive_noops=max_consecutive_noops,
    )
    env.reset(seed=SEED)
    return env


# ── Episode bookkeeping ────────────────────────────────────────────────
def reset_to_index(env, idx: int, max_consecutive_noops_override: Optional[int] = None):
    """Position env at `env.samples[idx]` with all counters / prev_* state zeroed.

    Unlike `eval_v2.run_inference`, this re-derives `prev_spatial_value` and
    `prev_spatial` from the *new* state — otherwise the first step's reward
    delta is contaminated by whatever random initial state env.reset() picked.
    """
    env._get_obs(idx)
    env.step_count = 0
    env.consecutive_noops = 0
    if max_consecutive_noops_override is not None:
        env.max_consecutive_noops = max_consecutive_noops_override
    tv = env._compute_total_value()
    env.prev_total_value = tv[0]
    env.prev_spatial_value = tv[1]
    env.prev_spatial = (tv[2], tv[3], tv[4], tv[5])
    env.initial_total_et = env._compute_total_et()
    return {
        "total_value": tv[0],
        "spatial_value": tv[1],
        "tree_contiguity": tv[2],
        "crop_contiguity": tv[3],
        "built_contiguity": tv[4],
        "buffer_penalty": tv[5],
    }


ActFn = Callable[[object], int]


def run_episode(
    act_fn: ActFn,
    env,
    idx: int,
    max_consecutive_noops_override: Optional[int] = None,
) -> dict:
    """Run one episode. `act_fn(env) -> int` is called each step until done.

    Returns a metrics dict with keys documented in `aggregate`.
    """
    initial = reset_to_index(env, idx, max_consecutive_noops_override)
    initial_total = initial["total_value"]
    initial_spatial = initial["spatial_value"]
    initial_eco = initial_total - initial_spatial

    terminated = False
    info = {}
    while not terminated:
        action = act_fn(env)
        _, _, terminated, _, info = env.step(action)

    final_total = info["total_value"]
    final_spatial = info["spatial_value"]
    final_eco = final_total - final_spatial
    delta = final_total - initial_total

    return {
        "episode_idx": idx,
        "initial_total_value": float(initial_total),
        "final_total_value": float(final_total),
        "delta_total_value": float(delta),
        "delta_pct": float(delta / (abs(initial_total) + 1e-9) * 100.0),
        "initial_eco": float(initial_eco),
        "final_eco": float(final_eco),
        "initial_spatial_value": float(initial_spatial),
        "final_spatial_value": float(final_spatial),
        "final_tree_contiguity": float(info["tree_contiguity"]),
        "final_crop_contiguity": float(info["crop_contiguity"]),
        "final_built_contiguity": float(info["built_contiguity"]),
        "final_buffer_penalty": float(info["buffer_penalty"]),
        "steps": int(info["step"]),
        "success": bool(delta > 0),
        "termination_reason": info.get("termination_reason") or "unknown",
    }


# ── Aggregation ────────────────────────────────────────────────────────
def aggregate(records: list[dict]) -> dict:
    """Reduce per-episode records to a summary dict (mean, std, success rate)."""
    if not records:
        return {}
    keys = ("delta_total_value", "delta_pct", "final_eco", "final_spatial_value",
            "final_tree_contiguity", "final_crop_contiguity",
            "final_built_contiguity", "final_buffer_penalty", "steps")
    out = {"n_episodes": len(records)}
    for k in keys:
        vals = np.array([r[k] for r in records], dtype=np.float64)
        out[f"{k}_mean"] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std(ddof=0))
    out["success_rate"] = float(np.mean([r["success"] for r in records]))
    return out


# ── Utilities ──────────────────────────────────────────────────────────
def snapshot_env_state(env):
    """Deep-copy the mutable parts of env for fitness replay (used by GA)."""
    return {
        "state": env.state.copy(),
        "_protected_water": env._protected_water.copy(),
        "step_count": env.step_count,
        "consecutive_noops": env.consecutive_noops,
        "prev_total_value": env.prev_total_value,
        "prev_spatial_value": env.prev_spatial_value,
        "prev_spatial": env.prev_spatial,
        "initial_total_et": env.initial_total_et,
        "max_consecutive_noops": env.max_consecutive_noops,
    }


def restore_env_state(env, snapshot):
    env.state = snapshot["state"].copy()
    env._protected_water = snapshot["_protected_water"].copy()
    env.step_count = snapshot["step_count"]
    env.consecutive_noops = snapshot["consecutive_noops"]
    env.prev_total_value = snapshot["prev_total_value"]
    env.prev_spatial_value = snapshot["prev_spatial_value"]
    env.prev_spatial = snapshot["prev_spatial"]
    env.initial_total_et = snapshot["initial_total_et"]
    env.max_consecutive_noops = snapshot["max_consecutive_noops"]

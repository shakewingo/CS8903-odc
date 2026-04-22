"""Shared helpers for baseline methods (Random / Greedy / GA) running against
LandUseEnvV2.

Exports:
- `make_eval_env(...)`  — build a configured LandUseEnvV2 and set the
  `train_v2` globals (ECO_MOD, ET_MOD, logger) needed for reward computation.
- `reset_to_index(env, idx)`  — deterministically position the env at sample
  `idx` and re-derive all `prev_*` state from the new initial state.
- `run_episode(act_fn, env, idx)`  — run one episode and return a metrics dict.
- `aggregate(records)`  — mean±std summary over a list of episode records.

Env-state snapshot / restore (used by GA) lives on the env itself:
`env.get_state_dict()` / `env.set_state_dict(snapshot)`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Protocol

import numpy as np

from src.config import SEED, log_dir
from src.utils import get_logger


# ── Env factory ────────────────────────────────────────────────────────
def make_eval_env(
    split: Literal["train_indices", "test_indices", "no_split"] = "test_indices",
    use_et: bool = False,
    spatial_scale: float = 1.0,
    w_tree: float = 1.0,
    w_crop: float = 3.0,
    w_built: float = 3.0,
    w_buf: float = 5.0,
    w_rip: float = 0.0,
    riparian_mask: bool = False,
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
    """Build a LandUseEnvV2 and initialise the `train_v2` globals it reads.

    Side effects (required before any reward computation):
    - Assigns `train_v2.ECO_MOD` / `ET_MOD` via `build_value_vecs`.
    - Installs a dedicated file logger on `train_v2.logger`.
    """
    import src.train_v2 as tv2

    tv2.build_value_vecs(use_et=use_et, assign_globals=True)
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
        w_rip=w_rip,
        riparian_mask=riparian_mask,
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
    """Position env at `env.samples[idx]` and re-derive all `prev_*` reward
    buffers from the new state so the first step's Δ reward is unbiased.

    Optionally override `env.max_consecutive_noops` for this episode (used by
    GA to let chromosomes replay their full horizon without early termination).
    """
    env._get_obs(idx)
    env.step_count = 0
    env.consecutive_noops = 0
    if max_consecutive_noops_override is not None:
        env.max_consecutive_noops = max_consecutive_noops_override
    tv = env._compute_total_value()
    env.prev_total_value = tv[0]
    env.prev_spatial_value = tv[1]
    env.prev_spatial = (tv[2], tv[3], tv[4], tv[5], tv[6])
    env.initial_total_et = env._compute_total_et()
    return {
        "total_value": tv[0],
        "spatial_value": tv[1],
        "tree_contiguity": tv[2],
        "crop_contiguity": tv[3],
        "built_contiguity": tv[4],
        "buffer_penalty": tv[5],
        "trees_riparian": tv[6],
    }


class Agent(Protocol):
    """Baseline agent interface: callable that picks an action given env."""
    def __call__(self, env) -> int: ...


def run_episode(
    act_fn: Agent,
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

    # Final class fractions (modifiable mass only; order: trees, crops, built,
    # bare, rangeland). Sanity check for mosaic vs monoculture policies.
    import src.train_v2 as tv2
    final_mass = float(env.state.sum())
    if final_mass > 1e-9:
        frac = env.state.sum(axis=(0, 1)) / final_mass
    else:
        frac = np.zeros(tv2.N_MOD, dtype=np.float32)

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
        "final_trees_riparian": float(info.get("trees_riparian", 0.0)),
        "final_frac_trees": float(frac[0]),
        "final_frac_crops": float(frac[1]),
        "final_frac_built": float(frac[2]),
        "final_frac_bare": float(frac[3]),
        "final_frac_range": float(frac[4]),
        "steps": int(info["step"]),
        "success": bool(delta > 0),
        "termination_reason": info.get("termination_reason") or "unknown",
    }


# ── Aggregation ────────────────────────────────────────────────────────
def aggregate(records: list[dict]) -> dict:
    if not records:
        return {}
    keys = ("delta_total_value", "delta_pct", "final_eco", "final_spatial_value",
            "final_tree_contiguity", "final_crop_contiguity",
            "final_built_contiguity", "final_buffer_penalty",
            "final_trees_riparian",
            "final_frac_trees", "final_frac_crops", "final_frac_built",
            "final_frac_bare", "final_frac_range",
            "steps")
    out = {"n_episodes": len(records)}
    for k in keys:
        vals = np.array([r[k] for r in records], dtype=np.float64)
        out[f"{k}_mean"] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std(ddof=0))
    out["success_rate"] = float(np.mean([r["success"] for r in records]))
    return out



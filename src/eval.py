#!/usr/bin/env python3
"""
Evaluate a trained MaskablePPO agent and visualise before/after land-use maps.

Usage examples:
    # Evaluate using the model saved by the last wandb run
    python src/eval.py --model-path models/<run_id>/model.zip

    # Evaluate on test split only
    python src/eval.py --model-path models/<run_id>/model.zip --split test

    # Evaluate on train split only
    python src/eval.py --model-path models/<run_id>/model.zip --split train

    # Custom output plot name
    python src/eval.py --model-path models/<run_id>/model.zip --plot-name my_experiment

    # Skip plotting, just print episode stats
    python src/eval.py --model-path models/<run_id>/model.zip --no-plot
"""

import argparse
import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from sb3_contrib import MaskablePPO

# ── Project imports ─────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import (
    PROTECTED_CLASSES, ECO_VALUES, ET_VALUES,
    N_CLASSES, N_PIXELS_PER_CELL, SEED, data_dir, log_dir,
)
from src.post_eda import plot_state_heatmap
from src.train import (
    LandUseEnv, MODIFIABLE_CLASSES, N_MOD,
    augment_data,
)
from src.utils import minmax_normalize, get_logger


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a trained MaskablePPO land-use agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--model-path", type=str, required=True,
                   help="Path to saved model.zip")
    p.add_argument("--split", type=str, default="both",
                   choices=["train", "test", "both"],
                   help="Which data split(s) to evaluate")
    p.add_argument("--deterministic", action="store_true",
                   help="Use deterministic actions during inference")

    # Environment args (should match training)
    env_g = p.add_argument_group("Environment (must match training)")
    env_g.add_argument("--max-steps", type=int, default=500)
    env_g.add_argument("--lambda-cont", type=float, default=0.05)
    env_g.add_argument("--lambda-buf", type=float, default=0.05)
    env_g.add_argument("--lambda-et", type=float, default=1.0)
    env_g.add_argument("--reward-scale", type=float, default=1.0)
    env_g.add_argument("--et-dcs-tolerance", type=float, default=1.0)
    env_g.add_argument("--pixels-per-transfer", type=int, default=5)
    env_g.add_argument("--max-consecutive-noops", type=int, default=10)
    env_g.add_argument("--min-mod-frac", type=float, default=0.1)
    env_g.add_argument("--no-contiguity-reward", action="store_true",
                       help="Disable tree contiguity bonus")
    env_g.add_argument("--no-buffer-penalty", action="store_true",
                       help="Disable water-buffer penalty")
    env_g.add_argument("--use-et", action="store_true",
                       help="Use real ET values from config instead of zeros")

    # Output
    out = p.add_argument_group("Output")
    out.add_argument("--plot-name", type=str, default=None,
                     help="Plot filename stem (auto-generated if omitted)")
    out.add_argument("--no-plot", action="store_true",
                     help="Skip generating the comparison plot")
    out.add_argument("--output-dir", type=str,
                     default=str(Path(data_dir, "processed")),
                     help="Directory to save plots")
    out.add_argument("--seed", type=int, default=SEED)

    return p.parse_args()


def _make_env(args, split):
    """Create an evaluation environment with no augmentation."""
    return LandUseEnv(
        split=split,
        max_steps=args.max_steps,
        lambda_cont=args.lambda_cont,
        lambda_buf=args.lambda_buf,
        lambda_et=args.lambda_et,
        reward_scale=args.reward_scale,
        n_augment=0,
        et_dcs_tolerance=args.et_dcs_tolerance,
        add_contiguity_reward=not args.no_contiguity_reward,
        add_buffer_penalty=not args.no_buffer_penalty,
        min_mod_frac=args.min_mod_frac,
        pixels_per_transfer=args.pixels_per_transfer,
        max_consecutive_noops=args.max_consecutive_noops,
    )


def run_inference(model, env, data, split_key, deterministic=False):
    """Run the agent on every sample in the given split. Returns {(r,c): obs}."""
    indices = data[split_key]
    final_obs = {}

    for ep in range(len(indices)):
        coord_idx = indices[ep]
        _, _ = env.reset()
        obs = env._get_obs(ep).copy()
        env.prev_total_value, _, _, _ = env._compute_total_value()
        env.initial_total_et = env._compute_total_et()

        total_reward, done = 0.0, False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        final_obs[tuple(coord_idx)] = obs
        print(f"  {split_key} ep {ep+1}/{len(indices)} "
              f"(coord={coord_idx}): reward={total_reward:.4f}, steps={info['step']}")

    return final_obs


def reconstruct_map(data, final_obs):
    """Overlay agent outputs onto the full map (fraction space)."""
    initial_obs = data["pixel_counts"].astype(np.float32) / N_PIXELS_PER_CELL
    final_obs_recst = initial_obs.copy()

    for (r, c), obs in final_obs.items():
        for k, cls in enumerate(MODIFIABLE_CLASSES):
            final_obs_recst[r:r+10, c:c+10, cls] = obs[:, :, k]

    return initial_obs, final_obs_recst


def print_diff_table(initial_obs, final_obs_recst):
    """Print cells that changed between initial and final."""
    diff_mask = ~np.isclose(initial_obs, final_obs_recst, atol=1e-6)
    diff_positions = np.argwhere(diff_mask)

    diffs = []
    for pos in diff_positions:
        r, c, cls = pos
        delta = final_obs_recst[r, c, cls] - initial_obs[r, c, cls]
        diffs.append((r, c, cls, initial_obs[r, c, cls], final_obs_recst[r, c, cls], delta))

    diffs.sort(key=lambda x: abs(x[5]), reverse=True)

    print(f"\nTotal differing entries: {len(diffs)}")
    print(f"Unique (row, col) cells: {len(set((r, c) for r, c, *_ in diffs))}")
    print(f"\n{'row':>3s} {'col':>3s} {'cls':>3s}  {'initial':>8s}  {'final':>8s}  {'delta':>8s}")
    print("-" * 42)
    for r, c, cls, init_val, final_val, delta in diffs[:20]:
        print(f"{r:3d} {c:3d} {cls:3d}  {init_val:8.4f}  {final_val:8.4f}  {delta:+8.4f}")

    return diffs


def plot_comparison(initial_obs, final_obs_recst, diffs, value_vec, save_path):
    """Side-by-side before/after heatmap with changed cells highlighted."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(50, 20))
    plot_state_heatmap(initial_obs, value_vec, title="Before", ax=ax1)
    plot_state_heatmap(final_obs_recst, value_vec, title="After", ax=ax2)

    n_rows = initial_obs.shape[0]
    changed_cells = set((r, c) for r, c, *_ in diffs)
    for (r, c) in changed_cells:
        for ax in (ax1, ax2):
            rect = patches.Rectangle(
                (c, n_rows - 1 - r), 1, 1,
                linewidth=2, edgecolor="black", facecolor="none",
            )
            ax.add_patch(rect)

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"\nPlot saved to {save_path}")
    plt.close(fig)


def main():
    import src.train as train_module

    args = parse_args()

    # ── Set value vectors (must match training) ────────────────────
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

    train_module.ECO_MOD = eco_per_class[MODIFIABLE_CLASSES]
    train_module.ET_MOD = et_per_class[MODIFIABLE_CLASSES]

    train_module.logger = get_logger(
        "eval", stream=False, level="WARNING",
        log_file=str(Path(log_dir, "eval.log")),
    )

    value_vec = eco_per_class + et_per_class

    # ── Load model & data ──────────────────────────────────────────
    data = np.load(Path(data_dir, "processed", "rl_dataset.npz"))
    final_obs = {}

    # Create a dummy env for loading (SB3 needs it)
    dummy_env = _make_env(args, "test_indices")
    model = MaskablePPO.load(args.model_path, env=dummy_env)
    print(f"Loaded model from {args.model_path}")

    # ── Run inference ──────────────────────────────────────────────
    if args.split in ("test", "both"):
        print("\n=== Test split ===")
        env = _make_env(args, "test_indices")
        test_obs = run_inference(model, env, data, "test_indices",
                                 deterministic=args.deterministic)
        final_obs.update(test_obs)

    if args.split in ("train", "both"):
        print("\n=== Train split ===")
        env = _make_env(args, "train_indices")
        train_obs = run_inference(model, env, data, "train_indices",
                                  deterministic=args.deterministic)
        final_obs.update(train_obs)

    # ── Reconstruct & diff ─────────────────────────────────────────
    initial_obs, final_obs_recst = reconstruct_map(data, final_obs)
    diffs = print_diff_table(initial_obs, final_obs_recst)

    # ── Plot ───────────────────────────────────────────────────────
    if not args.no_plot and diffs:
        plot_name = args.plot_name or Path(args.model_path).parent.name
        save_path = Path(args.output_dir, f"{plot_name}.png")
        plot_comparison(initial_obs, final_obs_recst, diffs, value_vec, save_path)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()

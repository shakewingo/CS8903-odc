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
    ECO_VALUES, ET_VALUES,
    N_CLASSES, N_PIXELS_PER_CELL, SEED, data_dir, log_dir,
)
from src.post_eda import plot_state_heatmap
from src.train import LandUseEnv, MODIFIABLE_CLASSES
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
    env_g.add_argument("--spatial-scale", type=float, default=1.0)
    env_g.add_argument("--w-tree", type=float, default=3.0)
    env_g.add_argument("--w-crop", type=float, default=1.0)
    env_g.add_argument("--w-built", type=float, default=1.0)
    env_g.add_argument("--w-buf", type=float, default=2.0)
    env_g.add_argument("--lambda-et", type=float, default=1.0)
    env_g.add_argument("--reward-scale", type=float, default=1.0)
    env_g.add_argument("--et-dcs-tolerance", type=float, default=1.0)
    env_g.add_argument("--pixels-per-transfer", type=int, default=5)
    env_g.add_argument("--max-consecutive-noops", type=int, default=10)
    env_g.add_argument("--min-mod-frac", type=float, default=0.1)
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


def make_env(args, split):
    """Create an evaluation environment with no augmentation.

    ``args`` may be the full eval-CLI Namespace or any object exposing the
    env-related attributes listed below.
    """
    return LandUseEnv(
        split=split,
        max_steps=getattr(args, "max_steps", 500),
        spatial_scale=getattr(args, "spatial_scale", 1.0),
        w_tree=getattr(args, "w_tree", 3.0),
        w_crop=getattr(args, "w_crop", 1.0),
        w_built=getattr(args, "w_built", 1.0),
        w_buf=getattr(args, "w_buf", 2.0),
        lambda_et=getattr(args, "lambda_et", 1.0),
        reward_scale=getattr(args, "reward_scale", 1.0),
        n_augment=0,
        et_dcs_tolerance=getattr(args, "et_dcs_tolerance", 1.0),
        min_mod_frac=getattr(args, "min_mod_frac", 0.1),
        pixels_per_transfer=getattr(args, "pixels_per_transfer", 5),
        max_consecutive_noops=getattr(args, "max_consecutive_noops", 10),
    )


def build_value_vecs(use_et=False, assign_globals=True):
    """Construct normalised per-class eco/ET value vectors.

    Mirrors the setup used at training time. When ``assign_globals`` is
    True (the default) the matching slices are also written into
    ``src.train.ECO_MOD`` / ``ET_MOD`` so a loaded env can compute rewards
    consistently during inference.

    Returns ``(eco_per_class, et_per_class)`` — combined ``value_vec`` is
    just their sum.
    """
    et_values = ET_VALUES if use_et else {k: 0 for k in ET_VALUES}
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

    if assign_globals:
        import src.train as train_module
        train_module.ECO_MOD = eco_per_class[MODIFIABLE_CLASSES]
        train_module.ET_MOD = et_per_class[MODIFIABLE_CLASSES]

    return eco_per_class, et_per_class


def compute_diff_cells(initial_obs, final_obs_recst, atol=1e-6):
    """Return the set of ``(row, col)`` coordinates that changed."""
    diff_mask = ~np.isclose(initial_obs, final_obs_recst, atol=atol)
    return {(int(r), int(c)) for r, c, _ in np.argwhere(diff_mask)}


def run_inference(model, env, data, split_key, deterministic=False):
    """Run the agent on every sample in the given split. Returns {(r,c): obs}."""
    indices = data[split_key]
    final_obs = {}

    for ep in range(len(indices)):
        coord_idx = indices[ep]
        _, _ = env.reset()
        obs = env._get_obs(ep).copy()
        env.prev_total_value = env._compute_total_value()[0]
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
    eco_per_class, et_per_class = build_value_vecs(use_et=args.use_et)
    value_vec = eco_per_class + et_per_class

    train_module.logger = get_logger(
        "eval", stream=False, level="WARNING",
        log_file=str(Path(log_dir, "eval.log")),
    )

    # ── Load model & data ──────────────────────────────────────────
    data = np.load(Path(data_dir, "processed", "rl_dataset.npz"))
    final_obs = {}

    # Create a dummy env for loading (SB3 needs it)
    dummy_env = make_env(args, "test_indices")
    model = MaskablePPO.load(args.model_path, env=dummy_env)
    print(f"Loaded model from {args.model_path}")

    # ── Run inference ──────────────────────────────────────────────
    if args.split in ("test", "both"):
        print("\n=== Test split ===")
        env = make_env(args, "test_indices")
        test_obs = run_inference(model, env, data, "test_indices",
                                 deterministic=args.deterministic)
        final_obs.update(test_obs)

    if args.split in ("train", "both"):
        print("\n=== Train split ===")
        env = make_env(args, "train_indices")
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

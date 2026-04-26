#!/usr/bin/env python3
"""Evaluate an agent over the test/train split and render a before/after
heatmap of the resulting land-use allocation.

Supports --method {ppo, greedy, random, genetic}. Reward-config flags
(--w-tree/--w-crop/... --riparian-mask) must match training for meaningful
PPO reward numbers; geometric metrics are reward-independent.

Usage examples:
    # PPO
    python src/eval.py --method ppo --model-path models/<run_id>/model.zip

    # Greedy baseline with custom reward
    python src/eval.py --method greedy --w-crop 4 --w-buf 6 --w-rip 5 --riparian-mask

    # Genetic algorithm (slow; ~1-10 min/episode)
    python src/eval.py --method genetic --ga-pop 20 --ga-gens 20
"""

import argparse
import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from sb3_contrib import MaskablePPO

# See train.py for rationale — disables strict simplex validation.
torch.distributions.Distribution.set_default_validate_args(False)

# ── Project imports ─────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import (
    ECO_VALUES, ET_VALUES, LAND_COVER_COLORS, LAND_COVER_LABELS,
    N_CLASSES, N_PIXELS_PER_CELL, SEED, data_dir, log_dir,
)
from src.post_eda import plot_state_heatmap
from src.baselines.common import reset_to_index
from src.train import LandUseEnv, MODIFIABLE_CLASSES, build_value_vecs
from src.utils import minmax_normalize, get_logger


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a trained MaskablePPO land-use agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--method", type=str, default="ppo",
                   choices=["ppo", "greedy", "random", "genetic"],
                   help="Which agent to run. 'greedy' is 1-step-optimal via "
                        "env._simulate_delta. 'random' samples uniformly over "
                        "legal actions. 'genetic' evolves a per-episode "
                        "chromosome via GeneticAgent (slow: ~1–10 min/episode "
                        "depending on --ga-pop / --ga-gens).")
    p.add_argument("--model-path", type=str, default=None,
                   help="Path to saved model.zip (required when --method=ppo)")

    # GA knobs — lower than eval_baselines defaults by default since we're
    # generating a visualization, not a statistical summary.
    ga_g = p.add_argument_group("Genetic Algorithm (only used when --method=genetic)")
    ga_g.add_argument("--ga-pop", type=int, default=20)
    ga_g.add_argument("--ga-gens", type=int, default=20)
    ga_g.add_argument("--ga-tournament-k", type=int, default=3)
    ga_g.add_argument("--ga-crossover-prob", type=float, default=0.8)
    ga_g.add_argument("--ga-mutation-prob", type=float, default=0.02)
    ga_g.add_argument("--ga-n-elite", type=int, default=2)
    ga_g.add_argument("--ga-verbose", action="store_true")
    p.add_argument("--split", type=str, default="both",
                   choices=["train", "test", "both"],
                   help="Which data split(s) to evaluate")
    p.add_argument("--deterministic", action="store_true",
                   help="Use deterministic actions during inference (PPO only)")

    # Environment args (defaults match train.py; override if you trained with different values)
    env_g = p.add_argument_group("Environment (must match training)")
    env_g.add_argument("--max-steps", type=int, default=500)
    env_g.add_argument("--spatial-scale", type=float, default=1.0)
    env_g.add_argument("--w-tree", type=float, default=1.0)
    env_g.add_argument("--w-crop", type=float, default=3.0)
    env_g.add_argument("--w-built", type=float, default=3.0)
    env_g.add_argument("--w-buf", type=float, default=5.0)
    env_g.add_argument("--w-rip", type=float, default=0.0,
                       help="Match training's --w-rip (affects printed reward only; "
                            "PPO policy itself is frozen on load)")
    env_g.add_argument("--riparian-mask", action="store_true",
                       help="Match training's --riparian-mask (affects which actions "
                            "PPO is allowed to take at eval time)")
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
    """Create an eval env (no augmentation) and call reset() to populate
    `env.samples` so `reset_to_index(env, ep)` in run_inference can address
    specific grids."""
    env = LandUseEnv(
        split=split,
        max_steps=getattr(args, "max_steps", 500),
        spatial_scale=getattr(args, "spatial_scale", 1.0),
        w_tree=getattr(args, "w_tree", 1.0),
        w_crop=getattr(args, "w_crop", 3.0),
        w_built=getattr(args, "w_built", 3.0),
        w_buf=getattr(args, "w_buf", 5.0),
        w_rip=getattr(args, "w_rip", 0.0),
        riparian_mask=getattr(args, "riparian_mask", False),
        lambda_et=getattr(args, "lambda_et", 1.0),
        reward_scale=getattr(args, "reward_scale", 1.0),
        n_augment=0,
        et_dcs_tolerance=getattr(args, "et_dcs_tolerance", 1.0),
        min_mod_frac=getattr(args, "min_mod_frac", 0.1),
        pixels_per_transfer=getattr(args, "pixels_per_transfer", 5),
        max_consecutive_noops=getattr(args, "max_consecutive_noops", 10),
    )
    env.reset(seed=SEED)
    return env


def compute_diff_cells(initial_obs, final_obs_recst, atol=1e-6):
    """Return the set of ``(row, col)`` coordinates that changed."""
    diff_mask = ~np.isclose(initial_obs, final_obs_recst, atol=atol)
    return {(int(r), int(c)) for r, c, _ in np.argwhere(diff_mask)}


def run_inference(make_act_fn, env, data, split_key):
    """Run one episode per sample in `data[split_key]` and collect final obs.

    `make_act_fn(env, ep) -> act_fn` is a factory. For stateless agents (PPO,
    Greedy, Random) it ignores `ep` and returns the same callable each time.
    For GA it runs per-episode evolution in `solve(env, ep)` and returns the
    chromosome-replay closure.

    Returns `{(row, col): final_obs}` keyed by each sample's coord in the
    50×50 map.
    """
    indices = data[split_key]
    final_obs = {}

    for ep in range(len(indices)):
        coord_idx = indices[ep]
        reset_to_index(env, ep)
        act_fn = make_act_fn(env, ep)  # may re-reset env (GA) — idempotent.
        obs = env.state

        total_reward, done = 0.0, False
        while not done:
            action = act_fn(env)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        final_obs[tuple(coord_idx)] = obs
        print(f"  {split_key} ep {ep+1}/{len(indices)} "
              f"(coord={coord_idx}): reward={total_reward:.4f}, steps={info['step']}"
              + (", no_actions_left" if info.get('no_actions_left') else ""))

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


def land_use_ratios(obs):
    """Per-class global ratio across the map (obs is (H, W, N_CLASSES) fractions)."""
    return obs.reshape(-1, obs.shape[-1]).mean(axis=0)


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

    before = land_use_ratios(initial_obs)
    after = land_use_ratios(final_obs_recst)
    handles, labels = [], []
    for cls in sorted(LAND_COVER_LABELS):
        if before[cls] == 0 and after[cls] == 0:
            continue
        b, a = before[cls] * 100, after[cls] * 100
        handles.append(patches.Patch(color=LAND_COVER_COLORS[cls]))
        labels.append(f"{LAND_COVER_LABELS[cls]}: {b:.1f}% → {a:.1f}% ({a-b:+.1f}%)")
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5),
               fontsize=20, frameon=False, title="Land-use ratio", title_fontsize=22)
    fig.subplots_adjust(right=0.85)

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

    # ── Load data + build act_fn based on --method ─────────────────
    data = np.load(Path(data_dir, "processed", "rl_dataset.npz"))
    final_obs = {}

    if args.method == "ppo":
        if args.model_path is None:
            raise SystemExit("--method=ppo requires --model-path")
        dummy_env = make_env(args, "test_indices")
        model = MaskablePPO.load(args.model_path, env=dummy_env)
        print(f"Loaded model from {args.model_path}")
        deterministic = args.deterministic

        def ppo_act(env):
            action, _ = model.predict(env.state,
                                      action_masks=env.action_masks(),
                                      deterministic=deterministic)
            return int(action)
        make_act_fn = lambda env, ep: ppo_act
    elif args.method == "greedy":
        from src.baselines.greedy_agent import GreedyAgent
        greedy = GreedyAgent()
        make_act_fn = lambda env, ep: greedy
        print("Using GreedyAgent (1-step-optimal w.r.t. current reward config)")
    elif args.method == "random":
        from src.baselines.random_agent import RandomAgent
        rand = RandomAgent(seed=args.seed)
        make_act_fn = lambda env, ep: rand
        print(f"Using RandomAgent(seed={args.seed})")
    elif args.method == "genetic":
        from src.baselines.genetic_agent import GeneticAgent
        ga = GeneticAgent(
            pop_size=args.ga_pop,
            generations=args.ga_gens,
            tournament_k=args.ga_tournament_k,
            crossover_prob=args.ga_crossover_prob,
            mutation_prob=args.ga_mutation_prob,
            n_elite=args.ga_n_elite,
            seed=args.seed,
            verbose=args.ga_verbose,
        )
        make_act_fn = lambda env, ep: ga.solve(env, ep)
        print(f"Using GeneticAgent (pop={args.ga_pop}, gens={args.ga_gens}) — "
              f"expect ~{args.ga_pop * args.ga_gens} evaluations per episode")

    # ── Run inference ──────────────────────────────────────────────
    if args.split in ("test", "both"):
        print("\n=== Test split ===")
        env = make_env(args, "test_indices")
        test_obs = run_inference(make_act_fn, env, data, "test_indices")
        final_obs.update(test_obs)

    if args.split in ("train", "both"):
        print("\n=== Train split ===")
        env = make_env(args, "train_indices")
        train_obs = run_inference(make_act_fn, env, data, "train_indices")
        final_obs.update(train_obs)

    # ── Reconstruct & diff ─────────────────────────────────────────
    initial_obs, final_obs_recst = reconstruct_map(data, final_obs)
    diffs = print_diff_table(initial_obs, final_obs_recst)

    # ── Plot ───────────────────────────────────────────────────────
    if not args.no_plot and diffs:
        if args.plot_name:
            plot_name = args.plot_name
        elif args.method != "ppo":
            plot_name = args.method
        else:
            plot_name = Path(args.model_path).parent.name
        save_path = Path(args.output_dir, f"{plot_name}.png")
        plot_comparison(initial_obs, final_obs_recst, diffs, value_vec, save_path)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()

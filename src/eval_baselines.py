#!/usr/bin/env python3
"""Run one or more baseline methods (random / greedy / genetic / ppo) on the
test split and emit per-episode CSV + summary table CSV + box-plot.

Use --w-tree/--w-crop/--w-built/--w-buf/--w-rip/--riparian-mask to match the
reward config under which the PPO model was trained; otherwise Greedy and GA
optimise against a different objective than PPO saw.

Usage:
    # All methods, aligned reward
    python src/eval_baselines.py --ppo-model models/<run_id>/model.zip \\
        --w-crop 4 --w-buf 6 --w-rip 5 --riparian-mask

    # Smoke test: skip GA, 2 episodes
    python src/eval_baselines.py --methods random greedy ppo \\
        --n-episodes 2 --ppo-model models/<run_id>/model.zip

    # GA only, tiny budget
    python src/eval_baselines.py --methods genetic --ga-pop 5 --ga-gens 3 --n-episodes 2
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# ── Project path setup ─────────────────────────────────────────────────
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Matches train_v2.py (avoids false-positive simplex errors on large softmax)
torch.distributions.Distribution.set_default_validate_args(False)

from src.config import data_dir
from src.baselines.common import (
    aggregate,
    make_eval_env,
    run_episode,
)
from src.baselines.random_agent import RandomAgent
from src.baselines.greedy_agent import GreedyAgent
from src.baselines.genetic_agent import GeneticAgent


# ── PPO wrapper ────────────────────────────────────────────────────────
class PPOAgent:
    """Adapter: loaded MaskablePPO model presented as `act_fn(env) -> int`."""

    def __init__(self, model, deterministic: bool = True):
        self.model = model
        self.deterministic = deterministic

    def __call__(self, env) -> int:
        obs = env.state
        action_masks = env.action_masks()
        action, _ = self.model.predict(
            obs, action_masks=action_masks, deterministic=self.deterministic
        )
        return int(action)


# ── Method registry ────────────────────────────────────────────────────
def _make_random(args, env, ppo_model, ga_kwargs):
    return RandomAgent(seed=args.seed)


def _make_greedy(args, env, ppo_model, ga_kwargs):
    return GreedyAgent()


def _make_genetic(args, env, ppo_model, ga_kwargs):
    return GeneticAgent(seed=args.seed, **ga_kwargs)


def _make_ppo(args, env, ppo_model, ga_kwargs):
    assert ppo_model is not None, "PPO method requires --ppo-model"
    return PPOAgent(ppo_model, deterministic=True)


METHOD_REGISTRY = {
    "random":  {"factory": _make_random,  "needs_solve": False, "noops_override": None},
    "greedy":  {"factory": _make_greedy,  "needs_solve": False, "noops_override": None},
    "genetic": {"factory": _make_genetic, "needs_solve": True,  "noops_override": "horizon_plus_1"},
    "ppo":     {"factory": _make_ppo,     "needs_solve": False, "noops_override": None},
}
ALL_METHODS = tuple(METHOD_REGISTRY.keys())


# ── Episode driver ─────────────────────────────────────────────────────
def evaluate_method(method, args, env, ppo_model, ga_kwargs, verbose=True):
    spec = METHOD_REGISTRY[method]
    agent = spec["factory"](args, env, ppo_model, ga_kwargs)
    noops_override = env.max_steps + 1 if spec["noops_override"] == "horizon_plus_1" else spec["noops_override"]

    records: list[dict] = []
    t0 = time.time()
    n_episodes = args.n_episodes or len(env.samples)
    n_episodes = min(n_episodes, len(env.samples))
    for ep in range(n_episodes):
        act_fn = agent.solve(env, ep) if spec["needs_solve"] else agent
        metrics = run_episode(act_fn, env, ep, max_consecutive_noops_override=noops_override)
        metrics["method"] = method
        records.append(metrics)
        if verbose:
            elapsed = time.time() - t0
            print(f"  [{method}] ep {ep+1}/{n_episodes}: "
                  f"Δ={metrics['delta_total_value']:+.4f} "
                  f"({metrics['delta_pct']:+.2f}%), "
                  f"steps={metrics['steps']}, "
                  f"term={metrics['termination_reason']}, "
                  f"elapsed={elapsed:.1f}s")
    return records


# ── Output writers ─────────────────────────────────────────────────────
def write_per_episode_csv(records: list[dict], path: Path) -> None:
    if not records:
        return
    keys = list(records[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(records)
    print(f"Per-episode CSV → {path}")


def write_summary_table(per_method: dict[str, dict], path: Path) -> None:
    """Write a wide-format CSV: one row per method."""
    if not per_method:
        return
    first_summary = next(iter(per_method.values()))
    cols = ["method"] + list(first_summary.keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for method, summary in per_method.items():
            row = {"method": method, **summary}
            w.writerow(row)
    print(f"Summary table CSV → {path}")


def plot_box(all_records: list[dict], path: Path, title: str = "Baseline comparison on Exp III") -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping box plot")
        return

    by_method: dict[str, list[float]] = {}
    for r in all_records:
        by_method.setdefault(r["method"], []).append(r["delta_total_value"])

    methods = list(by_method.keys())
    data = [by_method[m] for m in methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, tick_labels=methods, showmeans=True)
    ax.axhline(0, linestyle="--", color="grey", linewidth=0.8)
    ax.set_ylabel("Δ total_value")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Box plot → {path}")


def plot_line(records: list[dict], path: Path,
              title: str = "Per-episode Δ total_value on effective grids") -> None:
    """Line plot: x = effective-episode ordinal, y = Δ total_value, one line per method."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping line plot")
        return

    by_method: dict[str, list[tuple[int, float]]] = {}
    for r in records:
        by_method.setdefault(r["method"], []).append(
            (int(r["episode_idx"]), float(r["delta_total_value"]))
        )

    fig, ax = plt.subplots(figsize=(9, 5))
    for method in sorted(by_method):
        pairs = sorted(by_method[method])
        xs = list(range(1, len(pairs) + 1))
        ys = [p[1] for p in pairs]
        ax.plot(xs, ys, marker="o", label=method, linewidth=1.2, markersize=4)
    ax.axhline(0, linestyle="--", color="grey", linewidth=0.8)
    ax.set_xlabel("Effective episode (sorted by grid index)")
    ax.set_ylabel("Δ total_value")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Line plot → {path}")


# ── Effective-episode filtering / reprocessing ────────────────────────
def filter_effective(records: list[dict], threshold: float = 1.0) -> list[dict]:
    """Keep episodes where the starting grid is non-trivial (initial_total_value > threshold).
    Trivial grids are dominated by protected classes and no method can move the score."""
    return [r for r in records if float(r["initial_total_value"]) > threshold]


def write_effective_artifacts(records: list[dict], out_dir: Path,
                              threshold: float = 1.0) -> None:
    effective = filter_effective(records, threshold)
    if not effective:
        print(f"No episodes with initial_total_value > {threshold}; skipping effective artifacts")
        return

    per_method: dict[str, dict] = {}
    by_method: dict[str, list[dict]] = {}
    for r in effective:
        by_method.setdefault(r["method"], []).append(r)
    for method, rs in by_method.items():
        per_method[method] = aggregate(rs)

    write_per_episode_csv(effective, out_dir / "per_episode_effective.csv")
    write_summary_table(per_method, out_dir / "table_effective.csv")
    plot_box(effective, out_dir / "boxplot_effective.png",
             title=f"Exp III (effective grids, initial_total_value > {threshold})")
    plot_line(effective, out_dir / "lineplot.png")
    print(f"Effective artifacts → {out_dir} "
          f"({len(effective)} rows across {len(per_method)} methods)")


def _coerce_row(row: dict) -> dict:
    """Convert a CSV DictReader row back to typed fields used downstream."""
    STR_KEYS = {"method", "termination_reason"}
    INT_KEYS = {"episode_idx", "steps"}
    out: dict = {}
    for k, v in row.items():
        if k in STR_KEYS:
            out[k] = v
        elif k == "success":
            out[k] = str(v).lower() in ("true", "1")
        elif k in INT_KEYS:
            out[k] = int(v) if v not in ("", None) else 0
        else:
            out[k] = float(v) if v not in ("", None) else 0.0
    return out


def reprocess_from_csv(input_dir: Path, threshold: float = 1.0) -> None:
    """Load per_episode.csv from `input_dir` and re-emit effective-only artifacts."""
    src = Path(input_dir) / "per_episode.csv"
    if not src.exists():
        raise SystemExit(f"per_episode.csv not found at {src}")
    with open(src, newline="") as f:
        records = [_coerce_row(r) for r in csv.DictReader(f)]
    print(f"Loaded {len(records)} records from {src}")
    write_effective_artifacts(records, Path(input_dir), threshold)


# ── CLI ───────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--methods", nargs="+", default=list(ALL_METHODS),
                   choices=list(ALL_METHODS),
                   help="Subset of baselines to run.")
    p.add_argument("--ppo-model", type=str, default="models/gbdpkgiq/model.zip",
                   help="Path to MaskablePPO V2 checkpoint.")
    p.add_argument("--n-episodes", type=int, default=None,
                   help="If set, only run this many episodes (for smoke tests).")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory (default: data/processed/baselines/<timestamp>).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-plot", action="store_true")
    p.add_argument("--reprocess", type=str, default=None,
                   help="Path to an existing baseline output dir. When set, "
                        "skips episode rollouts and re-emits effective-only "
                        "artifacts (per_episode_effective.csv, table_effective.csv, "
                        "boxplot_effective.png, lineplot.png) from per_episode.csv.")
    p.add_argument("--feasibility-threshold", type=float, default=1.0,
                   help="initial_total_value strictly-greater-than threshold for "
                        "an episode to count as 'effective'. Trivial grids covered "
                        "by protected classes have initial_total_value ≈ 0 and no "
                        "method can move the score.")

    # Env overrides (defaults match Exp III / tier-1 training defaults)
    env_g = p.add_argument_group("Env (Exp III defaults)")
    env_g.add_argument("--n-augment", type=int, default=5)
    env_g.add_argument("--max-steps", type=int, default=500)
    env_g.add_argument("--spatial-scale", type=float, default=1.0)
    env_g.add_argument("--w-tree", type=float, default=1.0,
                       help="Match training's --w-tree so Greedy/GA score under the same reward")
    env_g.add_argument("--w-crop", type=float, default=3.0,
                       help="Match training's --w-crop")
    env_g.add_argument("--w-built", type=float, default=3.0,
                       help="Match training's --w-built")
    env_g.add_argument("--w-buf", type=float, default=5.0,
                       help="Match training's --w-buf (final, post-anneal value)")
    env_g.add_argument("--w-rip", type=float, default=0.0,
                       help="Match training's --w-rip (riparian-trees bonus)")
    env_g.add_argument("--riparian-mask", action="store_true",
                       help="Match training's --riparian-mask. When set, Greedy and PPO "
                            "both see a restricted action space (no crop/built at "
                            "water-adjacent cells), which is the intended fair "
                            "comparison for Phase C (§8.10.7).")
    env_g.add_argument("--use-et", action="store_true",
                       help="Use real ET values (must match training --use-et)")

    # GA
    ga_g = p.add_argument_group("GA hyperparameters")
    ga_g.add_argument("--ga-pop", type=int, default=30)
    ga_g.add_argument("--ga-gens", type=int, default=50)
    ga_g.add_argument("--ga-tournament-k", type=int, default=3)
    ga_g.add_argument("--ga-crossover-prob", type=float, default=0.8)
    ga_g.add_argument("--ga-mutation-prob", type=float, default=0.02)
    ga_g.add_argument("--ga-n-elite", type=int, default=2)
    ga_g.add_argument("--ga-verbose", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    # Reprocess-only shortcut: skip rollouts, re-emit effective artifacts.
    if args.reprocess:
        reprocess_from_csv(Path(args.reprocess), threshold=args.feasibility_threshold)
        return

    # Output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(data_dir, "processed", "baselines", stamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # Env
    env = make_eval_env(
        split="test_indices",
        use_et=args.use_et,
        spatial_scale=args.spatial_scale,
        w_tree=args.w_tree,
        w_crop=args.w_crop,
        w_built=args.w_built,
        w_buf=args.w_buf,
        w_rip=args.w_rip,
        riparian_mask=args.riparian_mask,
        max_steps=args.max_steps,
        n_augment=args.n_augment,
    )
    n_available = len(env.samples)
    n_episodes = args.n_episodes or n_available
    n_episodes = min(n_episodes, n_available)
    print(f"Test samples available: {n_available}; running {n_episodes} episodes/method")

    # Load PPO if requested
    ppo_model = None
    if "ppo" in args.methods:
        from sb3_contrib import MaskablePPO
        print(f"Loading PPO model: {args.ppo_model}")
        ppo_model = MaskablePPO.load(args.ppo_model, env=env)

    ga_kwargs = dict(
        pop_size=args.ga_pop,
        generations=args.ga_gens,
        tournament_k=args.ga_tournament_k,
        crossover_prob=args.ga_crossover_prob,
        mutation_prob=args.ga_mutation_prob,
        n_elite=args.ga_n_elite,
        verbose=args.ga_verbose,
    )

    all_records: list[dict] = []
    per_method: dict[str, dict] = {}

    for method in args.methods:
        print(f"\n=== Method: {method} ===")
        records = evaluate_method(method, args, env, ppo_model, ga_kwargs, verbose=True)
        summary = aggregate(records)
        per_method[method] = summary
        all_records.extend(records)
        print(f"  summary: Δ_mean={summary['delta_total_value_mean']:+.4f} "
              f"(±{summary['delta_total_value_std']:.4f}), "
              f"success_rate={summary['success_rate']:.2%}")

    # Write artifacts
    write_per_episode_csv(all_records, out_dir / "per_episode.csv")
    write_summary_table(per_method, out_dir / "table.csv")
    if not args.skip_plot:
        plot_box(all_records, out_dir / "boxplot.png")
        write_effective_artifacts(all_records, out_dir,
                                  threshold=args.feasibility_threshold)

    print(f"\nDone. Artifacts in {out_dir}")


if __name__ == "__main__":
    main()

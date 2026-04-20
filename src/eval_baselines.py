#!/usr/bin/env python3
"""Run baseline comparison against MaskablePPO V2 on Exp III.

Methods: random / greedy / genetic / ppo.
Output: per-episode CSV, summary-table CSV, box-plot PNG in an output dir.

Usage:
    # Full run (all 4 methods, ~48 episodes each)
    python src/eval_baselines.py --ppo-model models/gbdpkgiq/model.zip

    # Smoke test: 2 episodes, skip GA
    python src/eval_baselines.py \\
        --methods random greedy ppo \\
        --n-episodes 2 \\
        --ppo-model models/gbdpkgiq/model.zip

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

# Disable torch simplex validation (same rationale as train_v2.py)
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
    """Wraps a loaded MaskablePPO model to match the baseline act_fn interface."""

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


def plot_box(all_records: list[dict], path: Path) -> None:
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
    ax.set_title("Baseline comparison on Exp III (spatial_scale=1, regen-ag)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Box plot → {path}")


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

    # Env overrides (defaults match Exp III)
    env_g = p.add_argument_group("Env (Exp III defaults)")
    env_g.add_argument("--n-augment", type=int, default=5)
    env_g.add_argument("--max-steps", type=int, default=500)
    env_g.add_argument("--eco-variant", type=str, default="regen",
                       choices=["standard", "regen"])

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
        eco_variant=args.eco_variant,
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

    print(f"\nDone. Artifacts in {out_dir}")


if __name__ == "__main__":
    main()

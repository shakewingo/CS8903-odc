#!/usr/bin/env python3
"""Generate a zoom-in comparison figure across multiple agents.

Runs each configured experiment (PPO checkpoint or baseline method) over the
full 50×50 grid, reconstructs the final land-use allocation, and renders a
single figure with a map overview plus an (N_experiments+1) × n_regions zoom
grid (row 1 = original).

Usage:
    # Default: 3 PPO checkpoints from EXPERIMENTS constant
    python src/eval_zoom.py

    # Mixed methods via CLI override
    python src/eval_zoom.py --experiments \\
        "Greedy=greedy;Tier-1=ppo:ajhudfa1;Phase 4=ppo:7ybfz89t"

    # Manual region layout
    python src/eval_zoom.py --riparian-mask --deterministic --regions "2,7,5,5;34,0,5,5;2,30,5,5"

    # Smoke test (truncate samples per split)
    python src/eval_zoom.py --max-samples 1
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sb3_contrib import MaskablePPO

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import N_PIXELS_PER_CELL, data_dir, log_dir
from src.eval import (
    build_value_vecs, compute_diff_cells, make_env,
    reconstruct_map, run_inference,
)
from src.post_eda import plot_zoom_comparison
from src.utils import get_logger


# ── Experiment registry ───────────────────────────────────────────────
# (label, method, spec). Method ∈ {"ppo", "greedy", "random", "genetic"}.
# Spec is the wandb run id (PPO only); ignored/None for baselines.
EXPERIMENTS = [
    ("Exp I",   "ppo", "n2947wpo"),
    ("Exp II",  "ppo", "umgje44z"),
    ("Exp III", "ppo", "7ybfz89t"),
]

# Distinct colours for up to 4 regions; matches matplotlib tab palette.
REGION_COLORS = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate the zoom-in comparison figure",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--models-dir", type=str,
                   default=str(project_root / "models"))
    p.add_argument("--output", type=str,
                   default=str(project_root / "report/final_paper/figures/zoom_comparison.png"))
    p.add_argument("--region-size", type=int, default=4,
                   help="Side length (in grid cells) of each square zoom region "
                        "(used for random placement and for --regions entries "
                        "that don't supply a size)")
    p.add_argument("--n-regions", type=int, default=3)
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for random region placement (ignored when --regions is set)")
    p.add_argument(
        "--regions", type=str, default=None,
        help="Manual region list. Semicolon-separated entries, each "
             "'r0,c0' (uses --region-size for h=w) or 'r0,c0,size' "
             "or 'r0,c0,h,w'. Example: "
             "'34,0,5,5;7,24,5,5;45,18,5,5'. Overrides random placement.",
    )
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--riparian-mask", action="store_true",
                   help="Apply the hard riparian action mask at eval. Must match "
                        "the training setting of every PPO checkpoint listed in "
                        "--experiments; otherwise policies see a different "
                        "action space than they were trained under.")
    p.add_argument("--max-samples", type=int, default=None,
                   help="(Smoke test) limit inference to the first N indices per split")
    p.add_argument(
        "--experiments", type=str, default=None,
        help="Override the default experiment list. Semicolon-separated "
             "entries of the form 'Label=method' or 'Label=method:spec'. "
             "method ∈ {ppo, greedy, random, genetic}; spec is the wandb "
             "run_id (PPO only). Example: "
             "'Greedy=greedy;Tier-1=ppo:ajhudfa1;Phase 4=ppo:7ybfz89t'",
    )
    # GA knobs (kept modest — per-episode solve is expensive; 25 eps × pop·gens
    # env rollouts each).
    ga_g = p.add_argument_group("GA (only when an experiment uses method=genetic)")
    ga_g.add_argument("--ga-pop", type=int, default=20)
    ga_g.add_argument("--ga-gens", type=int, default=20)
    ga_g.add_argument("--ga-tournament-k", type=int, default=3)
    ga_g.add_argument("--ga-crossover-prob", type=float, default=0.8)
    ga_g.add_argument("--ga-mutation-prob", type=float, default=0.02)
    ga_g.add_argument("--ga-n-elite", type=int, default=2)
    return p.parse_args()


def parse_experiments(spec: str):
    """Parse '--experiments' override. Returns list of (label, method, spec)."""
    out = []
    for entry in spec.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        if "=" not in entry:
            raise ValueError(f"Experiment '{entry}' missing '='")
        label, rhs = entry.split("=", 1)
        label = label.strip()
        if ":" in rhs:
            method, run_spec = rhs.split(":", 1)
            method = method.strip()
            run_spec = run_spec.strip() or None
        else:
            method = rhs.strip()
            run_spec = None
        if method not in ("ppo", "greedy", "random", "genetic"):
            raise ValueError(f"Unknown method '{method}' in experiment '{entry}'")
        if method == "ppo" and not run_spec:
            raise ValueError(f"PPO experiment '{entry}' requires ':<run_id>'")
        out.append((label, method, run_spec))
    if not out:
        raise ValueError("--experiments must contain at least one entry")
    return out


def parse_regions(spec, default_size):
    """Parse a ``--regions`` spec string into the canonical region tuple list.

    Each ``;``-separated entry is ``r0,c0`` / ``r0,c0,size`` / ``r0,c0,h,w``.
    Labels (A, B, C, ...) and colours are assigned in order.
    """
    regions = []
    for i, entry in enumerate(spec.split(";")):
        entry = entry.strip()
        if not entry:
            continue
        parts = [int(x) for x in entry.split(",")]
        if len(parts) == 2:
            r0, c0 = parts
            h = w = default_size
        elif len(parts) == 3:
            r0, c0, s = parts
            h = w = s
        elif len(parts) == 4:
            r0, c0, h, w = parts
        else:
            raise ValueError(
                f"Invalid region entry '{entry}': expected 2, 3, or 4 ints"
            )
        label = chr(ord("A") + i)
        color = REGION_COLORS[i % len(REGION_COLORS)]
        regions.append((label, r0, c0, h, w, color))
    if not regions:
        raise ValueError("--regions must contain at least one entry")
    return regions


def _build_make_act_fn(method, spec, models_dir, deterministic, env_args, ga_kwargs, seed):
    """Return an `(env, ep) -> act_fn` factory for the requested method.

    `spec` is the PPO run_id (under --models-dir) when method="ppo", else None.
    GA agents receive `ga_kwargs`; Random uses `seed`.
    """
    if method == "ppo":
        model_path = Path(models_dir, spec, "model.zip")
        print(f"  loading {model_path}")
        dummy_env = make_env(env_args, "test_indices")
        model = MaskablePPO.load(str(model_path), env=dummy_env)

        def ppo_act(env):
            action, _ = model.predict(env.state,
                                      action_masks=env.action_masks(),
                                      deterministic=deterministic)
            return int(action)
        return lambda env, ep: ppo_act
    elif method == "greedy":
        from src.baselines.greedy_agent import GreedyAgent
        agent = GreedyAgent()
        return lambda env, ep: agent
    elif method == "random":
        from src.baselines.random_agent import RandomAgent
        agent = RandomAgent(seed=seed)
        return lambda env, ep: agent
    elif method == "genetic":
        from src.baselines.genetic_agent import GeneticAgent
        agent = GeneticAgent(seed=seed, **ga_kwargs)
        print(f"  GeneticAgent pop={ga_kwargs['pop_size']}, "
              f"gens={ga_kwargs['generations']} — slow (~1 solve per episode)")
        return lambda env, ep: agent.solve(env, ep)
    else:
        raise ValueError(f"unknown method: {method}")


def run_experiment(label, method, spec, data, initial_obs,
                   models_dir, deterministic=False, max_samples=None,
                   ga_kwargs=None, seed=0, riparian_mask=False):
    """Run one experiment over train+test splits and return
    `(reconstructed_50x50_final_map, set_of_modified_cells)`."""
    print(f"\n=== {label} ({method}{':' + spec if spec else ''}) ===")
    env_args = SimpleNamespace(riparian_mask=riparian_mask)
    make_act_fn = _build_make_act_fn(
        method, spec, models_dir, deterministic, env_args,
        ga_kwargs or {}, seed,
    )

    combined = {}
    for split in ("test_indices", "train_indices"):
        env = make_env(env_args, split)
        data_view = data
        if max_samples is not None:
            data_view = {k: data[k] for k in data.files}
            data_view[split] = data[split][:max_samples]
        combined.update(run_inference(make_act_fn, env, data_view, split))

    _, final = reconstruct_map(data, combined)
    diff_cells = compute_diff_cells(initial_obs, final)
    print(f"  modified cells: {len(diff_cells)}")
    return final, diff_cells


def random_regions(n_rows, n_cols, size, n, rng, activity_mask=None,
                   min_active_frac=0.5):
    """Pick ``n`` non-overlapping square regions of the given size.

    When ``activity_mask`` (an (n_rows, n_cols) bool array marking cells that
    were modified by at least one experiment) is provided, candidate windows
    must contain ``min_active_frac`` × size² active cells — so we don't land
    on pure water / bare background with nothing to compare.
    """
    regions = []
    tries = 0
    threshold = int(np.ceil(min_active_frac * size * size))
    while len(regions) < n and tries < 2000:
        r0 = int(rng.integers(0, n_rows - size + 1))
        c0 = int(rng.integers(0, n_cols - size + 1))
        overlap = any(
            not (r0 + size <= r or r0 >= r + size or
                 c0 + size <= c or c0 >= c + size)
            for (_, r, c, _, _, _) in regions
        )
        if overlap:
            tries += 1
            continue
        if activity_mask is not None:
            window = activity_mask[r0:r0 + size, c0:c0 + size]
            if int(window.sum()) < threshold:
                tries += 1
                continue
        label = chr(ord("A") + len(regions))
        color = REGION_COLORS[len(regions) % len(REGION_COLORS)]
        regions.append((label, r0, c0, size, size, color))
        tries += 1
    if len(regions) < n:
        raise RuntimeError(f"Could not place {n} non-overlapping regions "
                           f"of size {size} in {n_rows}x{n_cols}")
    return regions


def main():
    args = parse_args()
    import src.train as train_module

    # Inference-time logger: keep quiet, write to a dedicated file.
    train_module.logger = get_logger(
        "eval_zoom", stream=False, level="WARNING",
        log_file=str(Path(log_dir, "eval_zoom.log")),
    )

    # Shared value vectors — identical across experiments so the zoom grid
    # uses a single ESV colormap. Also sets train_module.{ECO,ET}_MOD.
    eco_per_class, et_per_class = build_value_vecs(use_et=False)
    value_vec = eco_per_class + et_per_class

    # Load dataset once
    data = np.load(Path(data_dir, "processed", "rl_dataset.npz"))
    initial_obs = data["pixel_counts"].astype(np.float32) / N_PIXELS_PER_CELL
    n_rows, n_cols, _ = initial_obs.shape

    # Resolve experiment list: CLI override takes precedence over default.
    experiments = parse_experiments(args.experiments) if args.experiments else EXPERIMENTS

    ga_kwargs = dict(
        pop_size=args.ga_pop,
        generations=args.ga_gens,
        tournament_k=args.ga_tournament_k,
        crossover_prob=args.ga_crossover_prob,
        mutation_prob=args.ga_mutation_prob,
        n_elite=args.ga_n_elite,
    )

    # Run inference for each experiment
    finals_by_exp = {}
    diffs_by_exp = {}
    for label, method, spec in experiments:
        final, diff_cells = run_experiment(
            label, method, spec, data, initial_obs,
            models_dir=args.models_dir,
            deterministic=args.deterministic,
            max_samples=args.max_samples,
            ga_kwargs=ga_kwargs,
            seed=args.seed,
            riparian_mask=args.riparian_mask,
        )
        finals_by_exp[label] = final
        diffs_by_exp[label] = diff_cells

    # Region selection
    if args.regions:
        regions = parse_regions(args.regions, default_size=args.region_size)
        # Validate that manual regions fit inside the grid.
        for (label, r0, c0, h, w, _) in regions:
            if r0 < 0 or c0 < 0 or r0 + h > n_rows or c0 + w > n_cols:
                raise ValueError(
                    f"Region {label} ({r0},{c0},{h},{w}) is out of bounds "
                    f"for a {n_rows}x{n_cols} grid"
                )
    else:
        # Build a union activity mask: cells modified by ANY experiment,
        # then pick random non-overlapping windows with ≥50% activity.
        activity = np.zeros((n_rows, n_cols), dtype=bool)
        for diff_set in diffs_by_exp.values():
            for (rr, cc) in diff_set:
                activity[rr, cc] = True
        print(f"\nActive cells (union across experiments): {int(activity.sum())}")
        rng = np.random.default_rng(args.seed)
        regions = random_regions(n_rows, n_cols, args.region_size,
                                 args.n_regions, rng,
                                 activity_mask=activity,
                                 min_active_frac=0.5)

    print("\nRegions:")
    for (label, r0, c0, h, w, color) in regions:
        print(f"  {label}: rows {r0}..{r0+h-1}, cols {c0}..{c0+w-1}  ({color})")

    # Plot
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_zoom_comparison(
        initial_obs=initial_obs,
        finals_by_exp=finals_by_exp,
        regions=regions,
        value_vec=value_vec,
        diffs_by_exp=diffs_by_exp,
        save_path=out_path,
    )
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()


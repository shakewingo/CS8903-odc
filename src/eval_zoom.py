#!/usr/bin/env python3
"""
Generate the zoom-in comparison figure for the flagship experiments.

Loads the trained MaskablePPO checkpoints, runs inference on the full 50×50
grid, reconstructs each experiment's final land-use allocation, and renders
a single figure with:

  1. An overview of the original allocation with coloured locator boxes.
  2. A (4 × n_regions) zoom grid: Original / Exp I / Exp II / Exp III rows,
     one column per region.

Regions are chosen randomly (seeded); the intent is to let the author check
the layout before committing to curated coordinates.

Usage:
    python src/eval_zoom.py                       # random regions, full eval
    python src/eval_zoom.py --region-size 5       # custom zoom size
    python src/eval_zoom.py --seed 0              # reproducible random pick
    python src/eval_zoom.py --max-samples 1       # smoke test
    python src/eval_zoom.py --regions "9,4;24,6;23,17,4,6"  # manual regions
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
# (label, wandb run id / model dir, spatial_scale)
EXPERIMENTS = [
    ("Exp I",   "6f0ta58i", 0.0),
    ("Exp II",  "2sk0pnp3", 1.0),
    ("Exp III", "pzy2mxod", 1.0),
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
    p.add_argument("--max-samples", type=int, default=None,
                   help="(Smoke test) limit inference to the first N indices per split")
    return p.parse_args()


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


def run_experiment(label, run_id, spatial_scale, data, initial_obs,
                   models_dir, deterministic=False, max_samples=None):
    """Load one checkpoint, run inference over both splits, and return
    ``(reconstructed_final_map, set_of_modified_cells)``.

    All env / inference / reconstruction plumbing is reused from
    :mod:`src.eval` — this function is just the per-experiment glue.
    """
    model_path = Path(models_dir, run_id, "model.zip")
    print(f"\n=== {label} ({run_id}) ===")
    print(f"  loading {model_path}")

    # The env kwargs here affect reward computation but not the agent's
    # action choices, so only spatial_scale matters for consistency with
    # training; everything else falls back to make_env's defaults.
    env_args = SimpleNamespace(spatial_scale=spatial_scale)
    dummy_env = make_env(env_args, "test_indices")
    model = MaskablePPO.load(str(model_path), env=dummy_env)

    combined = {}
    for split in ("test_indices", "train_indices"):
        env = make_env(env_args, split)
        data_view = data
        if max_samples is not None:
            # Truncate just this split; leave the others untouched.
            data_view = {k: data[k] for k in data.files}
            data_view[split] = data[split][:max_samples]
        combined.update(
            run_inference(model, env, data_view, split, deterministic)
        )

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

    # Run inference for each experiment
    finals_by_exp = {}
    diffs_by_exp = {}
    for label, run_id, spatial_scale in EXPERIMENTS:
        final, diff_cells = run_experiment(
            label, run_id, spatial_scale, data, initial_obs,
            models_dir=args.models_dir,
            deterministic=args.deterministic,
            max_samples=args.max_samples,
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


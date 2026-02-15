"""dataset.py – Generate RL training / test datasets
from the 50×50 economic + ET grids.

Each cell in the 50×50 grid is 5×5 downsampled pixels (500 m × 500 m).
The breakdown stores land-cover fractions per cell:
    [(class_id, fraction), ...]

Rewards are computed as:
    reward = (eco - mean_eco) / std_eco + (et - mean_et) / std_et
where mean/std are computed across all valid cells.

Samples are non-overlapping 10×10 cell blocks split 70/30 into
train / test.
"""

from pathlib import Path

import numpy as np

from config import *
from utils import get_logger
from post_eda import compute_economic_grid, compute_et_grid

logger = get_logger(__name__)

def build_grid(center, **grid_kwargs):
    """Compute eco + ET grids and merge into dense numpy arrays.

    Returns
    -------
    fractions : ndarray, shape (n_rows, n_cols, N_CLASSES)
        Land-cover fraction per class per cell.
    eco_values : ndarray, shape (n_rows, n_cols)
        Weighted economic value (USD/ha/yr) per cell.
    et_values : ndarray, shape (n_rows, n_cols)
        Weighted ET (kg/m²/yr) per cell.
    valid_mask : ndarray, shape (n_rows, n_cols), dtype bool
        True for cells with valid data.
    coords : ndarray, shape (n_rows, n_cols, 2)
        (lat, lon) for each cell centre.
    """
    kw = {**grid_kwargs}
    n_rows, n_cols = kw["n_rows"], kw["n_cols"]

    logger.info("Computing economic grid...")
    eco_grid, eco_meta = compute_economic_grid(
        center_lat=center[0], center_lon=center[1], **kw
    )

    logger.info("Computing ET grid...")
    et_grid, et_meta = compute_et_grid(
        center_lat=center[0], center_lon=center[1], **kw
    )

    fractions = np.zeros((n_rows, n_cols, N_CLASSES), dtype=np.float32)
    eco_values = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    et_values = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    valid_mask = np.zeros((n_rows, n_cols), dtype=bool)
    coords = np.zeros((n_rows, n_cols, 2), dtype=np.float64)

    for r in range(n_rows):
        for c in range(n_cols):
            eco_cell = eco_grid[r][c]
            et_cell = et_grid[r][c]

            if eco_cell is None or et_cell is None:
                continue

            valid_mask[r, c] = True
            eco_values[r, c] = eco_cell["economic_value"]
            et_values[r, c] = et_cell["et_value"]
            coords[r, c, 0] = eco_cell["center_lat"]
            coords[r, c, 1] = eco_cell["center_lon"]

            for class_id, frac in eco_cell["breakdown"]:
                fractions[r, c, class_id] = frac

    n_valid = int(valid_mask.sum())
    logger.info(f"Valid cells: {n_valid} / {n_rows * n_cols}")
    return fractions, eco_values, et_values, valid_mask, coords


def normalize_and_compute_rewards(eco_values, et_values, valid_mask):
    """Normalize eco/ET using cell-level stats and compute per-cell reward.

    reward_i = (eco_i - μ_eco) / σ_eco  +  (et_i - μ_et) / σ_et

    Returns
    -------
    rewards : ndarray, shape same as eco_values
        Per-cell reward (NaN for invalid cells).
    norm_stats : dict
        {mean_eco, std_eco, mean_et, std_et} for reuse at inference.
    """
    valid_eco = eco_values[valid_mask]
    valid_et = et_values[valid_mask]

    mean_eco, std_eco = float(valid_eco.mean()), float(valid_eco.std())
    mean_et, std_et = float(valid_et.mean()), float(valid_et.std())

    logger.info(f"Eco  — mean: {mean_eco:.2f}, std: {std_eco:.2f}")
    logger.info(f"ET   — mean: {mean_et:.2f},  std: {std_et:.2f}")

    norm_stats = {
        "mean_eco": mean_eco,
        "std_eco": std_eco,
        "mean_et": mean_et,
        "std_et": std_et,
    }

    rewards = np.full_like(eco_values, np.nan)
    rewards[valid_mask] = (
        (eco_values[valid_mask] - mean_eco) / std_eco
        + (et_values[valid_mask] - mean_et) / std_et
    )

    valid_rewards = rewards[valid_mask]
    logger.info(
        f"Rewards — min: {valid_rewards.min():.3f}, "
        f"max: {valid_rewards.max():.3f}, "
        f"mean: {valid_rewards.mean():.3f}"
    )
    return rewards, norm_stats


def split_dataset(
    valid_mask,
    n_rows=50,
    n_cols=50,
    sample_size=SAMPLE_SIZE,
    train_ratio=TRAIN_RATIO,
    seed=SEED,
):
    """Partition the grid into train / test samples (no validation set).

    1. Divide the grid into non-overlapping blocks of *sample_size × sample_size*.
    2. Shuffle and split 70/30 into train / test.

    Returns
    -------
    train_indices : ndarray, shape (N_train, 2)
        (row_start, col_start) of each 10×10 train block.
    test_indices : ndarray, shape (N_test, 2)
        (row_start, col_start) of each 10×10 test block.
    """
    rng = np.random.RandomState(seed)

    blocks_per_row = n_rows // sample_size
    blocks_per_col = n_cols // sample_size

    all_blocks = [
        (br * sample_size, bc * sample_size)
        for br in range(blocks_per_row)
        for bc in range(blocks_per_col)
    ]
    rng.shuffle(all_blocks)

    n_train = int(len(all_blocks) * train_ratio)
    train_indices = np.array(all_blocks[:n_train])
    test_indices = np.array(all_blocks[n_train:])

    logger.info(
        f"Non-overlapping blocks: {len(all_blocks)} "
        f"({blocks_per_row}×{blocks_per_col})"
    )
    logger.info(
        f"Split — train: {len(train_indices)}, "
        f"test: {len(test_indices)}"
    )
    return train_indices, test_indices


def save_dataset(
    fractions,
    eco_values,
    et_values,
    rewards,
    valid_mask,
    coords,
    norm_stats,
    train_indices,
    test_indices,
    output_dir=None,
):
    """Persist the RL dataset as a compressed ``.npz`` archive.
    """
    if output_dir is None:
        output_dir = Path(data_dir, "processed")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / "rl_dataset.npz"

    np.savez_compressed(
        npz_path,
        fractions=fractions,
        eco_values=eco_values,
        et_values=et_values,
        rewards=rewards,
        valid_mask=valid_mask,
        coords=coords,
        norm_stats=np.array(
            [
                norm_stats["mean_eco"],
                norm_stats["std_eco"],
                norm_stats["mean_et"],
                norm_stats["std_et"],
            ]
        ),
        train_indices=train_indices,
        test_indices=test_indices,
    )

    logger.info(f"Dataset saved to {npz_path}")
    logger.info(f"  fractions:     {fractions.shape}")
    logger.info(f"  eco_values:    {eco_values.shape}")
    logger.info(f"  et_values:     {et_values.shape}")
    logger.info(f"  rewards:       {rewards.shape}")
    logger.info(f"  train samples: {train_indices.shape[0]}")
    logger.info(f"  test samples:  {test_indices.shape[0]}")
    return npz_path

if __name__ == "__main__":
    fractions, eco_values, et_values, valid_mask, coords = build_grid(CENTER, **GRID_KWARGS)

    rewards, norm_stats = normalize_and_compute_rewards(
        eco_values, et_values, valid_mask
    )

    train_idx, test_idx = split_dataset(
        valid_mask,
        n_rows=GRID_KWARGS["n_rows"],
        n_cols=GRID_KWARGS["n_cols"],
    )

    save_dataset(
        fractions, eco_values, et_values, rewards,
        valid_mask, coords, norm_stats,
        train_idx, test_idx,
    )

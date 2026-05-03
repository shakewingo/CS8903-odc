"""Shared helpers for converting numpy arrays to JSON-serialisable dicts.

Used by both grid_service (pixel_counts → grid JSON) and
infer_service (observation arrays → before/after grids + ESV changes).
"""

import numpy as np

from src.config import ECO_VALUES, LAND_COVER_LABELS, N_PIXELS_PER_CELL

ALL_CLASS_IDS = sorted(LAND_COVER_LABELS.keys())

FRAC_THRESHOLD = 0.001   # omit classes below this fraction
DELTA_THRESHOLD = 0.01   # minimum fractional change for from/to type detection


def pixel_counts_to_grid_json(pixel_counts, eco_values, coords):
    """Convert dense numpy arrays into the JSON grid structure the frontend expects.

    Mirrors the logic in ``scripts/export_web_data.py:export_grid_data()``.

    Parameters
    ----------
    pixel_counts : ndarray (n_rows, n_cols, N_CLASSES)
    eco_values   : ndarray (n_rows, n_cols)
    coords       : ndarray (n_rows, n_cols, 2)  — (lat, lon) per cell

    Returns
    -------
    list[list[dict]]  — ``[{f: {cls: frac}, e: esv, lat, lng}, ...]``
    """
    n_rows, n_cols = pixel_counts.shape[:2]
    grid = []
    for r in range(n_rows):
        row = []
        for c in range(n_cols):
            counts = pixel_counts[r, c]
            total = max(int(counts.sum()), 1)
            fractions = {
                str(cls_id): round(float(counts[cls_id]) / total, 4)
                for cls_id in ALL_CLASS_IDS
                if cls_id < len(counts) and float(counts[cls_id]) / total > FRAC_THRESHOLD
            }
            row.append({
                "f": fractions,
                "e": round(float(eco_values[r, c]), 1),
                "lat": round(float(coords[r, c, 0]), 6),
                "lng": round(float(coords[r, c, 1]), 6),
            })
        grid.append(row)
    return grid


def obs_to_fractions_grid(obs):
    """Convert a (n_rows, n_cols, N_CLASSES) fraction array into per-cell dicts.

    Mirrors ``grid_to_fractions()`` in ``scripts/export_web_data.py``.

    Parameters
    ----------
    obs : ndarray (n_rows, n_cols, N_CLASSES), values in [0, 1]

    Returns
    -------
    list[list[dict]]  — ``[{cls_str: frac}, ...]``
    """
    n_rows, n_cols = obs.shape[:2]
    grid = []
    for r in range(n_rows):
        row = []
        for c in range(n_cols):
            fracs = {
                str(cls_id): round(float(obs[r, c, cls_id]), 4)
                for cls_id in ALL_CLASS_IDS
                if cls_id < obs.shape[2] and float(obs[r, c, cls_id]) > FRAC_THRESHOLD
            }
            row.append(fracs)
        grid.append(row)
    return grid


def compute_esv_changes(initial_obs, final_obs, changed_cells, esv_map):
    """Compute per-cell ESV deltas and dominant from→to land-cover types.

    Mirrors the ESV change block in ``scripts/export_web_data.py``.

    Parameters
    ----------
    initial_obs, final_obs : ndarray (n_rows, n_cols, N_CLASSES)
    changed_cells          : set of (row, col) tuples
    esv_map                : dict {class_id: USD/ha/yr}

    Returns
    -------
    list[dict]  — sorted descending by ``esvDelta``
    """
    changes = []
    for r, c in changed_cells:
        before_esv = sum(
            float(initial_obs[r, c, cls_id]) * esv_map.get(cls_id, 0)
            for cls_id in ALL_CLASS_IDS if cls_id < initial_obs.shape[2]
        )
        after_esv = sum(
            float(final_obs[r, c, cls_id]) * esv_map.get(cls_id, 0)
            for cls_id in ALL_CLASS_IDS if cls_id < final_obs.shape[2]
        )

        # Dominant decrease / increase class
        max_dec_cls, max_dec = 0, 0.0
        max_inc_cls, max_inc = 0, 0.0
        for cls_id in ALL_CLASS_IDS:
            if cls_id >= initial_obs.shape[2]:
                continue
            delta = float(final_obs[r, c, cls_id] - initial_obs[r, c, cls_id])
            if delta < -DELTA_THRESHOLD and abs(delta) > abs(max_dec):
                max_dec, max_dec_cls = delta, cls_id
            if delta > DELTA_THRESHOLD and delta > max_inc:
                max_inc, max_inc_cls = delta, cls_id

        changes.append({
            "row": r, "col": c,
            "fromType": max_dec_cls,
            "toType": max_inc_cls,
            "esvDelta": round(after_esv - before_esv, 1),
        })

    changes.sort(key=lambda x: x["esvDelta"], reverse=True)
    return changes


def build_results_json(experiment, initial_obs, final_obs, changed_cells, esv_map):
    """Assemble the full inference response dict.

    Parameters
    ----------
    experiment   : str
    initial_obs  : ndarray (n_rows, n_cols, N_CLASSES)
    final_obs    : ndarray (n_rows, n_cols, N_CLASSES)
    changed_cells: set of (row, col)
    esv_map      : dict {class_id: USD/ha/yr}
    """
    before_grid = obs_to_fractions_grid(initial_obs)
    after_grid = obs_to_fractions_grid(final_obs)
    esv_changes = compute_esv_changes(initial_obs, final_obs, changed_cells, esv_map)

    n_cells = initial_obs.shape[0] * initial_obs.shape[1]
    return {
        "experiment": experiment,
        "before": before_grid,
        "after": after_grid,
        "changedCells": [list(cc) for cc in sorted(changed_cells)],
        "esvChanges": esv_changes,
        "summary": {
            "cellsChanged": len(changed_cells),
            "totalEsvGain": round(sum(x["esvDelta"] for x in esv_changes), 1),
            "maxCellGain": round(max((x["esvDelta"] for x in esv_changes), default=0), 1),
            "pctAreaModified": round(len(changed_cells) / n_cells * 100, 1),
        },
    }

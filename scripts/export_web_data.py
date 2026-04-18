"""Export rl_dataset and model inference results as JSON for the web app."""
import json
import sys
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.config import ECO_VALUES, ET_VALUES, LAND_COVER_LABELS, LAND_COVER_COLORS, N_PIXELS_PER_CELL
from src.train import MODIFIABLE_CLASSES

OUT_DIR = os.path.join(str(project_root), 'web-app', 'public', 'data')
DATA_PATH = os.path.join(str(project_root), 'data', 'processed', 'rl_dataset.npz')

ALL_CLASS_IDS = sorted(LAND_COVER_LABELS.keys())


def export_grid_data():
    """Export the 50x50 grid as JSON with pixel fractions and ESV per cell."""
    d = np.load(DATA_PATH)
    pixel_counts = d['pixel_counts']  # (50, 50, 12)
    eco_values = d['eco_values']       # (50, 50)
    coords = d['coords']              # (50, 50, 2)
    valid_mask = d['valid_mask']       # (50, 50)

    grid = []
    for r in range(50):
        row_data = []
        for c in range(50):
            counts = pixel_counts[r, c]
            total = counts.sum()
            if total == 0:
                total = 1
            fractions = {}
            for idx, cls_id in enumerate(ALL_CLASS_IDS):
                if idx < len(counts):
                    frac = float(counts[idx]) / float(total)
                    if frac > 0.001:
                        fractions[str(cls_id)] = round(frac, 4)
            row_data.append({
                'f': fractions,
                'e': round(float(eco_values[r, c]), 1),
                'lat': round(float(coords[r, c, 0]), 6),
                'lng': round(float(coords[r, c, 1]), 6),
            })
        grid.append(row_data)

    out = {
        'rows': 50,
        'cols': 50,
        'grid': grid,
        'classLabels': {str(k): v for k, v in LAND_COVER_LABELS.items()},
        'classColors': {str(k): v for k, v in LAND_COVER_COLORS.items()},
        'ecoValues': {str(k): v for k, v in ECO_VALUES.items()},
    }

    path = os.path.join(OUT_DIR, 'grid_data.json')
    with open(path, 'w') as f:
        json.dump(out, f)
    print(f'Exported grid data to {path} ({os.path.getsize(path) / 1024:.0f} KB)')


def export_inference_results():
    """Run inference for all 3 experiments and export before/after as JSON."""
    from src.eval import reconstruct_map, run_inference, make_env, build_value_vecs
    from src.utils import get_logger
    from sb3_contrib import MaskablePPO
    import src.train as train_module

    # Initialize the logger that train.py expects at module level
    train_module.logger = get_logger("export", level="WARNING")

    EXPERIMENTS = [
        ('exp1', '6f0ta58i', 0.0),
        ('exp2', '2sk0pnp3', 1.0),
        ('exp3', 'pzy2mxod', 1.0),
    ]

    d = np.load(DATA_PATH)

    # Build value vecs once (sets globals in src.train)
    build_value_vecs(use_et=False, assign_globals=True)

    for exp_id, run_id, spatial_scale in EXPERIMENTS:
        print(f'\n=== Running inference for {exp_id} (run {run_id}) ===')
        model_path = os.path.join(str(project_root), 'models', run_id, 'model.zip')

        if not os.path.exists(model_path):
            print(f'  Model not found at {model_path}, skipping')
            continue

        try:
            env_args = SimpleNamespace(spatial_scale=spatial_scale)
            dummy_env = make_env(env_args, 'test_indices')
            model = MaskablePPO.load(model_path, env=dummy_env)

            combined = {}
            for split in ('test_indices', 'train_indices'):
                env = make_env(env_args, split)
                split_results = run_inference(model, env, d, split, deterministic=True)
                combined.update(split_results)

            initial_obs, final_obs = reconstruct_map(d, combined)
            # initial_obs and final_obs are (50, 50, N_CLASSES) in fraction space

            # Convert to per-cell fractions dict
            def grid_to_fractions(obs):
                grid = []
                for r in range(50):
                    row = []
                    for c in range(50):
                        fracs = {}
                        for idx, cls_id in enumerate(ALL_CLASS_IDS):
                            if idx < obs.shape[2]:
                                frac = float(obs[r, c, idx])
                                if frac > 0.001:
                                    fracs[str(cls_id)] = round(frac, 4)
                        row.append(fracs)
                    grid.append(row)
                return grid

            before_grid = grid_to_fractions(initial_obs)
            after_grid = grid_to_fractions(final_obs)

            # Find changed cells
            diff_mask = ~np.isclose(initial_obs, final_obs, atol=1e-4)
            changed_cells = {(int(r), int(c)) for r, c, _ in np.argwhere(diff_mask)}

            # Compute ESV changes
            esv_changes = []
            for r, c in changed_cells:
                before_esv = sum(
                    float(before_grid[r][c].get(str(cls_id), 0)) * ECO_VALUES.get(cls_id, 0)
                    for cls_id in ALL_CLASS_IDS
                )
                after_esv = sum(
                    float(after_grid[r][c].get(str(cls_id), 0)) * ECO_VALUES.get(cls_id, 0)
                    for cls_id in ALL_CLASS_IDS
                )

                # Find dominant from/to type
                max_decrease_cls = 0
                max_decrease = 0
                max_increase_cls = 0
                max_increase = 0
                for idx, cls_id in enumerate(ALL_CLASS_IDS):
                    if idx < initial_obs.shape[2]:
                        delta = float(final_obs[r, c, idx] - initial_obs[r, c, idx])
                        if delta < -0.01 and abs(delta) > abs(max_decrease):
                            max_decrease = delta
                            max_decrease_cls = cls_id
                        if delta > 0.01 and delta > max_increase:
                            max_increase = delta
                            max_increase_cls = cls_id

                esv_changes.append({
                    'row': r, 'col': c,
                    'fromType': max_decrease_cls,
                    'toType': max_increase_cls,
                    'esvDelta': round(after_esv - before_esv, 1),
                })

            esv_changes.sort(key=lambda x: x['esvDelta'], reverse=True)

            out = {
                'experiment': exp_id,
                'runId': run_id,
                'spatialScale': spatial_scale,
                'before': before_grid,
                'after': after_grid,
                'changedCells': [list(cc) for cc in sorted(changed_cells)],
                'esvChanges': esv_changes,
                'summary': {
                    'cellsChanged': len(changed_cells),
                    'totalEsvGain': round(sum(x['esvDelta'] for x in esv_changes), 1),
                    'maxCellGain': round(max((x['esvDelta'] for x in esv_changes), default=0), 1),
                    'pctAreaModified': round(len(changed_cells) / 2500 * 100, 1),
                },
            }

            path = os.path.join(OUT_DIR, f'results_{exp_id}.json')
            with open(path, 'w') as f:
                json.dump(out, f)
            print(f'  Exported to {path} ({os.path.getsize(path) / 1024:.0f} KB)')
            print(f'  Changed cells: {len(changed_cells)}')

        except Exception as e:
            print(f'  Error: {e}')
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    export_grid_data()
    export_inference_results()
    print('\nDone!')

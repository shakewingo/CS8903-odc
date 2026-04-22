from pathlib import Path

root_dir = Path(__file__).parent.parent
data_dir = Path(root_dir, 'data')
script_dir = Path(root_dir, 'src')
log_dir  = Path(data_dir).parent / "log"
model_dir = Path(data_dir).parent / "model"

LAND_COVER_LABELS = {
    1: 'Water',
    2: 'Trees',
    4: 'Flooded',
    5: 'Crops',
    7: 'Built Area',
    8: 'Bare Ground',
    9: 'Snow/Ice',
    10: 'Clouds',
    11: 'Rangeland',
}

LAND_COVER_COLORS = {
    1: '#419bdf',
    2: '#397d49',
    4: '#7a87c6',
    5: '#e49635',
    7: '#c4281b',
    8: '#a59b8f',
    9: '#a8ebff',
    10: '#616161',
    11: '#e3e2c3',
}

# Per-class annual ET (kg/m²/yr), extracted from
# data/processed/et_per_landcover_2024.json
ET_VALUES = {
    1:  616.93,   # Water
    2:  933.57,   # Trees
    4:  767.62,   # Flooded
    5:  675.66,   # Crops
    7:  648.95,   # Built Area
    8:  591.53,   # Bare Ground
    9:  0,        # Snow/Ice — no data
    10: 845.87,   # Clouds
    11: 745.94,   # Rangeland
}

# Per-class economic value (USD / ha / year). Crops is boosted 1.35× to
# reflect regenerative-agriculture uplift used in the headline reward config.
ECO_VALUES = {
    1:  554,          # Water (anchor)
    2:  238,          # Trees
    4:  1136,         # Flooded
    5:  246 * 1.35,   # Crops (regen-ag uplift)
    7:  295,          # Built Area
    8:  0,            # Bare Ground
    9:  0,            # Snow/Ice
    10: 0,            # Clouds
    11: 184,          # Rangeland
}

# Classes the RL agent cannot modify (water bodies, wetlands, clouds, snow).
PROTECTED_CLASSES = frozenset({0, 1, 3, 4, 6, 9, 10})

# Study-area centre (Lake Malawi sub-region) used as the AOI anchor for grid
# building. Bounds are derived by the dataset builder from GRID_KWARGS.
CENTER = (-13.934564, 34.542859)
N_CLASSES = max(LAND_COVER_LABELS.keys()) + 1  # 12 class IDs (0-11)
GRID_KWARGS = {
    "year": 2024,
    "sample_rate": 0.1,
    "grid_size": 5,   # 5×5 pixels per cell (downsampled)
    "n_rows": 50,     # 50×50 cells total
    "n_cols": 50,
}
N_PIXELS_PER_CELL = GRID_KWARGS['grid_size'] ** 2
SAMPLE_SIZE = 10  # Each RL episode sees a 10×10 cell block
TRAIN_RATIO = 0.7
SEED = 42

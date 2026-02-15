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

# Average economic value per land-cover class (USD / ha / year)
# Sources: SeedCo Malawi, MwAPATA Institute, ecosystem-service valuations
ECONOMIC_VALUES = {
    1:  554,    # Water        – fisheries + ecosystem services
    2:  325,    # Trees/Forest – firewood, carbon, water regulation
    4:  554,    # Flooded/Wetlands – same as water/wetlands valuation
    5:  400,    # Crops        – rainfed maize midpoint ($300-$500)
    7:  2000,   # Built Area   – urban real-estate (constraint: very high)
    8:  25,     # Bare Ground  – minimal value
    9:  0,      # Snow/Ice     – no economic value
    10: 0,      # Clouds       – no economic value (masked)
    11: 75,     # Rangeland    – low-intensity grazing
}

# Land-cover classes that cannot be modified by the RL agent
PROTECTED_CLASSES = frozenset({1, 3, 4, 6, 9, 10})  # Water, Flooded, Snow/Ice, Clouds


# Training params setup
CENTER = (-13.934564, 34.542859)
N_CLASSES = max(LAND_COVER_LABELS.keys()) + 1
GRID_KWARGS = {
    "year": 2024,
    "sample_rate": 0.1,
    "grid_size": 5, # 5x5 pixels per cell
    "n_rows": 50, # 50*50 cells as total area
    "n_cols": 50,
}
SAMPLE_SIZE = 10  # 10×10 cells per sample
TRAIN_RATIO = 0.7
SEED = 42

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

# Extracted from data/processed/et_per_landcover_2024.json
ET_VALUES = {
    1:  616.93,   # Water
    2:  933.57,   # Trees
    4:  767.62,   # Flooded
    5:  675.66,   # Crops
    7:  648.95,   # Built Area
    8:  591.53,   # Bare Ground
    9:  0,        # Snow/Ice     – no data
    10: 845.87,   # Clouds
    11: 745.94,   # Rangeland
}

# Economic value per land-cover class (USD / ha / year)
# Costanza 2014 ratios scaled to Malawi anchor (Zuze 2013: Lake Chiuta wetland $554/ha/yr)
# ECO_VALUES = {
#     1:  554,    # Water (Lakes/Rivers)  – anchor value, ratio 1.00
#     2:  238,    # Trees (Tropical Forest) – ratio 0.43
#     4:  1136,   # Flooded (Inland Wetlands) – ratio 2.05
#     5:  246,    # Crops (Cropland) – ratio 0.44
#     7:  295,    # Built Area (Urban) – ratio 0.53
#     8:  0,      # Bare Ground (Desert) – ~0
#     9:  0,      # Snow/Ice – no economic value
#     10: 0,      # Clouds – no economic value (masked)
#     11: 184,    # Rangeland (Grass/Rangelands) – ratio 0.33
# }

# # With regenerative agriculture considered
ECO_VALUES = {
    1:  554,    # Water (Lakes/Rivers)  – anchor value, ratio 1.00
    2:  238,    # Trees (Tropical Forest) – ratio 0.43
    4:  1136,   # Flooded (Inland Wetlands) – ratio 2.05
    5:  246 * 1.35,    # Crops (Cropland) – ratio 0.44
    7:  295,    # Built Area (Urban) – ratio 0.53
    8:  0,      # Bare Ground (Desert) – ~0
    9:  0,      # Snow/Ice – no economic value
    10: 0,      # Clouds – no economic value (masked)
    11: 184,    # Rangeland (Grass/Rangelands) – ratio 0.33
}



# Land-cover classes that cannot be modified by the RL agent
PROTECTED_CLASSES = frozenset({0, 1, 3, 4, 6, 9, 10})  # Water, Flooded, Snow/Ice, Clouds

# Training params setup
# Northern Boundary (Max Latitude): -13.822144
# Southern Boundary (Min Latitude): -14.046984
# Eastern Boundary (Max Longitude): 34.658687
# Western Boundary (Min Longitude): 34.427031
CENTER = (-13.934564, 34.542859) 
N_CLASSES = max(LAND_COVER_LABELS.keys()) + 1  # 12 classes (0-11), where rest is unused
GRID_KWARGS = {
    "year": 2024,
    "sample_rate": 0.1,
    "grid_size": 5, # 5x5 pixels per cell
    "n_rows": 50, # 50*50 cells as total area
    "n_cols": 50,
}
N_PIXELS_PER_CELL  = GRID_KWARGS['grid_size'] ** 2
SAMPLE_SIZE = 10  # 10×10 cells per sample
TRAIN_RATIO = 0.7
SEED = 42

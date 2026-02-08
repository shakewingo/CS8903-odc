from pathlib import Path

root_dir = Path(__file__).parent.parent
data_dir = Path(root_dir, 'data')
script_dir = Path(root_dir, 'scripts')

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


"""Grid generation endpoint — builds a 50x50 land-cover grid from GeoTIFF."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from shapely.geometry import shape, Point

from src.config import (
    ECO_VALUES, LAND_COVER_LABELS, LAND_COVER_COLORS,
    N_CLASSES, N_PIXELS_PER_CELL, GRID_KWARGS, SEED,
)
from src.dataset import build_grid, normalize_and_compute_rewards, split_dataset
from src.utils import get_logger

logger = get_logger("grid_service", level="INFO")

router = APIRouter()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # web-app/backend → web-app → CS8903-odc
BUFFER_PATH = PROJECT_ROOT / "data" / "processed" / "lake_malawi_expanded_25km.json"

ALL_CLASS_IDS = sorted(LAND_COVER_LABELS.keys())


# ---------------------------------------------------------------------------
# Buffer polygon for validation
# ---------------------------------------------------------------------------
_buffer_polygon = None


def _load_buffer_polygon():
    global _buffer_polygon
    if _buffer_polygon is None:
        with open(BUFFER_PATH) as f:
            geojson = json.load(f)
        if geojson.get("type") == "FeatureCollection":
            geojson = geojson["features"][0]["geometry"]
        elif geojson.get("type") == "Feature":
            geojson = geojson["geometry"]
        _buffer_polygon = shape(geojson)
    return _buffer_polygon


def is_within_buffer(lat: float, lng: float) -> bool:
    poly = _load_buffer_polygon()
    return poly.contains(Point(lng, lat))


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class GridRequest(BaseModel):
    lat: float
    lng: float
    grid_size: int = GRID_KWARGS["grid_size"]
    sample_rate: float = GRID_KWARGS["sample_rate"]
    year: int = GRID_KWARGS["year"]
    n_rows: int = GRID_KWARGS["n_rows"]
    n_cols: int = GRID_KWARGS["n_cols"]


# ---------------------------------------------------------------------------
# Grid generation (cached by rounded coords)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=32)
def _generate_grid(lat_r: float, lng_r: float, year: int, sample_rate: float,
                   grid_size: int, n_rows: int, n_cols: int):
    """Generate grid and dataset arrays. Cached by rounded lat/lng."""
    center = (lat_r, lng_r)
    kw = dict(year=year, sample_rate=sample_rate, grid_size=grid_size,
              n_rows=n_rows, n_cols=n_cols)

    pixel_counts, eco_values, et_values, valid_mask, coords = build_grid(center, **kw)

    rewards, norm_stats = normalize_and_compute_rewards(eco_values, et_values, valid_mask)
    train_indices, test_indices = split_dataset(valid_mask, n_rows=n_rows, n_cols=n_cols)

    # Build JSON grid
    grid_json = []
    for r in range(n_rows):
        row_data = []
        for c in range(n_cols):
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
                "f": fractions,
                "e": round(float(eco_values[r, c]), 1),
                "lat": round(float(coords[r, c, 0]), 6),
                "lng": round(float(coords[r, c, 1]), 6),
            })
        grid_json.append(row_data)

    # Store numpy arrays for inference
    dataset = {
        "pixel_counts": pixel_counts,
        "eco_values": eco_values,
        "et_values": et_values,
        "valid_mask": valid_mask,
        "coords": coords,
        "rewards": rewards,
        "norm_stats": np.array([
            norm_stats["mean_eco"], norm_stats["std_eco"],
            norm_stats["mean_et"], norm_stats["std_et"],
        ]),
        "train_indices": train_indices,
        "test_indices": test_indices,
    }

    return grid_json, dataset


def get_cached_dataset(lat: float, lng: float, **kw):
    """Return the numpy dataset dict for a given center (for inference)."""
    lat_r = round(lat, 6)
    lng_r = round(lng, 6)
    _, dataset = _generate_grid(
        lat_r, lng_r,
        kw.get("year", GRID_KWARGS["year"]),
        kw.get("sample_rate", GRID_KWARGS["sample_rate"]),
        kw.get("grid_size", GRID_KWARGS["grid_size"]),
        kw.get("n_rows", GRID_KWARGS["n_rows"]),
        kw.get("n_cols", GRID_KWARGS["n_cols"]),
    )
    return dataset


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
@router.post("/grid")
def generate_grid(req: GridRequest):
    if not is_within_buffer(req.lat, req.lng):
        raise HTTPException(
            status_code=400,
            detail="Coordinates are outside the Lake Malawi 25km buffer zone.",
        )

    lat_r = round(req.lat, 6)
    lng_r = round(req.lng, 6)

    logger.info(f"Generating grid for ({lat_r}, {lng_r})")
    grid_json, _ = _generate_grid(
        lat_r, lng_r, req.year, req.sample_rate,
        req.grid_size, req.n_rows, req.n_cols,
    )

    return {
        "rows": req.n_rows,
        "cols": req.n_cols,
        "center": {"lat": lat_r, "lng": lng_r},
        "grid": grid_json,
        "classLabels": {str(k): v for k, v in LAND_COVER_LABELS.items()},
        "classColors": {str(k): v for k, v in LAND_COVER_COLORS.items()},
        "ecoValues": {str(k): v for k, v in ECO_VALUES.items()},
    }

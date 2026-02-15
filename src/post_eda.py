import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import pyproj
import rasterio
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.transform import Affine
from rasterio.warp import reproject
from rasterio.windows import Window
from shapely.geometry import shape
from shapely.ops import transform

from config import LAND_COVER_LABELS, LAND_COVER_COLORS, ECONOMIC_VALUES, data_dir
from utils import get_logger

logger = get_logger(__name__)


def fetch_modis_data(year, area_of_interest):
    """Fetch, merge, and mask MODIS ET data for a given year and AOI.
    Note: data is calculated with 8days compound interval and eventually become a yearly aggregation per 500*500m resolution
    """
    logger.info(f"Fetching MODIS data for {year}...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=["modis-16A3GF-061"],
        intersects=area_of_interest,
        datetime=f"{year}-01-01/{year}-12-31",
    )
    items = search.item_collection()
    logger.info(f"Found {len(items)} MODIS tiles for {year}")

    datasets = [rasterio.open(item.assets["ET_500m"].href) for item in items]
    modis_crs = datasets[0].crs
    modis_nodata = datasets[0].nodata
    modis_profile = datasets[0].profile.copy()
    logger.info(f"MODIS CRS: {modis_crs}, nodata: {modis_nodata}")

    merged_data, merged_transform = merge(datasets)
    for ds in datasets:
        ds.close()
    logger.info(f"Merged shape: {merged_data.shape}")

    transformer = pyproj.Transformer.from_crs("EPSG:4326", modis_crs, always_xy=True)
    aoi_reprojected = transform(transformer.transform, shape(area_of_interest))

    modis_profile.update(
        {
            "height": merged_data.shape[1],
            "width": merged_data.shape[2],
            "transform": merged_transform,
        }
    )

    with MemoryFile() as memfile:
        with memfile.open(**modis_profile) as mem_ds:
            mem_ds.write(merged_data)
            masked_et, masked_et_transform = mask(
                mem_ds, [aoi_reprojected], crop=True, nodata=modis_nodata
            )

    return masked_et, masked_et_transform, modis_profile, modis_crs


def load_landcover_data(year, sample_rate=0.1):
    """Load and downsample the masked Sentinel-2 land-cover raster.

    Parameters
    ----------
    year : int
        Year of the land-cover raster.
    sample_rate : float
        Downsample factor (0.1 → 10 m to 100 m).

    Returns
    -------
    nodata, src_crs, new_height, new_width, downsampled_data, ds_transform, transformer
    """
    lc_path = Path(data_dir, "processed", f"malawi_masked_{year}.tif")

    with rasterio.open(lc_path) as src:
        orig_height, orig_width = src.height, src.width
        nodata = src.nodata
        src_crs = src.crs

        new_height = int(orig_height * sample_rate)
        new_width = int(orig_width * sample_rate)
        logger.info(
            f"Downsampling from ({orig_height}, {orig_width}) to "
            f"({new_height}, {new_width}) with mode resampling"
        )
        downsampled_data = src.read(
            1,
            out_shape=(new_height, new_width),
            resampling=Resampling.mode,
        )

        ds_transform = src.transform * Affine.scale(
            orig_width / new_width,
            orig_height / new_height,
        )

    transformer = pyproj.Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)

    return (
        nodata,
        src_crs,
        new_height,
        new_width,
        downsampled_data,
        ds_transform,
        transformer,
    )


def _compute_fraction_for_class(args):
    """
    Worker function: compute the fraction map for a single LC class
    by reading the LC raster in row-strips and reprojecting each strip.

    This runs in a subprocess so it gets its own memory space.
    """
    cls, lc_path, et_height, et_width, et_transform_tuple, et_crs_str, strip_height = (
        args
    )
    et_transform = Affine(*et_transform_tuple)
    dst_fraction = np.zeros((et_height, et_width), dtype=np.float64)

    with rasterio.open(lc_path) as lc_src:
        lc_height = lc_src.height
        lc_width = lc_src.width
        lc_transform = lc_src.transform
        lc_crs = lc_src.crs
        lc_nodata = lc_src.nodata

        # Process in horizontal strips to limit memory
        for row_off in range(0, lc_height, strip_height):
            h = min(strip_height, lc_height - row_off)
            window = Window(0, row_off, lc_width, h)
            chunk = lc_src.read(1, window=window)

            # Binary mask for this class within the strip
            binary = np.where(chunk == cls, 1.0, 0.0).astype(np.float32)

            # Skip strips with zero presence of this class
            if binary.sum() == 0:
                continue

            # Compute the transform for this strip
            strip_transform = lc_transform * Affine.translation(0, row_off)

            # Accumulate into the destination fraction map
            strip_dst = np.zeros((et_height, et_width), dtype=np.float32)
            reproject(
                source=binary,
                destination=strip_dst,
                src_transform=strip_transform,
                src_crs=lc_crs,
                dst_transform=et_transform,
                dst_crs=et_crs_str,
                resampling=Resampling.average,
            )
            dst_fraction += strip_dst

    return cls, dst_fraction


def compute_fractions_parallel(
    lc_classes,
    lc_path,
    et_height,
    et_width,
    et_transform,
    et_crs,
    strip_height=2048,
    max_workers=None,
):
    """
    Compute fraction maps for all LC classes in parallel using ProcessPoolExecutor.
    Each class is processed in a separate subprocess, and within each subprocess
    the LC raster is read in row-strips to limit peak memory.
    """
    if max_workers is None:
        # Use at most 8 workers or cpu_count - 2, whichever is smaller
        max_workers = min(8, max(1, multiprocessing.cpu_count() - 2))

    logger.info(
        f"Computing fractions with {max_workers} parallel workers, "
        f"strip_height={strip_height} rows"
    )

    # Serialize transform to a tuple for pickling
    et_transform_tuple = tuple(et_transform)[:6]
    et_crs_str = str(et_crs)

    args_list = [
        (
            int(cls),
            str(lc_path),
            et_height,
            et_width,
            et_transform_tuple,
            et_crs_str,
            strip_height,
        )
        for cls in lc_classes
    ]

    fraction_maps = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_compute_fraction_for_class, args): args[0]
            for args in args_list
        }
        for future in as_completed(futures):
            cls = futures[future]
            label = LAND_COVER_LABELS.get(cls, f"Class {cls}")
            try:
                cls_id, frac_map = future.result()
                fraction_maps[cls_id] = frac_map.astype(np.float32)
                logger.info(f"  {label:>12s} (class {cls:2d}): done")
            except Exception as e:
                logger.error(f"  {label:>12s} (class {cls:2d}): FAILED — {e}")

    return fraction_maps


# ---------------------------------------------------------------------------
#  Average-ET per land-cover class
# ---------------------------------------------------------------------------


def get_et_per_landcover(year=2024, skip_fetch=True, strip_height=2048, max_workers=8):

    MODIS_ET_PATH = Path(data_dir, "processed", f"modis_et_masked_{year}.tif")
    # Note: LC data is 10*10m resolution
    LC_PATH = Path(data_dir, "processed", f"malawi_masked_{year}.tif")

    with open(Path(data_dir, "processed", "lake_malawi_expanded_25km.json"), "r") as f:
        malawi_data = json.load(f)
    area_of_interest = malawi_data["geometry"]
    aoi_shape = shape(area_of_interest)
    logger.info(f"AOI type: {aoi_shape.geom_type}, bounds: {aoi_shape.bounds}")

    # Fetch MODIS ET
    if not skip_fetch or not MODIS_ET_PATH.exists():
        masked_et, masked_et_transform, modis_profile, modis_crs = fetch_modis_data(
            year, area_of_interest
        )

        save_profile = modis_profile.copy()
        save_profile.update(
            {
                "height": masked_et.shape[1],
                "width": masked_et.shape[2],
                "transform": masked_et_transform,
            }
        )
        with rasterio.open(MODIS_ET_PATH, "w", **save_profile) as dst:
            dst.write(masked_et)
        logger.info(f"Saved MODIS ET to {MODIS_ET_PATH}")
        logger.info(f"  Shape: {masked_et.shape}, CRS: {modis_crs}")
    else:
        logger.info(f"Skipping fetch — using saved file: {MODIS_ET_PATH}")

    # Load rasters
    with rasterio.open(MODIS_ET_PATH) as src:
        et_data = src.read(1).astype(np.float64)
        et_transform = src.transform
        et_crs = src.crs
        et_nodata = src.nodata
        et_height, et_width = src.height, src.width
    logger.info(
        f"MODIS ET:  {et_data.shape}, CRS={et_crs}, res={et_transform.a:.1f}m, nodata={et_nodata}"
    )

    # Peek at LC metadata and size
    with rasterio.open(LC_PATH) as src:
        lc_crs = src.crs
        lc_nodata = src.nodata
        lc_height, lc_width = src.height, src.width
        lc_res = src.transform.a
    logger.info(
        f"Land Cover: ({lc_height}, {lc_width}), CRS={lc_crs}, res={lc_res:.1f}m, nodata={lc_nodata}"
    )
    logger.info(f"  LC raster size: ~{lc_height * lc_width * 1 / 1e9:.2f} GB (uint8)")

    # Mark invalid ET pixels as NaN
    et_valid_mask = (et_data != et_nodata) & (et_data >= 0) & (et_data < 32767)
    et_data[~et_valid_mask] = np.nan
    et_data *= 0.1  # Scale to kg/m²/year
    n_valid = np.count_nonzero(et_valid_mask)
    logger.info(
        f"Valid MODIS pixels: {n_valid} / {et_data.size} ({n_valid / et_data.size * 100:.1f}%)"
    )
    logger.info(
        f"ET range: {np.nanmin(et_data):.1f} — {np.nanmax(et_data):.1f} kg/m²/year"
    )

    # Discover LC classes
    logger.info("Scanning LC raster for unique classes...")
    all_classes = set()
    with rasterio.open(LC_PATH) as src:
        # Scan in strips to save memory
        for row_off in range(0, lc_height, strip_height):
            h = min(strip_height, lc_height - row_off)
            chunk = src.read(1, window=Window(0, row_off, lc_width, h))
            valid_chunk = (
                chunk[chunk != lc_nodata]
                if lc_nodata is not None
                else chunk[chunk != 0]
            )
            all_classes.update(np.unique(valid_chunk).tolist())
    lc_classes = sorted(all_classes)
    logger.info(
        f"LC classes found: {[(c, LAND_COVER_LABELS.get(c, '?')) for c in lc_classes]}"
    )

    # Compute fractions in parallel
    fraction_maps = compute_fractions_parallel(
        lc_classes,
        LC_PATH,
        et_height,
        et_width,
        et_transform,
        et_crs,
        strip_height=strip_height,
        max_workers=max_workers,
    )

    # Log coverage stats
    for cls in sorted(fraction_maps.keys()):
        frac = fraction_maps[cls]
        coverage = np.sum(frac[et_valid_mask]) / np.sum(et_valid_mask) * 100
        label = LAND_COVER_LABELS.get(cls, f"Class {cls}")
        logger.info(
            f"  {label:>12s} (class {cls:2d}): mean fraction = {frac[et_valid_mask].mean():.4f}, "
            f"coverage = {coverage:.2f}%"
        )

    # Sanity check
    total_fraction = np.zeros((et_height, et_width), dtype=np.float32)
    for frac in fraction_maps.values():
        total_fraction += frac
    valid_totals = total_fraction[et_valid_mask]
    logger.info("Fraction sum stats (should be ~1.0):")
    logger.info(f"  Mean: {valid_totals.mean():.4f}")
    logger.info(f"  Min:  {valid_totals.min():.4f}")
    logger.info(f"  Max:  {valid_totals.max():.4f}")
    logger.warning(
        f"  Pixels with sum < 0.9: {np.sum(valid_totals < 0.9)} / {len(valid_totals)}"
    )

    # Compute Weighted Mean ET
    results = {}
    for cls, frac_map in fraction_maps.items():
        label = LAND_COVER_LABELS.get(cls, f"Class {cls}")
        valid = et_valid_mask & (frac_map > 0)
        if not np.any(valid):
            logger.warning(f"  {label}: no overlap with valid ET pixels")
            continue
        et_vals = et_data[valid]
        weights = frac_map[valid]
        weighted_mean = np.average(et_vals, weights=weights)
        weighted_var = np.average((et_vals - weighted_mean) ** 2, weights=weights)
        weighted_std = np.sqrt(weighted_var)
        sorted_idx = np.argsort(et_vals)
        sorted_et = et_vals[sorted_idx]
        sorted_w = weights[sorted_idx]
        cum_w = np.cumsum(sorted_w) / np.sum(sorted_w)
        p25 = float(sorted_et[np.searchsorted(cum_w, 0.25)])
        p75 = float(sorted_et[np.searchsorted(cum_w, 0.75)])
        results[cls] = {
            "label": label,
            "color": LAND_COVER_COLORS.get(cls, "#000000"),
            "weighted_mean_et": round(float(weighted_mean), 2),
            "weighted_std_et": round(float(weighted_std), 2),
            "weighted_p25_et": round(p25, 2),
            "weighted_p75_et": round(p75, 2),
            "min_et": round(float(np.min(et_vals)), 2),
            "max_et": round(float(np.max(et_vals)), 2),
            "total_fraction": round(float(np.sum(weights)), 2),
            "n_modis_pixels": int(np.sum(valid)),
        }
        logger.info(
            f"  {label:>12s}: mean ET = {weighted_mean:7.2f} ± {weighted_std:6.2f} kg/m²/yr "
            f"(p25={p25:.1f}, p75={p75:.1f}, n={np.sum(valid)})"
        )

    # Save and Visualize
    output = {
        "year": year,
        "description": "Fractional-weighted mean MODIS ET (kg/m²/year) per Sentinel-2 land cover class, 25km Lake Malawi buffer",
        "method": "Binary LC masks reprojected to MODIS grid via average resampling (strip-based, parallel), then weighted aggregation",
        "classes": {str(cls): data for cls, data in sorted(results.items())},
    }
    json_path = Path(data_dir, "processed", f"et_per_landcover_{year}.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved to {json_path}")

    sorted_classes = sorted(
        results.keys(), key=lambda c: results[c]["weighted_mean_et"], reverse=True
    )
    bar_labels = [results[c]["label"] for c in sorted_classes]
    bar_means = [results[c]["weighted_mean_et"] for c in sorted_classes]
    bar_p25 = [results[c]["weighted_p25_et"] for c in sorted_classes]
    bar_p75 = [results[c]["weighted_p75_et"] for c in sorted_classes]
    err_low = [m - p for m, p in zip(bar_means, bar_p25)]
    err_high = [p - m for m, p in zip(bar_means, bar_p75)]
    bar_colors = [results[c]["color"] for c in sorted_classes]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(
        bar_labels, bar_means, color=bar_colors, edgecolor="black", linewidth=0.5
    )
    ax.errorbar(
        bar_labels,
        bar_means,
        yerr=[err_low, err_high],
        fmt="none",
        ecolor="black",
        capsize=5,
        linewidth=1.5,
    )
    ax.set_ylabel("Evapotranspiration (kg/m²/year)", fontsize=12)
    ax.set_title(f"Mean MODIS ET per Land Cover Class — {year}", fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    for bar, mean_val in zip(bars, bar_means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{mean_val:.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    plt.tight_layout()
    chart_path = Path(data_dir, "processed", f"et_per_landcover_{year}.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    logger.info(f"Chart saved to {chart_path}")
    plt.close()

    summary_data = []
    for cls in sorted(results.keys()):
        r = results[cls]
        summary_data.append(
            {
                "Class ID": cls,
                "Label": r["label"],
                "Mean ET (kg/m²/yr)": r["weighted_mean_et"],
                "Std ET": r["weighted_std_et"],
                "P25": r["weighted_p25_et"],
                "P75": r["weighted_p75_et"],
                "Min ET": r["min_et"],
                "Max ET": r["max_et"],
                "Total Fraction": r["total_fraction"],
                "# MODIS Pixels": r["n_modis_pixels"],
            }
        )
    summary_df = pd.DataFrame(summary_data)
    logger.info(f"Summary:\n{summary_df.to_string(index=False)}")


# ---------------------------------------------------------------------------
#  Understand land-cover distribution
# ---------------------------------------------------------------------------


def find_most_diverse_grids(year=2024, sample_rate=0.1, grid_size=5, top_n=10):
    """Find the top-N most land-cover-diverse 5×5 grids in a downsampled raster.

    Workflow
    --------
    1. Load the masked Sentinel-2 land-cover GeoTIFF (10 m resolution).
    2. Downsample to 100 m (sample_rate=0.1) using mode resampling.
    3. Slide a 5×5 window (500 m × 500 m) across every non-overlapping position.
    4. Count distinct valid land-cover classes in each window.
    5. Return the *top_n* windows with the highest class diversity, breaking ties
       by Shannon entropy (higher entropy = more even distribution).

    Parameters
    ----------
    year : int
        Year of the land-cover raster to use.
    sample_rate : float
        Downsample factor applied to both height and width (0.1 → 10 m ➜ 100 m).
    grid_size : int
        Side length of the scanning window in downsampled pixels.
    top_n : int
        Number of most-diverse grids to return.

    Returns
    -------
    list[dict]
        Each dict contains grid location (row/col indices, lat/lon of centre),
        the number of unique classes, class breakdown, and Shannon entropy.
    """
    nodata, _, new_height, new_width, downsampled, ds_transform, transformer = (
        load_landcover_data(year, sample_rate)
    )

    # Scan every non-overlapping grid_size × grid_size window
    n_rows = new_height // grid_size
    n_cols = new_width // grid_size
    logger.info(
        f"Scanning {n_rows} × {n_cols} = {n_rows * n_cols} "
        f"non-overlapping {grid_size}×{grid_size} grids"
    )

    grid_records = []
    for gr in range(n_rows):
        r_start = gr * grid_size
        r_end = r_start + grid_size
        for gc in range(n_cols):
            c_start = gc * grid_size
            c_end = c_start + grid_size

            block = downsampled[r_start:r_end, c_start:c_end]

            # Filter out nodata
            if nodata is not None:
                valid = block[block != nodata]
            else:
                valid = block[block != 0]

            if len(valid) == 0:
                continue

            unique_classes, counts = np.unique(valid, return_counts=True)
            n_classes = len(unique_classes)

            # Shannon entropy (higher = more evenly distributed)
            proportions = counts / counts.sum()
            entropy = float(-np.sum(proportions * np.log2(proportions + 1e-12)))

            grid_records.append(
                {
                    "grid_row": gr,
                    "grid_col": gc,
                    "row_start": r_start,
                    "col_start": c_start,
                    "n_classes": n_classes,
                    "entropy": round(entropy, 4),
                    "unique_classes": unique_classes.tolist(),
                    "counts": counts.tolist(),
                }
            )

    # Rank by n_classes desc, then entropy desc
    grid_records.sort(key=lambda g: (g["n_classes"], g["entropy"]), reverse=True)
    top_grids = grid_records[:top_n]

    results = []
    for rank, g in enumerate(top_grids, start=1):
        # Centre pixel in downsampled grid
        center_col = g["col_start"] + grid_size / 2.0
        center_row = g["row_start"] + grid_size / 2.0

        # Project to map coordinates then to lat/lon
        x, y = ds_transform * (center_col, center_row)
        lon, lat = transformer.transform(x, y)

        breakdown = []
        total_pixels = sum(g["counts"])
        for cls_id, cnt in zip(g["unique_classes"], g["counts"]):
            breakdown.append(
                {
                    "class_id": int(cls_id),
                    "label": LAND_COVER_LABELS.get(int(cls_id), f"Class {cls_id}"),
                    "color": LAND_COVER_COLORS.get(int(cls_id), "#000000"),
                    "pixel_count": int(cnt),
                    "percentage": round(cnt / total_pixels * 100, 2),
                }
            )

        record = {
            "rank": rank,
            "grid_row": g["grid_row"],
            "grid_col": g["grid_col"],
            "center_lat": round(lat, 6),
            "center_lon": round(lon, 6),
            "n_classes": g["n_classes"],
            "entropy": g["entropy"],
            "breakdown": breakdown,
        }
        results.append(record)

        logger.info(
            f"  #{rank}: grid({g['grid_row']}, {g['grid_col']})  "
            f"lat={lat:.4f}, lon={lon:.4f}  "
            f"classes={g['n_classes']}  entropy={g['entropy']:.3f}  "
            f"types={[b['label'] for b in breakdown]}"
        )

    # Save results
    output = {
        "year": year,
        "description": (
            f"Top {top_n} most land-cover-diverse {grid_size}×{grid_size} grids "
            f"(each {grid_size * int(1 / sample_rate * 10)}m × "
            f"{grid_size * int(1 / sample_rate * 10)}m) "
            f"from Sentinel-2 land cover, 25 km Lake Malawi buffer"
        ),
        "method": (
            f"10 m raster downsampled to {int(10 / sample_rate)}m via mode resampling, "
            f"then scanned with non-overlapping {grid_size}×{grid_size} windows. "
            "Ranked by number of unique classes (desc), then Shannon entropy (desc)."
        ),
        "grids": results,
    }
    json_path = Path(data_dir, "processed", f"diverse_grids_{year}.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved diverse-grid results to {json_path}")

    return results


def get_landcover_block_at_latlon(
    year, center_lat, center_lon, grid_size=5, sample_rate=0.1
):
    """
    Get a block of pixels surrounding a lat/lon coordinate.
    Returns:
        dict with the pixel block and class breakdown
    """
    (
        nodata,
        src_crs,
        new_height,
        new_width,
        downsampled_data,
        ds_transform,
        to_latlon,
    ) = load_landcover_data(year, sample_rate)

    # to_latlon: src_crs→4326 is already done, but for lat/lon→projected we need the reverse direction
    to_projected = pyproj.Transformer.from_crs("EPSG:4326", src_crs, always_xy=True)
    x, y = to_projected.transform(center_lon, center_lat)

    # Convert projected coords to row/col in downsampled grid
    inv_transform = ~ds_transform
    col_f, row_f = inv_transform * (x, y)
    center_row, center_col = int(row_f), int(col_f)

    # Define window around center
    radius_pixels = grid_size // 2
    r_start = max(0, center_row - radius_pixels)
    c_start = max(0, center_col - radius_pixels)
    r_end = min(new_height, center_row + radius_pixels + 1)
    c_end = min(new_width, center_col + radius_pixels + 1)

    data = downsampled_data[r_start:r_end, c_start:c_end]

    # Get class breakdown
    valid = data[data != nodata] if nodata is not None else data[data != 0]
    unique, counts = np.unique(valid, return_counts=True)

    breakdown = []
    for val, cnt in zip(unique, counts):
        breakdown.append(
            {
                "class_id": int(val),
                "label": LAND_COVER_LABELS.get(int(val), f"Class {int(val)}"),
                "pixel_count": int(cnt),
                "percentage": round(float(cnt / len(valid) * 100), 2),
            }
        )

    return {
        "center": {"lat": center_lat, "lon": center_lon},
        "block_shape": data.shape,
        "details": breakdown,
        "raw_data": data.tolist(),
    }


# ---------------------------------------------------------------------------
#  Value-grid computation & heatmap plotting
# ---------------------------------------------------------------------------
# TODO: record each cell's raw_data for land-cover class
def compute_value_grid(
    center_lat,
    center_lon,
    value_map,
    value_key="value",
    year=2024,
    sample_rate=0.1,
    grid_size=5,
    n_rows=50,
    n_cols=50,
):
    """Build an n_rows×n_cols grid of cells centred on a lat/lon coordinate.

    Each cell spans *grid_size × grid_size* downsampled pixels (default 5×5 at
    100 m = 500 m).  For every cell the weighted mean of *value_map* is
    computed from the land-cover fractions.

    Parameters
    ----------
    center_lat, center_lon : float
        WGS-84 centre of the grid.
    value_map : dict[int, float]
        Mapping from land-cover class ID to the per-class value
        (e.g. economic USD/ha/yr, or ET kg/m²/yr).
    value_key : str
        Key name used in the returned cell dicts (e.g. ``"economic_value"``
        or ``"et_value"``).
    year : int
        Year of the land-cover raster.
    sample_rate : float
        Downsample factor (0.1 → 10 m to 100 m).
    grid_size : int
        Side length of each grid cell in downsampled pixels.
    n_rows, n_cols : int
        Number of grid cells along each axis.

    Returns
    -------
    grid : list[list[dict | None]]
        Each element is ``None`` or a dict with ``"breakdown"``,
        *value_key*, ``"center_lat"``, ``"center_lon"``.
    meta : dict
        Grid-level metadata including ``"value_key"``.
    """
    nodata, src_crs, ds_h, ds_w, data, ds_transform, to_latlon = load_landcover_data(
        year, sample_rate
    )

    # Convert centre lat/lon → pixel coords in the downsampled grid
    to_projected = pyproj.Transformer.from_crs("EPSG:4326", src_crs, always_xy=True)
    cx, cy = to_projected.transform(center_lon, center_lat)
    inv_transform = ~ds_transform
    center_col, center_row = inv_transform * (cx, cy)
    center_row, center_col = int(center_row), int(center_col)

    # Compute offsets so the grid is centred on that pixel
    total_px_rows = n_rows * grid_size
    total_px_cols = n_cols * grid_size
    row_offset = center_row - total_px_rows // 2
    col_offset = center_col - total_px_cols // 2

    # Clamp to raster bounds
    row_offset = max(0, min(row_offset, ds_h - total_px_rows))
    col_offset = max(0, min(col_offset, ds_w - total_px_cols))
    logger.info(
        f"[{value_key}] Grid centred on ({center_lat:.4f}, {center_lon:.4f}) → "
        f"pixel ({center_row}, {center_col}), origin ({row_offset}, {col_offset}), "
        f"covering {total_px_rows}×{total_px_cols} of {ds_h}×{ds_w}"
    )

    grid: list[list[dict | None]] = []
    for gr in range(n_rows):
        row_cells = []
        r0 = row_offset + gr * grid_size
        r1 = min(r0 + grid_size, ds_h)
        for gc in range(n_cols):
            c0 = col_offset + gc * grid_size
            c1 = min(c0 + grid_size, ds_w)

            block = data[r0:r1, c0:c1]
            valid = (
                block[(block != nodata) & (block != 0)]
                if nodata is not None
                else block[block != 0]
            )

            if len(valid) == 0:
                row_cells.append(None)
                continue

            unique, counts = np.unique(valid, return_counts=True)
            total = counts.sum()
            fractions = counts / total

            # Weighted value
            weighted_val = sum(
                value_map.get(int(cls), 0) * frac
                for cls, frac in zip(unique, fractions)
            )

            # Breakdown sorted by fraction desc
            breakdown = sorted(
                [(int(cls), float(frac)) for cls, frac in zip(unique, fractions)],
                key=lambda x: x[1],
                reverse=True,
            )

            # Cell centre → lat/lon
            cell_cx = col_offset + gc * grid_size + grid_size / 2.0
            cell_cy = row_offset + gr * grid_size + grid_size / 2.0
            mx, my = ds_transform * (cell_cx, cell_cy)
            cell_lon, cell_lat = to_latlon.transform(mx, my)

            row_cells.append(
                {
                    "breakdown": breakdown,
                    value_key: round(float(weighted_val), 2),
                    "center_lat": round(cell_lat, 6),
                    "center_lon": round(cell_lon, 6),
                }
            )
        grid.append(row_cells)

    # Collect value range for colour-map normalisation
    values = [cell[value_key] for row in grid for cell in row if cell is not None]
    meta = {
        "year": year,
        "value_key": value_key,
        "center_lat": center_lat,
        "center_lon": center_lon,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "cell_res_m": grid_size * int(10 / sample_rate),
        "value_min": min(values) if values else 0,
        "value_max": max(values) if values else 0,
    }
    logger.info(
        f"[{value_key}] Grid ready: {n_rows}×{n_cols}, "
        f"value range {meta['value_min']:.0f}–{meta['value_max']:.0f}"
    )
    return grid, meta


def plot_value_heatmap(grid, meta, title, unit, cmap_colors, filename, save=True):
    """Render a value heatmap with per-cell land-cover proportion bars.

    Parameters
    ----------
    grid, meta : as returned by ``compute_value_grid``.
    title : str
        Plot title (may contain ``{year}`` and ``{res}`` placeholders).
    unit : str
        Colour-bar label, e.g. ``"USD/ha/yr"`` or ``"kg/m²/yr"``.
    cmap_colors : list[str]
        Ordered colour stops for the background colour-map.
    filename : str
        Output filename template (may contain ``{year}``).
    """
    import matplotlib.colors as mcolors
    from matplotlib.patches import Rectangle

    value_key = meta["value_key"]
    n_rows = meta["n_rows"]
    n_cols = meta["n_cols"]

    cmap = mcolors.LinearSegmentedColormap.from_list("val", cmap_colors)
    vmin, vmax = meta["value_min"], meta["value_max"]
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    cell_w, cell_h = 1.0, 1.0
    fig_w = n_cols * cell_w * 0.25 + 2.5
    fig_h = n_rows * cell_h * 0.25 + 1.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    bar_height_frac = 0.30
    bar_y_offset = 0.10

    for gr in range(n_rows):
        for gc in range(n_cols):
            cell = grid[gr][gc]
            x0 = gc * cell_w
            y0 = (n_rows - 1 - gr) * cell_h

            if cell is None:
                ax.add_patch(
                    Rectangle(
                        (x0, y0),
                        cell_w,
                        cell_h,
                        facecolor="#f0f0f0",
                        edgecolor="grey",
                        linewidth=0.3,
                    )
                )
                continue

            val = cell[value_key]
            bg_color = cmap(norm(val))
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    cell_w,
                    cell_h,
                    facecolor=bg_color,
                    edgecolor="grey",
                    linewidth=0.3,
                )
            )

            # Stacked horizontal bar for land-cover proportions
            bar_y = y0 + bar_y_offset * cell_h
            bar_h = bar_height_frac * cell_h
            bar_x = x0 + 0.05 * cell_w
            bar_total_w = 0.90 * cell_w
            cursor = bar_x
            for cls_id, frac in cell["breakdown"]:
                seg_w = frac * bar_total_w
                ax.add_patch(
                    Rectangle(
                        (cursor, bar_y),
                        seg_w,
                        bar_h,
                        facecolor=LAND_COVER_COLORS.get(cls_id, "#000000"),
                        edgecolor="none",
                    )
                )
                cursor += seg_w

            # Value label
            label_color = "white" if norm(val) > 0.6 else "black"
            ax.text(
                x0 + cell_w / 2,
                y0 + cell_h * 0.72,
                f"{val:.0f}",
                ha="center",
                va="center",
                fontsize=3.5,
                fontweight="bold",
                color=label_color,
            )

    ax.set_xlim(0, n_cols * cell_w)
    ax.set_ylim(0, n_rows * cell_h)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        title.format(year=meta["year"], res=meta["cell_res_m"]),
        fontsize=12,
        fontweight="bold",
        pad=12,
    )

    # Colour-bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(unit, fontsize=9)

    # Land-cover legend
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=color,
            markersize=7,
            label=LAND_COVER_LABELS.get(cls, f"Class {cls}"),
        )
        for cls, color in sorted(LAND_COVER_COLORS.items())
        if cls in LAND_COVER_LABELS
    ]
    ax.legend(
        handles=legend_handles,
        title="Land Cover",
        loc="upper left",
        bbox_to_anchor=(1.06, 1.0),
        fontsize=6,
        title_fontsize=7,
        frameon=True,
    )

    plt.tight_layout()
    if save:
        chart_path = Path(data_dir, "processed", filename.format(year=meta["year"]))
        plt.savefig(chart_path, dpi=200, bbox_inches="tight")
        logger.info(f"Heatmap saved to {chart_path}")
    plt.close()


def compute_economic_grid(center_lat, center_lon, **kwargs):
    """Economic-value grid (USD/ha/yr)."""
    return compute_value_grid(
        center_lat,
        center_lon,
        value_map=ECONOMIC_VALUES,
        value_key="economic_value",
        **kwargs,
    )


def plot_economic_heatmap(grid, meta, **kwargs):
    """Plot economic-value heatmap."""
    plot_value_heatmap(
        grid,
        meta,
        title="Economic Value Heatmap — {year}  ({res}m cells, USD/ha/yr)",
        unit="Economic Value (USD/ha/yr)",
        cmap_colors=["#e8f5e9", "#ffffcc", "#ffccbc", "#ef5350", "#b71c1c"],
        filename="economic_heatmap_{year}.png",
        **kwargs,
    )


def load_et_value_map(year=2024):
    """Load weighted-mean ET per land-cover class from the precomputed JSON."""
    json_path = Path(data_dir, "processed", f"et_per_landcover_{year}.json")
    with open(json_path) as f:
        et_data = json.load(f)
    return {int(k): v["weighted_mean_et"] for k, v in et_data["classes"].items()}


def compute_et_grid(center_lat, center_lon, year=2024, **kwargs):
    """ET-value grid (kg/m²/yr)."""
    et_map = load_et_value_map(year)
    return compute_value_grid(
        center_lat,
        center_lon,
        value_map=et_map,
        value_key="et_value",
        year=year,
        **kwargs,
    )


def plot_et_heatmap(grid, meta, **kwargs):
    """Plot ET-value heatmap."""
    plot_value_heatmap(
        grid,
        meta,
        title="Evapotranspiration Heatmap — {year}  ({res}m cells, kg/m²/yr)",
        unit="ET (kg/m²/yr)",
        cmap_colors=["#fff7e6", "#fee8c8", "#fdbb84", "#e34a33", "#7a0177"],
        filename="et_heatmap_{year}.png",
        **kwargs,
    )


if __name__ == "__main__":
    # get_et_per_landcover(year=2024)
    # find_most_diverse_grids(year=2024, sample_rate=0.1, grid_size=5, top_n=10)

    # look into the most diverse grids
    # top_distributed_grid = get_landcover_block_at_latlon(year=2024, center_lat=-12.580513, center_lon=34.170748, grid_size=5)
    # print(np.array2string(np.array(top_distributed_grid["raw_data"]), separator=", "))

    center = (-13.934564, 34.542859)

    args = {
        "center_lat": center[0],
        "center_lon": center[1],
        "year": 2024,
        "sample_rate": 0.1,
        "grid_size": 5,
        "n_rows": 50,
        "n_cols": 50,
    }

    # Economic-value heatmap
    grid, meta = compute_economic_grid(**args)
    plot_economic_heatmap(grid, meta, **args)

    # ET-value heatmap
    grid, meta = compute_et_grid(**args)
    plot_et_heatmap(grid, meta, **args)

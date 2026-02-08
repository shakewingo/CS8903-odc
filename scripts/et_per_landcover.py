import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
from rasterio.io import MemoryFile
from rasterio.windows import Window
from shapely.geometry import shape
from shapely.ops import transform
import pyproj
import pystac_client
import planetary_computer
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from config import LAND_COVER_LABELS, LAND_COVER_COLORS, data_dir
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

    datasets = [rasterio.open(item.assets['ET_500m'].href) for item in items]
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

    modis_profile.update({
        'height': merged_data.shape[1],
        'width': merged_data.shape[2],
        'transform': merged_transform,
    })

    with MemoryFile() as memfile:
        with memfile.open(**modis_profile) as mem_ds:
            mem_ds.write(merged_data)
            masked_et, masked_et_transform = mask(
                mem_ds, [aoi_reprojected], crop=True, nodata=modis_nodata
            )

    return masked_et, masked_et_transform, modis_profile, modis_crs


def _compute_fraction_for_class(args):
    """
    Worker function: compute the fraction map for a single LC class
    by reading the LC raster in row-strips and reprojecting each strip.

    This runs in a subprocess so it gets its own memory space.
    """
    cls, lc_path, et_height, et_width, et_transform_tuple, et_crs_str, strip_height = args

    from rasterio.warp import reproject, Resampling
    from rasterio.transform import Affine
    import rasterio
    import numpy as np

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


def compute_fractions_parallel(lc_classes, lc_path, et_height, et_width,
                                et_transform, et_crs, strip_height=2048,
                                max_workers=None):
    """
    Compute fraction maps for all LC classes in parallel using ProcessPoolExecutor.
    Each class is processed in a separate subprocess, and within each subprocess
    the LC raster is read in row-strips to limit peak memory.
    """
    if max_workers is None:
        # Use at most 8 workers or cpu_count - 2, whichever is smaller
        max_workers = min(8, max(1, multiprocessing.cpu_count() - 2))

    logger.info(f"Computing fractions with {max_workers} parallel workers, "
                f"strip_height={strip_height} rows")

    # Serialize transform to a tuple for pickling
    et_transform_tuple = tuple(et_transform)[:6]
    et_crs_str = str(et_crs)

    args_list = [
        (int(cls), str(lc_path), et_height, et_width,
         et_transform_tuple, et_crs_str, strip_height)
        for cls in lc_classes
    ]

    fraction_maps = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_compute_fraction_for_class, args): args[0]
                   for args in args_list}
        for future in as_completed(futures):
            cls = futures[future]
            label = LAND_COVER_LABELS.get(cls, f'Class {cls}')
            try:
                cls_id, frac_map = future.result()
                fraction_maps[cls_id] = frac_map.astype(np.float32)
                logger.info(f"  {label:>12s} (class {cls:2d}): done")
            except Exception as e:
                logger.error(f"  {label:>12s} (class {cls:2d}): FAILED — {e}")

    return fraction_maps


def get_et_per_landcover():
    SKIP_FETCH = True
    YEAR = 2024
    STRIP_HEIGHT = 2048  # rows per strip 
    MAX_WORKERS = 8      # parallel subprocesses

    MODIS_ET_PATH = Path(data_dir, "processed", f"modis_et_masked_{YEAR}.tif")
    # Note: LC data is 10*10m resolution
    LC_PATH = Path(data_dir, "processed", f"malawi_masked_{YEAR}.tif")

    with open(Path(data_dir, "processed", "lake_malawi_expanded_25km.json"), 'r') as f:
        malawi_data = json.load(f)
    area_of_interest = malawi_data["geometry"]
    aoi_shape = shape(area_of_interest)
    logger.info(f"AOI type: {aoi_shape.geom_type}, bounds: {aoi_shape.bounds}")

    # Fetch MODIS ET
    if not SKIP_FETCH:
        masked_et, masked_et_transform, modis_profile, modis_crs = fetch_modis_data(YEAR, area_of_interest)

        save_profile = modis_profile.copy()
        save_profile.update({
            'height': masked_et.shape[1],
            'width': masked_et.shape[2],
            'transform': masked_et_transform,
        })
        with rasterio.open(MODIS_ET_PATH, 'w', **save_profile) as dst:
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
    logger.info(f"MODIS ET:  {et_data.shape}, CRS={et_crs}, res={et_transform.a:.1f}m, nodata={et_nodata}")

    # Peek at LC metadata and size
    with rasterio.open(LC_PATH) as src:
        lc_crs = src.crs
        lc_nodata = src.nodata
        lc_height, lc_width = src.height, src.width
        lc_res = src.transform.a
    logger.info(f"Land Cover: ({lc_height}, {lc_width}), CRS={lc_crs}, res={lc_res:.1f}m, nodata={lc_nodata}")
    logger.info(f"  LC raster size: ~{lc_height * lc_width * 1 / 1e9:.2f} GB (uint8)")

    # Mark invalid ET pixels as NaN
    et_valid_mask = (et_data != et_nodata) & (et_data >= 0) & (et_data < 32767)
    et_data[~et_valid_mask] = np.nan
    et_data *= 0.1  # Scale to kg/m²/year
    n_valid = np.count_nonzero(et_valid_mask)
    logger.info(f"Valid MODIS pixels: {n_valid} / {et_data.size} ({n_valid / et_data.size * 100:.1f}%)")
    logger.info(f"ET range: {np.nanmin(et_data):.1f} — {np.nanmax(et_data):.1f} kg/m²/year")

    # Discover LC classes
    logger.info("Scanning LC raster for unique classes...")
    all_classes = set()
    with rasterio.open(LC_PATH) as src:
        # Scan in strips to save memory
        for row_off in range(0, lc_height, STRIP_HEIGHT):
            h = min(STRIP_HEIGHT, lc_height - row_off)
            chunk = src.read(1, window=Window(0, row_off, lc_width, h))
            valid_chunk = chunk[chunk != lc_nodata] if lc_nodata is not None else chunk[chunk != 0]
            all_classes.update(np.unique(valid_chunk).tolist())
    lc_classes = sorted(all_classes)
    logger.info(f"LC classes found: {[(c, LAND_COVER_LABELS.get(c, '?')) for c in lc_classes]}")

    # Compute fractions in parallel
    fraction_maps = compute_fractions_parallel(
        lc_classes, LC_PATH, et_height, et_width,
        et_transform, et_crs,
        strip_height=STRIP_HEIGHT,
        max_workers=MAX_WORKERS,
    )

    # Log coverage stats
    for cls in sorted(fraction_maps.keys()):
        frac = fraction_maps[cls]
        coverage = np.sum(frac[et_valid_mask]) / np.sum(et_valid_mask) * 100
        label = LAND_COVER_LABELS.get(cls, f'Class {cls}')
        logger.info(f"  {label:>12s} (class {cls:2d}): mean fraction = {frac[et_valid_mask].mean():.4f}, "
                    f"coverage = {coverage:.2f}%")

    # Sanity check
    total_fraction = np.zeros((et_height, et_width), dtype=np.float32)
    for frac in fraction_maps.values():
        total_fraction += frac
    valid_totals = total_fraction[et_valid_mask]
    logger.info("Fraction sum stats (should be ~1.0):")
    logger.info(f"  Mean: {valid_totals.mean():.4f}")
    logger.info(f"  Min:  {valid_totals.min():.4f}")
    logger.info(f"  Max:  {valid_totals.max():.4f}")
    logger.warning(f"  Pixels with sum < 0.9: {np.sum(valid_totals < 0.9)} / {len(valid_totals)}")

    # Compute Weighted Mean ET
    results = {}
    for cls, frac_map in fraction_maps.items():
        label = LAND_COVER_LABELS.get(cls, f'Class {cls}')
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
            'label': label,
            'color': LAND_COVER_COLORS.get(cls, '#000000'),
            'weighted_mean_et': round(float(weighted_mean), 2),
            'weighted_std_et': round(float(weighted_std), 2),
            'weighted_p25_et': round(p25, 2),
            'weighted_p75_et': round(p75, 2),
            'min_et': round(float(np.min(et_vals)), 2),
            'max_et': round(float(np.max(et_vals)), 2),
            'total_fraction': round(float(np.sum(weights)), 2),
            'n_modis_pixels': int(np.sum(valid)),
        }
        logger.info(f"  {label:>12s}: mean ET = {weighted_mean:7.2f} ± {weighted_std:6.2f} kg/m²/yr "
                    f"(p25={p25:.1f}, p75={p75:.1f}, n={np.sum(valid)})")

    # Save and Visualize
    output = {
        'year': YEAR,
        'description': 'Fractional-weighted mean MODIS ET (kg/m²/year) per Sentinel-2 land cover class, 25km Lake Malawi buffer',
        'method': 'Binary LC masks reprojected to MODIS grid via average resampling (strip-based, parallel), then weighted aggregation',
        'classes': {str(cls): data for cls, data in sorted(results.items())},
    }
    json_path = Path(data_dir, "processed", f"et_per_landcover_{YEAR}.json")
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved to {json_path}")

    sorted_classes = sorted(results.keys(), key=lambda c: results[c]['weighted_mean_et'], reverse=True)
    bar_labels = [results[c]['label'] for c in sorted_classes]
    bar_means = [results[c]['weighted_mean_et'] for c in sorted_classes]
    bar_p25 = [results[c]['weighted_p25_et'] for c in sorted_classes]
    bar_p75 = [results[c]['weighted_p75_et'] for c in sorted_classes]
    err_low = [m - p for m, p in zip(bar_means, bar_p25)]
    err_high = [p - m for m, p in zip(bar_means, bar_p75)]
    bar_colors = [results[c]['color'] for c in sorted_classes]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(bar_labels, bar_means, color=bar_colors, edgecolor='black', linewidth=0.5)
    ax.errorbar(bar_labels, bar_means, yerr=[err_low, err_high],
                fmt='none', ecolor='black', capsize=5, linewidth=1.5)
    ax.set_ylabel('Evapotranspiration (kg/m²/year)', fontsize=12)
    ax.set_title(f'Mean MODIS ET per Land Cover Class — {YEAR}', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    for bar, mean_val in zip(bars, bar_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f'{mean_val:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    chart_path = Path(data_dir, "processed", f"et_per_landcover_{YEAR}.png")
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    logger.info(f"Chart saved to {chart_path}")
    plt.close()

    summary_data = []
    for cls in sorted(results.keys()):
        r = results[cls]
        summary_data.append({
            'Class ID': cls,
            'Label': r['label'],
            'Mean ET (kg/m²/yr)': r['weighted_mean_et'],
            'Std ET': r['weighted_std_et'],
            'P25': r['weighted_p25_et'],
            'P75': r['weighted_p75_et'],
            'Min ET': r['min_et'],
            'Max ET': r['max_et'],
            'Total Fraction': r['total_fraction'],
            '# MODIS Pixels': r['n_modis_pixels'],
        })
    summary_df = pd.DataFrame(summary_data)
    logger.info(f"Summary:\n{summary_df.to_string(index=False)}")


if __name__ == "__main__":
    get_et_per_landcover()
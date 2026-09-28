# -*- coding: utf-8 -*-
"""
01_whymap_hygeo2_rasterize.py

Clips the WHYMAP global aquifer polygon layer to the Germany AOI and
rasterizes the HYGEO2 (groundwater recharge classification) attribute
to a regular 500 m grid in EPSG:4326.

Processing steps:
    1. Read WHYMAP shapefile and clip to Germany bbox
    2. Build a cell-centred 500 m (1/240°) target grid in EPSG:4326
    3. Rasterize HYGEO2 at 5x finer resolution
    4. Aggregate fine grid -> 500 m via mode (ignoring no-data)
    5. Save via nc_io.save_nc (CF-1.8, uint8 flag var, descending lat)

Input:
    01_Input_Global/1_Original/WHYMAP/WHYMAP_GWR/shp/
        whymap_GW_aquifers_v1_poly.shp

Output:
    01_Input_Global/3_Final/WHYMAP/WHYMAP.nc

Variable in output:
    hygeo2 : uint8 (lat, lon)
        WHYMAP HYGEO2 groundwater recharge class.
        First digit  = aquifer system type
            1 = Major Groundwater Basins
            2 = Complex Hydrogeological Structures
            3 = Local and Shallow Aquifers
        Second digit = recharge class within type (see flag_meanings)

HYGEO2 classification (groundwater recharge in mm/yr):
    MAJOR GROUNDWATER BASINS
        15 very high (> 300)
        14 high      (100 - 300)
        13 medium    (20  - 100)
        12 low       (2   - 20)
        11 very low  (< 2)
    COMPLEX HYDROGEOLOGICAL STRUCTURES
        25 very high (> 300)
        24 high      (100 - 300)
        23 medium    (20  - 100)
        22 low - very low (< 20)
    LOCAL AND SHALLOW AQUIFERS
        34 very high - high     (> 100)
        33 medium    - very low (< 100)

Location: 02_Processing_Scripts/WHYMAP/
Run     : python 01_whymap_hygeo2_rasterize.py

Reference:
    BGR & UNESCO (2008): Groundwater Resources of the World - WHYMAP.
"""

import sys
from pathlib import Path
import time
import warnings

import numpy as np
import geopandas as gpd
import xarray as xr
from shapely.geometry import box
from pyproj import Transformer
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from scipy import stats

# =============================================================================
# SHARED HELPER
# =============================================================================
sys.path.append(str(Path(__file__).resolve().parents[1] / "0_Shared"))
import nc_io


# =============================================================================
# PATHS  (relative to the project root; adjust here if the layout changes)
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_GLOBAL = PROJECT_ROOT / "01_Input_Global"
ORIGINAL     = INPUT_GLOBAL / "1_Original"
INTERMEDIATE = INPUT_GLOBAL / "2_Intermediate"
FINAL        = INPUT_GLOBAL / "3_Final"

# Input shapefile (folder layout of the WHYMAP download kept under 1_Original/WHYMAP/)
INPUT_SHP = (ORIGINAL / "WHYMAP" / "WHYMAP_GWR" / "shp"
             / "whymap_GW_aquifers_v1_poly.shp")

# Output
OUT_PATH = FINAL / "WHYMAP" / "WHYMAP.nc"

# =============================================================================
# CONFIGURATION
# =============================================================================
# Germany bounding box in WGS84 (minx, miny, maxx, maxy)
BBOX_WGS84 = (5.5, 47.0, 15.5, 55.5)

# Target resolution: 500 m in EPSG:4326
#   1 degree of latitude ~ 111 km  ->  500 m ~ 1/222°
#   For consistency with the 1 km grid (1/120°) of the GLiM/GLHYMPS pipeline,
#   1/240° is used (~463 m in north-south direction).
RES_DEG = 1 / 240.0

# Oversample factor for area-weighted aggregation.
# Categorical mode: 5 -> 25 sub-pixels per coarse cell (sufficient & memory-safe)
OVERSAMPLE_FACTOR = 5


# =============================================================================
# HYGEO2 CLASSIFICATION
# =============================================================================
HYGEO2_VALUES = [11, 12, 13, 14, 15, 22, 23, 24, 25, 33, 34]

HYGEO2_MEANINGS = [
    "major_basin_very_low",         # 11  ( < 2     mm/yr)
    "major_basin_low",              # 12  ( 2-20    mm/yr)
    "major_basin_medium",           # 13  ( 20-100  mm/yr)
    "major_basin_high",             # 14  ( 100-300 mm/yr)
    "major_basin_very_high",        # 15  ( > 300   mm/yr)
    "complex_low_to_very_low",      # 22  ( < 20    mm/yr)
    "complex_medium",               # 23  ( 20-100  mm/yr)
    "complex_high",                 # 24  ( 100-300 mm/yr)
    "complex_very_high",            # 25  ( > 300   mm/yr)
    "local_medium_to_very_low",     # 33  ( < 100   mm/yr)
    "local_very_high_to_high",      # 34  ( > 100   mm/yr)
]

VAR_ATTRS = {
    "hygeo2": {
        "long_name": "WHYMAP HYGEO2 groundwater recharge class",
        "units": "1",
        "grid_mapping": "crs",
        "flag_values": np.array(HYGEO2_VALUES, dtype=np.uint8),
        "flag_meanings": " ".join(HYGEO2_MEANINGS),
        "comment": (
            "First digit = aquifer system type "
            "(1=Major Groundwater Basins, 2=Complex Hydrogeological Structures, "
            "3=Local and Shallow Aquifers). "
            "Second digit = groundwater recharge class within that type. "
            "Source: WHYMAP (BGR/UNESCO 2008)."
        ),
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_target_grid(bbox, res):
    """Cell-centred regular WGS84 grid (ascending lat S->N, ascending lon W->E)."""
    minx, miny, maxx, maxy = bbox
    lon = np.arange(minx + res / 2, maxx, res, dtype=np.float64)
    lat = np.arange(miny + res / 2, maxy, res, dtype=np.float64)
    return lat, lon


def clip_shapefile(shp_path, bbox_wgs84):
    """Read shapefile, prefilter by bbox, reproject to WGS84, exact clip."""
    print(f"  Reading {shp_path.name} ...")
    t0 = time.time()

    # Detect source CRS
    gdf_meta = gpd.read_file(shp_path, rows=0)
    src_crs = gdf_meta.crs
    print(f"    Source CRS: {src_crs}")

    # Transform bbox to source CRS for the spatial prefilter
    if src_crs is not None and src_crs.to_epsg() != 4326:
        transformer = Transformer.from_crs("EPSG:4326", src_crs, always_xy=True)
        xs, ys = transformer.transform(
            [bbox_wgs84[0], bbox_wgs84[2], bbox_wgs84[0], bbox_wgs84[2]],
            [bbox_wgs84[1], bbox_wgs84[1], bbox_wgs84[3], bbox_wgs84[3]],
        )
        bbox_src = (min(xs), min(ys), max(xs), max(ys))
    else:
        bbox_src = bbox_wgs84

    gdf = gpd.read_file(shp_path, bbox=bbox_src)
    print(f"    Read {len(gdf):,} features in {time.time() - t0:.1f}s")

    # Reproject to WGS84
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    # Precise clip to AOI
    clip_box = box(*bbox_wgs84)
    gdf = gpd.clip(gdf, clip_box)
    gdf = gdf[~gdf.is_empty].copy()
    print(f"    {len(gdf):,} features after clipping to AOI")
    return gdf


def rasterize_and_aggregate(gdf, target_lat, target_lon, value_col):
    """
    Rasterize a categorical attribute at OVERSAMPLE_FACTOR-finer resolution,
    then block-aggregate to the target grid using the mode (ignoring no-data).
    Returns a uint8 array aligned with ascending lat (S->N) / lon (W->E).
    """
    t0 = time.time()
    F = OVERSAMPLE_FACTOR

    lat_res = float(np.abs(np.diff(target_lat)).mean())
    lon_res = float(np.abs(np.diff(target_lon)).mean())
    h_c, w_c = len(target_lat), len(target_lon)

    west  = float(target_lon.min()) - lon_res / 2
    east  = float(target_lon.max()) + lon_res / 2
    south = float(target_lat.min()) - lat_res / 2
    north = float(target_lat.max()) + lat_res / 2

    h_f, w_f = h_c * F, w_c * F
    transform_fine = from_bounds(west, south, east, north, w_f, h_f)
    print(f"    Coarse grid: {h_c} x {w_c}   |   Fine grid: {h_f} x {w_f}")

    # Build (geom, value) shape list, drop missing
    values = gdf[value_col].astype(float).values
    valid  = ~np.isnan(values)
    geom_list = list(gdf.geometry)
    shapes = [(geom_list[i], int(values[i])) for i in range(len(gdf)) if valid[i]]

    if not shapes:
        print("    WARNING: No valid features to rasterize.")
        return np.full((h_c, w_c), 255, dtype=np.uint8)

    # Fine rasterization (fill=0 marks "no polygon here")
    fine = rasterize(
        shapes, out_shape=(h_f, w_f), transform=transform_fine,
        fill=0, dtype=np.int16,
    )
    print(f"    Rasterized in {time.time() - t0:.1f}s")

    # Reshape to (h_c, w_c, F*F) so we can mode-aggregate per coarse cell
    blocks = (fine.reshape(h_c, F, w_c, F)
                  .transpose(0, 2, 1, 3)
                  .reshape(h_c, w_c, F * F))

    # Mode while ignoring 0 (= no polygon)
    blocks_float = blocks.astype(np.float32)
    blocks_float[blocks_float == 0] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        res_mode = stats.mode(blocks_float, axis=2, nan_policy="omit")

    coarse = res_mode[0] if isinstance(res_mode, tuple) else res_mode.mode
    if hasattr(coarse, "squeeze"):
        coarse = coarse.squeeze()

    coarse = np.nan_to_num(coarse, nan=255).astype(np.uint8)
    # Flip vertical so row 0 corresponds to the minimum (southern) latitude
    coarse = coarse[::-1, :]
    print(f"    Aggregated to target grid in {time.time() - t0:.1f}s total")
    return coarse


def build_dataset(coarse_array, target_lat, target_lon):
    """
    Wrap the rasterized array in a minimal Dataset.
    Coordinate attrs, orientation and global metadata are applied by nc_io.save_nc.
    """
    ds = xr.Dataset()
    ds["crs"] = xr.DataArray(
        np.int32(0),
        attrs={"grid_mapping_name": "latitude_longitude", "epsg_code": "EPSG:4326"},
    )
    ds["hygeo2"] = xr.DataArray(
        coarse_array, dims=["lat", "lon"],
        coords={"lat": target_lat, "lon": target_lon},
        attrs=VAR_ATTRS["hygeo2"],
    )
    return ds


# =============================================================================
# EXECUTION
# =============================================================================
print("=" * 65)
print("WHYMAP HYGEO2 -> 500 m raster (Germany)")
print("=" * 65)
print(f"  AOI (WGS84): W={BBOX_WGS84[0]}°, S={BBOX_WGS84[1]}°, "
      f"E={BBOX_WGS84[2]}°, N={BBOX_WGS84[3]}°")
print(f"  Target resolution: {RES_DEG:.6f}°  (~500 m)")
print(f"  Oversample factor: {OVERSAMPLE_FACTOR}x")

print("\nSTEP 1: Read and clip WHYMAP shapefile")
gdf = clip_shapefile(INPUT_SHP, BBOX_WGS84)

# Find the HYGEO2 attribute (be tolerant of capitalization)
hygeo_col = None
for cand in ("HYGEO2", "hygeo2", "Hygeo2"):
    if cand in gdf.columns:
        hygeo_col = cand
        break
if hygeo_col is None:
    raise KeyError(
        f"HYGEO2 column not found. Available columns: {list(gdf.columns)}"
    )
print(f"  Using attribute column: '{hygeo_col}'")
unique_vals = sorted(gdf[hygeo_col].dropna().unique().tolist())
print(f"  Unique HYGEO2 values in AOI: {unique_vals}")

unexpected = [v for v in unique_vals if int(v) not in HYGEO2_VALUES]
if unexpected:
    print(f"  WARNING: Unexpected HYGEO2 codes encountered: {unexpected}")

print("\nSTEP 2: Build target grid and rasterize")
lat, lon = create_target_grid(BBOX_WGS84, RES_DEG)
print(f"  Grid shape: {len(lat)} (lat) x {len(lon)} (lon) "
      f"= {len(lat) * len(lon):,} pixels")

coarse = rasterize_and_aggregate(gdf, lat, lon, hygeo_col)

print("\nSTEP 3: Build NetCDF dataset and save")
ds = build_dataset(coarse, lat, lon)

global_attrs = nc_io.make_global_attrs(
    title="WHYMAP HYGEO2 groundwater recharge classification, Germany, ~500 m",
    source="WHYMAP Global Groundwater Resources (BGR/UNESCO, 2008)",
    references="BGR & UNESCO (2008): Groundwater Resources of the World - WHYMAP.",
    script=Path(__file__).name,
    extra={"spatial_resolution": f"~500 m (1/{int(1 / RES_DEG)} degree)"},
)

nc_io.save_nc(
    ds, OUT_PATH,
    fill_value={"hygeo2": np.uint8(255), "crs": None},
    dtype={"hygeo2": "uint8", "crs": "int32"},
    global_attrs=global_attrs,
    complevel=4,
)

# Class distribution summary
unique, counts = np.unique(coarse, return_counts=True)
total = counts.sum()
print("\n  Class distribution in raster:")
for u, c in zip(unique, counts):
    if u == 255:
        label = "no_data"
    elif int(u) in HYGEO2_VALUES:
        label = HYGEO2_MEANINGS[HYGEO2_VALUES.index(int(u))]
    else:
        label = "unknown"
    print(f"    {int(u):>3}  {label:<32}  {c:>10,} px  ({100 * c / total:5.2f}%)")

print(f"\n{'=' * 65}")
print(f"Done. Output: {OUT_PATH}")
print("=" * 65)

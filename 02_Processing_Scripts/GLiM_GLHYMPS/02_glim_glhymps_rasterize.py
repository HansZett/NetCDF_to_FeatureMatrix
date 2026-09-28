# -*- coding: utf-8 -*-
"""
02_glim_glhymps_rasterize.py

Rasterizes the merged GLiM+GLHYMPS GeoPackage to a ~1 km target grid
(1/120°, EPSG:4326) and saves it via nc_io.save_nc.

Rasterization strategy:
    1. Build a cell-centred ~1 km grid over the Germany AOI
    2. Rasterize at 10x finer resolution (center-of-pixel rule)
    3. Block-aggregate to target resolution:
       - Continuous vars (porosity, permeability): area-weighted mean
       - Categorical var  (lithology class): mode (ignoring missing values)

Input:
    01_Input_Global/2_Intermediate/GLiM_GLHYMPS/GLiM_GLHYMPS.gpkg
Output:
    01_Input_Global/3_Final/GLiM_GLHYMPS/GLiM_GLHYMPS.nc

Location: 02_Processing_Scripts/GLiM_GLHYMPS/
Run     : python 02_glim_glhymps_rasterize.py
"""

import sys
from pathlib import Path
import time
import warnings

import numpy as np
import xarray as xr
import geopandas as gpd
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

# Input GeoPackage (from script 01)
INPUT_GPKG = INTERMEDIATE / "GLiM_GLHYMPS" / "GLiM_GLHYMPS.gpkg"

# Output
OUT_PATH = FINAL / "GLiM_GLHYMPS" / "GLiM_GLHYMPS.nc"

# =============================================================================
# CONFIGURATION
# =============================================================================
BBOX_WGS84 = (5.5, 47.0, 15.5, 55.5)
OVERSAMPLE_FACTOR = 10


# =============================================================================
# CF ATTRIBUTES & MAPPINGS
# =============================================================================

# GLiM level-1 classes (field "xx") mapped to integer codes 1-16
GLIM_CLASSES = {
    'su': 1, 'sc': 2, 'ss': 3, 'sm': 4, 'py': 5, 'ev': 6, 'mt': 7,
    'pa': 8, 'pb': 9, 'pi': 10, 'va': 11, 'vb': 12, 'vi': 13,
    'wb': 14, 'ig': 15, 'nd': 16
}

GLIM_MEANINGS = [
    "unconsolidated_sediments", "carbonate_sedimentary_rocks", "siliciclastic_sedimentary_rocks",
    "mixed_sedimentary_rocks", "pyroclastic_rocks", "evaporites", "metamorphic_rocks",
    "acid_plutonic_rocks", "basic_plutonic_rocks", "intermediate_plutonic_rocks",
    "acid_volcanic_rocks", "basic_volcanic_rocks", "intermediate_volcanic_rocks",
    "water_bodies", "ice_and_glaciers", "no_data"
]

CONTINUOUS_VARS = {
    "Porosity":                        "porosity",
    "Permeability_no_permafrost":      "permeability_no_permafrost",
    "Permeability_permafrost":         "permeability_permafrost",
    "Permeability_standard_deviation": "permeability_std",
}

VAR_ATTRS = {
    "porosity": {
        "long_name": "Porosity", "units": "1", "grid_mapping": "crs"
    },
    "permeability_no_permafrost": {
        "long_name": "Log10 permeability (no permafrost)", "units": "log10(m2)", "grid_mapping": "crs"
    },
    "permeability_permafrost": {
        "long_name": "Log10 permeability (permafrost)", "units": "log10(m2)", "grid_mapping": "crs"
    },
    "permeability_std": {
        "long_name": "Permeability standard deviation", "units": "log10(m2)", "grid_mapping": "crs"
    },
    "litho_class": {
        "long_name": "Dominant lithology class",
        "units": "1",
        "grid_mapping": "crs",
        # GLiM 'nd' (code 16) is folded into the single no-data sentinel 255,
        # so it is intentionally excluded from the flag table below.
        "flag_values": np.array([v for k, v in GLIM_CLASSES.items() if k != "nd"],
                                dtype=np.uint8),
        "flag_meanings": " ".join(m for v, m in zip(GLIM_CLASSES.values(), GLIM_MEANINGS)
                                  if v != 16),
        "comment": ("255 = no data: cells either unclassified by GLiM "
                    "(former code 16 'nd') or not covered by any polygon."),
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_1km_grid(bbox):
    """Generates an evenly spaced, cell-centred ~1km grid (ascending lat/lon)."""
    minx, miny, maxx, maxy = bbox
    res = 1 / 120.0  # 30 arc-seconds ~ 1km
    lon = np.arange(minx + res/2, maxx, res, dtype=np.float64)
    lat = np.arange(miny + res/2, maxy, res, dtype=np.float64)
    return lat, lon


def rasterize_fine_and_aggregate(gdf, target_lat, target_lon, label):
    """
    Rasterize at OVERSAMPLE_FACTOR-finer resolution and block-aggregate:
      continuous -> area-weighted mean, lithology class -> mode.
    Coordinate attrs / orientation / global metadata are applied by nc_io.save_nc.
    """
    t0 = time.time()
    F = OVERSAMPLE_FACTOR

    lat_res = float(np.abs(np.diff(target_lat)).mean())
    lon_res = float(np.abs(np.diff(target_lon)).mean())
    h_c, w_c = len(target_lat), len(target_lon)

    west, east  = float(target_lon.min()) - lon_res/2, float(target_lon.max()) + lon_res/2
    south, north = float(target_lat.min()) - lat_res/2, float(target_lat.max()) + lat_res/2

    h_f, w_f = h_c * F, w_c * F
    transform_fine = from_bounds(west, south, east, north, w_f, h_f)
    print(f"    Coarse grid: {h_c} x {w_c}   |   Fine grid: {h_f} x {w_f}")

    ds_out = xr.Dataset()
    ds_out["crs"] = xr.DataArray(np.int32(0), attrs={
        "long_name": "CRS definition",
        "grid_mapping_name": "latitude_longitude",
        "epsg_code": "EPSG:4326",
    })

    geom_list = list(gdf.geometry)

    # 1. Continuous Variables
    for src_col, nc_name in CONTINUOUS_VARS.items():
        values = gdf[src_col].values.astype(np.float64)
        valid = ~np.isnan(values)
        shapes = [(geom_list[i], values[i]) for i in range(len(gdf)) if valid[i]]

        if shapes:
            fine = rasterize(shapes, out_shape=(h_f, w_f), transform=transform_fine, fill=np.nan, dtype=np.float64)
            blocks = fine.reshape(h_c, F, w_c, F)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                coarse = np.nanmean(blocks, axis=(1, 3)).astype(np.float32)

            coarse = coarse[::-1, :]  # FLIP vertical to align with ascending lat (S->N)
        else:
            coarse = np.full((h_c, w_c), np.nan, dtype=np.float32)

        ds_out[nc_name] = xr.DataArray(coarse, dims=["lat", "lon"], coords={"lat": target_lat, "lon": target_lon}, attrs=VAR_ATTRS[nc_name])

    # 2. Categorical Variable (litho_class)
    lc_values = gdf["xx"].map(GLIM_CLASSES).values.astype(np.float64)
    valid = ~np.isnan(lc_values)
    shapes = [(geom_list[i], int(lc_values[i])) for i in range(len(gdf)) if valid[i]]

    fine_lc = rasterize(shapes, out_shape=(h_f, w_f), transform=transform_fine, fill=0, dtype=np.int16)
    blocks_lc = fine_lc.reshape(h_c, F, w_c, F).transpose(0, 2, 1, 3).reshape(h_c, w_c, F*F)

    # Calculate mode ignoring 0
    blocks_lc_float = blocks_lc.astype(np.float32)
    blocks_lc_float[blocks_lc_float == 0] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        res_mode = stats.mode(blocks_lc_float, axis=2, nan_policy='omit')

    coarse_lc = res_mode[0] if isinstance(res_mode, tuple) else res_mode.mode
    if hasattr(coarse_lc, "squeeze"): coarse_lc = coarse_lc.squeeze()

    coarse_lc = np.nan_to_num(coarse_lc, nan=255).astype(np.uint8)
    # Collapse GLiM 'nd' (16) into the single no-data sentinel (255)
    coarse_lc[coarse_lc == 16] = 255
    coarse_lc = coarse_lc[::-1, :]  # FLIP vertical

    ds_out["litho_class"] = xr.DataArray(coarse_lc, dims=["lat", "lon"], coords={"lat": target_lat, "lon": target_lon}, attrs=VAR_ATTRS["litho_class"])

    print(f"    Rasterized + aggregated in {time.time() - t0:.1f}s")
    return ds_out


# =============================================================================
# EXECUTION
# =============================================================================
print("=" * 65)
print("GLiM + GLHYMPS -> ~1 km raster (Germany)")
print("=" * 65)

print("Loading vector data...")
gdf = gpd.read_file(INPUT_GPKG)
print(f"  {len(gdf):,} features")

print("\nProcessing ~1 km grid...")
k1_lat, k1_lon = create_1km_grid(BBOX_WGS84)
print(f"  Grid shape: {len(k1_lat)} (lat) x {len(k1_lon)} (lon) "
      f"= {len(k1_lat) * len(k1_lon):,} pixels")

ds_1k = rasterize_fine_and_aggregate(gdf, k1_lat, k1_lon, "1km")

global_attrs = nc_io.make_global_attrs(
    title="GLiM + GLHYMPS lithology & hydraulic properties, Germany, ~1 km",
    source="GLiM lithology (Hartmann & Moosdorf, 2012) + GLHYMPS hydraulic properties (Gleeson et al., 2014)",
    references=(
        "Gleeson, T., et al. (2014). doi:10.1002/2014gl059856 ; "
        "Hartmann, J. & Moosdorf, N. (2012). doi:10.1029/2012GC004370"
    ),
    script=Path(__file__).name,
    extra={"spatial_resolution": "~1 km (1/120 degree)"},
)

nc_io.save_nc(
    ds_1k, OUT_PATH,
    fill_value={
        "porosity": np.nan,
        "permeability_no_permafrost": np.nan,
        "permeability_permafrost": np.nan,
        "permeability_std": np.nan,
        "litho_class": np.uint8(255),
        "crs": None,
    },
    dtype={
        "porosity": "float32",
        "permeability_no_permafrost": "float32",
        "permeability_permafrost": "float32",
        "permeability_std": "float32",
        "litho_class": "uint8",
        "crs": "int32",
    },
    global_attrs=global_attrs,
    complevel=4,
)

# Lithology class distribution
lc = ds_1k["litho_class"].values
unique, counts = np.unique(lc, return_counts=True)
total = counts.sum()
inv = {v: k for k, v in GLIM_CLASSES.items()}
print("\n  Lithology class distribution in raster:")
for u, c in zip(unique, counts):
    if u == 255:
        label = "no_data"
    elif int(u) in inv.values() or int(u) in range(1, 17):
        idx = int(u) - 1
        label = GLIM_MEANINGS[idx] if 0 <= idx < len(GLIM_MEANINGS) else "unknown"
    else:
        label = "unknown"
    print(f"    {int(u):>3}  {label:<32}  {c:>10,} px  ({100 * c / total:5.2f}%)")

print(f"\n{'=' * 65}")
print(f"Done. Output: {OUT_PATH}")
print("=" * 65)

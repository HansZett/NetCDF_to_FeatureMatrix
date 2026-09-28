# -*- coding: utf-8 -*-
"""
01_gasy_merge_depths.py

Merges the 7 GASY-SoilGrids NetCDF files (one per SoilGrids standard depth)
into a single file with reconstructed metadata, optionally with a spatial
subset (Germany AOI). The file is written with nc_io.save_nc so that it
follows the project convention (descending lat, explicit _FillValue, zlib,
CF-1.8 global attributes). The merged file is the final product at native
resolution and is written directly to 3_Final.

The 7 files correspond to the 7 SoilGrids standard depths:
  Layer 1:   0 cm   (surface)
  Layer 2:   5 cm
  Layer 3:  15 cm
  Layer 4:  30 cm
  Layer 5:  60 cm
  Layer 6: 100 cm
  Layer 7: 200 cm

Input : 01_Input_Global/1_Original/GASY/soilgrids_sy_{1..7}_integrate.nc
Output: 01_Input_Global/3_Final/GASY/GASY.nc

References:
  - Lv et al. (2025): A global dataset of average specific yield for soils.
    Scientific Data, 12, 427. https://doi.org/10.1038/s41597-025-04742-1
  - Poggio et al. (2021): SoilGrids 2.0. SOIL, 7, 217-240.
    https://doi.org/10.5194/soil-7-217-2021
  - Hengl et al. (2017): SoilGrids250m. PLoS ONE, 12(2), e0169748.
    https://doi.org/10.1371/journal.pone.0169748

Location: 02_Processing_Scripts/GASY/
Run     : python 01_gasy_merge_depths.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

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

INPUT_DIR = ORIGINAL / "GASY"
OUT_PATH  = FINAL / "GASY" / "GASY.nc"

# =============================================================================
# CONFIGURATION
# =============================================================================
# SoilGrids standard depths in metres (Hengl et al. 2017; Lv et al. 2025)
DEPTH_VALUES_M = [0.0, 0.05, 0.15, 0.30, 0.60, 1.00, 2.00]

# Expected file names (sorted by layer number)
EXPECTED_FILES = [f"soilgrids_sy_{i}_integrate.nc" for i in range(1, 8)]

# Spatial subset (set SUBSET_BBOX = None to keep the full global grid)
# Study area: 47 - 55.5 deg N, 5.5 - 15.5 deg E
SUBSET_BBOX = {
    "lat_min": 47.0,
    "lat_max": 55.5,
    "lon_min":  5.5,
    "lon_max": 15.5,
}


# =============================================================================
# Find and check input files
# =============================================================================
print("=" * 70)
print("  GASY-SoilGrids  —  merge and metadata")
print("=" * 70)
print(f"  Input :  {INPUT_DIR}")
print(f"  Output:  {OUT_PATH}\n")

nc_files = []
for fname in EXPECTED_FILES:
    fp = INPUT_DIR / fname
    if not fp.is_file():
        sys.exit(f"ERROR: File not found: {fp}")
    nc_files.append(fp)

print("  ✓ All 7 files found.\n")


# =============================================================================
# Reference grid and subset indices (from layer 1, applied to all layers)
# =============================================================================
print("  Loading reference grid for the subset ...")
ds0 = nc_io.normalize_coords(xr.open_dataset(nc_files[0]))
lat_full = ds0.lat.values
lon_full = ds0.lon.values
ds0.close()

if SUBSET_BBOX is not None:
    lat_min, lat_max = SUBSET_BBOX["lat_min"], SUBSET_BBOX["lat_max"]
    lon_min, lon_max = SUBSET_BBOX["lon_min"], SUBSET_BBOX["lon_max"]

    # Indices of all grid points inside the bounding box
    lat_idx = np.where((lat_full >= lat_min) & (lat_full <= lat_max))[0]
    lon_idx = np.where((lon_full >= lon_min) & (lon_full <= lon_max))[0]

    if len(lat_idx) == 0:
        sys.exit(f"ERROR: No grid points in latitude range {lat_min}-{lat_max}.")
    if len(lon_idx) == 0:
        sys.exit(f"ERROR: No grid points in longitude range {lon_min}-{lon_max}.")

    print("  Subset selected:")
    print(f"    Latitude : {lat_min} - {lat_max} deg  →  {len(lat_idx)} points")
    print(f"    Longitude: {lon_min} - {lon_max} deg  →  {len(lon_idx)} points\n")
else:
    lat_idx = None
    lon_idx = None
    print("  No subset selected - the full global grid is processed.\n")


# =============================================================================
# Read, subset and stack the depth layers
# =============================================================================
print("  Reading and stacking depth layers ...")
t0 = time.time()

sy_layers = []
for i, fp in enumerate(nc_files):
    depth_cm = int(DEPTH_VALUES_M[i] * 100)
    print(f"  [{i+1}/7]  {fp.name}  →  depth = {depth_cm} cm  ...", end="", flush=True)

    ds_i = nc_io.normalize_coords(xr.open_dataset(fp))
    da = ds_i["sy"]
    if lat_idx is not None:
        da = da.isel(lat=lat_idx, lon=lon_idx)
    sy_layers.append(da.load())
    ds_i.close()
    print("  ✓")

# Concatenate along a new depth dimension
sy = xr.concat(sy_layers, dim="depth")
sy = sy.assign_coords(depth=("depth", np.array(DEPTH_VALUES_M, dtype="float32")))
sy = sy.transpose("depth", "lat", "lon")
sy.name = "sy"
sy.attrs = {
    "long_name": "Average specific yield",
    "standard_name": "specific_yield",
    "units": "1",
    "description": (
        "Gridded average specific yield (Sy) for soils, derived from SoilGrids "
        "soil texture fractions via the trilinear graph method (Johnson, 1967). "
        "Sy is dimensionless (volume of water drained per unit volume of soil "
        "under gravity)."
    ),
    "valid_min": np.float32(0.0),
    "valid_max": np.float32(0.5),
    "grid_mapping": "crs",
    "coordinates": "depth lat lon",
}

ds_out = sy.to_dataset()

# depth attributes (nc_io only stamps lat/lon/time)
ds_out["depth"].attrs = {
    "units": "m",
    "long_name": "Soil depth",
    "standard_name": "depth",
    "positive": "down",
    "axis": "Z",
    "description": "Standard SoilGrids prediction depths: 0, 5, 15, 30, 60, 100, 200 cm",
}

# CRS grid-mapping container (WGS84 / EPSG:4326)
ds_out["crs"] = xr.DataArray(
    np.int32(0),
    attrs={
        "long_name": "CRS definition",
        "grid_mapping_name": "latitude_longitude",
        "epsg_code": "EPSG:4326",
        "semi_major_axis": 6378137.0,
        "inverse_flattening": 298.257223563,
        "longitude_of_prime_meridian": 0.0,
        "crs_wkt": (
            'GEOGCS["WGS 84",'
            'DATUM["WGS_1984",'
            'SPHEROID["WGS 84",6378137,298.257223563]],'
            'PRIMEM["Greenwich",0],'
            'UNIT["degree",0.0174532925199433],'
            'AUTHORITY["EPSG","4326"]]'
        ),
    },
)

print(f"\n  Stacked in {time.time() - t0:.1f}s  |  "
      f"shape sy = {tuple(ds_out['sy'].shape)} (depth, lat, lon)\n")


# =============================================================================
# Global attributes and output
# =============================================================================
extra = {
    "summary": (
        "Gridded average specific yield (Sy) derived from SoilGrids soil texture "
        "data (sand, silt, clay fractions) using the trilinear graph method of "
        "Johnson (1967). Seven depth layers correspond to the standard SoilGrids "
        "prediction depths."
    ),
    "source_dataset": "GASY (Global Average Specific Yield)",
    "source_soil_texture": (
        "SoilGrids 1-km (Hengl et al., 2017); SoilGrids 2.0 250m (Poggio et al., 2021)"
    ),
    "data_source_url": "https://zenodo.org/uploads/14216083",
    "spatial_resolution": "30 arc-seconds (~1 km)",
}
if SUBSET_BBOX is not None:
    extra["subset_note"] = (
        f"File contains a spatial subset: "
        f"lat {SUBSET_BBOX['lat_min']}-{SUBSET_BBOX['lat_max']}, "
        f"lon {SUBSET_BBOX['lon_min']}-{SUBSET_BBOX['lon_max']}"
    )

global_attrs = nc_io.make_global_attrs(
    title="GASY-SoilGrids: Global Average Specific Yield based on SoilGrids",
    source=(
        "GASY specific yield (Lv et al., 2025) derived from SoilGrids soil "
        "texture (Hengl et al., 2017; Poggio et al., 2021)"
    ),
    institution="Lv et al. / ISRIC World Soil Information",
    references=(
        "Lv, M., Lv, M., Zha, Y., Wang, L., & Yang, Z.-L. (2025). "
        "A global dataset of average specific yield for soils. "
        "Scientific Data, 12, 427. https://doi.org/10.1038/s41597-025-04742-1 ; "
        "Poggio, L., de Sousa, L.M., Batjes, N.H., et al. (2021). "
        "SoilGrids 2.0. SOIL, 7, 217-240. "
        "https://doi.org/10.5194/soil-7-217-2021 ; "
        "Hengl, T., et al. (2017). SoilGrids250m. PLoS ONE, 12(2), e0169748. "
        "https://doi.org/10.1371/journal.pone.0169748"
    ),
    script=Path(__file__).name,
    extra=extra,
)

nc_io.save_nc(
    ds_out, OUT_PATH,
    fill_value={"sy": np.nan, "crs": None},
    dtype={"sy": "float32", "crs": "int32"},
    global_attrs=global_attrs,
    encoding_overrides={"depth": {"_FillValue": None}},
    complevel=4,
)

print(f"\n{'=' * 70}")
print("  ✓ Merge complete.")
print(f"    Depths:    {[f'{d*100:.0f} cm' for d in DEPTH_VALUES_M]}")
print("    Variable:  sy (depth, lat, lon)  float32")
print(f"{'=' * 70}")

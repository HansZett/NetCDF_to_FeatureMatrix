# -*- coding: utf-8 -*-
"""
01_hydrosheds_raw_to_nc.py

Reads 3 HydroSHEDS GeoTIFFs (DEM, ACC, DIR), subsets them to a common bounding
box, and writes an aligned NetCDF file using the project's nc_io convention.

Input : 01_Input_Global/1_Original/HydroSHEDS_RIVERS_BASINS/eu_*_3s*.tif
Output: 01_Input_Global/2_Intermediate/HydroSHEDS_RIVERS_BASINS/eu_hydrosheds_raw.nc

Location: 02_Processing_Scripts/HydroSHEDS/
Run     : python 01_hydrosheds_raw_to_nc.py
"""

import sys
import gc
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import from_bounds
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
ORIGINAL     = INPUT_GLOBAL / "1_Original" / "HydroSHEDS_RIVERS_BASINS"
INTERMEDIATE = INPUT_GLOBAL / "2_Intermediate" / "HydroSHEDS_RIVERS_BASINS"
FINAL        = INPUT_GLOBAL / "3_Final" / "HydroSHEDS_RIVERS_BASINS"

INTERMEDIATE.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = INTERMEDIATE / "eu_hydrosheds_raw.nc"

# =============================================================================
# CONFIGURATION
# =============================================================================
FILE_PATTERNS = {
    "dem": "eu_dem_3s*.tif",
    "acc": "eu_acc_3s*.tif",
    "dir": "eu_dir_3s*.tif",
}

NODATA = {
    "dem": 32767,
    "acc": 4294967295,
    "dir": 255,
}

# [lon_min, lat_min, lon_max, lat_max]
BBOX = [5.5, 47.0, 15.5, 55.5]

VAR_META = {
    "dem": {
        "long_name"  : "void-filled digital elevation model",
        "units"      : "m",
        "grid_code"  : "DEM",
    },
    "acc": {
        "long_name"  : "flow accumulation – number of upstream cells",
        "units"      : "count",
        "grid_code"  : "ACC",
        "note"       : "Proxy for upstream catchment area; use ACA for area in hectares",
    },
    "dir": {
        "long_name"  : "drainage direction (D8, ESRI convention)",
        "grid_code"  : "DIR",
        "flag_values": np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8),
        "flag_meanings": "E SE S SW W NW N NE",
    },
}

# =============================================================================
# PROCESSING
# =============================================================================
def resolve_paths(input_dir: Path, patterns: dict) -> dict:
    resolved = {}
    for key, pattern in patterns.items():
        matches = sorted(input_dir.glob(pattern))
        if not matches:
            sys.exit(f"[ERROR] No file found for '{pattern}' in {input_dir}")
        resolved[key] = matches[0]
    return resolved

def main():
    print("=" * 65)
    print("  HydroSHEDS TIF → NetCDF (nc_io alignment)")
    print("=" * 65)

    paths = resolve_paths(ORIGINAL, FILE_PATTERNS)
    for key, p in paths.items():
        print(f"  Found {key.upper()} : {p.name}")

    # 1. Read DEM as the reference framework
    print("\nReading reference grid (DEM) & subsetting...")
    with rasterio.open(paths["dem"]) as src:
        window = from_bounds(*BBOX, src.transform)
        transform = src.window_transform(window)

        # Calculate coordinates directly from the windowed transform
        h, w = int(round(window.height)), int(round(window.width))
        lons = np.array([transform.c + (j + 0.5) * transform.a for j in range(w)], dtype=np.float64)
        lats = np.array([transform.f + (i + 0.5) * transform.e for i in range(h)], dtype=np.float64)

        dem_data = src.read(1, window=window).astype(np.float32)
        dem_data[dem_data == NODATA["dem"]] = np.nan

    print(f"  Grid: {h} x {w} cells")

    # 2. Read ACC
    print("Reading ACC...")
    with rasterio.open(paths["acc"]) as src:
        acc_data = src.read(1, window=window).astype(np.float32)
        acc_data[acc_data == NODATA["acc"]] = np.nan

    # 3. Read DIR (integer flag variable)
    print("Reading DIR...")
    with rasterio.open(paths["dir"]) as src:
        # Read as float to handle nodata replacement, will cast back to uint8 on save
        dir_data = src.read(1, window=window).astype(np.float32)
        dir_data[dir_data == NODATA["dir"]] = np.nan

    # 4. Construct xarray Dataset
    print("\nBuilding xarray Dataset...")
    ds = xr.Dataset(
        data_vars={
            "dem": (["lat", "lon"], dem_data, VAR_META["dem"]),
            "acc": (["lat", "lon"], acc_data, VAR_META["acc"]),
            "dir": (["lat", "lon"], dir_data, VAR_META["dir"]),
        },
        coords={
            "lat": (["lat"], lats),
            "lon": (["lon"], lons),
        }
    )

    # Free memory before saving
    del dem_data, acc_data, dir_data
    gc.collect()

    # 5. Save using nc_io
    global_attrs = nc_io.make_global_attrs(
        title="HydroSHEDS raw static layers – Europe (3 arc-second)",
        source="HydroSHEDS v1.1 (Lehner, Verdin, Jarvis 2008; https://www.hydrosheds.org)",
        references="Lehner B., Verdin K., Jarvis A. (2008): New global hydrography derived from spaceborne elevation data. Eos, Transactions 89(10): 93-94.",
        script=Path(__file__).name
    )

    fill_values = {
        "dem": np.nan,
        "acc": np.nan,
        "dir": 255  # 255 is standard NoData for uint8
    }

    dtypes = {
        "dem": "float32",
        "acc": "float32",
        "dir": "uint8"
    }

    nc_io.save_nc(
        ds=ds,
        path=OUTPUT_FILE,
        fill_value=fill_values,
        dtype=dtypes,
        global_attrs=global_attrs,
        compress=True,
        require_tier=True,
        verify=True
    )

    print("\n✅ Done. Next steps: 02a and 02b.")

if __name__ == "__main__":
    main()

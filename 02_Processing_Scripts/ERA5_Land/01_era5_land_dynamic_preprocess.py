# -*- coding: utf-8 -*-
"""
01_era5_land_dynamic_preprocess.py

Cleans the raw ERA5-Land monthly file and writes the standardized final NetCDF.

Steps:
  - rename valid_time→time, latitude→lat, longitude→lon   (nc_io.normalize_coords)
  - drop extra coords: expver, number
  - enforce project orientation (descending lat / ascending lon)   (nc_io)
  - explicit _FillValue = NaN, dtype float32, zlib, CF global attrs (nc_io.save_nc)

Input : 01_Input_Global/1_Original/ERA5_Land/era5_1991_2022_big.nc
Output: 01_Input_Global/3_Final/ERA5_Land/era5_land.nc

Location: 02_Processing_Scripts/ERA5_Land/
Run     : python 01_era5_land_dynamic_preprocess.py
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr

# ============================================================================
# Shared helper
# ============================================================================
sys.path.append(str(Path(__file__).resolve().parents[1] / "0_Shared"))
import nc_io

# ============================================================================
# PATHS  (relative to the project root; adjust here if the layout changes)
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_GLOBAL = PROJECT_ROOT / "01_Input_Global"
ORIGINAL     = INPUT_GLOBAL / "1_Original"
INTERMEDIATE = INPUT_GLOBAL / "2_Intermediate"
FINAL        = INPUT_GLOBAL / "3_Final"

INPUT_FILE  = ORIGINAL / "ERA5_Land" / "era5_1991_2022_big.nc"
OUTPUT_FILE = FINAL / "ERA5_Land" / "era5_land.nc"

# ============================================================================
# Load
# ============================================================================
print("Loading raw ERA5-Land file …")
ds = xr.open_dataset(INPUT_FILE)
print(f"  dims : {dict(ds.sizes)}")
print(f"  vars : {len(ds.data_vars)}")

# ============================================================================
# Normalize coordinates  (valid_time/latitude/longitude → time/lat/lon)
# ============================================================================
ds = nc_io.normalize_coords(ds)

# ============================================================================
# Drop extra coordinates (expver, number)
# ============================================================================
drop = [c for c in ("expver", "number") if c in ds.coords]
if drop:
    ds = ds.drop_vars(drop)
    print(f"  dropped extra coords: {drop}")

# ============================================================================
# Sanity print (AOI + time)
# ============================================================================
print(f"  lat  : {float(ds.lat.min()):.2f}° – {float(ds.lat.max()):.2f}°")
print(f"  lon  : {float(ds.lon.min()):.2f}° – {float(ds.lon.max()):.2f}°")
print(f"  time : {ds.time.values[0]} – {ds.time.values[-1]}  "
      f"({ds.sizes['time']} steps)")

# ============================================================================
# Global attributes
# ============================================================================
attrs = nc_io.make_global_attrs(
    title="ERA5-Land monthly averaged reanalysis 1991–2022 — Germany & surroundings",
    source=("ECMWF ERA5-Land monthly averaged reanalysis, Copernicus CDS "
            "(reanalysis-era5-land-monthly-means)"),
    references=("Muñoz-Sabater et al. (2021), ESSD, "
                "https://doi.org/10.5194/essd-13-4349-2021"),
    script=Path(__file__).name,
)

# ============================================================================
# Save  (tier check, orientation, coord attrs, encoding, verify — all in nc_io)
# ============================================================================
nc_io.save_nc(
    ds, OUTPUT_FILE,
    fill_value=np.nan,        # all 43 variables are continuous floats
    dtype=np.float32,         # align dtype with the other datasets
    global_attrs=attrs,
)

ds.close()
print("✓ ERA5-Land preprocessing complete.")

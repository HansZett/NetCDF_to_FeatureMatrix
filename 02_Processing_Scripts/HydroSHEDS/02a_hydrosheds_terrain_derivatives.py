# -*- coding: utf-8 -*-
"""
02a_hydrosheds_terrain_derivatives.py

Calculates derived topographic variables from eu_hydrosheds_raw.nc:
  slope   – Slope in degrees
  aspect  – Aspect in degrees (0°=N, 90°=E)
  twi     – Topographic Wetness Index ln(acc_ha / tan(slope))

Input : 01_Input_Global/2_Intermediate/HydroSHEDS_RIVERS_BASINS/eu_hydrosheds_raw.nc
Output: 01_Input_Global/2_Intermediate/HydroSHEDS_RIVERS_BASINS/eu_hydrosheds_derived.nc

Location: 02_Processing_Scripts/HydroSHEDS/
Run     : python 02a_hydrosheds_terrain_derivatives.py
"""

import sys
import gc
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
ORIGINAL     = INPUT_GLOBAL / "1_Original" / "HydroSHEDS_RIVERS_BASINS"
INTERMEDIATE = INPUT_GLOBAL / "2_Intermediate" / "HydroSHEDS_RIVERS_BASINS"
FINAL        = INPUT_GLOBAL / "3_Final" / "HydroSHEDS_RIVERS_BASINS"

INPUT_FILE  = INTERMEDIATE / "eu_hydrosheds_raw.nc"
OUTPUT_FILE = INTERMEDIATE / "eu_hydrosheds_derived.nc"

# =============================================================================
# CONFIGURATION
# =============================================================================
EARTH_RADIUS_M = 6_371_000.0
SLOPE_MIN_DEG  = 0.1

VAR_META = {
    "slope": {
        "long_name"   : "terrain slope angle",
        "units"       : "degree",
        "standard_name": "slope_of_terrain",
        "valid_min"   : np.float32(0.0),
        "valid_max"   : np.float32(90.0),
        "note"        : "Derived from HydroSHEDS DEM (3\") via numpy.gradient with geographic correction (dx scaled by cos(lat)).",
    },
    "aspect": {
        "long_name"   : "terrain aspect (exposition)",
        "units"       : "degree",
        "note"        : "0° = North, 90° = East, 180° = South, 270° = West (clockwise). Flat cells (slope < 0.001°) assigned NaN.",
    },
    "twi": {
        "long_name"   : "topographic wetness index",
        "units"       : "1",
        "standard_name": "topographic_wetness_index",
        "note"        : f"ln(a / tan(beta)) where a = specific catchment area (ha), beta = slope (rad, minimum {SLOPE_MIN_DEG}°).",
    },
}

# =============================================================================
# COMPUTATION
# =============================================================================
def compute_cell_metrics(lats: np.ndarray, lons: np.ndarray):
    d_lat  = np.radians(abs(float(lats[1] - lats[0])))
    d_lon  = np.radians(abs(float(lons[1] - lons[0])))
    dy     = EARTH_RADIUS_M * d_lat
    dx     = EARTH_RADIUS_M * d_lon * np.cos(np.radians(lats))
    area_ha = (dy * dx) / 10_000.0
    return dy, dx, area_ha

def compute_slope_aspect(dem: np.ndarray, dy: float, dx: np.ndarray):
    dz_dy_px, dz_dx_px = np.gradient(dem)
    dz_dy_m = dz_dy_px / dy
    dz_dx_m = dz_dx_px / dx[:, np.newaxis]
    del dz_dy_px, dz_dx_px; gc.collect()

    slope_rad = np.arctan(np.sqrt(dz_dx_m**2 + dz_dy_m**2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)

    aspect_math = np.degrees(np.arctan2(dz_dy_m, dz_dx_m))
    aspect_geo  = (90.0 - aspect_math) % 360.0
    aspect_geo  = aspect_geo.astype(np.float32)

    flat_mask = slope_deg < 0.001
    aspect_geo[flat_mask] = np.nan

    del dz_dy_m, dz_dx_m, aspect_math, flat_mask; gc.collect()
    return slope_deg, aspect_geo

def compute_twi(acc: np.ndarray, slope_deg: np.ndarray, area_ha: np.ndarray, slope_min_deg: float):
    acc_ha = acc * area_ha[:, np.newaxis]
    slope_rad = np.radians(np.where(slope_deg < slope_min_deg, slope_min_deg, slope_deg))

    with np.errstate(invalid="ignore", divide="ignore"):
        twi = np.log(acc_ha / np.tan(slope_rad)).astype(np.float32)

    nan_mask = np.isnan(acc) | np.isnan(slope_deg)
    twi[nan_mask] = np.nan

    del acc_ha, slope_rad, nan_mask; gc.collect()
    return twi

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 65)
    print("  HydroSHEDS → Slope / Aspect / TWI")
    print("=" * 65)

    if not INPUT_FILE.exists():
        sys.exit(f"[ERROR] Input file not found: {INPUT_FILE}\nRun 01_hydrosheds_raw_to_nc.py first.")

    print("[1/4] Reading eu_hydrosheds_raw.nc ...")
    ds_in = xr.open_dataset(INPUT_FILE)
    lats = ds_in["lat"].values.astype(np.float64)
    lons = ds_in["lon"].values.astype(np.float64)
    dem  = ds_in["dem"].values.astype(np.float32)
    acc  = ds_in["acc"].values.astype(np.float32)

    h, w = dem.shape
    print(f"      Grid: {h} × {w}")

    print("\n[2/4] Computing cell metrics ...")
    dy, dx, area_ha = compute_cell_metrics(lats, lons)

    print("\n[3/4] Computing Slope and Aspect ...")
    slope_deg, aspect_deg = compute_slope_aspect(dem, dy, dx)

    print("\n[4/4] Computing TWI and assembling Dataset ...")
    twi = compute_twi(acc, slope_deg, area_ha, SLOPE_MIN_DEG)

    # Free memory
    del dem, acc; ds_in.close(); gc.collect()

    ds_out = xr.Dataset(
        data_vars={
            "slope":  (["lat", "lon"], slope_deg, VAR_META["slope"]),
            "aspect": (["lat", "lon"], aspect_deg, VAR_META["aspect"]),
            "twi":    (["lat", "lon"], twi, VAR_META["twi"]),
        },
        coords={
            "lat": (["lat"], lats),
            "lon": (["lon"], lons),
        }
    )

    del slope_deg, aspect_deg, twi; gc.collect()

    global_attrs = nc_io.make_global_attrs(
        title="HydroSHEDS derived terrain indices – Europe (3 arc-second)",
        source="Derived from eu_hydrosheds_raw.nc (HydroSHEDS v1.1)",
        script=Path(__file__).name
    )

    nc_io.save_nc(
        ds=ds_out,
        path=OUTPUT_FILE,
        fill_value=np.nan, # all vars are continuous floats
        dtype="float32",
        global_attrs=global_attrs,
        compress=True,
        require_tier=True,
        verify=True
    )

    print("\n✅ Done.")

if __name__ == "__main__":
    main()

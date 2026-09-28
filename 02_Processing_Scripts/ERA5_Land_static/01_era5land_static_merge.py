# -*- coding: utf-8 -*-
"""
01_era5land_static_merge.py

Reads all ERA5-Land *invariant* (static) NetCDF files from 1_Original,
clips them to Germany's bounding box, harmonizes coordinates to lat/lon,
cleans up categorical variables, and merges everything into a single
NetCDF using the project's nc_io convention.

Input : 01_Input_Global/1_Original/ERA5_Land_static/*.nc
Output: 01_Input_Global/3_Final/ERA5_Land_static/ERA5_Land_static_Germany.nc

Location: 02_Processing_Scripts/ERA5_Land_static/
Run     : python 01_era5land_static_merge.py

Categorical handling
--------------------
The class tables of slt, tvl and tvh follow the ECMWF parameter documentation
(documentation_era5_static.txt). Valid codes are checked by MEMBERSHIP in the
class list, not by a minimum/maximum range, because the codes are not
contiguous. A range test (for example 0 to 9) would discard the high
vegetation classes 18 (mixed forest/woodland) and 19 (interrupted forest).

The code 0 is not part of the documented tables. It marks the absence of the
respective cover type and is kept as a class for tvl and tvh. For slt it marks
a cell without soil and is treated as missing, so that no artificial texture
class is created.
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
ORIGINAL     = INPUT_GLOBAL / "1_Original" / "ERA5_Land_static"
INTERMEDIATE = INPUT_GLOBAL / "2_Intermediate" / "ERA5_Land_static"
FINAL        = INPUT_GLOBAL / "3_Final" / "ERA5_Land_static"

FINAL.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = FINAL / "ERA5_Land_static_Germany.nc"

# =============================================================================
# CONFIGURATION
# =============================================================================
# Germany bounding box
LAT_MIN, LAT_MAX = 47.0, 55.5
LON_MIN, LON_MAX = 5.5,  15.5

UNITS_FIX = {
    "cl":  "1",
    "cvl": "1",
    "cvh": "1",
    "lsm": "1",
    "glm": "1",
    "dl":  "m",
}

# -----------------------------------------------------------------------------
# CATEGORICAL DEFINITIONS
# -----------------------------------------------------------------------------
# Class names and codes are taken verbatim from the ECMWF parameter
# documentation. The shared vegetation table is used by both tvl and tvh; each
# parameter accepts only its own subset.
DOC_VEG_MEANING = {
    1:  "crops_mixed_farming",
    2:  "grass",
    3:  "evergreen_needleleaf_trees",
    4:  "deciduous_needleleaf_trees",
    5:  "deciduous_broadleaf_trees",
    6:  "evergreen_broadleaf_trees",
    7:  "tall_grass",
    8:  "desert",
    9:  "tundra",
    10: "irrigated_crops",
    11: "semidesert",
    12: "ice_caps_and_glaciers",
    13: "bogs_and_marshes",
    14: "inland_water",
    15: "ocean",
    16: "evergreen_shrubs",
    17: "deciduous_shrubs",
    18: "mixed_forest_woodland",
    19: "interrupted_forest",
    20: "water_and_land_mixtures",
}

DOC_SOIL_MEANING = {
    1: "coarse",
    2: "medium",
    3: "medium_fine",
    4: "fine",
    5: "very_fine",
    6: "organic",
    7: "tropical_organic",
}

HIGH_VEG_CLASSES = [3, 4, 5, 6, 18, 19]
LOW_VEG_CLASSES = [1, 2, 7, 9, 10, 11, 13, 16, 17, 20]
NO_SURFACE_VEG_CLASSES = [8, 12, 14, 15]   # desert, ice caps, inland water, ocean
SOIL_CLASSES = [1, 2, 3, 4, 5, 6, 7]

# Code 0 is not in the documentation. For tvl and tvh it marks the absence of
# the respective cover type and is a real value, so it is kept. For slt it marks
# a cell without soil and is treated as missing.
KEEP_ZERO = {"slt": False, "tvl": True, "tvh": True}

# The four no-surface-vegetation codes do not occur in the source files, neither
# globally nor over Germany. They are therefore not accepted. If a future
# download contains them, the warning in the cleaning loop reports them instead
# of discarding them silently.
KEEP_NO_SURFACE_VEG = False


def _flags(base: list, var: str) -> list:
    """Accepted class codes for one variable."""
    f = list(base)
    if KEEP_NO_SURFACE_VEG and var in ("tvl", "tvh"):
        f += NO_SURFACE_VEG_CLASSES
    if KEEP_ZERO[var]:
        f += [0]
    return sorted(f)


CATEGORICALS = {
    "slt": {"flags": _flags(SOIL_CLASSES, "slt"),
            "lut": {**DOC_SOIL_MEANING, 0: "no_soil"}},
    "tvl": {"flags": _flags(LOW_VEG_CLASSES, "tvl"),
            "lut": {**DOC_VEG_MEANING, 0: "no_low_vegetation"}},
    "tvh": {"flags": _flags(HIGH_VEG_CLASSES, "tvh"),
            "lut": {**DOC_VEG_MEANING, 0: "no_high_vegetation"}},
}


# =============================================================================
# HELPERS
# =============================================================================
def preprocess_file(ds: xr.Dataset, fname: str) -> xr.Dataset:
    """Normalize coords, drop time, fix longitude wrapping, and clip to bbox."""
    ds = nc_io.normalize_coords(ds)

    # 1. Squeeze out time/step/surface dimensions
    for tdim in ["time", "valid_time", "forecast_reference_time", "step",
                 "surface", "number", "expver"]:
        if tdim in ds.dims:
            ds = ds.isel({tdim: 0}, drop=True)
        if tdim in ds.coords or tdim in ds.data_vars:
            ds = ds.drop_vars(tdim, errors="ignore")

    # 2. Longitude Wrap (0..360 -> -180..180) if necessary
    if "lon" in ds.coords and float(ds.lon.max()) > 180:
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))

    # 3. Sort for robust clipping
    ds = ds.sortby("lat").sortby("lon")

    # 4. Clip to Bounding Box
    ds = ds.sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))

    return ds


def clean_categorical(da: xr.DataArray, var: str) -> xr.DataArray:
    """
    Round to integer codes and set every code outside the accepted class list
    to NaN. Membership test, not a range test, because the valid codes are not
    contiguous. NaN in the source stays NaN.
    """
    flags = CATEGORICALS[var]["flags"]
    lut = CATEGORICALS[var]["lut"]

    rounded = np.around(np.asarray(da.values, dtype="float64")).astype("float32")
    valid = np.isin(rounded, np.array(flags, dtype="float32"))

    # Code 0 is excluded on purpose where KEEP_ZERO is False. Report that
    # separately, so the warning stays reserved for genuinely unexpected codes.
    rejected = ~valid & np.isfinite(rounded)
    if not KEEP_ZERO[var]:
        n_zero = int((rejected & (rounded == 0)).sum())
        if n_zero:
            print(f"    code 0 set to missing by configuration: {n_zero} cells")
        rejected = rejected & (rounded != 0)

    n_bad = int(rejected.sum())
    if n_bad:
        bad = np.unique(rounded[rejected])
        print(f"    WARNING: {n_bad} cells carry codes outside the accepted "
              f"class list and were set to missing: {[int(b) for b in bad[:20]]}")

    out = da.copy(data=np.where(valid, rounded, np.nan))

    present = np.unique(rounded[valid]).astype("int64")
    counts = {int(c): int((rounded == c).sum()) for c in present}
    print(f"    accepted classes: "
          f"{ {int(c): lut.get(int(c), f'code_{int(c)}') for c in present} }")
    print(f"    cells per class : {counts}")
    print(f"    missing cells   : {int(np.isnan(out.values).sum())} of "
          f"{out.size}")

    out.attrs["flag_values"] = np.array(flags, dtype=np.uint8)
    out.attrs["flag_meanings"] = " ".join(
        lut.get(c, f"code_{c}") for c in flags)
    out.attrs["units"] = "1"
    return out


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 65)
    print("  ERA5-Land Static Variables → Germany Merge")
    print("=" * 65)

    if not ORIGINAL.exists():
        sys.exit(f"[ERROR] Input directory not found:\n  {ORIGINAL}")

    nc_files = sorted(ORIGINAL.glob("*.nc"))
    if not nc_files:
        sys.exit(f"[ERROR] No .nc files found in:\n  {ORIGINAL}")

    print(f"Found {len(nc_files)} NetCDF file(s). Processing...")

    datasets = []
    for f in nc_files:
        print(f"  Reading {f.name}...", end=" ", flush=True)
        ds_in = xr.open_dataset(f).load()

        # Drop stray CRS variables from source files
        for v in ("crs", "spatial_ref"):
            if v in ds_in.variables:
                ds_in = ds_in.drop_vars(v, errors="ignore")

        ds_clipped = preprocess_file(ds_in, f.name)
        datasets.append(ds_clipped)
        ds_in.close()
        print("Done.")

    print("\nMerging all datasets...")
    # join="override" ensures small coordinate precision differences don't create NaNs
    ds_merged = xr.merge(datasets, compat="override", join="override")
    del datasets; gc.collect()

    print("Post-processing variables...")

    # Drop geopotential if present
    if "z" in ds_merged.data_vars:
        ds_merged = ds_merged.drop_vars("z")
        print("  Dropped variable 'z' (geopotential)")

    fill_values = {}
    dtypes = {}

    for var in ds_merged.data_vars:
        # 1. Handle Categoricals
        if var in CATEGORICALS:
            print(f"  Cleaning categorical: {var} -> uint8")
            ds_merged[var] = clean_categorical(ds_merged[var], var)
            dtypes[var] = "uint8"
            fill_values[var] = 255

        # 2. Handle Continuous Variables
        else:
            print(f"  Formatting continuous: {var} -> float32")
            if var in UNITS_FIX:
                ds_merged[var].attrs["units"] = UNITS_FIX[var]

            dtypes[var] = "float32"
            fill_values[var] = np.nan

    # =========================================================================
    # ENCODING & METADATA
    # =========================================================================
    global_attrs = nc_io.make_global_attrs(
        title="ERA5-Land invariant fields, clipped to Germany",
        source="ECMWF ERA5-Land (Copernicus Climate Data Store)",
        references="Muñoz Sabater, J., (2019): ERA5-Land hourly data from 1950 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS).",
        script=Path(__file__).name,
        extra={
            "comment": (
                "Categorical variables slt, tvl and tvh were rounded to integer "
                "class codes, checked against the ECMWF class tables by "
                "membership, converted to uint8 and annotated with flag_values "
                "and flag_meanings. Codes outside the class table were set to "
                "the fill value 255. The undocumented code 0 was kept as a "
                "class for tvl and tvh, where it marks the absence of the "
                "respective cover type, and was set to missing for slt, where "
                "it marks a cell without soil. Geopotential (z) was dropped."
            )
        }
    )

    print("\nSaving final dataset...")
    nc_io.save_nc(
        ds=ds_merged,
        path=OUTPUT_FILE,
        fill_value=fill_values,
        dtype=dtypes,
        global_attrs=global_attrs,
        compress=True,
        require_tier=True,
        verify=True
    )

    print("\n✅ Script complete.")


if __name__ == "__main__":
    main()

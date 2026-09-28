# -*- coding: utf-8 -*-
"""
01_c3s_lcc_merge.py

Merges all annual C3S/CCI Land Cover NetCDF files (300 m) from two product
versions into one consolidated, CF-compliant NC file sorted by year.

Source versions:
  v2.0.7cds  → 1992–2015
  v2.1.1     → 2016–present

Integer / flag dataset — explicit NoData sentinels (distinct from true 0):
  lccs_class          uint8   NoData = 0   (0 is the product's genuine NoData)
  processed_flag      uint8   NoData = 255
  current_pixel_state uint8   NoData = 255
  change_count        uint8   NoData = 255
  observation_count   uint16  NoData = 65535

Top-of-range sentinels keep these variables compact while letting a real 0
(e.g. change_count = 0 "no changes") be told apart from missing data.
Orientation (descending lat / ascending lon), per-variable dtypes + fill,
zlib and the CF global-attribute block are applied by nc_io.save_nc.

Input : 01_Input_Global/1_Original/C3S_LCC/*.nc
Output: 01_Input_Global/3_Final/C3S_LCC/c3s_lcc.nc

Location: 02_Processing_Scripts/C3S_LCC/
Run     : python 01_c3s_lcc_merge.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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

INPUT_DIR   = ORIGINAL / "C3S_LCC"
OUTPUT_FILE = FINAL / "C3S_LCC" / "c3s_lcc.nc"

# ============================================================================
# Configuration
# ============================================================================
VARIABLES_PRIMARY   = ["lccs_class"]
VARIABLES_AUXILIARY = ["processed_flag", "current_pixel_state",
                       "observation_count", "change_count"]
VERSION_NEW_CUTOFF  = 2016    # ≥ 2016 → v2.1.1  |  < 2016 → v2.0.7cds

# Per-variable output dtype (everything else would default to float32)
DTYPE_MAP = {
    "lccs_class":          np.uint8,
    "processed_flag":      np.uint8,
    "current_pixel_state": np.uint8,
    "change_count":        np.uint8,
    "observation_count":   np.uint16,
}

# Per-variable NoData sentinel. Top-of-dtype-range so a true 0 stays distinct
# from missing. lccs_class keeps 0 (its only 0 is the product's NoData class).
FILL_MAP = {
    "lccs_class":          0,
    "processed_flag":      255,
    "current_pixel_state": 255,
    "change_count":        255,
    "observation_count":   65535,
}

# ============================================================================
# Helpers
# ============================================================================

def extract_year(path: Path) -> int:
    m = re.search(r"(\d{4})", path.name)
    if not m:
        raise ValueError(f"Cannot extract year from filename: {path.name}")
    return int(m.group(1))


def infer_version(year: int) -> str:
    return "v2.1.1" if year >= VERSION_NEW_CUTOFF else "v2.0.7cds"


def load_annual_slice(nc_path: Path, year: int) -> xr.Dataset:
    """Load one annual file → Dataset with dims (time=1, lat, lon)."""
    ds = nc_io.normalize_coords(xr.open_dataset(nc_path, decode_times=False))

    keep = [v for v in VARIABLES_PRIMARY + VARIABLES_AUXILIARY if v in ds.data_vars]
    missing_primary = [v for v in VARIABLES_PRIMARY if v not in ds.data_vars]
    if missing_primary:
        raise KeyError(
            f"Primary variable(s) {missing_primary} not found in {nc_path.name}. "
            f"Available: {list(ds.data_vars)}"
        )
    ds = ds[keep]

    # Drop degenerate singleton time dim if present
    for v in keep:
        if "time" in ds[v].dims and ds[v].sizes["time"] == 1:
            ds[v] = ds[v].isel(time=0)
    if "time" in ds.dims and ds.sizes["time"] == 1:
        ds = ds.isel(time=0)

    # NOTE: do NOT pre-cast / fillna here. Float flag variables are cast to
    # their target uint dtype at write time by nc_io.save_nc, where missing
    # values become the NoData sentinel (FILL_MAP) and a true 0 is preserved.

    # Mid-year timestamp
    ds = ds.expand_dims(dim={"time": [pd.Timestamp(f"{year}-07-02")]})

    # CF-compliant variable attributes
    _attrs = {
        "lccs_class": {
            "long_name":     "Land Cover Classification (LCCS)",
            "standard_name": "land_cover_lccs",
            "units":         "1",
            "flag_values":   ("0, 10, 11, 12, 20, 30, 40, 50, 60, 61, 62, "
                              "70, 71, 72, 80, 81, 82, 90, 100, 110, 120, 121, 122, "
                              "130, 140, 150, 151, 152, 153, 160, 170, 180, 190, "
                              "200, 201, 202, 210, 220"),
            "flag_meanings": (
                "NoData Rainfed_cropland Herbaceous_rainfed_cropland "
                "Tree_or_shrub_rainfed_cropland Irrigated_cropland "
                "Mosaic_cropland_gt50pct_naturalveg_lt50pct "
                "Mosaic_naturalveg_gt50pct_cropland_lt50pct "
                "Tree_broadleaved_evergreen_closed_to_open "
                "Tree_broadleaved_deciduous_closed_to_open "
                "Tree_broadleaved_deciduous_closed "
                "Tree_broadleaved_deciduous_open "
                "Tree_needleleaved_evergreen_closed_to_open "
                "Tree_needleleaved_evergreen_closed "
                "Tree_needleleaved_evergreen_open "
                "Tree_needleleaved_deciduous_closed_to_open "
                "Tree_needleleaved_deciduous_closed "
                "Tree_needleleaved_deciduous_open "
                "Tree_mixed_leaf_type "
                "Mosaic_tree_and_shrub_gt50pct_herbaceous_lt50pct "
                "Mosaic_herbaceous_gt50pct_tree_and_shrub_lt50pct "
                "Shrubland Evergreen_shrubland Deciduous_shrubland "
                "Grassland Lichens_and_mosses Sparse_vegetation "
                "Sparse_tree Sparse_shrub Sparse_herbaceous "
                "Tree_flooded_freshwater_or_brackish Tree_flooded_saline "
                "Shrub_or_herbaceous_flooded Urban Bare_areas "
                "Consolidated_bare_areas Unconsolidated_bare_areas "
                "Water Snow_and_ice"
            ),
            "comment": (
                "LCCS class codes mapped to 6 IPCC classes: "
                "Agriculture (10-12, 20, 30, 40), Forest (50-100, 160, 170), "
                "Grassland (110, 130), Wetland (180), Settlement (190), "
                "Other (120-122, 140, 150-153, 200-202, 210, 220). NoData = 0."
            ),
            "valid_range": np.array([0, 220], dtype=np.uint8),
        },
        "processed_flag": {
            "long_name":     "Processing flag",
            "flag_values":   "0, 1",
            "flag_meanings": "not_processed processed",
            "comment":       ("Whether the pixel was processed (1) for the baseline "
                              "LC map. Cast from float32 to uint8."),
        },
        "current_pixel_state": {
            "long_name":     "Pixel state in baseline LC map",
            "flag_values":   "0, 1, 2, 3, 4, 5",
            "flag_meanings": "invalid clear_land clear_water clear_snow_ice cloud cloud_shadow",
            "comment":       "Pre-processing pixel status. Cast from float32 to uint8.",
        },
        "observation_count": {
            "long_name": "Number of valid observations used for baseline classification",
            "units":     "1",
            "comment":   "Stored as uint16; values can exceed 255.",
        },
        "change_count": {
            "long_name": "Number of LC changes observed",
            "units":     "1",
            "comment":   "Updated with each new annual LC map release.",
        },
    }
    for v in keep:
        if v in _attrs:
            ds[v].attrs.update(_attrs[v])

    return ds


# ============================================================================
# Discover files
# ============================================================================
print("=" * 70)
print("C3S_LCC — Annual Merge")
print("=" * 70)
print(f"Input  : {INPUT_DIR}")
print(f"Output : {OUTPUT_FILE}")

nc_files = sorted(INPUT_DIR.glob("*.nc"))
if not nc_files:
    raise FileNotFoundError(f"No .nc files found in:\n  {INPUT_DIR}")

year_to_file: dict[int, Path] = {}
for f in nc_files:
    try:
        y = extract_year(f)
    except ValueError as e:
        print(f"  ⚠  Skipping (cannot parse year): {f.name}  — {e}")
        continue
    if y in year_to_file:
        print(f"  ⚠  Duplicate year {y}: keeping {f.name}, "
              f"dropping {year_to_file[y].name}")
    year_to_file[y] = f

years = sorted(year_to_file.keys())
print(f"\nYears detected : {years[0]}–{years[-1]}  ({len(years)} files)")

# ============================================================================
# Load & concatenate
# ============================================================================
slices = []
for year in years:
    f = year_to_file[year]
    print(f"  Loading {year}  [{infer_version(year)}]  {f.name} …")
    slices.append(load_annual_slice(f, year))

print("\nConcatenating along time …")
ds_merged = xr.concat(slices, dim="time").sortby("time")

# Custom time-coordinate comment (standard lat/lon/time attrs added by nc_io)
ds_merged["time"].attrs["comment"] = (
    "Annual mid-year timestamp (YYYY-07-02); represents the full calendar year."
)

# ============================================================================
# Global attributes
# ============================================================================
y0, y1 = years[0], years[-1]
attrs = nc_io.make_global_attrs(
    title=f"ESA CCI Land Cover — Annual LCCS maps 300 m, {y0}–{y1}",
    institution="ESA Climate Change Initiative – Land Cover project (CCI-LC)",
    source=(f"C3S/CCI Land Cover v2.0.7cds ({y0}–{VERSION_NEW_CUTOFF-1}) and "
            f"v2.1.1 ({VERSION_NEW_CUTOFF}–{y1}), Copernicus CDS"),
    references="https://cds.climate.copernicus.eu/datasets/satellite-land-cover",
    script=Path(__file__).name,
    extra={
        "product_version":    (f"v2.0.7cds (≤{VERSION_NEW_CUTOFF-1}) | "
                               f"v2.1.1 (≥{VERSION_NEW_CUTOFF})"),
        "spatial_resolution": "300 m (~0.00278°)",
        "comment": (
            "Annual land cover classification (LCCS, 38 class codes incl. "
            "snow_and_ice=220, NoData=0), combined from two product releases "
            "with a consistent class legend. 'processed_flag' and "
            "'current_pixel_state' were cast from float32 to uint8; "
            "'observation_count' is uint16 to preserve values > 255."
        ),
        "license": "Copernicus License v1.2",
    },
)

# ============================================================================
# Save  (integer fill = 0, per-variable dtypes, explicit CF time encoding)
# ============================================================================
nc_io.save_nc(
    ds_merged, OUTPUT_FILE,
    fill_value=FILL_MAP,      # per-variable NoData sentinels (true 0 ≠ fill)
    dtype=DTYPE_MAP,
    global_attrs=attrs,
    encoding_overrides={
        "time": {"dtype": "float64",
                 "units": "days since 1970-01-01",
                 "calendar": "proleptic_gregorian"},
    },
)

print("=" * 70)
print("✅  C3S_LCC merge complete.")
print("=" * 70)

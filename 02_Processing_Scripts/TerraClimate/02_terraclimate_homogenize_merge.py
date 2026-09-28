# -*- coding: utf-8 -*-
"""
02_terraclimate_homogenize_merge.py

TerraClimate – homogenize and merge all variables into a single NetCDF file.

Pipeline:  1_Original → 3_Final   (no 2_Intermediate stage).

- Reads the 14 individual TerraClimate NC files from 1_Original/TerraClimate/
- Decodes the 'days since 1900-01-01' (gregorian) time axis to CF datetimes
  -> clean first-of-month stamps, no drift to snap
- Renames variables to clean English names, attaches long_name + units
- Merges all variables into a single dataset
- Writes via nc_io.save_nc:
    * lat descending / lon ascending (enforced; already native here)
    * float32, explicit NaN _FillValue, zlib complevel 4
    * CF-1.8 global attrs (make_global_attrs) + auto geospatial/time bounds
    * time encoded as 'days since 1970-01-01' (float64)

Input : 01_Input_Global/1_Original/TerraClimate/TerraClimate_*.nc
        (from 01_terraclimate_download.py)
Output: 01_Input_Global/3_Final/TerraClimate/TerraClimate.nc

Location: 02_Processing_Scripts/TerraClimate/
Run     : python 02_terraclimate_homogenize_merge.py
"""

import sys
import re
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

INPUT_DIR   = ORIGINAL / "TerraClimate"
OUTPUT_DIR  = FINAL / "TerraClimate"
OUTPUT_FILE = OUTPUT_DIR / "TerraClimate.nc"

# =============================================================================
# VARIABLE MAPPING
# Maps filename stem pattern → clean variable name in merged NC
# =============================================================================
VARIABLE_MAP = {
    "ET_actual":              "ET_actual",
    "ET_potential":           "ET_potential",
    "water_deficit":          "water_deficit",
    "precipitation":          "precipitation",
    "runoff":                 "runoff",
    "soil_moisture":          "soil_moisture",
    "solar_radiation":        "solar_radiation",
    "snow_water":             "snow_water",
    "temp_max":               "temp_max",
    "temp_min":               "temp_min",
    "vapor_pressure":         "vapor_pressure",
    "vapor_pressure_deficit": "vapor_pressure_deficit",
    "drought_index_pdsi":     "drought_index_pdsi",
    "wind_speed":             "wind_speed",
}

# Per-variable long_name + units (CF metadata)
VARIABLE_ATTRS = {
    "ET_actual":              {"long_name": "Actual Evapotranspiration",       "units": "mm/month"},
    "ET_potential":           {"long_name": "Potential Evapotranspiration",    "units": "mm/month"},
    "water_deficit":          {"long_name": "Climate Water Deficit",           "units": "mm/month"},
    "precipitation":          {"long_name": "Precipitation",                   "units": "mm/month"},
    "runoff":                 {"long_name": "Runoff",                          "units": "mm/month"},
    "soil_moisture":          {"long_name": "Soil Moisture",                   "units": "mm"},
    "solar_radiation":        {"long_name": "Solar Radiation",                 "units": "W/m2"},
    "snow_water":             {"long_name": "Snow Water Equivalent",           "units": "mm"},
    "temp_max":               {"long_name": "Maximum Temperature",             "units": "degC"},
    "temp_min":               {"long_name": "Minimum Temperature",             "units": "degC"},
    "vapor_pressure":         {"long_name": "Vapor Pressure",                  "units": "kPa"},
    "vapor_pressure_deficit": {"long_name": "Vapor Pressure Deficit",          "units": "kPa"},
    "drought_index_pdsi":     {"long_name": "Palmer Drought Severity Index",   "units": "unitless"},
    "wind_speed":             {"long_name": "Wind Speed",                      "units": "m/s"},
}

# =============================================================================
# HELPERS
# =============================================================================
def decode_time(ds):
    """Decode the CF time axis ('days since 1900-01-01', gregorian) to datetimes.

    For these TerraClimate files the axis decodes cleanly to first-of-month
    stamps. If decoding fails, an error is raised instead of constructing a
    date range, so that no wrong time axis can enter the output.
    """
    if "time" not in ds.dims:
        return ds
    try:
        return xr.decode_cf(ds)
    except Exception as exc:
        raise RuntimeError(
            f"Could not decode the TerraClimate time axis "
            f"(units={ds.time.attrs.get('units', 'unknown')!r}): {exc}"
        )


def get_var_key(filename):
    """Extract the variable key from e.g. TerraClimate_ET_actual_1991-2022_AOI.nc"""
    stem = Path(filename).stem                      # TerraClimate_ET_actual_1991-2022_AOI
    stem = re.sub(r"^TerraClimate_", "", stem)      # ET_actual_1991-2022_AOI
    stem = re.sub(r"_\d{4}-\d{4}_AOI$", "", stem)   # ET_actual
    return stem


# =============================================================================
# MAIN
# =============================================================================
nc_files = sorted(INPUT_DIR.glob("TerraClimate_*.nc"))

if not nc_files:
    print(f"ERROR: No TerraClimate files found in {INPUT_DIR}")
    sys.exit()

print(f"Input:  {INPUT_DIR}")
print(f"Output: {OUTPUT_FILE}")
print(f"\nFound {len(nc_files)} file(s):")
for f in nc_files:
    print(f"  {f.name}")

datasets = []

for nc_file in nc_files:
    var_key    = get_var_key(nc_file.name)
    clean_name = VARIABLE_MAP.get(var_key)

    if clean_name is None:
        print(f"\n  WARNING: No mapping for '{var_key}' – skipping {nc_file.name}")
        continue

    print(f"\n  Processing: {nc_file.name}")
    print(f"    Variable key: {var_key} → {clean_name}")

    ds = xr.open_dataset(nc_file, decode_times=False)
    ds = decode_time(ds)

    # Keep only the real data variable (drop any coord-like vars stored as vars)
    data_vars = [v for v in ds.data_vars
                 if v not in ("lat", "lon", "time", "crs", "spatial_ref")]

    if len(data_vars) != 1:
        print(f"    WARNING: Expected 1 data variable, found {len(data_vars)}: {data_vars}")

    original_var = data_vars[0]
    ds = ds[[original_var]].rename({original_var: clean_name})

    # Replace source attrs with our clean long_name + units
    if clean_name in VARIABLE_ATTRS:
        ds[clean_name].attrs = VARIABLE_ATTRS[clean_name]

    datasets.append(ds)
    print(f"    ✓ Ready: {clean_name}")

if not datasets:
    print("ERROR: No datasets to merge.")
    sys.exit()

# =============================================================================
# MERGE & SAVE
# =============================================================================
print(f"\nMerging {len(datasets)} variables...")
merged = xr.merge(datasets)
print(f"Merged dimensions: {dict(merged.sizes)}")
print(f"Variables: {list(merged.data_vars)}")

global_attrs = nc_io.make_global_attrs(
    title="TerraClimate v1.1 — Germany 1991–2022",
    source="TerraClimate v1.1 (with ERA5 forcing); Climatology Lab, UC Merced",
    institution="Climatology Lab, University of California Merced",
    references="Abatzoglou et al. (2018), Scientific Data, doi:10.1038/sdata.2017.191",
    crs="EPSG:4326 (WGS84)",
    script=Path(__file__).name,
    extra={
        "spatial_coverage":   "Germany and surroundings (5.5–15.5°E, 47.0–55.5°N)",
        "spatial_resolution": "0.04167° (~4.5 km)",
        "time_resolution":    "monthly",
    },
)

print(f"\nSaving to: {OUTPUT_FILE}")
nc_io.save_nc(
    merged,
    OUTPUT_FILE,
    fill_value=np.nan,          # all 14 variables are continuous floats
    dtype="float32",
    global_attrs=global_attrs,
    compress=True,
    complevel=4,
    encoding_overrides={
        "time": {"dtype":    "float64",
                 "units":    "days since 1970-01-01",
                 "calendar": "standard"},
    },
)

for ds in datasets:
    ds.close()
merged.close()

print("\n✓ Done.")
print(f"  Output: {OUTPUT_FILE}")

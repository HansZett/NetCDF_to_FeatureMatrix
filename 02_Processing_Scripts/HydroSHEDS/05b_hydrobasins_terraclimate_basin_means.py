# -*- coding: utf-8 -*-
"""
05b_hydrobasins_terraclimate_basin_means.py

Per-basin monthly time series of TerraClimate v1.1 climate forcing for
HydroBASINS levels 06 and 07.

This is the TerraClimate counterpart of 05a_hydrobasins_era5_basin_means.py
(ERA5-Land). It is deliberately a SEPARATE script with SEPARATE outputs rather than extra columns
on the ERA5 CSVs, because TerraClimate lives on a different grid (~4.5 km vs
ERA5-Land ~9 km) and a different time axis (1991–2022). That means its
rasterized basin areas, coverage fractions, and even the set of basins
intersecting the AOI all differ from ERA5's, so the two cannot share the
basin_area / covered_area / coverage columns. To make a later join painless the
per-basin CSVs use the SAME <HYBAS_ID>.csv filenames in a parallel folder and
every TerraClimate column is prefixed `terra_`, so that

    pd.merge(era5_df, terra_df, on="time")

works without column collisions.

Variables extracted (14, native TerraClimate units — NO unit conversion):
  precipitation           monthly total            mm
  runoff                  monthly total            mm
  ET_actual               monthly total            mm
  ET_potential            monthly total            mm
  water_deficit           monthly total (climatic) mm
  soil_moisture           end-of-month state       mm
  snow_water              end-of-month state       mm
  solar_radiation         monthly mean             W/m2
  temp_max                monthly mean             degC
  temp_min                monthly mean             degC
  vapor_pressure          monthly mean             kPa
  vapor_pressure_deficit  monthly mean             kPa
  drought_index_pdsi      end-of-month state       unitless
  wind_speed              monthly mean             m/s

⚠ UNITS (important — differs from the ERA5 script):
  TerraClimate variables in TerraClimate.nc are ALREADY in their final units
  (the homogenize step attached mm / W/m2 / degC / kPa / m/s). There is NO
  metres→mm (×1000) step here. The area-weighting below produces the
  AREA-WEIGHTED BASIN MEAN of each variable in its native unit:
    - "monthly total" variables (precip, runoff, ET, deficit) → basin-mean of
      the monthly total, i.e. mm/month.
    - "monthly mean" variables (temp, radiation, vapor, wind) → basin-mean of
      the monthly mean.
    - "end-of-month state" variables (soil moisture, snow, PDSI) → basin-mean
      of the end-of-month value.
  The area-weighted mean is the correct basin aggregate for BOTH extensive
  (mm) and intensive (degC, W/m2, …) quantities; only the interpretation of the
  per-month number differs, which the column name + README make explicit.

Per basin / month / variable, exported column (names as listed in VAR_SPEC;
some carry a unit suffix, e.g. terra_temp_max_degC, others do not, e.g.
terra_precipitation):
  terra_<var>[_<unit>]  area-weighted basin mean, normalised over the part of
                        the basin actually COVERED by valid TerraClimate data
Plus:
  terra_basin_area_m2     full rasterized basin area on the TerraClimate grid (m²)
  terra_covered_area_m2   area with valid (non-NaN) TerraClimate data (m²)
  terra_coverage_frac     terra_covered_area_m2 / terra_basin_area_m2 (0..1)

NaN / coverage handling: TerraClimate is a land surface product, so ocean (and
out-of-domain) pixels are NaN, and that geometry is fixed in time. We average
each variable only over its own valid (non-NaN) pixels in BOTH numerator and
denominator. This avoids NaN poisoning (one ocean pixel turning a basin mean
into NaN) and the low bias from dividing a partial-coverage sum by the full
basin area. Coverage is reported using `precipitation` as the land reference
(it is defined everywhere over land); per-variable footprints are essentially
identical in TerraClimate, so this single fraction is representative. Basins
with no covered land at all yield NaN means.

Inputs
  TerraClimate : 01_Input_Global/3_Final/TerraClimate/TerraClimate.nc
                 (from TerraClimate/02_terraclimate_homogenize_merge.py)
  Shapefiles   : 01_Input_Global/1_Original/HydroSHEDS_RIVERS_BASINS/
                 hybas_eu_lev06_v1c/, hybas_eu_lev07_v1c/
Outputs (01_Input_Global/3_Final/HydroSHEDS_RIVERS_BASINS/)
  basin_terraclimate_lev06/<HYBAS_ID>.csv
  basin_terraclimate_lev07/<HYBAS_ID>.csv
  metadata_terraclimate_lev0X.csv
  basin_mask_terraclimate_lev0X.nc   (basin-ID raster on the TerraClimate grid)
  README_terraclimate_units.txt

Location: 02_Processing_Scripts/HydroSHEDS/
Run     : python 05b_hydrobasins_terraclimate_basin_means.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rasterio.features
import rasterio.transform
from scipy.ndimage import sum_labels
from shapely.geometry import box

warnings.filterwarnings("ignore")

# ============================================================================
# SHARED HELPER
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

NC_INPUT = FINAL / "TerraClimate" / "TerraClimate.nc"

SHP_DIR   = ORIGINAL / "HydroSHEDS_RIVERS_BASINS"
SHP_LEV06 = SHP_DIR / "hybas_eu_lev06_v1c" / "hybas_eu_lev06_v1c.shp"
SHP_LEV07 = SHP_DIR / "hybas_eu_lev07_v1c" / "hybas_eu_lev07_v1c.shp"

OUT_BASE = FINAL / "HydroSHEDS_RIVERS_BASINS"
OUT_BASE.mkdir(parents=True, exist_ok=True)

LEVELS = [
    {"name": "lev06", "shp": SHP_LEV06, "out_dir": OUT_BASE / "basin_terraclimate_lev06"},
    {"name": "lev07", "shp": SHP_LEV07, "out_dir": OUT_BASE / "basin_terraclimate_lev07"},
]

# ============================================================================
# CONFIGURATION
# Variable spec — single source of truth for names, units, column names.
# `nc_name`  : variable name inside TerraClimate.nc (the `terra__` prefix is
#              only added later in the point-extraction pipeline in
#              02_Dataframes_Processing_Scripts).
# `col`      : output CSV column (terra_ prefix, partly with a unit suffix).
# `kind`     : how to read the monthly number (documentation only).
# ============================================================================
VAR_SPEC = [
    # nc_name                   col                                 unit        kind
    ("precipitation",           "terra_precipitation",              "mm",       "monthly total"),
    ("runoff",                  "terra_runoff",                     "mm",       "monthly total"),
    ("ET_actual",               "terra_ET_actual",                  "mm",       "monthly total"),
    ("ET_potential",            "terra_ET_potential",               "mm",       "monthly total"),
    ("water_deficit",           "terra_water_deficit",              "mm",       "monthly total"),
    ("soil_moisture",           "terra_soil_moisture",              "mm",       "end-of-month state"),
    ("snow_water",              "terra_snow_water_mm",              "mm",       "end-of-month state"),
    ("solar_radiation",         "terra_solar_radiation_Wm2",        "W/m2",     "monthly mean"),
    ("temp_max",                "terra_temp_max_degC",              "degC",     "monthly mean"),
    ("temp_min",                "terra_temp_min_degC",              "degC",     "monthly mean"),
    ("vapor_pressure",          "terra_vapor_pressure_kPa",         "kPa",      "monthly mean"),
    ("vapor_pressure_deficit",  "terra_vapor_pressure_deficit_kPa", "kPa",      "monthly mean"),
    ("drought_index_pdsi",      "terra_drought_index_pdsi",         "unitless", "end-of-month state"),
    ("wind_speed",              "terra_wind_speed_ms",              "m/s",      "monthly mean"),
]

# Land reference for the single reported coverage fraction (defined over all land).
COVERAGE_REF_VAR = "precipitation"

# ============================================================================
# 1. Open TerraClimate & compute true pixel areas
# ============================================================================
print(f"Loading TerraClimate: {NC_INPUT}")
ds = xr.open_dataset(NC_INPUT)
ds = nc_io.normalize_coords(ds)   # latitude/longitude/x/y → lat/lon (defensive)

# Sanity check: every expected variable is present.
missing_vars = [nc for nc, *_ in VAR_SPEC if nc not in ds.data_vars]
if missing_vars:
    raise KeyError(
        f"TerraClimate.nc is missing expected variable(s): {missing_vars}\n"
        f"Present: {list(ds.data_vars)}"
    )

lon = ds.lon.values
lat = ds.lat.values
dlon = np.abs(lon[1] - lon[0])
dlat = np.abs(lat[1] - lat[0])

# True pixel area (m²) per latitude band on a sphere
R = 6371000.0
lat_top = np.radians(lat + dlat / 2)
lat_bot = np.radians(lat - dlat / 2)
area_1d = (R ** 2) * np.radians(dlon) * np.abs(np.sin(lat_top) - np.sin(lat_bot))
area_2d = np.tile(area_1d[:, np.newaxis], (1, len(lon)))

# AOI bbox from the NetCDF extent
aoi_polygon = box(float(lon.min()), float(lat.min()),
                  float(lon.max()), float(lat.max()))

n_time = ds.time.shape[0]
time_index = pd.DatetimeIndex(ds.time.values)
print(f"  Grid : {len(lat)} lat × {len(lon)} lon  "
      f"({time_index[0].date()} → {time_index[-1].date()}, {n_time} months)")

# ============================================================================
# README describing the TerraClimate unit / coverage convention (written once)
# ============================================================================
_readme_lines = [
    "Per-basin TerraClimate v1.1 monthly time series",
    "===============================================",
    "Columns terra_<var>[_<unit>] are AREA-WEIGHTED BASIN MEANS in the variable's",
    "NATIVE TerraClimate unit. There is NO metres→mm (×1000) conversion here —",
    "unlike the ERA5-Land script — because TerraClimate is already stored in its",
    "final units. The per-month interpretation depends on the variable kind:",
    "",
    "  variable                          column                                unit      kind",
    "  --------------------------------  ------------------------------------  --------  ------------------",
]
for nc_name, col, unit, kind in VAR_SPEC:
    _readme_lines.append(f"  {nc_name:<32}  {col:<36}  {unit:<8}  {kind}")
_readme_lines += [
    "",
    "  'monthly total'      : basin mean of the month's accumulated total (e.g. mm/month).",
    "  'monthly mean'       : basin mean of the month's mean value.",
    "  'end-of-month state' : basin mean of the snapshot at month end.",
    "",
    "Coverage / NaN handling",
    "-----------------------",
    "TerraClimate is a land-surface product: ocean and out-of-domain pixels are",
    "NaN, and that geometry is fixed in time. Each variable is averaged only over",
    "its own valid (non-NaN) pixels in BOTH numerator and denominator, so a single",
    "missing pixel does not turn the basin mean into NaN and there is no low bias",
    "from under-counting area. Basins with no covered land at all yield NaN means.",
    "",
    f"The reported coverage columns use `{COVERAGE_REF_VAR}` as the land reference",
    "(it is valid everywhere over land); the 14 variables share essentially the",
    "same land footprint, so this single fraction is representative.",
    "",
    "Area columns (all prefixed terra_ so they never collide with the ERA5 CSVs)",
    "---------------------------------------------------------------------------",
    "terra_basin_area_m2   : full rasterized basin area on the TerraClimate grid (m²).",
    "terra_covered_area_m2 : area with valid TerraClimate data (m²).",
    "terra_coverage_frac   : terra_covered_area_m2 / terra_basin_area_m2 (0..1). Use",
    "                        this to filter or down-weight poorly covered coastal",
    "                        basins; a value near 1 means the mean is representative.",
    "",
    "Joining with the ERA5-Land per-basin CSVs",
    "-----------------------------------------",
    "Same <HYBAS_ID>.csv filenames live in basin_era5_lev0X/ and",
    "basin_terraclimate_lev0X/. Join on 'time':",
    "    pd.merge(era5_df, terra_df, on='time', how='outer')",
    "Note the two products cover different periods/grids, so an outer join will",
    "leave NaNs outside the overlapping months.",
    "",
]
(OUT_BASE / "README_terraclimate_units.txt").write_text(
    "\n".join(_readme_lines), encoding="utf-8"
)

# ============================================================================
# 2. Process each level
# ============================================================================
for lvl in LEVELS:
    print(f"\n--- {lvl['name']} ---")
    lvl["out_dir"].mkdir(parents=True, exist_ok=True)

    # -- Load & filter shapefile to AOI ------------------------------------
    gdf = gpd.read_file(lvl["shp"])
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    gdf = gdf[gdf.intersects(aoi_polygon)].copy()
    print(f"  Basins in AOI : {len(gdf)}")

    # -- Metadata ----------------------------------------------------------
    meta_cols = [c for c in gdf.columns if c != "geometry"]
    gdf[meta_cols].to_csv(
        OUT_BASE / f"metadata_terraclimate_{lvl['name']}.csv", index=False
    )

    # -- Rasterize basins onto the TerraClimate grid -----------------------
    transform = rasterio.transform.from_bounds(
        lon.min() - dlon / 2, lat.min() - dlat / 2,
        lon.max() + dlon / 2, lat.max() + dlat / 2,
        len(lon), len(lat),
    )
    shapes = ((geom, val) for geom, val in zip(gdf.geometry, gdf["HYBAS_ID"]))
    mask = rasterio.features.rasterize(
        shapes=shapes, out_shape=(len(lat), len(lon)),
        transform=transform, fill=0, dtype="int64",
    )
    # rasterize() puts row 0 at the north edge; align to the NetCDF lat order
    if lat[0] < lat[-1]:
        mask = np.flipud(mask)

    # -- Save the basin-ID mask as a reference NC (via nc_io) --------------
    mask_da = xr.DataArray(mask, coords={"lat": lat, "lon": lon},
                           dims=["lat", "lon"], name="HYBAS_ID")
    mask_da.attrs.update({
        "long_name": "HydroBASINS basin identifier",
        "units":     "1",
        "comment":   "0 = outside any basin (NoData). Rasterized on the "
                     "TerraClimate grid (~4.5 km).",
    })
    mask_attrs = nc_io.make_global_attrs(
        title=f"HydroBASINS {lvl['name']} basin-ID mask on the TerraClimate grid",
        source=f"HydroSHEDS HydroBASINS v1c, Europe, {lvl['name']}",
        references="Lehner & Grill (2013), https://doi.org/10.1002/hyp.9740",
        script=Path(__file__).name,
    )
    nc_io.save_nc(
        mask_da.to_dataset(),
        OUT_BASE / f"basin_mask_terraclimate_{lvl['name']}.nc",
        fill_value={"HYBAS_ID": 0},      # 0 = no basin
        global_attrs=mask_attrs,
        verify=False,
    )

    # -- Zonal statistics --------------------------------------------------
    basins = np.unique(mask)
    basins = basins[basins > 0]
    print(f"  Basins on grid: {len(basins)}")

    # Full rasterized basin area (every pixel inside the polygon)
    basin_area_m2 = sum_labels(area_2d, labels=mask, index=basins)

    # Coverage diagnostic from the land-reference variable. TerraClimate's
    # land/ocean geometry is fixed in time, so one slice is enough.
    ref_valid = ~np.isnan(ds[COVERAGE_REF_VAR].isel(time=0).values)
    covered_area_m2 = sum_labels(area_2d * ref_valid, labels=mask, index=basins)
    coverage_frac = np.where(basin_area_m2 > 0,
                             covered_area_m2 / basin_area_m2, 0.0)

    n_partial = int(np.sum(coverage_frac < 0.999))
    n_empty   = int(np.sum(covered_area_m2 <= 0))
    print(f"  Coverage      : {n_partial} basin(s) partially covered, "
          f"{n_empty} with no data (NaN means)")

    # Area-weighted basin mean for every variable, normalised over that
    # variable's OWN valid (non-NaN) pixels. NO unit conversion.
    means = {}
    for nc_name, col, unit, kind in VAR_SPEC:
        arr = ds[nc_name].values                      # (time, lat, lon)
        vmask = ~np.isnan(arr[0])                      # static land mask for this var
        var_area = sum_labels(area_2d * vmask, labels=mask, index=basins)

        vol = np.zeros((n_time, len(basins)), dtype=np.float64)
        for t in range(n_time):
            # zero-fill invalid pixels so they add nothing to the numerator;
            # var_area already excludes them from the denominator
            field = np.where(vmask, arr[t], 0.0) * area_2d
            vol[t] = sum_labels(field, labels=mask, index=basins)

        with np.errstate(invalid="ignore", divide="ignore"):
            means[col] = np.where(
                var_area[np.newaxis, :] > 0,
                vol / var_area[np.newaxis, :],
                np.nan,
            )
        del arr                                        # keep peak memory low

    # -- Per-basin CSVs ----------------------------------------------------
    print(f"  Writing {len(basins)} CSVs …")
    for i, basin_id in enumerate(basins):
        df_dict = {
            "time":                  time_index,
            "terra_basin_area_m2":   basin_area_m2[i],    # full rasterized area
            "terra_covered_area_m2": covered_area_m2[i],  # land-reference area
            "terra_coverage_frac":   coverage_frac[i],    # 0..1
        }
        for _nc, col, *_ in VAR_SPEC:
            df_dict[col] = means[col][:, i]
        pd.DataFrame(df_dict).to_csv(lvl["out_dir"] / f"{basin_id}.csv", index=False)

    print(f"  Done {lvl['name']}.")

ds.close()
print("\n✓ All TerraClimate basin processing complete.")

# -*- coding: utf-8 -*-
"""
05a_hydrobasins_era5_basin_means.py

Per-basin monthly time series of ERA5-Land water-cycle fluxes for
HydroBASINS levels 06 and 07.

Variables extracted (accumulated ERA5-Land fluxes, monthly product):
  tp    total precipitation
  ro    runoff (total)
  ssro  sub-surface runoff
  sro   surface runoff

⚠ ERA5-Land MONTHLY unit convention (important):
  Accumulated fluxes in the ERA5-Land *monthly* product are daily-mean
  accumulations expressed PER DAY — units "m of water equivalent per day",
  NOT monthly totals. Converting to mm (×1000) therefore yields mm/day,
  i.e. the month's mean daily rate. A monthly total would be this value
  × number of days in the month. We keep the native per-day convention.

Per basin / month / variable, exported column:
  <var>_mm_day    area-weighted mean daily rate (mm/day), normalised over the
                  part of the basin actually COVERED by ERA5-Land data
Plus:
  basin_area_m2    full rasterized basin area on the ERA5 grid (m²)
  covered_area_m2  area with valid (non-NaN) ERA5-Land data — the actual
                   denominator used for the area weighting (m²)
  coverage_frac    covered_area_m2 / basin_area_m2 (0..1); a diagnostic of how
                   much of the basin contributed to the mean. Coastal basins
                   that extend over the ocean have coverage_frac < 1, since
                   ERA5-Land is a land-only product (ocean pixels are NaN).

NaN handling: ERA5-Land's land-sea geometry is fixed in time, so the missing
(ocean) pixels are the same every month. We build one validity mask and
average only over valid pixels in BOTH numerator and denominator. This avoids
NaN poisoning (one missing pixel turning the whole basin mean into NaN) and the
low bias that would result from dividing a partial-coverage volume by the full
basin area. Basins with no covered land at all yield NaN means.

Inputs
  ERA5-Land : 01_Input_Global/3_Final/ERA5_Land/era5_land.nc
              (from ERA5_Land/01_era5_land_dynamic_preprocess.py)
  Shapefiles: 01_Input_Global/1_Original/HydroSHEDS_RIVERS_BASINS/
              hybas_eu_lev06_v1c/, hybas_eu_lev07_v1c/
Outputs (01_Input_Global/3_Final/HydroSHEDS_RIVERS_BASINS/)
  basin_era5_lev06/<HYBAS_ID>.csv
  basin_era5_lev07/<HYBAS_ID>.csv
  metadata_lev0X.csv
  basin_mask_lev0X.nc          (basin-ID raster on the ERA5 grid)
  README_units.txt

Location: 02_Processing_Scripts/HydroSHEDS/
Run     : python 05a_hydrobasins_era5_basin_means.py
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

NC_INPUT = FINAL / "ERA5_Land" / "era5_land.nc"

SHP_DIR   = ORIGINAL / "HydroSHEDS_RIVERS_BASINS"
SHP_LEV06 = SHP_DIR / "hybas_eu_lev06_v1c" / "hybas_eu_lev06_v1c.shp"
SHP_LEV07 = SHP_DIR / "hybas_eu_lev07_v1c" / "hybas_eu_lev07_v1c.shp"

OUT_BASE = FINAL / "HydroSHEDS_RIVERS_BASINS"
OUT_BASE.mkdir(parents=True, exist_ok=True)

LEVELS = [
    {"name": "lev06", "shp": SHP_LEV06, "out_dir": OUT_BASE / "basin_era5_lev06"},
    {"name": "lev07", "shp": SHP_LEV07, "out_dir": OUT_BASE / "basin_era5_lev07"},
]

# ============================================================================
# CONFIGURATION
# ============================================================================
VARS = ["tp", "ro", "ssro", "sro"]

# ============================================================================
# 1. Load ERA5-Land & compute true pixel areas
# ============================================================================
print(f"Loading ERA5-Land: {NC_INPUT}")
ds = xr.open_dataset(NC_INPUT)

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

# ERA5-Land accumulated fluxes are in metres (per day). Load once.
data_arrays = {v: ds[v].values for v in VARS}      # (time, lat, lon)
n_time = ds.time.shape[0]
time_index = pd.DatetimeIndex(ds.time.values)

# ============================================================================
# README describing the per-day unit convention (written once)
# ============================================================================
(OUT_BASE / "README_units.txt").write_text(
    "Per-basin ERA5-Land monthly time series\n"
    "=======================================\n"
    "Columns <var>_mm_day are AREA-WEIGHTED MEAN DAILY RATES in mm/day.\n"
    "ERA5-Land monthly accumulated fluxes (tp, ro, ssro, sro) are stored as\n"
    "daily-mean accumulations 'per day' (m/day); ×1000 → mm/day.\n"
    "For a monthly TOTAL: multiply <var>_mm_day by the number of days in the month.\n"
    "\n"
    "Coverage / NaN handling\n"
    "-----------------------\n"
    "ERA5-Land is a land-only product, so ocean pixels are NaN. Coastal basins\n"
    "extend over the ocean and are therefore only partially covered. The means\n"
    "are computed over the COVERED (valid, non-NaN) part of each basin only, in\n"
    "both numerator and denominator, so a single missing pixel does not turn the\n"
    "whole basin mean into NaN and there is no low bias from under-counting area.\n"
    "Basins with no covered land at all yield NaN means.\n"
    "\n"
    "Area columns\n"
    "------------\n"
    "basin_area_m2   : full rasterized basin area on the ERA5 grid (m²).\n"
    "covered_area_m2 : area with valid ERA5-Land data — the actual weighting\n"
    "                  denominator (m²).\n"
    "coverage_frac   : covered_area_m2 / basin_area_m2 (0..1). Use this to filter\n"
    "                  or down-weight poorly covered coastal basins; a value near\n"
    "                  1 means the mean is representative of the whole basin.\n",
    encoding="utf-8",
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
    gdf[meta_cols].to_csv(OUT_BASE / f"metadata_{lvl['name']}.csv", index=False)

    # -- Rasterize basins onto the ERA5 grid -------------------------------
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
        "comment":   "0 = outside any basin (NoData).",
    })
    mask_attrs = nc_io.make_global_attrs(
        title=f"HydroBASINS {lvl['name']} basin-ID mask on the ERA5-Land grid",
        source=f"HydroSHEDS HydroBASINS v1c, Europe, {lvl['name']}",
        references="Lehner & Grill (2013), https://doi.org/10.1002/hyp.9740",
        script=Path(__file__).name,
    )
    nc_io.save_nc(
        mask_da.to_dataset(), OUT_BASE / f"basin_mask_{lvl['name']}.nc",
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

    # ERA5-Land is land-only: ocean pixels are NaN. The land–sea geometry is
    # fixed in time, so derive ONE validity mask (intersection across vars).
    # Averaging only over valid pixels avoids NaN poisoning of the basin sum
    # and the low bias from dividing a partial-coverage volume by full area.
    valid = np.ones(mask.shape, dtype=bool)
    for v in VARS:
        valid &= ~np.isnan(data_arrays[v][0])

    # Area actually covered by data — the correct weighting denominator
    valid_area_m2 = sum_labels(area_2d * valid, labels=mask, index=basins)
    coverage_frac = np.where(basin_area_m2 > 0,
                             valid_area_m2 / basin_area_m2, 0.0)

    n_partial = int(np.sum(coverage_frac < 0.999))
    n_empty   = int(np.sum(valid_area_m2 <= 0))
    print(f"  Coverage      : {n_partial} basin(s) partially covered, "
          f"{n_empty} with no data (NaN means)")

    # Area-weighted mean daily rate (mm/day), normalised by the COVERED area
    means = {v: np.full((n_time, len(basins)), np.nan, dtype=np.float64)
             for v in VARS}
    for v in VARS:
        vol = np.zeros((n_time, len(basins)), dtype=np.float64)
        for t in range(n_time):
            # zero-fill ocean pixels so they contribute nothing to the sum;
            # the denominator (valid_area_m2) already excludes them
            field = np.where(valid, data_arrays[v][t], 0.0) * area_2d
            vol[t] = sum_labels(field, labels=mask, index=basins)
        # depth(m) = volume/covered_area ; ×1000 → mm/day (per-day convention)
        with np.errstate(invalid="ignore", divide="ignore"):
            means[v] = np.where(
                valid_area_m2[np.newaxis, :] > 0,
                vol / valid_area_m2[np.newaxis, :],
                np.nan,
            ) * 1000.0

    # -- Per-basin CSVs ----------------------------------------------------
    print(f"  Writing {len(basins)} CSVs …")
    for i, basin_id in enumerate(basins):
        df_dict = {
            "time":            time_index,
            "basin_area_m2":   basin_area_m2[i],   # full rasterized area
            "covered_area_m2": valid_area_m2[i],   # area used for weighting
            "coverage_frac":   coverage_frac[i],   # 0..1
        }
        for v in VARS:
            df_dict[f"{v}_mm_day"] = means[v][:, i]
        pd.DataFrame(df_dict).to_csv(lvl["out_dir"] / f"{basin_id}.csv", index=False)

    print(f"  Done {lvl['name']}.")

ds.close()
print("\n✓ All basin processing complete.")

# -*- coding: utf-8 -*-
"""
06_hydrobasins_level_statistics.py

Descriptive statistics of the HydroBASINS level 06 and level 07 delineation over
the study area, computed from the outputs of

    05a_hydrobasins_era5_basin_means.py
    05b_hydrobasins_terraclimate_basin_means.py

and from the static feature table (for the well based statistics). The static
feature table is produced by the scripts in 02_Dataframes_Processing_Scripts,
so the well based part of this script can only run after that pipeline. If the
table is missing, the well statistics are skipped.

Purpose
-------
Provide the numbers that justify the choice of Pfafstetter levels 6 and 7, i.e.
bound the choice from two sides:

  UPPER BOUND (the level must not be too coarse)
      Averaging a climate field over a catchment removes spatial variability.
      The between-basin share of the total spatial variance of a long-term mean
      climate field states how much of that variability the basin means retain.

  LOWER BOUND (the level must not be too fine)
      A basin mean computed from a single grid cell adds nothing to the value
      already sampled at the well. The number of native climate grid cells per
      catchment quantifies this.

In addition, the script answers a question that arises directly from the data:
for how many wells are the level 6 and level 7 catchments the SAME polygon?
HydroBASINS allows a small sub-basin to skip the subdivision at the next level,
so the two levels are not necessarily distinct everywhere. The HYBAS_ID encodes
region (digit 1), level (digits 2 and 3), a unique identifier within the
HydroSHEDS network (digits 4 to 9) and the side (digit 10), so two identifiers
that differ only in the level digits denote the same polygon.

No shapefiles and no geopandas are required. Every input is a product of the
two processing scripts or the static feature table.

Inputs (relative to the project root)
  01_Input_Global/3_Final/HydroSHEDS_RIVERS_BASINS/
      metadata_lev0X.csv                     HydroBASINS attributes, incl. SUB_AREA
      basin_mask_lev0X.nc                    basin-ID raster, ERA5-Land grid
      basin_mask_terraclimate_lev0X.nc       basin-ID raster, TerraClimate grid
      basin_era5_lev0X/<HYBAS_ID>.csv
      basin_terraclimate_lev0X/<HYBAS_ID>.csv
  01_Input_Global/3_Final/ERA5_Land/era5_land.nc          (variance step only)
  01_Input_Global/3_Final/TerraClimate/TerraClimate.nc    (variance step only)
  03_Dataframes/02_Static_Features/02_global/static_features.csv   (optional)

Outputs
  01_Input_Global/3_Final/HydroSHEDS_RIVERS_BASINS/basin_level_stats/
      per_basin_lev0X.csv          one row per catchment, all diagnostics
      summary_levels.csv           the numbers for the thesis text
      variance_decomposition.csv   retained spatial variability per level
      level_comparison.csv         how levels 6 and 7 relate to each other
      report.txt                   the same numbers as readable text

Location: 02_Processing_Scripts/HydroSHEDS/
Run     : python 06_hydrobasins_level_statistics.py
          python 06_hydrobasins_level_statistics.py "<path to project root>"
          (the optional argument overrides the derived project root)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")

# ============================================================================
# PATHS  (relative to the project root; adjust here if the layout changes)
# ============================================================================
if len(sys.argv) > 1:
    PROJECT_ROOT = Path(sys.argv[1]).resolve()
else:
    # 02_Processing_Scripts/HydroSHEDS/<this file>
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_GLOBAL = PROJECT_ROOT / "01_Input_Global"
FINAL        = INPUT_GLOBAL / "3_Final"
BASIN_DIR    = FINAL / "HydroSHEDS_RIVERS_BASINS"
OUT_DIR      = BASIN_DIR / "basin_level_stats"

NC_ERA5  = FINAL / "ERA5_Land" / "era5_land.nc"
NC_TERRA = FINAL / "TerraClimate" / "TerraClimate.nc"

STATIC_CSV = (PROJECT_ROOT / "03_Dataframes" / "02_Static_Features" /
              "02_global" / "static_features.csv")

if not BASIN_DIR.exists():
    raise SystemExit(
        f"Basin directory not found:\n  {BASIN_DIR}\n"
        f"Derived project root: {PROJECT_ROOT}\n"
        f"Pass the project root as the first argument if the script is not "
        f"stored in 02_Processing_Scripts/HydroSHEDS/."
    )
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================
LEVELS = ["lev06", "lev07"]

# label, per-basin folder, column prefix, column STEM of the reference
# variable, mask file pattern, native NetCDF, native variable, native scale.
#
# The stem is resolved against the actual CSV header at runtime, because the
# per-basin CSVs may or may not carry a unit suffix on the column name
# (for example terra_precipitation or terra_precipitation_mm).
#
# NATIVE SCALE: the variance decomposition compares the native field with the
# basin means from the CSVs, so both must be in the same unit. ERA5-Land stores
# the accumulated fluxes in metres per day, while script 05a writes the basin
# means in millimetres per day, hence the factor 1000. TerraClimate is already
# stored in its final unit, hence 1. A wrong factor inflates the reported
# variance share by the square of that factor.
PRODUCTS = [
    ("era5",  "basin_era5_{lvl}",         "",       "tp",
     "basin_mask_{lvl}.nc",               NC_ERA5,  "tp",            1000.0),
    ("terra", "basin_terraclimate_{lvl}", "terra_", "terra_precipitation",
     "basin_mask_terraclimate_{lvl}.nc",  NC_TERRA, "precipitation",    1.0),
]

# A catchment with fewer than this many valid cells is counted as
# "effectively a point" for the lower-bound argument.
MIN_CELLS = 5

# Set to False to skip the variance decomposition (it reads the native cubes).
DO_VARIANCE = True


# ============================================================================
# Helpers
# ============================================================================
def pixel_area_2d(lat, lon):
    """True spherical pixel area in m^2 for a regular lat/lon grid."""
    R = 6371000.0
    dlat = np.abs(lat[1] - lat[0])
    dlon = np.abs(lon[1] - lon[0])
    lat_top = np.radians(lat + dlat / 2)
    lat_bot = np.radians(lat - dlat / 2)
    a1 = (R ** 2) * np.radians(dlon) * np.abs(np.sin(lat_top) - np.sin(lat_bot))
    return np.tile(a1[:, np.newaxis], (1, len(lon)))


def read_mask(path):
    """
    Read a basin-ID mask written by scripts 05a/05b.

    The mask was saved with _FillValue = 0. A CF-decoding read would therefore
    turn every 0 into NaN and cast the field to float, so decoding is disabled
    and 0 is treated as 'outside any basin'.
    """
    with xr.open_dataset(path, mask_and_scale=False) as d:
        return (d["HYBAS_ID"].values.astype("int64"),
                d["lat"].values, d["lon"].values)


def time_mean_field(nc_path, var):
    """Long-term mean of one variable, accumulated in float64 over time."""
    with xr.open_dataset(nc_path) as d:
        n_t = d[var].sizes["time"]
        shp = (d.sizes["lat"], d.sizes["lon"])
        acc = np.zeros(shp, dtype=np.float64)
        cnt = np.zeros(shp, dtype=np.int64)
        for t in range(n_t):
            sl = d[var].isel(time=t).values
            ok = ~np.isnan(sl)
            acc[ok] += sl[ok]
            cnt[ok] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)


def weighted_var(values, weights):
    """Area-weighted variance of a 1-D sample."""
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[ok], w[ok]
    if w.sum() <= 0:
        return np.nan
    m = np.sum(w * v) / np.sum(w)
    return float(np.sum(w * (v - m) ** 2) / np.sum(w))


def q(series, p):
    a = np.asarray(series, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, p)) if a.size else np.nan


def hybas_core(series):
    """
    Strip the two level digits from a HYBAS_ID so that the same polygon carries
    the same key across levels. Digit 1 = region, digits 2-3 = level,
    digits 4-9 = unique identifier, digit 10 = side.
    """
    s = pd.Series(series).astype("Int64").astype(str)
    return s.str.slice(0, 1) + s.str.slice(3)


def resolve_columns(folder, prefix, stem):
    """
    Resolve the column names of a per-basin CSV against its actual header.

    The reference variable may carry a unit suffix (terra_precipitation or
    terra_precipitation_mm, tp_mm_day and so on), so it is matched by stem and
    the shortest match is taken. Returns (coverage, covered_area, reference)
    and the sample file that was inspected.
    """
    sample = next(folder.glob("*.csv"), None)
    if sample is None:
        raise SystemExit(f"No per-basin CSV found in {folder}")
    header = list(pd.read_csv(sample, nrows=0).columns)

    def pick(name, exact_first=True):
        if exact_first and name in header:
            return name
        hits = sorted([c for c in header if c.startswith(name)], key=len)
        if not hits:
            raise SystemExit(
                f"Column starting with '{name}' not found in {sample.name}.\n"
                f"Available columns: {header}"
            )
        return hits[0]

    return (pick(f"{prefix}coverage_frac"),
            pick(f"{prefix}covered_area_m2"),
            pick(stem),
            sample)


def read_static_csv(path, columns):
    """Read selected columns, tolerating a non-UTF-8 encoding."""
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, usecols=columns, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Could not decode {path}")


# ============================================================================
# 0. Wells: basin assignment straight from the static feature table
# ============================================================================
wells = None
if STATIC_CSV.exists():
    wells = read_static_csv(
        STATIC_CSV,
        ["MW_ID", "basin_lev06__HYBAS_ID", "basin_lev07__HYBAS_ID"],
    )
    print(f"Wells loaded: {len(wells)}  ({STATIC_CSV.name})")
else:
    print(f"Static feature table not found, well statistics are skipped:\n"
          f"  {STATIC_CSV}")


# ============================================================================
# 1. Native grids and long-term mean fields (once, reused by both levels)
# ============================================================================
fields = {}
if DO_VARIANCE:
    for label, _f, _p, _c, _m, ncpath, ncvar, scale in PRODUCTS:
        if not ncpath.exists():
            print(f"  {label}: {ncpath.name} not found, variance step skipped")
            continue
        print(f"Long-term mean field: {label} / {ncvar}")
        fields[label] = time_mean_field(ncpath, ncvar) * scale


# ============================================================================
# 2. Per level
# ============================================================================
summary_rows, variance_rows, per_basin = [], [], {}

for lvl in LEVELS:
    print(f"\n--- {lvl} ---")

    # -- HydroBASINS polygon attributes -------------------------------------
    meta = pd.read_csv(BASIN_DIR / f"metadata_{lvl}.csv")
    meta["HYBAS_ID"] = meta["HYBAS_ID"].astype("int64")
    keep = [c for c in ["HYBAS_ID", "SUB_AREA", "UP_AREA", "COAST", "ENDO",
                        "ORDER", "PFAF_ID", "NEXT_DOWN"] if c in meta.columns]
    tab = meta[keep].copy()
    if "PFAF_ID" in tab.columns:
        tab["PFAF_ID"] = tab["PFAF_ID"].astype("Int64").astype(str)
    tab["id_core"] = hybas_core(tab["HYBAS_ID"])

    # -- Cell counts, coverage and long-term means per product --------------
    for label, folder, pfx, col, maskpat, _nc, _var, _sc in PRODUCTS:
        mask, lat, lon = read_mask(BASIN_DIR / maskpat.format(lvl=lvl))
        area = pixel_area_2d(lat, lon)

        ids, counts = np.unique(mask[mask > 0], return_counts=True)
        cells = pd.DataFrame({"HYBAS_ID": ids.astype("int64"),
                              f"{label}_cells": counts})

        if label in fields:
            valid = np.isfinite(fields[label])
            iv, cv = np.unique(mask[(mask > 0) & valid], return_counts=True)
            cells = cells.merge(
                pd.DataFrame({"HYBAS_ID": iv.astype("int64"),
                              f"{label}_cells_valid": cv}),
                on="HYBAS_ID", how="left")
            cells[f"{label}_cells_valid"] = (
                cells[f"{label}_cells_valid"].fillna(0).astype(int))
        else:
            cells[f"{label}_cells_valid"] = cells[f"{label}_cells"]

        # coverage, covered area and the long-term mean from the per-basin CSVs
        cdir = BASIN_DIR / folder.format(lvl=lvl)
        c_cov, c_area, c_ref, sample = resolve_columns(cdir, pfx, col)
        print(f"  {label:<6} columns resolved from {sample.name}: "
              f"{c_ref}, {c_cov}, {c_area}")

        cov, cova, ltm = [], [], []
        for bid in cells["HYBAS_ID"]:
            f = cdir / f"{bid}.csv"
            if not f.exists():
                cov.append(np.nan); cova.append(np.nan); ltm.append(np.nan)
                continue
            d = pd.read_csv(f, usecols=[c_cov, c_area, c_ref])
            cov.append(float(d[c_cov].iloc[0]))
            cova.append(float(d[c_area].iloc[0]))
            ltm.append(float(np.nanmean(d[c_ref].values)))
        cells[f"{label}_coverage_frac"] = cov
        cells[f"{label}_covered_area_m2"] = cova
        cells[f"{label}_longterm_mean"] = ltm
        cells[f"{label}_cell_km2"] = float(np.median(area)) / 1e6

        tab = tab.merge(cells, on="HYBAS_ID", how="left")

    # -- Completeness within the study area ---------------------------------
    # A catchment reaching beyond the study area is only partly rasterized, so
    # its cell count understates the true catchment. The ratio of the
    # rasterized area on the fine TerraClimate grid to the polygon area
    # SUB_AREA identifies these edge cases without needing the geometry.
    if "SUB_AREA" in tab.columns:
        tab["rasterized_area_ratio"] = (
            (tab["terra_cells"] * tab["terra_cell_km2"]) / tab["SUB_AREA"])
        tab["inside_aoi"] = tab["rasterized_area_ratio"] >= 0.90
    else:
        tab["inside_aoi"] = True

    # -- Wells per catchment ------------------------------------------------
    if wells is not None:
        wcol = f"basin_{lvl}__HYBAS_ID"
        n_w = (wells.dropna(subset=[wcol])
                    .astype({wcol: "int64"})
                    .groupby(wcol)["MW_ID"].nunique()
                    .rename("n_wells").reset_index()
                    .rename(columns={wcol: "HYBAS_ID"}))
        tab = tab.merge(n_w, on="HYBAS_ID", how="left")
        tab["n_wells"] = tab["n_wells"].fillna(0).astype(int)

    tab.to_csv(OUT_DIR / f"per_basin_{lvl}.csv", index=False)
    per_basin[lvl] = tab

    # -- Variance decomposition ---------------------------------------------
    # The area-weighted spatial variance of the long-term mean field over all
    # cells inside any catchment decomposes exactly into a between-catchment
    # and a within-catchment part. The between part is the spatial signal that
    # the basin means retain.
    for label, _f, _p, _c, maskpat, _nc, ncvar, _sc in PRODUCTS:
        if label not in fields:
            continue
        mask, lat, lon = read_mask(BASIN_DIR / maskpat.format(lvl=lvl))
        area = pixel_area_2d(lat, lon)
        sel = (mask > 0) & np.isfinite(fields[label])
        tot = weighted_var(fields[label][sel], area[sel])
        btw = weighted_var(tab[f"{label}_longterm_mean"].values,
                           tab[f"{label}_covered_area_m2"].values)
        share = btw / tot if tot and np.isfinite(tot) else np.nan
        if np.isfinite(share) and share > 1.0:
            print(f"  WARNING {label}: retained share {share:.2f} exceeds 1. "
                  f"The basin means and the native field are not the same "
                  f"population; check that the per-basin CSVs and "
                  f"{ncvar} cover the same period.")
        variance_rows.append({
            "level": lvl, "product": label, "variable": ncvar,
            "total_spatial_variance": tot, "between_basin_variance": btw,
            "between_basin_share": share,
            "n_basins": int(np.isfinite(tab[f"{label}_longterm_mean"]).sum()),
        })
        print(f"  {label:<6} retained share of spatial variance: {share:.3f}")

    # -- Summary row ---------------------------------------------------------
    ins = tab["inside_aoi"].fillna(False).values
    row = {
        "level": lvl,
        "n_basins_on_grid": int(len(tab)),
        "n_basins_inside_aoi": int(ins.sum()),
    }
    for pop, selm in [("all", np.ones(len(tab), bool)), ("inside", ins)]:
        row[f"sub_area_km2_median_{pop}"] = q(tab.loc[selm, "SUB_AREA"], 50)
        row[f"sub_area_km2_p25_{pop}"] = q(tab.loc[selm, "SUB_AREA"], 25)
        row[f"sub_area_km2_p75_{pop}"] = q(tab.loc[selm, "SUB_AREA"], 75)
    row["sub_area_km2_min_inside"] = q(tab.loc[ins, "SUB_AREA"], 0)
    row["sub_area_km2_max_inside"] = q(tab.loc[ins, "SUB_AREA"], 100)
    row["n_coastal"] = int((tab.get("COAST", pd.Series(dtype=int)) == 1).sum())

    for label, *_ in [(p[0],) for p in PRODUCTS]:
        c = tab.loc[ins, f"{label}_cells_valid"]
        row[f"{label}_cell_km2"] = float(tab[f"{label}_cell_km2"].iloc[0])
        row[f"{label}_cells_median_inside"] = q(c, 50)
        row[f"{label}_cells_p10_inside"] = q(c, 10)
        row[f"{label}_cells_min_inside"] = q(c, 0)
        row[f"{label}_share_below_{MIN_CELLS}_cells"] = float((c < MIN_CELLS).mean())
        row[f"{label}_coverage_median"] = q(tab[f"{label}_coverage_frac"], 50)
        row[f"{label}_n_partial_coverage"] = int(
            (tab[f"{label}_coverage_frac"] < 0.999).sum())

    if "n_wells" in tab.columns:
        w = tab["n_wells"] > 0
        row["n_basins_with_wells"] = int(w.sum())
        row["n_wells_assigned"] = int(tab["n_wells"].sum())
        row["wells_per_basin_median"] = q(tab.loc[w, "n_wells"], 50)
        row["wells_per_basin_max"] = q(tab["n_wells"], 100)
        row["sub_area_km2_median_wellbasins"] = q(tab.loc[w, "SUB_AREA"], 50)
        row["sub_area_km2_p25_wellbasins"] = q(tab.loc[w, "SUB_AREA"], 25)
        row["sub_area_km2_p75_wellbasins"] = q(tab.loc[w, "SUB_AREA"], 75)
        for label, *_ in [(p[0],) for p in PRODUCTS]:
            cw = tab.loc[w, f"{label}_cells_valid"]
            row[f"{label}_cells_median_wellbasins"] = q(cw, 50)
            row[f"{label}_cells_p10_wellbasins"] = q(cw, 10)
            row[f"{label}_share_below_{MIN_CELLS}_cells_wellbasins"] = float(
                (cw < MIN_CELLS).mean())
    summary_rows.append(row)

    print(f"  catchments on grid / inside study area: "
          f"{row['n_basins_on_grid']} / {row['n_basins_inside_aoi']}")
    print(f"  median SUB_AREA (inside): {row['sub_area_km2_median_inside']:.0f} km2")
    print(f"  median valid cells: ERA5-Land {row['era5_cells_median_inside']:.0f}, "
          f"TerraClimate {row['terra_cells_median_inside']:.0f}")


# ============================================================================
# 3. How the two levels relate to each other
# ============================================================================
cmp_rows = []
t6, t7 = per_basin["lev06"], per_basin["lev07"]

# Identity of a level 7 polygon with its level 6 parent.
#
# CAUTION: a matching identifier alone is NOT sufficient. The six digits of the
# HYBAS_ID identify the outlet within the HydroSHEDS network, not the polygon,
# and the most downstream sub-basin of a level 6 unit shares that outlet with
# its parent while covering a smaller area. Identity therefore requires the
# same outlet AND the same polygon area.
AREA_TOL_KM2 = 0.05
pair = t6[["id_core", "SUB_AREA"]].merge(
    t7[["id_core", "SUB_AREA"]], on="id_core", how="inner",
    suffixes=("_l6", "_l7"))
pair["identical"] = (pair["SUB_AREA_l6"] - pair["SUB_AREA_l7"]).abs() <= AREA_TOL_KM2
identical_cores = set(pair.loc[pair["identical"], "id_core"])
cmp_rows += [
    {"quantity": "level 7 polygons sharing the outlet of a level 6 unit",
     "value": len(pair), "of": len(t7),
     "share": len(pair) / len(t7) if len(t7) else np.nan},
    {"quantity": "level 7 polygons identical to their level 6 parent "
                 "(same outlet and same area)",
     "value": len(identical_cores), "of": len(t7),
     "share": len(identical_cores) / len(t7) if len(t7) else np.nan},
]

# Number of level 7 units per level 6 unit, from the Pfafstetter codes.
if "PFAF_ID" in t6.columns and "PFAF_ID" in t7.columns:
    children = (t7["PFAF_ID"].str.slice(0, 6).value_counts()
                .rename("n_children").reset_index()
                .rename(columns={"index": "PFAF_ID"}))
    t6c = t6.merge(children, on="PFAF_ID", how="left")
    t6c["n_children"] = t6c["n_children"].fillna(0).astype(int)
    t6c.to_csv(OUT_DIR / "level6_children.csv", index=False)
    cmp_rows += [
        {"quantity": "level 7 units per level 6 unit (median)",
         "value": q(t6c["n_children"], 50), "of": np.nan, "share": np.nan},
        {"quantity": "level 6 units not subdivided at level 7",
         "value": int((t6c["n_children"] <= 1).sum()), "of": len(t6c),
         "share": float((t6c["n_children"] <= 1).mean())},
    ]

# The question that matters for the model comparison: for how many wells do the
# two levels deliver the same catchment? Resolved through the basin table, so
# that the area condition above is applied here as well.
if wells is not None:
    w = wells.dropna(subset=["basin_lev06__HYBAS_ID",
                             "basin_lev07__HYBAS_ID"]).copy()
    a6 = t6.set_index("HYBAS_ID")["SUB_AREA"]
    a7 = t7.set_index("HYBAS_ID")["SUB_AREA"]
    c6 = hybas_core(w["basin_lev06__HYBAS_ID"]).values
    c7 = hybas_core(w["basin_lev07__HYBAS_ID"]).values
    s6 = w["basin_lev06__HYBAS_ID"].astype("int64").map(a6).values
    s7 = w["basin_lev07__HYBAS_ID"].astype("int64").map(a7).values
    same_outlet = c6 == c7
    same = same_outlet & (np.abs(s6 - s7) <= AREA_TOL_KM2)
    cmp_rows += [
        {"quantity": "wells whose two catchments share an outlet",
         "value": int(same_outlet.sum()), "of": int(len(w)),
         "share": float(same_outlet.mean())},
        {"quantity": "wells whose level 6 and level 7 catchment are the "
                     "same polygon",
         "value": int(np.nansum(same)), "of": int(len(w)),
         "share": float(np.nansum(same) / len(w))},
    ]
    print(f"\nWells with an identical catchment at both levels: "
          f"{int(np.nansum(same))} of {len(w)} "
          f"({np.nansum(same) / len(w):.1%})")

cmp_df = pd.DataFrame(cmp_rows)
cmp_df.to_csv(OUT_DIR / "level_comparison.csv", index=False)


# ============================================================================
# 4. Write the tables and a readable report
# ============================================================================
sum_df = pd.DataFrame(summary_rows)
var_df = pd.DataFrame(variance_rows)
sum_df.to_csv(OUT_DIR / "summary_levels.csv", index=False)
var_df.to_csv(OUT_DIR / "variance_decomposition.csv", index=False)

lines = ["HydroBASINS level 6 and level 7 over the study area", "=" * 52, ""]
lines.append(sum_df.T.to_string())
lines += ["", "Variance decomposition of the long-term mean precipitation field",
          "-" * 62, var_df.to_string(index=False)]
lines += ["", "Relation between the two levels", "-" * 31,
          cmp_df.to_string(index=False)]
(OUT_DIR / "report.txt").write_text("\n".join(lines), encoding="utf-8")

print(f"\n✓ Written to {OUT_DIR}")
for f in ["summary_levels.csv", "variance_decomposition.csv",
          "level_comparison.csv", "report.txt", "per_basin_lev06.csv",
          "per_basin_lev07.csv"]:
    print(f"    {f}")

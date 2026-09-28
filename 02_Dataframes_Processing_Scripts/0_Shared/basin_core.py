# -*- coding: utf-8 -*-
"""
basin_core.py — HydroBASINS basin features (shared by both store builders)
==========================================================================

Basin features are NOT point-sampled from a raster; they come from a
point-in-polygon join (well -> HYBAS_ID) plus the per-basin monthly series
produced by the basin-aggregation scripts. This module is the single place that
logic lives, so the static and dynamic store builders stay consistent.

Location: 02_Dataframes_Processing_Scripts/0_Shared/basin_core.py
          (imported by the scripts in 2_Feature_Stores/)

TWO climate products on the same basins
---------------------------------------
The well -> HYBAS_ID join is a property of geometry, so it is computed ONCE and
shared. On top of that join we aggregate TWO climate products, each on its own
grid, into the same basins:

  era5l : ERA5-Land fluxes      (05a_hydrobasins_era5_basin_means.py)          -> basin_era5_levX/
  terra : TerraClimate forcing  (05b_hydrobasins_terraclimate_basin_means.py)  -> basin_terraclimate_levX/
  (both scripts are in 02_Processing_Scripts/HydroSHEDS/)

They live under distinct, self-describing prefixes so a column's product, level
and group are readable from the name alone:

  basin_era5l_levX__<var>   e.g. basin_era5l_lev06__tp / __ro / __ssro / __sro
  basin_terra_levX__<var>   e.g. basin_terra_lev06__precipitation / __temp_max / …

The split that matters (per the project's de-duplication rule)
--------------------------------------------------------------
HYBAS_ID / UP_AREA / COAST are STATIC basin GEOMETRY attributes — identical for
both products (they come from the shapefile, not from any grid). They are NOT
duplicated per product: they keep the product-neutral basin_levX__ prefix and
are written ONLY into the static store. coverage_frac / basin_area_m2 /
covered_area_m2 ARE grid-dependent (ERA5-Land ~9 km vs TerraClimate ~4.5 km
rasterize to different areas), so those carry a product prefix.

  STATIC  store gets:
      basin_levX__HYBAS_ID, basin_levX__UP_AREA, basin_levX__COAST   (shared, once)
      basin_era5l_levX__coverage_frac / __basin_area_m2 / __covered_area_m2
      basin_terra_levX__coverage_frac / __basin_area_m2 / __covered_area_m2
  DYNAMIC store gets:
      basin_era5l_levX__<flux>           (tp/ro/ssro/sro)            ONLY
      basin_terra_levX__<var>            (14 TerraClimate variables) ONLY

Unit handling: store columns are SHORT (no unit suffix). Units live in the
feature-metadata table / label overrides, exactly like the ERA5 columns. The
ERA5 source CSVs carry <var>_mm_day; the TerraClimate source CSVs carry
terra_<var>, partly with a unit suffix (e.g. terra_temp_max_degC); both are
stripped to the bare <var> on the way into the store. For TerraClimate
variables whose CSV column name has appeared with and without a unit suffix,
both spellings are accepted (see PRODUCTS below).

Per-product source layout
--------------------------
  01_Input_Global/3_Final/HydroSHEDS_RIVERS_BASINS/
      basin_era5_lev06/<HYBAS_ID>.csv          time, basin_area_m2,
          covered_area_m2, coverage_frac, tp_mm_day, ro_mm_day, ssro_mm_day,
          sro_mm_day
      basin_era5_lev07/<HYBAS_ID>.csv
      basin_terraclimate_lev06/<HYBAS_ID>.csv  time, terra_basin_area_m2,
          terra_covered_area_m2, terra_coverage_frac, terra_precipitation, …
      basin_terraclimate_lev07/<HYBAS_ID>.csv
      metadata_lev06.csv     HYBAS_ID, …, UP_AREA, COAST, …  (shapefile attrs,
      metadata_lev07.csv                                       product-neutral)
  01_Input_Global/1_Original/HydroSHEDS_RIVERS_BASINS/
      hybas_eu_lev06_v1c/hybas_eu_lev06_v1c.shp
      hybas_eu_lev07_v1c/hybas_eu_lev07_v1c.shp

A product whose CSV folder is absent is simply skipped — its columns do not
appear — so the store can be built before TerraClimate basins exist and gains
the terra_ columns automatically once 05b_hydrobasins_terraclimate_basin_means.py
has run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LEVELS = ["lev06", "lev07"]

# Static per-product diagnostics carried into the static store (one value/well).
STAT_KEYS = ["coverage_frac", "basin_area_m2", "covered_area_m2"]

# ----------------------------------------------------------------------------
# PRODUCT REGISTRY — the single source of truth for both products.
#   prefix      : token used in the column group  -> basin_<prefix>_levX__<var>
#   csv_subdir  : per-level source folder under …/HydroSHEDS_RIVERS_BASINS/
#   vars        : list of (store_var, source_csv_column); store_var is the SHORT
#                 name written to the store, source_csv_column is what the basin
#                 script wrote (a tuple lists accepted alternative spellings;
#                 the first one present in the CSV is used). Order is preserved
#                 in the output columns.
#   stat        : {canonical_stat_key: source_csv_column} for the static diags.
# ----------------------------------------------------------------------------
PRODUCTS = {
    "era5l": {
        "csv_subdir": {"lev06": "basin_era5_lev06", "lev07": "basin_era5_lev07"},
        "vars": [
            ("tp",   "tp_mm_day"),
            ("ro",   "ro_mm_day"),
            ("ssro", "ssro_mm_day"),
            ("sro",  "sro_mm_day"),
        ],
        "stat": {
            "coverage_frac":   "coverage_frac",
            "basin_area_m2":   "basin_area_m2",
            "covered_area_m2": "covered_area_m2",
        },
    },
    "terra": {
        "csv_subdir": {"lev06": "basin_terraclimate_lev06",
                       "lev07": "basin_terraclimate_lev07"},
        "vars": [
            ("precipitation",          ("terra_precipitation", "terra_precipitation_mm")),
            ("runoff",                 ("terra_runoff", "terra_runoff_mm")),
            ("ET_actual",              ("terra_ET_actual", "terra_ET_actual_mm")),
            ("ET_potential",           ("terra_ET_potential", "terra_ET_potential_mm")),
            ("water_deficit",          ("terra_water_deficit", "terra_water_deficit_mm")),
            ("soil_moisture",          ("terra_soil_moisture", "terra_soil_moisture_mm")),
            ("snow_water",             "terra_snow_water_mm"),
            ("solar_radiation",        "terra_solar_radiation_Wm2"),
            ("temp_max",               "terra_temp_max_degC"),
            ("temp_min",               "terra_temp_min_degC"),
            ("vapor_pressure",         "terra_vapor_pressure_kPa"),
            ("vapor_pressure_deficit", "terra_vapor_pressure_deficit_kPa"),
            ("drought_index_pdsi",     "terra_drought_index_pdsi"),
            ("wind_speed",             "terra_wind_speed_ms"),
        ],
        "stat": {
            "coverage_frac":   "terra_coverage_frac",
            "basin_area_m2":   "terra_basin_area_m2",
            "covered_area_m2": "terra_covered_area_m2",
        },
    },
}

# Short names of the ERA5-Land basin fluxes (for code that imports bc.BASIN_VARS)
BASIN_VARS = [v for v, _ in PRODUCTS["era5l"]["vars"]]


# ============================================================================
# PATHS  (relative to the project root; adjust here if the layout changes)
# ============================================================================
# Per-basin CSVs, basin metadata and masks (from 02_Processing_Scripts/HydroSHEDS)
BASIN_FINAL_REL = Path("01_Input_Global") / "3_Final" / "HydroSHEDS_RIVERS_BASINS"
# HydroBASINS shapefiles (original download)
BASIN_SHP_REL = Path("01_Input_Global") / "1_Original" / "HydroSHEDS_RIVERS_BASINS"
SHP_FILES = {
    "lev06": Path("hybas_eu_lev06_v1c") / "hybas_eu_lev06_v1c.shp",
    "lev07": Path("hybas_eu_lev07_v1c") / "hybas_eu_lev07_v1c.shp",
}


# ============================================================================
# PATH RESOLUTION / AVAILABILITY
# ============================================================================
def basin_paths(root: Path) -> dict:
    base = root / BASIN_FINAL_REL
    shp_base = root / BASIN_SHP_REL
    return {
        "shp": {lvl: shp_base / SHP_FILES[lvl] for lvl in LEVELS},
        "meta": {lvl: base / f"metadata_{lvl}.csv" for lvl in LEVELS},
        # product -> {level -> per-basin CSV directory}
        "csv_dir": {prod: {lvl: base / cfg["csv_subdir"][lvl] for lvl in LEVELS}
                    for prod, cfg in PRODUCTS.items()},
    }


def present_products(paths: dict) -> list:
    """Products whose per-basin CSV folder exists for EVERY level."""
    return [p for p in PRODUCTS
            if all(paths["csv_dir"][p][lvl].exists() for lvl in LEVELS)]


def available(root: Path) -> bool:
    """True if geopandas + the join inputs + at least one product are present."""
    try:
        import geopandas  # noqa: F401
    except ImportError:
        return False
    P = basin_paths(root)
    base_ok = all(P["shp"][l].exists() and P["meta"][l].exists() for l in LEVELS)
    return base_ok and len(present_products(P)) > 0


# ============================================================================
# SPATIAL JOIN  well -> HYBAS_ID (point-in-polygon, per level) — product-neutral
# ============================================================================
def join_wells(lons, lats, shp_paths: dict) -> dict:
    """Return {level: float array of HYBAS_ID per well} (NaN where unmatched)."""
    import geopandas as gpd
    lons = np.asarray(lons, float)
    lats = np.asarray(lats, float)
    pts = gpd.GeoDataFrame({"_i": np.arange(len(lons))},
                           geometry=gpd.points_from_xy(lons, lats),
                           crs="EPSG:4326")
    out = {}
    for lvl in LEVELS:
        basins = gpd.read_file(shp_paths[lvl])
        if basins.crs is None or basins.crs.to_epsg() != 4326:
            basins = basins.to_crs(epsg=4326)
        basins = basins[["HYBAS_ID", "geometry"]]
        j = gpd.sjoin(pts, basins, how="left", predicate="intersects")
        # shared boundaries can multi-match -> keep first; restore well order
        j = j.drop_duplicates(subset="_i", keep="first").sort_values("_i")
        out[lvl] = j["HYBAS_ID"].to_numpy(dtype="float64")
    return out


def needed_basins(hybas_by_level: dict) -> dict:
    return {lvl: np.unique(h[~np.isnan(h)]).astype("int64")
            for lvl, h in hybas_by_level.items()}


# ============================================================================
# LOAD per-basin CSVs once  (per product: static first-row diags + series)
# ============================================================================
def _read_basin_csv(path: Path, cfg: dict) -> dict:
    """Read one per-basin CSV for one product; short-rename the series columns."""
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"]).dt.to_period("M").dt.to_timestamp()
    df = df.set_index("time")

    stat = {canon: (float(df[src].iloc[0]) if src in df.columns else np.nan)
            for canon, src in cfg["stat"].items()}

    rename = {}
    for short, src in cfg["vars"]:
        for cand in ((src,) if isinstance(src, str) else src):
            if cand in df.columns:
                rename[cand] = short
                break
    series = df[list(rename.keys())].rename(columns=rename)   # cols = short names
    return {"stat": stat, "series": series}


def load_basin_csvs(needed_by_level: dict, paths: dict) -> dict:
    """
    Return a cache:
        cache["products"]          -> list of products actually present
        cache[level][int(bid)]     -> {product: {"stat": {...}, "series": df}}
    A basin missing for a given product is simply absent from that product's
    sub-dict (handled as NaN downstream).
    """
    prods = present_products(paths)
    cache: dict = {"products": prods}
    for lvl in LEVELS:
        d: dict = {}
        for bid in needed_by_level.get(lvl, []):
            bid = int(bid)
            entry = {}
            for prod in prods:
                f = paths["csv_dir"][prod][lvl] / f"{bid}.csv"
                if f.exists():
                    entry[prod] = _read_basin_csv(f, PRODUCTS[prod])
            if entry:
                d[bid] = entry
        cache[lvl] = d
    return cache


# ============================================================================
# STATIC basin columns  (HYBAS_ID/UP_AREA/COAST shared; diags per product)
# ============================================================================
def static_basin_frame(ids, hybas_by_level: dict, cache: dict, paths: dict) -> pd.DataFrame:
    prods = cache["products"] if "products" in cache else present_products(paths)
    out = pd.DataFrame({"well_id": [f"MW_{i}" for i in ids]})

    for lvl in LEVELS:
        hyb = hybas_by_level[lvl]
        meta = pd.read_csv(paths["meta"][lvl]).drop_duplicates(subset="HYBAS_ID")
        meta = meta.set_index("HYBAS_ID")
        up_s    = meta["UP_AREA"] if "UP_AREA" in meta.columns else None
        coast_s = meta["COAST"]   if "COAST"   in meta.columns else None

        hid, up, coast = [], [], []
        # per-product static diagnostics
        pstat = {prod: {k: [] for k in STAT_KEYS} for prod in prods}

        for h in hyb:
            if np.isnan(h):
                hid.append(pd.NA); up.append(np.nan); coast.append(pd.NA)
                for prod in prods:
                    for k in STAT_KEYS:
                        pstat[prod][k].append(np.nan)
                continue
            hi = int(h)
            hid.append(hi)
            up.append(float(up_s.get(hi, np.nan)) if up_s is not None else np.nan)
            coast.append(int(coast_s.get(hi)) if (coast_s is not None and hi in coast_s.index
                                                  and pd.notna(coast_s.get(hi))) else pd.NA)
            entry = cache[lvl].get(hi, {})
            for prod in prods:
                st = entry.get(prod, {}).get("stat", {}) if entry else {}
                for k in STAT_KEYS:
                    pstat[prod][k].append(st.get(k, np.nan))

        # shared identity / geometry (product-neutral, written once)
        out[f"basin_{lvl}__HYBAS_ID"] = pd.array(hid, dtype="Int64")
        out[f"basin_{lvl}__UP_AREA"]  = up
        out[f"basin_{lvl}__COAST"]    = pd.array(coast, dtype="Int64")
        # grid-dependent diagnostics, one block per product
        for prod in prods:
            for k in STAT_KEYS:
                out[f"basin_{prod}_{lvl}__{k}"] = pstat[prod][k]

    return out


# ============================================================================
# DYNAMIC basin columns for one well  (NO HYBAS_ID; both products)
# ============================================================================
def well_series_columns(cache: dict, hybas_by_level: dict, w: int,
                        months: pd.DatetimeIndex) -> dict:
    prods = cache["products"] if "products" in cache else list(PRODUCTS)
    cols = {}
    for lvl in LEVELS:
        h = hybas_by_level[lvl][w]
        entry = None if np.isnan(h) else cache[lvl].get(int(h))
        for prod in prods:
            pe = entry.get(prod) if entry else None
            for short, _src in PRODUCTS[prod]["vars"]:
                name = f"basin_{prod}_{lvl}__{short}"   # unit -> metadata
                if pe is not None and short in pe["series"].columns:
                    cols[name] = pe["series"][short].reindex(months).to_numpy()
                else:
                    cols[name] = np.full(len(months), np.nan, dtype="float64")
    return cols

# -*- coding: utf-8 -*-
"""
02_build_dynamic_store.py — assemble the single dynamic feature store
=====================================================================

Runs the extraction engine (0_Shared/extract_core.py) over every dynamic point
dataset in the manifest, builds the land cover features (0_Shared/transforms.py)
and adds the monthly HydroBASINS series (0_Shared/basin_core.py). It produces,
per well, a single tidy table:

    <OUT_DIR>/03_global_monthly/MW_<id>.csv
        time, <GWL + any extra cols from the monthly CSV>,
        era5l__*, terra__*, twsa__*, C3S__LCC, C3S__LCC_previous,
        C3S__LCC_changed, C3S__months_since_LCC_change, C3S__LCC_stable,
        basin_era5l_lev06__*, basin_era5l_lev07__*,
        basin_terra_lev06__*, basin_terra_lev07__*

plus two side files:
    <OUT_DIR>/03_global_provenance.csv    one row per well: <ds>__sample_dist_km,
                                          <ds>__rescued, <ds>__outcome
    <OUT_DIR>/03_global_build_report.csv  per-well month count / time span / status

Input : 03_Dataframes/02_Static_Features/01_GEMS-GER/02_adapted_well_metadata.csv
        03_Dataframes/01_Dynamic_Features/02_GEMS-GER_monthly/MW_<id>_monthly.csv
        01_Input_Global/3_Final/<Dataset>/<file>.nc   (from 02_Processing_Scripts)
        HydroBASINS shapefiles and per-basin CSVs     (see 0_Shared/basin_core.py)
Output: 03_Dataframes/01_Dynamic_Features/            (<OUT_DIR>)

Design
------
* The well's monthly GWL CSV is the ROW SPINE; features are attached by `time`
  (left join), so every GWL month is kept and features are NaN outside their
  dataset's coverage (e.g. TWSA after 2020). NO global trimming — completeness
  is a per-experiment decision applied later on the columns each model uses.
* All datasets share one well order; extraction runs ONCE per dataset.
* Feature values are float32. LCC engineered features are namespaced with the
  C3S__ prefix (C3S__LCC, C3S__LCC_previous, ...) so every column in the store
  carries a clear dataset prefix.

Location: 02_Dataframes_Processing_Scripts/2_Feature_Stores/
Run
---
    python 02_build_dynamic_store.py --limit 20      # dry run: 20 wells, low RAM
    python 02_build_dynamic_store.py                 # full run (all wells)
    python 02_build_dynamic_store.py --root "<path to project root>"

If the project lies in a cloud-synchronised folder (e.g. OneDrive), pause the
synchronisation during a full run (thousands of small CSV writes).
"""

from __future__ import annotations

import argparse
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# SHARED MODULES  (02_Dataframes_Processing_Scripts/0_Shared)
# =============================================================================
sys.path.append(str(Path(__file__).resolve().parents[1] / "0_Shared"))
import dataset_manifest as dm     # noqa: E402
import extract_core as ec         # noqa: E402 (self-bootstraps nc_io)
import transforms as tr           # noqa: E402
import basin_core as bc           # noqa: E402

try:
    from tqdm import tqdm as _tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ============================================================================
# PATHS  (relative to the project root; adjust here if the layout changes)
# The project root itself can be overridden with --root.
# ============================================================================
WELL_METADATA_REL = (Path("03_Dataframes") / "02_Static_Features" / "01_GEMS-GER"
                     / "02_adapted_well_metadata.csv")
GWL_DIR_REL = Path("03_Dataframes") / "01_Dynamic_Features" / "02_GEMS-GER_monthly"
OUTPUT_DIR_REL = Path("03_Dataframes") / "01_Dynamic_Features"
WELLS_SUBDIR = "03_global_monthly"                     # per-well CSVs
PROVENANCE_FILE = "03_global_provenance.csv"
REPORT_FILE = "03_global_build_report.csv"


def make_paths(root: Path):
    return {
        "meta":   root / WELL_METADATA_REL,
        "gwl":    root / GWL_DIR_REL,
        "out":    root / OUTPUT_DIR_REL,
    }


# ============================================================================
# CONFIGURATION
# ============================================================================
TIME_COL = "time"

# Redundant columns from the monthly GWL CSV that should NOT enter the store.
# time_days = "days since 1970-01-01" is a NetCDF/CF storage convention, not a
# CSV one; it just duplicates `time` and is trivially recomputable, so drop it.
GWL_KEEP_COLS = ["time", "GWL"]

# Append HydroBASINS aggregated climate columns for BOTH products:
#   basin_era5l_levX__<flux>   (tp/ro/ssro/sro, ERA5-Land)
#   basin_terra_levX__<var>    (14 TerraClimate variables)
# Products with no per-basin CSV folder are skipped automatically. HYBAS_ID and
# the other static basin attributes are NOT written here — they live only in the
# static store.
INCLUDE_BASIN = True

# Prefix for the engineered LCC features so the whole group is namespaced like
# every other dataset (C3S__LCC, C3S__LCC_previous, ...). Source = C3S/ESA-CCI
# land cover. Change here if a different prefix is wanted (e.g. "lcc").
LCC_PREFIX = "C3S"


def progress(it, desc="", total=None):
    return _tqdm(it, desc=desc, total=total, ncols=80) if HAS_TQDM else it


def log(step, total, msg):
    print(f"\n{'─'*60}\n  Step {step}/{total}: {msg}\n{'─'*60}")


def month_start(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.to_period("M").dt.to_timestamp()


# ============================================================================
# MAIN
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="project root (optional)")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only the first N wells (dry run)")
    args = ap.parse_args()

    root = Path(args.root).resolve() if args.root else dm.PROJECT_ROOT
    P = make_paths(root)
    wells_out = P["out"] / WELLS_SUBDIR
    wells_out.mkdir(parents=True, exist_ok=True)

    t_total = _time.perf_counter()
    TOTAL = 4

    # ---- 1. wells ----------------------------------------------------------
    log(1, TOTAL, "Load wells")
    ids, lons, lats = ec._load_wells(root, WELL_METADATA_REL.parts)
    if args.limit:
        ids, lons, lats = ids[:args.limit], lons[:args.limit], lats[:args.limit]
        print(f"  DRY RUN: limited to first {len(ids)} wells")
    print(f"  project root : {root}")
    print(f"  wells        : {len(ids)}")
    print(f"  GWL CSV dir  : {P['gwl']}")
    print(f"  output       : {P['out']}")

    # ---- 2. extract every dynamic point dataset (once) ---------------------
    point_specs = [s for s in dm.dynamic_features() if s.mode == "point"]
    log(2, TOTAL, f"Extract {len(point_specs)} dynamic datasets: "
                  f"{[s.key for s in point_specs]}")

    monthly = {}     # key -> {'idx': DatetimeIndex, 'cols': {name: (T, n_wells)}}
    lcc_annual = None
    prov_cols = {}   # provenance columns to assemble later
    for spec in point_specs:
        res = ec.extract_dataset(spec, ids, lons, lats, root=root, verbose=True)
        p = res["prov"]
        prov_cols[f"{spec.key}__sample_dist_km"] = np.round(p["dist_km"], 3)
        prov_cols[f"{spec.key}__rescued"] = p["rescued"]
        prov_cols[f"{spec.key}__outcome"] = p["outcome"]

        if spec.transform == "lcc_features":
            arr = res["columns"][f"{spec.key}__lccs_class"]      # (n_years, n_wells)
            yrs = pd.DatetimeIndex(res["times"]).year.values
            lcc_annual = {"years": yrs, "arr": arr}
        else:
            idx = month_start(pd.Series(res["times"])).values
            monthly[spec.key] = {"idx": pd.DatetimeIndex(idx), "cols": res["columns"]}

    # ---- 2b. basin join (optional) ----------------------------------------
    use_basin = INCLUDE_BASIN and bc.available(root)
    hybas = basin_cache = None
    if use_basin:
        bpaths = bc.basin_paths(root)
        hybas = bc.join_wells(lons, lats, bpaths["shp"])
        basin_cache = bc.load_basin_csvs(bc.needed_basins(hybas), bpaths)
        msg = ", ".join(f"{lvl}={int(np.isfinite(hybas[lvl]).sum())}/{len(ids)}"
                        for lvl in bc.LEVELS)
        print(f"  basin: matched wells to HydroBASINS -> {msg} "
              f"(flux cols only; HYBAS_ID stays in the static store)")
    elif INCLUDE_BASIN:
        print("  basin: inputs not found -> skipping basin flux columns")

    # ---- 3. per-well assembly ---------------------------------------------
    log(3, TOTAL, "Assemble per-well CSVs")
    report_rows, prov_rows = [], []
    saved = skipped = 0
    t_read = t_build = t_write = 0.0

    for w, wid in enumerate(progress(ids, desc="  Wells", total=len(ids))):
        # provenance row (always recorded, even if GWL CSV missing)
        prov_rows.append({"well_id": f"MW_{wid}",
                          **{c: prov_cols[c][w] for c in prov_cols}})

        gwl_csv = P["gwl"] / f"MW_{wid}_monthly.csv"
        if not gwl_csv.exists():
            report_rows.append({"well_id": f"MW_{wid}", "n_months": 0,
                                "t_min": "", "t_max": "", "status": "NO_GWL_CSV"})
            skipped += 1
            continue

        _t = _time.perf_counter()
        df = pd.read_csv(gwl_csv, usecols=lambda c: c in GWL_KEEP_COLS)
        # keep only 'time' and 'GWL', in this order
        keep = [c for c in GWL_KEEP_COLS if c in df.columns]
        df = df[keep]

        df[TIME_COL] = month_start(df[TIME_COL])
        df = df.sort_values(TIME_COL).reset_index(drop=True)
        months = pd.DatetimeIndex(df[TIME_COL])
        t_read += _time.perf_counter() - _t

        _t = _time.perf_counter()
        feat = {}
        # monthly datasets: align this well's series to the well's months
        for key, blk in monthly.items():
            ser = pd.DataFrame({c: blk["cols"][c][:, w] for c in blk["cols"]},
                               index=blk["idx"]).reindex(months)
            for c in ser.columns:
                feat[c] = ser[c].to_numpy()
        # LCC engineered features via the faithful transform
        if lcc_annual is not None:
            annual = pd.Series(lcc_annual["arr"][:, w],
                               index=pd.Index(lcc_annual["years"], name="year"))
            lcc_df = tr.lcc_features(annual, months)
            for c in lcc_df.columns:
                feat[f"{LCC_PREFIX}__{c}"] = lcc_df[c].to_numpy()

        # basin flux columns (NO HYBAS_ID — that is static)
        if use_basin:
            feat.update(bc.well_series_columns(basin_cache, hybas, w, months))

        out_df = df.copy()
        for c, vals in feat.items():
            out_df[c] = vals
        t_build += _time.perf_counter() - _t

        _t = _time.perf_counter()
        out_df.to_csv(wells_out / f"MW_{wid}.csv", index=False)
        t_write += _time.perf_counter() - _t

        report_rows.append({
            "well_id":  f"MW_{wid}",
            "n_months": len(out_df),
            "t_min":    months.min().strftime("%Y-%m"),
            "t_max":    months.max().strftime("%Y-%m"),
            "status":   "OK",
        })
        saved += 1

    # ---- 4. side files + summary ------------------------------------------
    log(4, TOTAL, "Write provenance + report")
    prov_df = pd.DataFrame(prov_rows).sort_values("well_id").reset_index(drop=True)
    prov_df.to_csv(P["out"] / PROVENANCE_FILE, index=False)
    report_df = pd.DataFrame(report_rows).sort_values("well_id").reset_index(drop=True)
    report_df.to_csv(P["out"] / REPORT_FILE, index=False)

    print(f"\n  read  CSVs : {t_read:6.1f}s")
    print(f"  build rows : {t_build:6.1f}s")
    print(f"  write CSVs : {t_write:6.1f}s   (slow => pause OneDrive)")
    print(f"\n  Saved      : {saved} well CSVs -> {wells_out}")
    print(f"  Skipped    : {skipped} (no GWL CSV)")
    print(f"  Provenance : {P['out'] / PROVENANCE_FILE}")
    print(f"  Report     : {P['out'] / REPORT_FILE}")

    print("\n  Status:")
    for s, c in report_df["status"].value_counts().items():
        print(f"    {s:<14} {c}")

    if saved:
        sample = report_df[report_df["status"] == "OK"]["well_id"].iloc[0]
        sdf = pd.read_csv(wells_out / f"{sample}.csv", nrows=2)
        print(f"\n  Sample {sample}.csv  ({len(sdf.columns)} cols):")
        print(f"    {list(sdf.columns)}")

    print(f"\n  Total: {_time.perf_counter() - t_total:.1f}s   Done ✅")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
01_build_static_store.py — assemble the single static feature store
===================================================================

Runs the extraction engine (0_Shared/extract_core.py) over every static point
raster in the manifest (0_Shared/dataset_manifest.py), adds the static
HydroBASINS attributes (0_Shared/basin_core.py) and writes ONE table, one row
per well:

    <OUT_DIR>/static_features.csv
        <all metadata cols incl. MW_ID, lon, lat>,
        era5l_static__*, soilgrids__*, gasy__sy_<depth>, glim__*,
        whymap__hygeo2, hydsheds_terrain__*, hydrivers_dist__*,
        basin_levX__*, basin_era5l_levX__*, basin_terra_levX__*

    <OUT_DIR>/_provenance.csv     one row per well: <ds>__sample_dist_km,
                                  <ds>__rescued, <ds>__outcome
    <OUT_DIR>/_build_report.csv   per-well NaN-feature count
    <OUT_DIR>/_feature_nan_summary.csv   per-feature NaN count and percentage

Input : 03_Dataframes/02_Static_Features/01_GEMS-GER/02_adapted_well_metadata.csv
        01_Input_Global/3_Final/<Dataset>/<file>.nc   (from 02_Processing_Scripts)
        HydroBASINS shapefiles and per-basin CSVs     (see 0_Shared/basin_core.py)
Output: 03_Dataframes/02_Static_Features/02_global/   (<OUT_DIR>)

Notes
-----
* Row spine = the well metadata CSV (02_adapted_well_metadata.csv): all its
  columns are carried through unchanged, features are appended. So the static
  store is self-contained (well attributes + extracted features).
* Index rasters (basin_mask_*) are not sampled (role='index'). The basin
  attributes come from the polygon join in basin_core.py (INCLUDE_BASIN).
* Categorical features (manifest `categorical`) are written as nullable Int64
  class codes (e.g. litho_class, hygeo2, dir, slt/tvh/tvl), not floats.

Location: 02_Dataframes_Processing_Scripts/2_Feature_Stores/
Run
---
    python 01_build_static_store.py
    python 01_build_static_store.py --limit 20                  # dry run
    python 01_build_static_store.py --root "<path to project root>"

If the project lies in a cloud-synchronised folder (e.g. OneDrive), pause the
synchronisation during the run.
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
import basin_core as bc           # noqa: E402

# =============================================================================
# PATHS  (relative to the project root; adjust here if the layout changes)
# =============================================================================
WELL_METADATA_REL = (Path("03_Dataframes") / "02_Static_Features" / "01_GEMS-GER"
                     / "02_adapted_well_metadata.csv")
OUTPUT_DIR_REL = Path("03_Dataframes") / "02_Static_Features" / "02_global"

# =============================================================================
# CONFIGURATION
# =============================================================================
# Append HydroBASINS static attributes. Shared identity/geometry stays product-
# neutral and is written ONCE (basin_levX__HYBAS_ID/UP_AREA/COAST); the grid-
# dependent coverage/area diagnostics are written PER product
# (basin_era5l_levX__coverage_frac/basin_area_m2/covered_area_m2 and the
# basin_terra_levX__ equivalents). HYBAS_ID lives here — once — and never in the
# dynamic store.
INCLUDE_BASIN = True


def make_paths(root: Path):
    return {
        "meta": root / WELL_METADATA_REL,
        "out":  root / OUTPUT_DIR_REL,
    }


def log(step, total, msg):
    print(f"\n{'─'*60}\n  Step {step}/{total}: {msg}\n{'─'*60}")


def load_meta(meta_path: Path):
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    df = pd.read_csv(meta_path)
    id_col = "MW_ID" if "MW_ID" in df.columns else df.columns[0]
    for c in ("lon", "lat"):
        if c not in df.columns:
            raise KeyError(f"'{c}' missing in {meta_path.name}; have {list(df.columns)}")
    df = df.dropna(subset=["lon", "lat"]).reset_index(drop=True)
    ids = (df[id_col].astype(str).str.replace("MW_", "", regex=False).str.strip()).tolist()
    return df, id_col, ids, df["lon"].to_numpy(float), df["lat"].to_numpy(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="project root (optional)")
    ap.add_argument("--limit", type=int, default=None, help="first N wells (dry run)")
    args = ap.parse_args()

    root = Path(args.root).resolve() if args.root else dm.PROJECT_ROOT
    P = make_paths(root)
    P["out"].mkdir(parents=True, exist_ok=True)

    t_total = _time.perf_counter()
    TOTAL = 3

    # ---- 1. wells ----------------------------------------------------------
    log(1, TOTAL, "Load well metadata")
    meta, id_col, ids, lons, lats = load_meta(P["meta"])
    if args.limit:
        meta = meta.iloc[:args.limit].reset_index(drop=True)
        ids, lons, lats = ids[:args.limit], lons[:args.limit], lats[:args.limit]
        print(f"  DRY RUN: first {len(ids)} wells")
    print(f"  project root : {root}")
    print(f"  metadata     : {P['meta'].name}  (shape {meta.shape})")
    print(f"  wells        : {len(ids)}")
    print(f"  output       : {P['out']}")

    # ---- 2. extract every static point raster ------------------------------
    static_specs = [s for s in dm.static_features() if s.mode == "point"]
    log(2, TOTAL, f"Extract {len(static_specs)} static rasters: "
                  f"{[s.key for s in static_specs]}")

    out_df = meta.copy()
    prov = {"well_id": [f"MW_{i}" for i in ids]}
    feature_cols = []

    for spec in static_specs:
        res = ec.extract_dataset(spec, ids, lons, lats, root=root, verbose=True)
        cat_cols = {f"{spec.key}__{v}" for v in spec.categorical}
        for col, vals in res["columns"].items():
            s = pd.Series(vals, index=out_df.index)
            if col in cat_cols:
                s = s.round().astype("Int64")     # clean class codes, <NA> for missing
            out_df[col] = s
            feature_cols.append(col)
        p = res["prov"]
        prov[f"{spec.key}__sample_dist_km"] = np.round(p["dist_km"], 3)
        prov[f"{spec.key}__rescued"] = p["rescued"]
        prov[f"{spec.key}__outcome"] = p["outcome"]

    # ---- 2b. basin static attributes (HYBAS_ID lives here — once) ----------
    if INCLUDE_BASIN and bc.available(root):
        print("\n  basin: joining wells to HydroBASINS (lev06/lev07)…")
        bpaths = bc.basin_paths(root)
        hybas = bc.join_wells(lons, lats, bpaths["shp"])
        cache = bc.load_basin_csvs(bc.needed_basins(hybas), bpaths)
        sf = bc.static_basin_frame(ids, hybas, cache, bpaths)
        for c in sf.columns:
            if c == "well_id":
                continue
            out_df[c] = sf[c].values
            if c.split("__")[-1] != "HYBAS_ID":   # HYBAS_ID is an identifier, not a feature
                feature_cols.append(c)
        for lvl in bc.LEVELS:
            print(f"    {lvl}: matched {int(sf[f'basin_{lvl}__HYBAS_ID'].notna().sum())}/{len(ids)} wells")
    elif INCLUDE_BASIN:
        print("\n  basin: inputs not found -> skipping basin attributes")

    # ---- 3. write store + provenance + report ------------------------------
    log(3, TOTAL, "Write store + provenance + report")
    out_csv = P["out"] / "static_features.csv"
    out_df.to_csv(out_csv, index=False)

    prov_df = pd.DataFrame(prov).sort_values("well_id").reset_index(drop=True)
    prov_df.to_csv(P["out"] / "_provenance.csv", index=False)

    nan_per_well = out_df[feature_cols].isna().sum(axis=1)
    report_df = pd.DataFrame({
        "well_id":     [f"MW_{i}" for i in ids],
        "n_features":  len(feature_cols),
        "n_nan":       nan_per_well.values,
    }).sort_values("well_id").reset_index(drop=True)
    report_df.to_csv(P["out"] / "_build_report.csv", index=False)

    # per-FEATURE NaN breakdown — the actionable view: a column NaN for ALL wells
    # is usually expected (e.g. a permafrost-only layer over Germany), a column
    # NaN for SOME wells is coverage at the raster edge.
    nan_per_feat = out_df[feature_cols].isna().sum(axis=0).sort_values(ascending=False)
    feat_summary = pd.DataFrame({
        "feature": nan_per_feat.index,
        "n_nan":   nan_per_feat.values,
        "pct":     (nan_per_feat.values / len(ids) * 100).round(1),
    })
    feat_summary.to_csv(P["out"] / "_feature_nan_summary.csv", index=False)

    print(f"\n  Saved store : {out_csv}")
    print(f"    shape     : {out_df.shape}  ({len(feature_cols)} feature cols)")
    print(f"  Provenance  : {P['out'] / '_provenance.csv'}")
    print(f"  Report      : {P['out'] / '_build_report.csv'}")

    # how many wells have all features present vs some NaN
    n_full = int((nan_per_well == 0).sum())
    print(f"\n  Wells with all static features present : {n_full} / {len(ids)}")

    nz = feat_summary[feat_summary["n_nan"] > 0]
    if len(nz):
        print("\n  Features with missing values (which column, NaN count / %):")
        for _, r in nz.head(20).iterrows():
            print(f"    {r['feature']:<36} {int(r['n_nan']):>5} / {len(ids)}  ({r['pct']}%)")
        n_allnan = int((feat_summary['n_nan'] == len(ids)).sum())
        if n_allnan:
            print(f"  -> {n_allnan} feature(s) NaN for EVERY well "
                  f"(systematic — check if expected, e.g. permafrost-only layer)")
    else:
        print("  No missing feature values.")

    # per-dataset rescue / no-valid summary from provenance
    print("\n  Per-dataset outcomes:")
    for spec in static_specs:
        oc = prov[f"{spec.key}__outcome"]
        vals, counts = np.unique(oc, return_counts=True)
        summ = "  ".join(f"{v}={c}" for v, c in zip(vals, counts))
        print(f"    {spec.key:<18} {summ}")

    print(f"\n  Sample row (MW_{ids[0]}), first feature of each dataset:")
    show = ["MW_ID", "lon", "lat"] + [next(c for c in feature_cols if c.startswith(s.key + "__"))
                                      for s in static_specs]
    show = [c for c in show if c in out_df.columns]
    print(out_df[show].head(1).to_string(index=False))

    print(f"\n  Total: {_time.perf_counter() - t_total:.1f}s   Done ✅")


if __name__ == "__main__":
    main()

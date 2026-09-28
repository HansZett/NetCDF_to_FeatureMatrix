# -*- coding: utf-8 -*-
"""
03_extraction_quality_report.py - evaluate the raster extraction quality
=========================================================================

Reads the side files written by 01_build_static_store.py and
02_build_dynamic_store.py
and answers two questions in a reproducible, citable way:

    1) Which datasets carry MISSING VALUES at the well locations?
       (no valid cell within the search radius, or a fill value in a single
        variable while the reference variable of that dataset was valid)

    2) Which datasets required the value of a NEIGHBOURING PIXEL?
       (outcome == 'rescued': the strict nearest cell was NoData, so the closest
        valid cell inside the search radius was used instead)

Inputs (all optional; whatever is found is used)
------------------------------------------------
    03_Dataframes/02_Static_Features/02_global/
        _provenance.csv                 well x dataset: dist_km/rescued/outcome
        _build_report.csv               well: n_features / n_nan
        _feature_nan_summary.csv        feature: n_nan / pct
        static_features.csv             the static store itself
    03_Dataframes/01_Dynamic_Features/
        03_global_provenance.csv        well x dataset: dist_km/rescued/outcome
        03_global_build_report.csv      well: n_months / t_min / t_max / status
        03_global_monthly/MW_<id>.csv   the dynamic store itself

Alternative file names (static_provenance.csv, static_build_report.csv,
static_feature_nan_summary.csv, dynamic_provenance.csv, ...) are recognised
as well, see NAMES below.

Outputs (written to --out)
--------------------------
    qc_by_dataset.csv        one row per dataset: outcome counts, rescue rate,
                             sampling distance statistics, manifest radius
    qc_by_feature.csv        one row per static feature column with NaN count
    qc_rescued_wells.csv     long table: every (well, dataset) that was rescued
                             or left without a valid cell, with its distance
    qc_completeness_by_dataset.csv
                             missing values per dataset, computed from the
                             stores (dynamic: only inside the coverage window)
    qc_thesis_table.csv      the per-dataset table in the column order and with
                             the dataset names used in the thesis appendix
    qc_summary.txt           the same information as plain text

Location: 02_Dataframes_Processing_Scripts/2_Feature_Stores/
Run
---
    python 03_extraction_quality_report.py
    python 03_extraction_quality_report.py --dynamic-scan sample
    python 03_extraction_quality_report.py --root "<path to project root>"
    python 03_extraction_quality_report.py \
        --static-dir  path/to/02_Static_Features/02_global \
        --dynamic-dir path/to/01_Dynamic_Features \
        --out         path/to/00_Extraction_Quality

Without arguments every well of both stores is evaluated, and the project root
is derived from the location of this file (two levels up, matching
02_Dataframes_Processing_Scripts/2_Feature_Stores/<script>.py) or from the
environment variable GWL_ROOT, and the output goes to
03_Dataframes/00_Extraction_Quality. The script therefore runs unchanged from
an editor such as Spyder; an extra --wdir argument is ignored. The full scan of
the dynamic store reads one CSV per well, which is the slow part of the run.
Pass --dynamic-scan sample for a quick check during development.

Notes
-----
* The pixel is chosen ONCE per (well, dataset) from the dataset's reference
  variable (the first variable in the manifest). A feature can therefore be NaN
  even when the outcome is 'home': the reference variable was valid at that
  cell, but the individual variable carried its fill value there. Both views are
  reported, because only their combination describes the extraction quality.
* Search radii and the on_no_valid policy are read from dataset_manifest.py when
  that module is importable; otherwise those columns stay empty.
"""

from __future__ import annotations

import argparse
import os
import sys
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# PATHS  (relative to the project root; adjust here if the layout changes)
# =============================================================================
STATIC_SUBDIR = ("03_Dataframes", "02_Static_Features", "02_global")
DYNAMIC_SUBDIR = ("03_Dataframes", "01_Dynamic_Features")
DYNAMIC_WELLS_SUBDIR = "03_global_monthly"
OUT_SUBDIR = ("03_Dataframes", "00_Extraction_Quality")
# Folder of dataset_manifest.py, relative to the project root
MANIFEST_DIR_REL = ("02_Dataframes_Processing_Scripts", "0_Shared")

# =============================================================================
# CONFIGURATION
# =============================================================================
OUTCOMES = ["home", "rescued", "nearest_global", "no_valid", "dropped"]

# Accepted file names per report, in priority order. The first entry is the
# current name, the rest are earlier names kept for backward compatibility.
NAMES = {
    "stat_prov":   ["static_provenance.csv", "_provenance.csv"],
    "stat_report": ["static_build_report.csv", "_build_report.csv"],
    "feat_nan":    ["static_feature_nan_summary.csv", "_feature_nan_summary.csv"],
    "dyn_prov":    ["03_global_provenance.csv", "dynamic_provenance.csv",
                    "_provenance.csv"],
    "dyn_report":  ["03_global_build_report.csv", "dynamic_build_report.csv",
                    "_build_report.csv"],
}

# Display names for the thesis table, keyed by the manifest dataset key.
DISPLAY = {
    "twsa": "TWSTORE",
    "era5l": "ERA5-Land time-varying",
    "era5l_static": "ERA5-Land time-invariant",
    "terra": "TerraClimate",
    "soilgrids": "SoilGrids 2.0",
    "lcc": "C3S Land Cover",
    "gasy": "GASY",
    "glim": "GLiM and GLHYMPS",
    "whymap": "WHYMAP",
    "hydsheds_terrain": "HydroSHEDS terrain",
    "hydrivers_dist": "HydroRIVERS distance",
}

# Row order of the thesis table, following the thematic order of chapter 3.
ORDER = ["twsa", "era5l", "era5l_static", "terra", "soilgrids", "lcc",
         "gasy", "glim", "whymap", "hydsheds_terrain", "hydrivers_dist"]

# Native resolution as it is stated in chapter 3, so that the table does not
# introduce a second notation for the same number.
NATIVE_RESOLUTION = {
    "twsa": "0.5 deg (about 55 km)",
    "era5l": "0.1 deg",
    "era5l_static": "0.1 deg",
    "terra": "1/24 deg (about 4.6 km)",
    "soilgrids": "250 m",
    "lcc": "300 m",
    "gasy": "1 km",
    "glim": "1/120 deg (rasterised)",
    "whymap": "1/240 deg (rasterised)",
    "hydsheds_terrain": "3 arcsec (about 90 m)",
    "hydrivers_dist": "15 arcsec (about 500 m)",
}

TEMPORAL = {"static": "static", "monthly": "monthly", "yearly": "yearly"}


# ---------------------------------------------------------------------------
# input discovery
# ---------------------------------------------------------------------------
def resolve_root(explicit: str | None) -> Path:
    """Project root from --root, from GWL_ROOT, or from this file's location."""
    if explicit:
        return Path(explicit)
    env = os.environ.get("GWL_ROOT")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for cand in (here.parents[2] if len(here.parents) > 2 else here.parent,
                 here.parents[1] if len(here.parents) > 1 else here.parent,
                 Path.cwd()):
        if (cand / "03_Dataframes").exists():
            return cand
    return here.parents[2] if len(here.parents) > 2 else here.parent


def find_inputs(args, root: Path) -> dict:
    """Resolve the five report files from the root or the explicit directories."""
    f = {k: None for k in NAMES}

    sdir = Path(args.static_dir) if args.static_dir else root.joinpath(*STATIC_SUBDIR)
    ddir = Path(args.dynamic_dir) if args.dynamic_dir else root.joinpath(*DYNAMIC_SUBDIR)

    if sdir.exists():
        for k in ("stat_prov", "stat_report", "feat_nan"):
            f[k] = _first(sdir, NAMES[k])
    else:
        print(f"  static directory not found: {sdir}")
    if ddir.exists():
        for k in ("dyn_prov", "dyn_report"):
            f[k] = _first(ddir, NAMES[k])
    else:
        print(f"  dynamic directory not found: {ddir}")

    # explicit per-file overrides always win
    for key, val in (("stat_prov", args.static_provenance),
                     ("stat_report", args.static_report),
                     ("feat_nan", args.feature_nan_summary),
                     ("dyn_prov", args.dynamic_provenance),
                     ("dyn_report", args.dynamic_report)):
        if val:
            f[key] = Path(val)
    return f


def _first(directory: Path, names) -> Path | None:
    for n in names:
        p = directory / n
        if p.exists():
            return p
    return None


def load_manifest_config(root: Path | None = None) -> dict:
    """
    Radius, coverage and on_no_valid per dataset key, read from the manifest.

    The manifest lives in 02_Dataframes_Processing_Scripts/0_Shared, which is
    not the folder this script sits in, so the project root and the sibling
    folder 0_Shared are searched. Without the manifest the report still runs, but the radius,
    resolution and coverage columns stay empty and the dynamic completeness
    scan counts every month instead of only those inside a dataset's coverage
    window.
    """
    cands = []
    if root:
        cands.append(root.joinpath(*MANIFEST_DIR_REL))
    cands += [Path(__file__).resolve().parent.parent / "0_Shared",
              Path(__file__).resolve().parent,
              Path.cwd()]
    for cand in cands:
        if (cand / "dataset_manifest.py").exists():
            sys.path.insert(0, str(cand))
            print(f"  manifest      : {cand / 'dataset_manifest.py'}")
            break
    else:
        print("  manifest      : NOT FOUND, radius and coverage columns stay empty")
    try:
        import dataset_manifest as dm
    except Exception as e:  # noqa: BLE001
        print(f"  manifest could not be imported: {e}")
        return {}
    cfg = {s.key: {"resolution_deg": s.resolution_deg,
                   "search_radius_km": dm.resolved_radius_km(s),
                   "on_no_valid": s.on_no_valid,
                   "temporal": s.temporal,
                   "coverage": s.coverage}
           for s in dm.MANIFEST}
    # The basin column groups carry the coverage of the climate dataset they
    # were aggregated from, so the completeness scan uses the same window.
    basin_cov = cfg.get("basin_hydro", {}).get("coverage")
    for prod, src in (("basin_era5l", "era5l"), ("basin_terra", "terra")):
        for lvl in ("lev06", "lev07"):
            cfg[f"{prod}_{lvl}"] = {
                "resolution_deg": None, "search_radius_km": None,
                "on_no_valid": None, "temporal": "monthly",
                "coverage": cfg.get(src, {}).get("coverage") or basin_cov}
        cfg[f"basin_{lvl}"] = {"resolution_deg": None, "search_radius_km": None,
                               "on_no_valid": None, "temporal": "static",
                               "coverage": None}
    return cfg


# ---------------------------------------------------------------------------
# provenance -> per-dataset table
# ---------------------------------------------------------------------------
def datasets_in(prov: pd.DataFrame) -> list:
    return [c[: -len("__outcome")] for c in prov.columns if c.endswith("__outcome")]


def summarise_provenance(prov: pd.DataFrame, store: str, cfg: dict) -> pd.DataFrame:
    rows = []
    n_wells = len(prov)
    for key in datasets_in(prov):
        oc = prov[f"{key}__outcome"].astype(str)
        dist = pd.to_numeric(prov.get(f"{key}__sample_dist_km"), errors="coerce")
        resc = oc.eq("rescued")

        row = {"dataset": key, "store": store, "n_wells": n_wells}
        for o in OUTCOMES:
            row[f"n_{o}"] = int(oc.eq(o).sum())
        row["n_other"] = int((~oc.isin(OUTCOMES)).sum())
        row["pct_rescued"] = round(100.0 * row["n_rescued"] / n_wells, 2)
        row["pct_without_valid_cell"] = round(
            100.0 * (row["n_no_valid"] + row["n_dropped"]) / n_wells, 2)

        d = dist.dropna()
        row["dist_km_median"] = round(float(d.median()), 3) if len(d) else np.nan
        row["dist_km_p95"] = round(float(d.quantile(0.95)), 3) if len(d) else np.nan
        row["dist_km_max"] = round(float(d.max()), 3) if len(d) else np.nan
        dr = dist[resc].dropna()
        row["rescue_dist_km_median"] = round(float(dr.median()), 3) if len(dr) else np.nan
        row["rescue_dist_km_max"] = round(float(dr.max()), 3) if len(dr) else np.nan

        c = cfg.get(key, {})
        row["resolution_deg"] = c.get("resolution_deg")
        row["search_radius_km"] = c.get("search_radius_km")
        row["on_no_valid"] = c.get("on_no_valid")
        rows.append(row)
    return pd.DataFrame(rows)


def long_flagged_wells(prov: pd.DataFrame, store: str) -> pd.DataFrame:
    """Every (well, dataset) whose value did not come from the strict home cell."""
    out = []
    id_col = prov.columns[0]
    for key in datasets_in(prov):
        oc = prov[f"{key}__outcome"].astype(str)
        sel = ~oc.eq("home")
        if not sel.any():
            continue
        sub = pd.DataFrame({
            "well_id": prov.loc[sel, id_col].values,
            "store": store,
            "dataset": key,
            "outcome": oc[sel].values,
            "sample_dist_km": pd.to_numeric(
                prov.loc[sel, f"{key}__sample_dist_km"], errors="coerce").values,
        })
        out.append(sub)
    if not out:
        return pd.DataFrame(columns=["well_id", "store", "dataset",
                                     "outcome", "sample_dist_km"])
    return (pd.concat(out, ignore_index=True)
              .sort_values(["dataset", "outcome", "well_id"])
              .reset_index(drop=True))


# ---------------------------------------------------------------------------
# completeness per dataset, computed from the stores themselves
# ---------------------------------------------------------------------------
# Column prefixes in the stores that do not equal the manifest dataset key.
PREFIX_TO_KEY = {"C3S": "lcc"}

# Column suffixes that are identifiers rather than features.
NON_FEATURE_SUFFIX = {"HYBAS_ID"}


def feature_columns(df: pd.DataFrame) -> list:
    """Feature columns of a store: prefixed with '<dataset>__', no identifiers."""
    return [c for c in df.columns
            if "__" in c and c.split("__")[-1] not in NON_FEATURE_SUFFIX]


def group_of(col: str) -> str:
    return PREFIX_TO_KEY.get(col.split("__")[0], col.split("__")[0])


def static_completeness(store: pd.DataFrame) -> pd.DataFrame:
    """Per dataset: how much of the static store is missing at the wells."""
    cols = feature_columns(store)
    groups: dict = {}
    for c in cols:
        groups.setdefault(group_of(c), []).append(c)

    n_wells = len(store)
    rows = []
    for key, gc in sorted(groups.items()):
        miss = store[gc].isna()
        n_missing = int(miss.values.sum())
        n_slots = n_wells * len(gc)
        per_col = miss.sum(axis=0)
        worst = per_col.idxmax()
        rows.append({
            "dataset": key,
            "store": "static",
            "n_variables": len(gc),
            "n_wells": n_wells,
            "wells_affected": int(miss.any(axis=1).sum()),
            "wells_all_missing": int(miss.all(axis=1).sum()),
            "pct_wells_affected": round(100.0 * miss.any(axis=1).sum() / n_wells, 2),
            "n_values_missing": n_missing,
            "n_values_total": n_slots,
            "pct_values_missing": round(100.0 * n_missing / n_slots, 2) if n_slots else np.nan,
            "worst_variable": worst.split("__", 1)[-1],
            "worst_variable_n_missing": int(per_col.max()),
        })
    return pd.DataFrame(rows)


def coverage_mask(times: pd.Series, cov) -> np.ndarray:
    """Boolean mask of the months that lie inside a dataset's coverage window."""
    if not cov:
        return np.ones(len(times), dtype=bool)
    start = pd.Timestamp(str(cov[0])) if len(str(cov[0])) > 4 else pd.Timestamp(f"{cov[0]}-01-01")
    end_raw = str(cov[1])
    end = (pd.Timestamp(f"{end_raw}-01") + pd.offsets.MonthEnd(0)
           if len(end_raw) > 4 else pd.Timestamp(f"{end_raw}-12-31"))
    t = pd.to_datetime(times)
    return ((t >= start) & (t <= end)).to_numpy()


def dynamic_completeness(wells_dir: Path, cfg: dict, mode: str,
                         sample: int) -> pd.DataFrame:
    """
    Per dataset: how much of the dynamic store is missing INSIDE the dataset's
    own coverage window. Months outside the window are excluded, because a value
    is not expected there (for example the terrestrial water storage anomaly
    after 2020) and counting them would report a design decision as a data gap.
    """
    files = sorted(wells_dir.glob("MW_*.csv"))
    if not files:
        print(f"  no well CSVs in {wells_dir}")
        return pd.DataFrame()
    n_all = len(files)
    if mode == "sample" and sample and n_all > sample:
        step = max(1, n_all // sample)
        files = files[::step][:sample]
    print(f"  scanning {len(files)} of {n_all} well CSVs in "
          f"{wells_dir.name} (mode={mode})")
    if mode == "full":
        print("    a full scan reads every file once; with a synchronising "
              "cloud folder this takes a few minutes")

    t0 = _time.perf_counter()
    acc: dict = {}
    for i, fp in enumerate(files, 1):
        if i % 250 == 0 or i == len(files):
            el = _time.perf_counter() - t0
            eta = el / i * (len(files) - i)
            print(f"    {i}/{len(files)}  elapsed {el:5.1f}s  "
                  f"remaining {eta:5.1f}s")
        df = pd.read_csv(fp)
        if "time" not in df.columns:
            continue
        cols = feature_columns(df)
        for c in cols:
            key = group_of(c)
            a = acc.setdefault(key, {"vars": set(), "missing": 0, "total": 0,
                                     "wells_affected": set(),
                                     "wells_all_missing": set(),
                                     "worst": {}})
            m = coverage_mask(df["time"], cfg.get(key, {}).get("coverage"))
            vals = df[c].to_numpy()[m]
            n_miss = int(pd.isna(vals).sum())
            a["vars"].add(c)
            a["missing"] += n_miss
            a["total"] += int(m.sum())
            a["worst"][c] = a["worst"].get(c, 0) + n_miss
            if n_miss:
                a["wells_affected"].add(fp.stem)
                if n_miss == int(m.sum()):
                    a["wells_all_missing"].add((fp.stem, c))

    n_wells = len(files)
    rows = []
    for key, a in sorted(acc.items()):
        worst_col = max(a["worst"], key=a["worst"].get)
        all_missing_wells = {w for w, _ in a["wells_all_missing"]}
        rows.append({
            "dataset": key,
            "store": "dynamic",
            "n_variables": len(a["vars"]),
            "n_wells": n_wells,
            "wells_affected": len(a["wells_affected"]),
            "wells_all_missing": len(all_missing_wells),
            "pct_wells_affected": round(100.0 * len(a["wells_affected"]) / n_wells, 2),
            "n_values_missing": a["missing"],
            "n_values_total": a["total"],
            "pct_values_missing": (round(100.0 * a["missing"] / a["total"], 2)
                                   if a["total"] else np.nan),
            "worst_variable": worst_col.split("__", 1)[-1],
            "worst_variable_n_missing": a["worst"][worst_col],
        })
    return pd.DataFrame(rows)


def well_overlap(store: pd.DataFrame, dyn_prov: pd.DataFrame,
                 priority_key: str = "twsa") -> dict:
    """
    Cross-tabulate the two reasons a well is not fully covered.

    One dataset is treated as the priority dataset, by default the terrestrial
    water storage reconstruction, because a well without a value there cannot
    enter an experiment that uses it. The remaining wells are then split by
    whether they carry any missing static value. This produces the figures for
    the closing summary of the extraction quality subsection.
    """
    if store is None or dyn_prov is None:
        return {}
    col = f"{priority_key}__outcome"
    if col not in dyn_prov.columns:
        return {}

    id_col = dyn_prov.columns[0]
    lost = set(dyn_prov.loc[dyn_prov[col].astype(str).isin(["dropped", "no_valid"]),
                            id_col].astype(str))

    sid = ("MW_ID" if "MW_ID" in store.columns else store.columns[0])
    ids = store[sid].astype(str)
    ids = ids.where(ids.str.startswith("MW_"), "MW_" + ids)
    cols = feature_columns(store)
    incomplete = set(ids[store[cols].isna().any(axis=1)].values)

    all_wells = set(ids.values)
    return {
        "n_wells": len(all_wells),
        "priority_key": priority_key,
        "n_without_priority": len(lost),
        "n_incomplete_static": len(incomplete),
        "n_both": len(lost & incomplete),
        "n_kept_and_complete": len(all_wells - lost - incomplete),
        "n_kept_but_incomplete": len(incomplete - lost),
    }


def merge_completeness(comp: pd.DataFrame, by_ds: pd.DataFrame,
                       cfg: dict) -> pd.DataFrame:
    """Join the completeness view with the pixel selection view."""
    p = by_ds.set_index("dataset")
    rows = []
    for _, r in comp.iterrows():
        key = r["dataset"]
        c = cfg.get(key, {})
        has_prov = key in p.index
        pr = p.loc[key] if has_prov else None
        rows.append({
            "dataset": DISPLAY.get(key, key),
            "dataset_key": key,
            "store": r["store"],
            "sampling": "grid cell at the well" if has_prov else "polygon join",
            "grid_spacing_deg": c.get("resolution_deg"),
            "search_radius_km": c.get("search_radius_km"),
            "n_variables": r["n_variables"],
            "n_wells": r["n_wells"],
            "pct_values_missing": r["pct_values_missing"],
            "wells_affected": r["wells_affected"],
            "pct_wells_affected": r["pct_wells_affected"],
            "wells_all_missing": r["wells_all_missing"],
            "worst_variable": r["worst_variable"],
            "worst_variable_n_missing": r["worst_variable_n_missing"],
            "wells_rescued_by_neighbour": int(pr["n_rescued"]) if has_prov else None,
            "pct_wells_rescued": pr["pct_rescued"] if has_prov else None,
            "wells_without_valid_cell": (int(pr["n_no_valid"]) + int(pr["n_dropped"])
                                         if has_prov else None),
            "median_sampling_distance_km": pr["dist_km_median"] if has_prov else None,
            "max_sampling_distance_km": pr["dist_km_max"] if has_prov else None,
            "max_rescue_distance_km": pr["rescue_dist_km_max"] if has_prov else None,
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["store", "pct_values_missing"],
                           ascending=[True, False]).reset_index(drop=True)


def thesis_table(by_ds: pd.DataFrame, comp: pd.DataFrame | None,
                 cfg: dict) -> pd.DataFrame:
    """
    The per-dataset table for the thesis appendix. Row order and dataset names
    follow chapter 3, the native resolution is quoted as chapter 3 states it,
    and the sampling distances are given in metres, because one decimal in
    kilometres would round the fine grids to zero.
    """
    p = by_ds.set_index("dataset")
    c = (comp.set_index("dataset_key") if comp is not None and len(comp)
         else None)
    rows = []
    for key in [k for k in ORDER if k in p.index]:
        r = p.loc[key]
        cr = (c.loc[key] if c is not None and key in c.index else None)
        if cr is not None and isinstance(cr, pd.DataFrame):
            cr = cr.iloc[0]
        rows.append({
            "dataset": DISPLAY.get(key, key),
            "native_resolution": NATIVE_RESOLUTION.get(key, ""),
            "search_radius_km": (None if cfg.get(key, {}).get("search_radius_km") is None
                                 else round(float(cfg[key]["search_radius_km"]), 1)),
            "n_variables": (int(cr["n_variables"]) if cr is not None else None),
            "missing_pct_of_expected": (cr["pct_values_missing"]
                                        if cr is not None else None),
            "wells_with_missing_values": (int(cr["wells_affected"])
                                          if cr is not None else None),
            "wells_from_neighbouring_cell": int(r["n_rescued"]),
            "wells_without_valid_cell": int(r["n_no_valid"]) + int(r["n_dropped"]),
            "median_distance_m": (None if not np.isfinite(r["dist_km_median"])
                                  else round(r["dist_km_median"] * 1000)),
            "maximum_distance_m": (None if not np.isfinite(r["dist_km_max"])
                                   else round(r["dist_km_max"] * 1000)),
        })
    return pd.DataFrame(rows)


def summarise_features(feat: pd.DataFrame) -> pd.DataFrame:
    df = feat.copy()
    df["dataset"] = df["feature"].str.split("__").str[0]
    df["variable"] = df["feature"].str.split("__").str[-1]
    if "pct" not in df.columns:
        df["pct"] = np.nan
    return (df[["dataset", "variable", "feature", "n_nan", "pct"]]
            .sort_values(["n_nan", "feature"], ascending=[False, True])
            .reset_index(drop=True))


# ---------------------------------------------------------------------------
# text report
# ---------------------------------------------------------------------------
def write_text(path: Path, by_ds: pd.DataFrame, by_feat: pd.DataFrame | None,
               stat_report: pd.DataFrame | None, dyn_report: pd.DataFrame | None,
               n_wells: int, comp: pd.DataFrame | None = None,
               ov: dict | None = None) -> str:
    L = []
    add = L.append
    add("EXTRACTION QUALITY REPORT")
    add("=" * 78)
    add(f"wells in the provenance tables : {n_wells}")
    add("")

    add("1) PIXEL SELECTION PER DATASET")
    add("-" * 78)
    add(f"{'dataset':<18}{'store':<9}{'home':>7}{'resc':>7}{'no_val':>8}"
        f"{'drop':>7}{'%resc':>8}{'medkm':>9}{'maxkm':>9}{'radkm':>8}")
    for _, r in by_ds.iterrows():
        add(f"{r['dataset']:<18}{r['store']:<9}{int(r['n_home']):>7}"
            f"{int(r['n_rescued']):>7}{int(r['n_no_valid']):>8}"
            f"{int(r['n_dropped']):>7}{r['pct_rescued']:>8.2f}"
            f"{_num(r['dist_km_median']):>9}{_num(r['dist_km_max']):>9}"
            f"{_num(r['search_radius_km']):>8}")
    add("")

    resc = by_ds[by_ds["n_rescued"] > 0]
    add("   datasets that required a neighbouring pixel for at least one well:")
    if len(resc):
        for _, r in resc.iterrows():
            add(f"     {r['dataset']:<18} {int(r['n_rescued'])}/{int(r['n_wells'])} wells "
                f"({r['pct_rescued']}%), rescue distance median "
                f"{_num(r['rescue_dist_km_median'])} km, max {_num(r['rescue_dist_km_max'])} km")
    else:
        add("     none")
    add("")

    nov = by_ds[(by_ds["n_no_valid"] + by_ds["n_dropped"]) > 0]
    add("   datasets with wells that had NO valid cell inside the search radius:")
    if len(nov):
        for _, r in nov.iterrows():
            add(f"     {r['dataset']:<18} no_valid={int(r['n_no_valid'])} "
                f"dropped={int(r['n_dropped'])} of {int(r['n_wells'])} "
                f"(policy: {r['on_no_valid']})")
    else:
        add("     none")
    add("")

    if by_feat is not None:
        add("2) MISSING VALUES PER STATIC FEATURE COLUMN")
        add("-" * 78)
        nz = by_feat[by_feat["n_nan"] > 0]
        if len(nz):
            add(f"{'feature':<44}{'n_nan':>8}{'pct':>8}")
            for _, r in nz.iterrows():
                add(f"{r['feature']:<44}{int(r['n_nan']):>8}{r['pct']:>8.1f}")
            grp = (nz.groupby("dataset")["n_nan"]
                     .agg(["size", "max"]).sort_values("max", ascending=False))
            add("")
            add("   affected datasets (columns with NaN, worst column):")
            for ds, r in grp.iterrows():
                add(f"     {ds:<18} {int(r['size'])} column(s), worst {int(r['max'])} wells")
        else:
            add("   no missing values in any static feature column")
        add("")

    if stat_report is not None and "n_nan" in stat_report.columns:
        n = len(stat_report)
        full = int((stat_report["n_nan"] == 0).sum())
        add("3) MISSING VALUES PER WELL (static store)")
        add("-" * 78)
        add(f"   wells with all static features present : {full} / {n} "
            f"({100.0 * full / n:.1f}%)")
        add(f"   median NaN features per well           : "
            f"{stat_report['n_nan'].median():.0f} of "
            f"{int(stat_report['n_features'].iloc[0])}")
        add(f"   maximum NaN features for a single well : {int(stat_report['n_nan'].max())}")
        vc = stat_report["n_nan"].value_counts().sort_index()
        add("   distribution (NaN features -> wells): "
            + ", ".join(f"{int(k)}->{int(v)}" for k, v in vc.items()))
        add("")

    if comp is not None and len(comp):
        add("5) COMPLETENESS PER DATASET (from the stores themselves)")
        add("-" * 78)
        add("   For the dynamic store only months inside the dataset's own")
        add("   coverage window are counted, so the end of a dataset is not")
        add("   reported as a data gap.")
        add("")
        add(f"   {'dataset':<34}{'store':<9}{'vars':>5}{'%miss':>8}"
            f"{'wells':>7}{'of':>6}{'%wells':>8}{'resc':>6}{'medkm':>8}{'maxkm':>8}")
        for _, r in comp.iterrows():
            add(f"   {str(r['dataset'])[:33]:<34}{str(r['store']):<9}"
                f"{int(r['n_variables']):>5}"
                f"{_num(r['pct_values_missing']):>8}{int(r['wells_affected']):>7}"
                f"{int(r['n_wells']):>6}{_num(r['pct_wells_affected']):>8}"
                f"{_num(r['wells_rescued_by_neighbour']):>6}"
                f"{_num(r['median_sampling_distance_km']):>8}"
                f"{_num(r['max_sampling_distance_km']):>8}")
        add("")
        add("   vars   = feature columns belonging to the dataset")
        add("   %miss  = share of all value slots that are missing, where a slot")
        add("            is one well and one variable in the static store, and")
        add("            one well, one variable and one month in the dynamic one")
        add("   wells  = wells with at least one missing value in the dataset")
        add("   of     = wells examined; a smaller number in the dynamic store")
        add("            means a sampled scan, see --dynamic-scan")
        add("   resc   = wells whose value came from a neighbouring cell")
        add("   the sampling columns are empty where a dataset was joined by")
        add("   polygon rather than sampled from a grid")
        add("")

    if ov:
        add("6) WELLS BY REASON FOR INCOMPLETE COVERAGE")
        add("-" * 78)
        add(f"   priority dataset: {ov['priority_key']}")
        add(f"   wells in total                                 : {ov['n_wells']}")
        add(f"   without a value from the priority dataset      : "
            f"{ov['n_without_priority']}")
        add(f"   with at least one missing static value         : "
            f"{ov['n_incomplete_static']}")
        add(f"   both of the above                              : {ov['n_both']}")
        add(f"   retained, but with a missing static value      : "
            f"{ov['n_kept_but_incomplete']}")
        add(f"   retained and static features complete          : "
            f"{ov['n_kept_and_complete']}")
        add("")

    if dyn_report is not None and "status" in dyn_report.columns:
        add("4) DYNAMIC STORE BUILD STATUS")
        add("-" * 78)
        for s, c in dyn_report["status"].value_counts().items():
            add(f"   {s:<14} {c}")
        if "n_months" in dyn_report.columns:
            ok = dyn_report[dyn_report["status"] == "OK"]
            add(f"   months per well: median {ok['n_months'].median():.0f}, "
                f"min {ok['n_months'].min()}, max {ok['n_months'].max()}")
        add("")

    txt = "\n".join(L)
    path.write_text(txt, encoding="utf-8")
    return txt


def _num(v) -> str:
    return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:g}"


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--root", default=None, help="project root (derives both dirs)")
    ap.add_argument("--static-dir", default=None)
    ap.add_argument("--dynamic-dir", default=None)
    ap.add_argument("--static-provenance", default=None)
    ap.add_argument("--static-report", default=None)
    ap.add_argument("--feature-nan-summary", default=None)
    ap.add_argument("--dynamic-provenance", default=None)
    ap.add_argument("--dynamic-report", default=None)
    ap.add_argument("--out", default=None,
                    help="output directory (default: "
                         "<root>/03_Dataframes/00_Extraction_Quality)")
    ap.add_argument("--static-store", default=None,
                    help="static_features.csv (default: <static_dir>/static_features.csv)")
    ap.add_argument("--dynamic-wells-dir", default=None,
                    help="per-well CSVs (default: <dynamic_dir>/"
                         f"{DYNAMIC_WELLS_SUBDIR})")
    ap.add_argument("--dynamic-scan", choices=["none", "sample", "full"],
                    default="full",
                    help="how much of the dynamic store to scan for the "
                         "completeness table (default: full, every well)")
    ap.add_argument("--dynamic-sample", type=int, default=300,
                    help="number of wells to scan when --dynamic-scan=sample")
    args, _unknown = ap.parse_known_args()   # tolerate Spyder's --wdir

    root = resolve_root(args.root)
    out = Path(args.out) if args.out else root.joinpath(*OUT_SUBDIR)
    out.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("  EXTRACTION QUALITY REPORT")
    print("=" * 78)
    print(f"  project root : {root}")
    print(f"  output       : {out}")
    cfg = load_manifest_config(root)
    f = find_inputs(args, root)
    print("  inputs:")
    for k, v in f.items():
        print(f"    {k:<12} {v.name if v else '(not found)'}")

    frames, flagged, n_wells = [], [], 0
    for key, store in (("stat_prov", "static"), ("dyn_prov", "dynamic")):
        if f[key] is None:
            continue
        prov = pd.read_csv(f[key])
        n_wells = max(n_wells, len(prov))
        frames.append(summarise_provenance(prov, store, cfg))
        flagged.append(long_flagged_wells(prov, store))

    if not frames:
        sys.exit("no provenance file found - nothing to summarise")

    by_ds = pd.concat(frames, ignore_index=True)
    by_ds.to_csv(out / "qc_by_dataset.csv", index=False)

    fl = pd.concat(flagged, ignore_index=True) if flagged else pd.DataFrame()
    fl.to_csv(out / "qc_rescued_wells.csv", index=False)

    by_feat = None
    if f["feat_nan"] is not None:
        by_feat = summarise_features(pd.read_csv(f["feat_nan"]))
        by_feat.to_csv(out / "qc_by_feature.csv", index=False)

    stat_report = pd.read_csv(f["stat_report"]) if f["stat_report"] else None
    dyn_report = pd.read_csv(f["dyn_report"]) if f["dyn_report"] else None

    # ---- completeness per dataset, computed from the stores -----------------
    sdir = (Path(args.static_dir) if args.static_dir
            else root.joinpath(*STATIC_SUBDIR))
    ddir = (Path(args.dynamic_dir) if args.dynamic_dir
            else root.joinpath(*DYNAMIC_SUBDIR))
    store_path = (Path(args.static_store) if args.static_store
                  else sdir / "static_features.csv")
    wells_dir = (Path(args.dynamic_wells_dir) if args.dynamic_wells_dir
                 else ddir / DYNAMIC_WELLS_SUBDIR)

    parts = []
    if store_path.exists():
        print(f"\n  reading static store: {store_path.name}")
        parts.append(static_completeness(pd.read_csv(store_path, low_memory=False)))
    else:
        print(f"\n  static store not found, completeness for the static "
              f"datasets is skipped: {store_path}")
    if args.dynamic_scan != "none" and wells_dir.exists():
        parts.append(dynamic_completeness(wells_dir, cfg, args.dynamic_scan,
                                          args.dynamic_sample))
    elif args.dynamic_scan != "none":
        print(f"  dynamic well CSVs not found: {wells_dir}")

    comp = None
    parts = [p for p in parts if p is not None and len(p)]
    if parts:
        comp = merge_completeness(pd.concat(parts, ignore_index=True), by_ds, cfg)
        comp.to_csv(out / "qc_completeness_by_dataset.csv", index=False)
    thesis_table(by_ds, comp, cfg).to_csv(out / "qc_thesis_table.csv", index=False)

    ov = {}
    if store_path.exists() and f["dyn_prov"] is not None:
        ov = well_overlap(pd.read_csv(store_path, low_memory=False),
                          pd.read_csv(f["dyn_prov"]))

    txt = write_text(out / "qc_summary.txt", by_ds, by_feat,
                     stat_report, dyn_report, n_wells, comp, ov)
    print("\n" + txt)
    print(f"  written to {out.resolve()}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
02a_check_era5land_static_classes.py
Verification of the categorical ERA5-Land static fields.

Diagnostic only, writes nothing. Run after 01_era5land_static_merge.py. Prints the value histogram of the categorical
variables (slt, tvl, tvh) BEFORE the merge (from 1_Original) and AFTER the merge
(from 3_Final), so the effect of the class filter in 01_era5land_static_merge.py
can be seen directly.

Run
---
    python 02a_check_era5land_static_classes.py
    python 02a_check_era5land_static_classes.py --root "<path to project root>"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

# =============================================================================
# PATHS  (relative to the project root; adjust here if the layout changes)
# =============================================================================
# The project root can also be passed on the command line with --root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_SUBDIR = Path("01_Input_Global") / "1_Original" / "ERA5_Land_static"
MERGED_FILE = (Path("01_Input_Global") / "3_Final" / "ERA5_Land_static"
               / "ERA5_Land_static_Germany.nc")

# =============================================================================
# CONFIGURATION
# =============================================================================
CATS = ["slt", "tvl", "tvh"]

# Class tables exactly as given in documentation_era5_static.txt. The code 0 is
# NOT part of that documentation and is therefore listed separately below.
IFS_VEG = {
    1: "crops_mixed_farming", 2: "grass",
    3: "evergreen_needleleaf_trees", 4: "deciduous_needleleaf_trees",
    5: "deciduous_broadleaf_trees", 6: "evergreen_broadleaf_trees",
    7: "tall_grass", 8: "desert", 9: "tundra", 10: "irrigated_crops",
    11: "semidesert", 12: "ice_caps_and_glaciers", 13: "bogs_and_marshes",
    14: "inland_water", 15: "ocean", 16: "evergreen_shrubs",
    17: "deciduous_shrubs", 18: "mixed_forest_woodland",
    19: "interrupted_forest", 20: "water_and_land_mixtures",
    0: "UNDOCUMENTED code 0",
}
IFS_SOIL = {
    1: "coarse", 2: "medium", 3: "medium_fine", 4: "fine", 5: "very_fine",
    6: "organic", 7: "tropical_organic", 0: "UNDOCUMENTED code 0",
}

# Documented membership. The four codes 8, 12, 14, 15 indicate no land surface
# vegetation and belong to neither group, so they are listed for both fields as
# codes that may legitimately occur.
NO_SURFACE_VEG = [8, 12, 14, 15]
TVH_CLASSES = [3, 4, 5, 6, 18, 19] + NO_SURFACE_VEG
TVL_CLASSES = [1, 2, 7, 9, 10, 11, 13, 16, 17, 20] + NO_SURFACE_VEG
SLT_CLASSES = [1, 2, 3, 4, 5, 6, 7]
EXPECTED = {"tvh": TVH_CLASSES, "tvl": TVL_CLASSES, "slt": SLT_CLASSES}

# Class codes kept by 01_era5land_static_merge.py (membership test; code 0 is
# kept for tvl and tvh, and set to missing for slt; the no-surface-vegetation
# codes are not kept). Keep this in sync with the merge script.
KEPT_BY_MERGE = {
    "slt": SLT_CLASSES,
    "tvl": [0, 1, 2, 7, 9, 10, 11, 13, 16, 17, 20],
    "tvh": [0, 3, 4, 5, 6, 18, 19],
}


def histogram(da: xr.DataArray, name: str) -> None:
    a = np.asarray(da.values, dtype="float64").ravel()
    n_total = a.size
    n_nan = int(np.isnan(a).sum())
    a = a[~np.isnan(a)]
    if a.size == 0:
        print(f"    all {n_total} cells are NaN")
        return
    vals, counts = np.unique(np.round(a).astype("int64"), return_counts=True)
    lut = IFS_SOIL if name == "slt" else IFS_VEG
    print(f"    cells {n_total}, NaN {n_nan} ({100.0 * n_nan / n_total:.1f} %)")
    print(f"    {'class':>6}  {'cells':>10}  {'%':>6}  {'kept by merge script':<24}  meaning")
    for v, c in zip(vals, counts):
        kept = "yes" if v in KEPT_BY_MERGE[name] else "NO  <-- masked to NaN"
        if v in EXPECTED[name]:
            valid = ""
        elif v == 0:
            valid = "  [code 0: not in the documentation, decide explicitly]"
        else:
            valid = "  [not in the documented class table]"
        print(f"    {v:>6}  {c:>10}  {100.0 * c / n_total:>5.1f}  {kept:<24}  "
              f"{lut.get(int(v), '?')}{valid}")


def scan(path: Path, label: str) -> None:
    print(f"\n  {label}: {path}")
    try:
        ds = xr.open_dataset(path, mask_and_scale=True)
    except Exception as e:  # noqa: BLE001
        print(f"    could not open: {e}")
        return
    found = [v for v in CATS if v in ds.data_vars]
    if not found:
        print(f"    none of {CATS} present (variables: {list(ds.data_vars)})")
        ds.close()
        return
    for v in found:
        da = ds[v]
        for d in da.dims:
            if d not in ("lat", "lon", "latitude", "longitude"):
                da = da.isel({d: 0})
        print(f"\n    variable '{v}'  attrs: "
              f"_FillValue={ds[v].encoding.get('_FillValue')} "
              f"dtype={ds[v].dtype}")
        histogram(da, v)
    ds.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="project root (optional)")
    args = ap.parse_args()

    root = Path(args.root) if args.root else PROJECT_ROOT
    original = root / ORIGINAL_SUBDIR
    final = root / MERGED_FILE

    print("=" * 78)
    print("  ERA5-Land static categorical check")
    print("=" * 78)
    print(f"  project root : {root}")

    print("\n" + "-" * 78)
    print("  BEFORE THE MERGE (1_Original)")
    print("-" * 78)
    if original.exists():
        for f in sorted(original.glob("*.nc")):
            scan(f, "source")
    else:
        print(f"  input directory not found: {original}")

    print("\n" + "-" * 78)
    print("  AFTER THE MERGE (3_Final)")
    print("-" * 78)
    if final.exists():
        scan(final, "merged")
    else:
        print(f"  merged file not found: {final}")

    print("\n  Documented classes (documentation_era5_static.txt):")
    print(f"    tvh  high vegetation {[3, 4, 5, 6, 18, 19]} "
          f"plus no-surface-vegetation codes {NO_SURFACE_VEG}")
    print(f"    tvl  low vegetation  {[1, 2, 7, 9, 10, 11, 13, 16, 17, 20]} "
          f"plus no-surface-vegetation codes {NO_SURFACE_VEG}")
    print(f"    slt  soil texture    {SLT_CLASSES}")
    print("    code 0 is not documented for any of the three fields; if it "
          "occurs above, decide whether it is a class or a gap.")
    print("\n  Done.")


if __name__ == "__main__":
    main()

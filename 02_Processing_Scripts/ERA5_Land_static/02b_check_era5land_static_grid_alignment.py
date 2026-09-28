# -*- coding: utf-8 -*-
"""
02b_check_era5land_static_grid_alignment.py
Check of the grid alignment of the ERA5-Land static fields.

Diagnostic only, writes nothing. Run after 01_era5land_static_merge.py.

Motivation
----------
Inside the Germany box the code 0 occurs in 1312 cells of slt, 1136 cells of
tvh and 1504 cells of tvl. If 0 marked the same water cells in all three
fields, tvh and tvl would have to carry at least as many zeros as slt, because
on land they add every cell without the respective cover type. tvh carries
fewer. This script separates the two possible causes.

  Case A, harmless: the source files share one grid, and the fields simply use
  slightly different land masks. The coordinate vectors of all source files are
  then identical to floating point tolerance.

  Case B, serious: the source files do NOT share one grid, and
  xr.merge(..., join="override") in 01_era5land_static_merge.py has silently
  taken the coordinates of the first file and stacked the others onto them. In
  that case whole fields are shifted against each other, and every ERA5-Land
  static feature is affected, not only the categorical ones.

What is reported
----------------
  1. Per source file, the lat/lon vector length, first value, last value and
     spacing, plus the maximum absolute deviation from the reference file.
  2. The agreement between the land-sea mask and the code 0 of slt, tvl and tvh
     on the merged grid, as a contingency count.
  3. For each pair of fields, the number of cells where one is water by the
     land-sea mask and the other is not zero.

Run
---
    python 02b_check_era5land_static_grid_alignment.py
    python 02b_check_era5land_static_grid_alignment.py --root "<path to project root>"
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
LSM_WATER_THRESHOLD = 0.5   # cells with a land fraction below this count as water


def coord_report(original: Path) -> None:
    print("\n" + "-" * 78)
    print("  1) COORDINATE VECTORS OF THE SOURCE FILES")
    print("-" * 78)
    files = sorted(original.glob("*.nc"))
    if not files:
        print(f"  no files in {original}")
        return

    ref_lat = ref_lon = ref_name = None
    print(f"  {'file':<14}{'n_lat':>7}{'n_lon':>7}{'lat0':>10}{'lat1':>10}"
          f"{'dlat':>9}{'max|dlat|':>12}{'max|dlon|':>12}")
    for f in files:
        ds = xr.open_dataset(f)
        lat_name = "lat" if "lat" in ds.coords else "latitude"
        lon_name = "lon" if "lon" in ds.coords else "longitude"
        if lat_name not in ds.coords or lon_name not in ds.coords:
            print(f"  {f.name:<14} no lat/lon coordinates")
            ds.close()
            continue
        lat = np.asarray(ds[lat_name].values, dtype="float64")
        lon = np.asarray(ds[lon_name].values, dtype="float64")
        ds.close()

        dlat = float(np.median(np.diff(lat))) if lat.size > 1 else np.nan
        if ref_lat is None:
            ref_lat, ref_lon, ref_name = lat, lon, f.name
            dmax_lat = dmax_lon = 0.0
        else:
            dmax_lat = (float(np.abs(lat - ref_lat).max())
                        if lat.shape == ref_lat.shape else np.nan)
            dmax_lon = (float(np.abs(lon - ref_lon).max())
                        if lon.shape == ref_lon.shape else np.nan)
        print(f"  {f.name:<14}{lat.size:>7}{lon.size:>7}{lat[0]:>10.4f}"
              f"{lat[-1]:>10.4f}{dlat:>9.4f}"
              f"{_fmt(dmax_lat):>12}{_fmt(dmax_lon):>12}")

    print(f"\n  reference file: {ref_name}")
    print("  A shape mismatch or a deviation above about 1e-6 degrees means the"
          " grids differ.")
    print("  In that case join=\"override\" in the merge script is unsafe and"
          " must be replaced")
    print("  by an explicit reindex or interpolation onto one target grid.")


def _fmt(v) -> str:
    if v is None or (isinstance(float(v), float) and not np.isfinite(v)):
        return "shape diff"
    return f"{v:.3e}"


def mask_report(merged: Path) -> None:
    print("\n" + "-" * 78)
    print("  2) LAND-SEA MASK AGAINST THE CODE 0 ON THE MERGED GRID")
    print("-" * 78)
    if not merged.exists():
        print(f"  merged file not found: {merged}")
        return
    ds = xr.open_dataset(merged, mask_and_scale=True)
    need = ["lsm", "slt", "tvl", "tvh"]
    missing = [v for v in need if v not in ds.data_vars]
    if missing:
        print(f"  missing variables: {missing} (present: {list(ds.data_vars)})")
        ds.close()
        return

    lsm = np.asarray(ds["lsm"].values, dtype="float64").ravel()
    water = lsm < LSM_WATER_THRESHOLD
    land = ~water
    n = lsm.size
    print(f"  cells {n}, water {int(water.sum())} "
          f"({100.0 * water.sum() / n:.1f} %), land {int(land.sum())}")

    print(f"\n  {'field':<6}{'zeros':>8}{'NaN':>8}{'zero&water':>12}"
          f"{'zero&land':>11}{'water&notzero':>15}{'water&NaN':>11}")
    for v in ("slt", "tvl", "tvh"):
        a = np.asarray(ds[v].values, dtype="float64").ravel()
        nan = np.isnan(a)
        zero = (~nan) & (np.round(a) == 0)
        print(f"  {v:<6}{int(zero.sum()):>8}{int(nan.sum()):>8}"
              f"{int((zero & water).sum()):>12}{int((zero & land).sum()):>11}"
              f"{int((water & ~zero & ~nan).sum()):>15}"
              f"{int((water & nan).sum()):>11}")

    print("\n  Reading of the table:")
    print("  zero&land > 0 for tvl and tvh confirms that 0 is a land class and"
          " means")
    print("  that the respective cover type is absent, not that the value is"
          " missing.")
    print("  water&notzero > 0 for a field means that the field disagrees with"
          " the")
    print("  land-sea mask, which points to different land masks or to a grid"
          " shift.")

    print("\n" + "-" * 78)
    print("  3) PAIRWISE DISAGREEMENT ON WATER CELLS")
    print("-" * 78)
    codes = {}
    for v in ("slt", "tvl", "tvh"):
        a = np.asarray(ds[v].values, dtype="float64").ravel()
        codes[v] = np.where(np.isnan(a), -1, np.round(a))
    for a, b in (("slt", "tvl"), ("slt", "tvh"), ("tvl", "tvh")):
        za = codes[a] == 0
        zb = codes[b] == 0
        print(f"  {a} zero but {b} not zero : {int((za & ~zb).sum()):>6}")
        print(f"  {b} zero but {a} not zero : {int((zb & ~za).sum()):>6}")
    ds.close()

    print("\n  A large asymmetry that does not follow the land-sea mask is the"
          " signature")
    print("  of a grid shift. A small, spatially scattered difference is a"
          " different")
    print("  land mask per field and is harmless.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="project root (optional)")
    args = ap.parse_args()
    root = Path(args.root) if args.root else PROJECT_ROOT

    original = root / ORIGINAL_SUBDIR
    merged = root / MERGED_FILE

    print("=" * 78)
    print("  ERA5-Land static grid alignment check")
    print("=" * 78)
    print(f"  project root : {root}")
    coord_report(original)
    mask_report(merged)
    print("\n  Done.")


if __name__ == "__main__":
    main()

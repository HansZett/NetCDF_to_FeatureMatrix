# -*- coding: utf-8 -*-
"""
00_inspect_gasy_nc.py

Inspection of all original GASY NetCDF files (one file per SoilGrids depth).
For every file the global attributes, dimensions, coordinate ranges and
spacing, data variables with basic statistics, and CRS information are
printed. Diagnostic only, writes nothing. Run before 01_gasy_merge_depths.py.

Input : 01_Input_Global/1_Original/GASY/*.nc

Location: 02_Processing_Scripts/GASY/
Run     : python 00_inspect_gasy_nc.py
          python 00_inspect_gasy_nc.py "<other folder with .nc files>"
"""

import sys
import os
import glob
from pathlib import Path

import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# PATHS  (relative to the project root; adjust here if the layout changes)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR  = PROJECT_ROOT / "01_Input_Global" / "1_Original" / "GASY"

nc_dir = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_DIR)

if not os.path.isdir(nc_dir):
    sys.exit(f"Folder not found: {nc_dir}")

nc_files = sorted(glob.glob(os.path.join(nc_dir, "*.nc")))

if not nc_files:
    sys.exit(f"No .nc files in: {nc_dir}")

print(f"{'='*80}")
print(f"  NetCDF metadata inspection  —  {len(nc_files)} file(s) found")
print(f"  Folder: {nc_dir}")
print(f"{'='*80}\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def print_section(title):
    print(f"\n  --- {title} ---")


def describe_coord(ds, name):
    """Print min, max, mean spacing and length of a coordinate."""
    if name not in ds.coords:
        return
    vals = ds.coords[name].values
    n = len(vals)
    if n == 0:
        print(f"    {name}: empty")
        return
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    step = np.nanmean(np.diff(vals)) if n > 1 else None
    step_str = f",  Δ ≈ {step:.6f}" if step is not None else ""
    print(f"    {name}: min={vmin},  max={vmax},  n={n}{step_str}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for i, fp in enumerate(nc_files, 1):
    fname = os.path.basename(fp)
    fsize_mb = os.path.getsize(fp) / (1024 * 1024)

    print(f"\n{'─'*80}")
    print(f"  [{i}/{len(nc_files)}]  {fname}  ({fsize_mb:.1f} MB)")
    print(f"{'─'*80}")

    ds = xr.open_dataset(fp)

    # -- Global attributes --
    print_section("Global attributes")
    if ds.attrs:
        for k, v in ds.attrs.items():
            print(f"    {k}: {v}")
    else:
        print("    (none)")

    # -- Dimensions --
    print_section("Dimensions")
    for dim, size in ds.dims.items():
        print(f"    {dim}: {size}")

    # -- Coordinates --
    print_section("Coordinates (range and spacing)")
    for coord_name in ["lat", "latitude", "y", "lon", "longitude", "x",
                        "time", "band", "layer"]:
        if coord_name in ds.coords:
            describe_coord(ds, coord_name)
    # Also print all remaining coordinates not covered above
    known = {"lat", "latitude", "y", "lon", "longitude", "x",
             "time", "band", "layer"}
    for coord_name in ds.coords:
        if coord_name not in known:
            describe_coord(ds, coord_name)

    # -- Data variables --
    print_section("Data variables")
    for vname, var in ds.data_vars.items():
        print(f"    {vname}")
        print(f"      dtype: {var.dtype},  shape: {var.shape},  dims: {var.dims}")
        # Variable attributes (units, long_name, _FillValue, ...)
        for ak, av in var.attrs.items():
            print(f"      {ak}: {av}")
        # Basic statistics
        try:
            sample = var.values
            valid = sample[np.isfinite(sample)]
            if valid.size > 0:
                print(f"      Value range: min={np.nanmin(valid):.6g}, "
                      f"max={np.nanmax(valid):.6g}, "
                      f"mean={np.nanmean(valid):.6g}")
                print(f"      NaN/Inf share: "
                      f"{1 - valid.size / sample.size:.1%} of {sample.size} cells")
            else:
                print("      (only NaN/Inf values)")
        except Exception as e:
            print(f"      (statistics not readable: {e})")

    # -- CRS / spatial_ref --
    print_section("CRS / projection")
    crs_found = False
    for crs_name in ["crs", "spatial_ref", "projection", "grid_mapping"]:
        if crs_name in ds:
            crs_found = True
            crs_var = ds[crs_name]
            for ak, av in crs_var.attrs.items():
                print(f"    {ak}: {av}")
    # Alternatively: grid_mapping attribute on a data variable
    if not crs_found:
        for vname, var in ds.data_vars.items():
            gm = var.attrs.get("grid_mapping")
            if gm and gm in ds:
                crs_found = True
                print(f"    (via grid_mapping '{gm}' on variable '{vname}')")
                for ak, av in ds[gm].attrs.items():
                    print(f"    {ak}: {av}")
                break
    if not crs_found:
        print("    (no explicit CRS found)")

    ds.close()

print(f"\n{'='*80}")
print("  Inspection complete.")
print(f"{'='*80}\n")

# -*- coding: utf-8 -*-
"""
00_inspect_terraclimate.py

Prints the structure (dimensions, coordinates, variables, attributes) of one
downloaded TerraClimate file. Diagnostic only, writes nothing. Change
INPUT_FILE to inspect another variable.

Input : 01_Input_Global/1_Original/TerraClimate/<one downloaded file>.nc

Location: 02_Processing_Scripts/TerraClimate/
Run     : python 00_inspect_terraclimate.py
"""

from pathlib import Path

import xarray as xr

# =============================================================================
# PATHS  (relative to the project root; adjust here if the layout changes)
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORIGINAL     = PROJECT_ROOT / "01_Input_Global" / "1_Original"
INPUT_FILE   = (ORIGINAL / "TerraClimate"
                / "TerraClimate_drought_index_pdsi_1991-2022_AOI.nc")

nc_path = INPUT_FILE
ds = xr.open_dataset(nc_path, decode_times=False)

print("=" * 70)
print(f"INSPECTING: {nc_path.name}")
print("=" * 70)
print("FULL REPR")
print("=" * 70)
print(ds)

print("\n" + "=" * 70)
print("DIMS:", dict(ds.sizes))
print("=" * 70)

print("\nCOORDS:")
for c in ds.coords:
    arr = ds[c].values
    print(f"  {c}: shape={arr.shape}, dtype={arr.dtype}")
    print(f"     attrs={dict(ds[c].attrs)}")
    if arr.ndim == 1 and arr.size > 0:
        print(f"     first={arr[0]}, last={arr[-1]}")
        if arr.size > 1:
            print(f"     step={arr[1]-arr[0]}")

print("\nDATA VARIABLES:")
for v in ds.data_vars:
    print(f"  {v}: shape={ds[v].shape}, dtype={ds[v].dtype}")
    print(f"     attrs={dict(ds[v].attrs)}")

print("\nGLOBAL ATTRS:")
for k, v in ds.attrs.items():
    print(f"  {k}: {v}")

ds.close()

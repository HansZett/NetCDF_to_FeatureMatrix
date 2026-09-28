# -*- coding: utf-8 -*-
"""
03_hydrosheds_merge.py

Merges all HydroSHEDS intermediate NetCDF files into a single final file.

Input (01_Input_Global/2_Intermediate/HydroSHEDS_RIVERS_BASINS/):
  eu_hydrosheds_raw.nc      → dem, acc, dir     (01_hydrosheds_raw_to_nc.py)
  eu_hydrosheds_derived.nc  → slope, aspect, twi (02a_hydrosheds_terrain_derivatives.py)
  eu_hydrosheds_hand.nc     → hand              (02b_hydrosheds_hand.py)

Output:
  01_Input_Global/3_Final/HydroSHEDS_RIVERS_BASINS/hydrosheds.nc

Location: 02_Processing_Scripts/HydroSHEDS/
Run     : python 03_hydrosheds_merge.py
"""

import sys
import gc
from pathlib import Path
import numpy as np
import xarray as xr

# =============================================================================
# SHARED HELPER
# =============================================================================
sys.path.append(str(Path(__file__).resolve().parents[1] / "0_Shared"))
import nc_io

# =============================================================================
# PATHS  (relative to the project root; adjust here if the layout changes)
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_GLOBAL = PROJECT_ROOT / "01_Input_Global"
ORIGINAL     = INPUT_GLOBAL / "1_Original" / "HydroSHEDS_RIVERS_BASINS"
INTERMEDIATE = INPUT_GLOBAL / "2_Intermediate" / "HydroSHEDS_RIVERS_BASINS"
FINAL        = INPUT_GLOBAL / "3_Final" / "HydroSHEDS_RIVERS_BASINS"

# Ensure final directory exists
FINAL.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = FINAL / "hydrosheds.nc"

INPUT_FILES = [
    "eu_hydrosheds_raw.nc",
    "eu_hydrosheds_derived.nc",
    "eu_hydrosheds_hand.nc"
]

# =============================================================================
# MAIN MERGE ROUTINE
# =============================================================================
def main():
    print("=" * 65)
    print("  HydroSHEDS: Final Merge")
    print("=" * 65)

    datasets = []

    for fname in INPUT_FILES:
        in_path = INTERMEDIATE / fname
        if not in_path.exists():
            sys.exit(f"[ERROR] Missing required file: {in_path}")

        print(f"Reading: {fname}")
        # 'dir' is decoded to float here; nc_io.save_nc casts it back to uint8.
        ds = xr.open_dataset(in_path)

        # Drop stray CRS variables, if present
        drop_vars = [v for v in ds.data_vars if v in ("crs", "spatial_ref")]
        if drop_vars:
            ds = ds.drop_vars(drop_vars)

        print(f"  → Variables: {list(ds.data_vars.keys())}")
        datasets.append(ds)

    print("\nMerging datasets...")
    ds_merged = xr.merge(datasets)

    # Free up the individual dataset memory
    for ds in datasets:
        ds.close()
    del datasets
    gc.collect()

    print(f"Merged Dataset Size: {ds_merged.nbytes / 1e9:.2f} GB in RAM")

    # =========================================================================
    # ENCODING & METADATA
    # =========================================================================
    global_attrs = nc_io.make_global_attrs(
        title="HydroSHEDS merged static layers (native resolution)",
        source="HydroSHEDS v1.1 (Lehner et al. 2008, doi:10.1029/2008EO100001)",
        references="Lehner B., Verdin K., Jarvis A. (2008): New global hydrography derived from spaceborne elevation data. Eos, Transactions 89(10): 93-94.",
        script=Path(__file__).name
    )

    # Explicitly map types and fill values
    dtypes = {v: "float32" for v in ds_merged.data_vars}
    dtypes["dir"] = "uint8"

    fill_values = {v: np.nan for v in ds_merged.data_vars}
    fill_values["dir"] = 255

    print("\nSaving final dataset...")
    nc_io.save_nc(
        ds=ds_merged,
        path=OUTPUT_FILE,
        fill_value=fill_values,
        dtype=dtypes,
        global_attrs=global_attrs,
        compress=True,
        require_tier=True,
        verify=True
    )

    print("\n✅ Done.")

if __name__ == "__main__":
    main()

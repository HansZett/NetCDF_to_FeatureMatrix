# -*- coding: utf-8 -*-
"""
02b_hydrosheds_hand.py

Calculates Height Above Nearest Drainage (HAND) from eu_hydrosheds_raw.nc
using reverse-flow propagation.

Input : 01_Input_Global/2_Intermediate/HydroSHEDS_RIVERS_BASINS/eu_hydrosheds_raw.nc
Output: 01_Input_Global/2_Intermediate/HydroSHEDS_RIVERS_BASINS/eu_hydrosheds_hand.nc

Location: 02_Processing_Scripts/HydroSHEDS/
Run     : python 02b_hydrosheds_hand.py
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

INPUT_FILE  = INTERMEDIATE / "eu_hydrosheds_raw.nc"
OUTPUT_FILE = INTERMEDIATE / "eu_hydrosheds_hand.nc"

# =============================================================================
# CONFIGURATION
# =============================================================================
ACC_THRESHOLD = 1000
MAX_ITER = 5000

D8 = {
    1  : ( 0,  1),   # E
    2  : ( 1,  1),   # SE
    4  : ( 1,  0),   # S
    8  : ( 1, -1),   # SW
    16 : ( 0, -1),   # W
    32 : (-1, -1),   # NW
    64 : (-1,  0),   # N
    128: (-1,  1),   # NE
}

VAR_META = {
    "hand": {
        "long_name": "height above nearest drainage",
        "units": "m",
        "valid_min": np.float32(0.0),
        "note": (
            f"HAND = DEM - elevation of nearest drainage cell along D8 flow path. "
            f"Drainage defined as ACC >= {ACC_THRESHOLD} upstream cells. "
            "Algorithm: reverse-flow propagation (BFS). "
            "NaN = ocean or cell with no downstream drainage outlet in dataset extent."
        )
    }
}

# =============================================================================
# COMPUTATION
# =============================================================================
def _slices(dr: int, dc: int):
    if dr > 0:
        r_up, r_dn = slice(None, -dr), slice(dr, None)
    elif dr < 0:
        r_up, r_dn = slice(-dr, None), slice(None, dr)
    else:
        r_up = r_dn = slice(None)

    if dc > 0:
        c_up, c_dn = slice(None, -dc), slice(dc, None)
    elif dc < 0:
        c_up, c_dn = slice(-dc, None), slice(None, dc)
    else:
        c_up = c_dn = slice(None)

    return r_up, c_up, r_dn, c_dn

def compute_hand(dem: np.ndarray, dir_data: np.ndarray, acc: np.ndarray, acc_threshold: int, max_iter: int):
    is_drainage = (acc >= acc_threshold) & ~np.isnan(acc) & ~np.isnan(dem)
    n_total    = int((~np.isnan(dem)).sum())
    n_drainage = int(is_drainage.sum())
    print(f"      Drainage cells : {n_drainage:>10,}  ({100*n_drainage/n_total:.2f}% of land area)")
    print(f"      To propagate   : {n_total - n_drainage:>10,} cells")

    drain_elev = np.full(dem.shape, np.nan, dtype=np.float32)
    drain_elev[is_drainage] = dem[is_drainage]
    del is_drainage; gc.collect()

    dir_int = np.where(np.isnan(dir_data), -1, dir_data).astype(np.int16)
    del dir_data; gc.collect()

    print(f"      Starting iteration (max {max_iter}) ...")
    prev_remaining = int(np.isnan(drain_elev).sum())

    for it in range(1, max_iter + 1):
        filled_this_iter = 0
        for d, (dr, dc) in D8.items():
            r_up, c_up, r_dn, c_dn = _slices(dr, dc)
            de_up = drain_elev[r_up, c_up]
            de_dn = drain_elev[r_dn, c_dn]
            di_up = dir_int[r_up, c_up]

            mask = (di_up == d) & np.isnan(de_up) & ~np.isnan(de_dn)
            n = int(mask.sum())
            if n > 0:
                de_up[mask] = de_dn[mask]
                filled_this_iter += n

        remaining = int(np.isnan(drain_elev).sum())

        if it % 50 == 0 or filled_this_iter == 0:
            print(f"      Iter {it:4d}: {filled_this_iter:>8,} filled  |  {remaining:>8,} remaining")

        if filled_this_iter == 0:
            print(f"      → Converged after {it} iterations")
            break

        prev_remaining = remaining
    else:
        print(f"      [WARNING] MAX_ITER={max_iter} reached; {prev_remaining:,} cells without drainage outlet.")

    del dir_int; gc.collect()

    hand = (dem - drain_elev).astype(np.float32)
    hand[np.isnan(drain_elev)] = np.nan
    hand[np.isnan(dem)]        = np.nan
    hand[hand < 0]             = np.float32(0)

    del drain_elev; gc.collect()
    return hand

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 65)
    print("  HydroSHEDS → HAND")
    print("=" * 65)

    if not INPUT_FILE.exists():
        sys.exit(f"[ERROR] Input file not found: {INPUT_FILE}\nRun 01_hydrosheds_raw_to_nc.py first.")

    print("[1/3] Reading eu_hydrosheds_raw.nc ...")
    ds_in = xr.open_dataset(INPUT_FILE)
    lats     = ds_in["lat"].values.astype(np.float64)
    lons     = ds_in["lon"].values.astype(np.float64)
    dem      = ds_in["dem"].values.astype(np.float32)
    acc      = ds_in["acc"].values.astype(np.float32)
    dir_data = ds_in["dir"].values.astype(np.float32)
    ds_in.close()
    gc.collect()

    h, w = dem.shape
    print(f"      Grid: {h} × {w}")

    print("\n[2/3] Computing HAND ...")
    hand = compute_hand(dem, dir_data, acc, ACC_THRESHOLD, MAX_ITER)
    del acc, dem, dir_data; gc.collect()

    print("\n[3/3] Assembling Dataset and Saving ...")
    ds_out = xr.Dataset(
        data_vars={
            "hand": (["lat", "lon"], hand, VAR_META["hand"]),
        },
        coords={
            "lat": (["lat"], lats),
            "lon": (["lon"], lons),
        }
    )
    del hand; gc.collect()

    global_attrs = nc_io.make_global_attrs(
        title="HydroSHEDS HAND – Europe (3 arc-second)",
        source="Derived from eu_hydrosheds_raw.nc (HydroSHEDS v1.1)",
        script=Path(__file__).name
    )

    nc_io.save_nc(
        ds=ds_out,
        path=OUTPUT_FILE,
        fill_value=np.nan,
        dtype="float32",
        global_attrs=global_attrs,
        compress=True,
        require_tier=True,
        verify=True
    )

    print("\n✅ Done.")

if __name__ == "__main__":
    main()

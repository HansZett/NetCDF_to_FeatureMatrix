# -*- coding: utf-8 -*-
"""
01_twsa_process.py

GRACE gap-filled reconstruction (TWSA) — subset and homogenize in one pass.

Pipeline:  1_Original → 3_Final   (no 2_Intermediate stage).

Steps:
- Read global GRACE_1984_2020.nc
  (GRACE_REC_VCE_SSLRfull_SRECES_monthly, Hacker et al., Univ. Bonn / CRC 1502)
- Decode the non-standard 'Seconds since 1984-01-01' time axis to datetimes
- Crop to Germany AOI (5.5–15.5°E, 47.0–55.5°N) and time ≥ 1991-01-01
- Mask the source FillValue (-999999.0; non-CF 'FillValue' attr) to NaN
- Rename TWSTORE → TWS, Sigma_TWSTORE → TWS_variance, attach clean attrs
- Snap monthly-ish stamps to first-of-month (MS), with a sanity check
- Write via nc_io.save_nc:
    * lat descending / lon ascending (native here)
    * float32, explicit NaN _FillValue, zlib complevel 4
    * CF-1.8 global attrs (CRS carried as the global 'crs' attribute — no crs var)
    * time encoded as 'days since 1970-01-01' (float64)

Input : 01_Input_Global/1_Original/TWSA/GRACE_1984_2020.nc
Output: 01_Input_Global/3_Final/TWSA/TWSA.nc

Location: 02_Processing_Scripts/TWSA/
Run     : python 01_twsa_process.py

NOTE: 'Sigma_TWSTORE' is documented (units mm^2, long_name "Variance of the
reconstructed GRACE-like TWSA maps") as the VARIANCE (sigma^2), not the standard
deviation — hence the name TWS_variance. For std: sqrt(TWS_variance) [mm].
"""

import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd
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
ORIGINAL     = INPUT_GLOBAL / "1_Original"
INTERMEDIATE = INPUT_GLOBAL / "2_Intermediate"
FINAL        = INPUT_GLOBAL / "3_Final"

INPUT_FILE  = ORIGINAL / "TWSA" / "GRACE_1984_2020.nc"

OUTPUT_DIR  = FINAL / "TWSA"
OUTPUT_FILE = OUTPUT_DIR / "TWSA.nc"

# =============================================================================
# AOI & TIME RANGE
# =============================================================================
LAT_MIN, LAT_MAX = 47.0, 55.5
LON_MIN, LON_MAX = 5.5, 15.5
TIME_START       = "1991-01-01"   # inclusive; end taken from the dataset

# =============================================================================
# VARIABLE MAPPING & ATTRS
# =============================================================================
VARIABLE_MAP = {
    "TWSTORE":       "TWS",
    "Sigma_TWSTORE": "TWS_variance",
}

VARIABLE_ATTRS = {
    "TWS": {
        "long_name":   "Terrestrial Water Storage Anomaly",
        "units":       "mm",
        "description": ("Reconstructed GRACE-like TWS anomaly "
                        "(Hacker et al., Univ. Bonn, CRC 1502 C03)."),
    },
    "TWS_variance": {
        "long_name":   "Variance of TWS Anomaly",
        "units":       "mm^2",
        "description": ("Variance (sigma^2) of the reconstructed GRACE-like TWS "
                        "anomaly. Source variable name 'Sigma_TWSTORE' is "
                        "misleading — units (mm^2) and source long_name confirm "
                        "this is variance, not standard deviation."),
    },
}

# =============================================================================
# HELPERS
# =============================================================================
def decode_time(ds):
    """Decode 'Seconds since YYYY-MM-DD' time axis to pandas datetimes.

    The CF decoder of xarray does not accept the capital 'S' in 'Seconds', so
    the units string is parsed manually.
    """
    time_units = ds.time.attrs.get("units", "")
    m = re.search(r"[Ss]econds since (\d{4}-\d{2}-\d{2})", time_units)
    if m is None:
        raise ValueError(f"Cannot parse time units: {time_units!r}")
    origin = pd.Timestamp(m.group(1))
    times  = origin + pd.to_timedelta(ds.time.values, unit="s")
    ds = ds.assign_coords(time=pd.to_datetime(times))
    ds.time.attrs.pop("units", None)
    ds.time.attrs.pop("calendar", None)
    return ds


def snap_time_to_month_start(ds):
    """Snap monthly-ish stamps to the first day of each month (MS frequency).

    The source uses ~monthly increments that can drift slightly within the
    calendar month. A clean month-start sequence is rebuilt, anchored to the
    month of the first timestamp, and each original stamp is checked to fall
    in the calendar month the snap assigns it to.
    """
    if "time" not in ds.coords:
        return ds

    n      = ds.sizes["time"]
    first  = pd.Timestamp(ds.time.values[0])
    anchor = first.to_period("M").to_timestamp()           # first of that month
    new    = pd.date_range(start=anchor, periods=n, freq="MS")

    orig_months     = pd.DatetimeIndex(ds.time.values).to_period("M")
    expected_months = new.to_period("M")
    mismatches      = int((orig_months != expected_months).sum())
    if mismatches:
        print(f"    WARNING: {mismatches}/{n} time stamps fall in a different "
              f"calendar month than the snap target. Check input cadence.")
    else:
        print(f"    Snap check: all {n} stamps map to their own calendar month ✓")

    return ds.assign_coords(time=new)


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("GRACE GAP-FILLED (TWSA) — SUBSET + HOMOGENIZE")
    print("=" * 70)
    print(f"Input:  {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"AOI:    lon {LON_MIN}–{LON_MAX}°E,  lat {LAT_MIN}–{LAT_MAX}°N")
    print(f"Time:   from {TIME_START} to end of dataset")
    print("=" * 70)

    if not INPUT_FILE.exists():
        print(f"ERROR: Input file not found:\n  {INPUT_FILE}")
        sys.exit(1)

    # Open WITHOUT auto time-decode (non-standard 'Seconds' units would trip it)
    ds = xr.open_dataset(INPUT_FILE, decode_times=False)
    print(f"\nOriginal dims: {dict(ds.sizes)}")

    # --- Decode time ---
    ds = decode_time(ds)
    print(f"Time range (full): "
          f"{pd.Timestamp(ds.time.values[0]).date()}  →  "
          f"{pd.Timestamp(ds.time.values[-1]).date()}")

    # --- Spatial crop (lat descending → high-to-low; lon ascending → low-to-high) ---
    ds = ds.sel(lat=slice(LAT_MAX, LAT_MIN),
                lon=slice(LON_MIN, LON_MAX))

    # --- Temporal crop ---
    ds = ds.sel(time=slice(TIME_START, None))
    print(f"\nSubset dims:   {dict(ds.sizes)}")

    # --- Mask the source FillValue (non-CF 'FillValue' attr, = -999999.0) → NaN ---
    for var in ds.data_vars:
        fv = ds[var].attrs.pop("FillValue", None)
        if fv is not None:
            ds[var] = ds[var].where(ds[var] != fv)

    # --- Rename data variables & keep only mapped ones ---
    rename  = {old: new for old, new in VARIABLE_MAP.items() if old in ds.data_vars}
    missing = [v for v in VARIABLE_MAP if v not in ds.data_vars]
    if missing:
        print(f"  WARNING: Expected variables not found in input: {missing}")
    ds = ds.rename(rename)
    ds = ds[[v for v in ds.data_vars if v in VARIABLE_MAP.values()]]

    # --- Apply clean variable attributes ---
    for v in ds.data_vars:
        if v in VARIABLE_ATTRS:
            ds[v].attrs = VARIABLE_ATTRS[v]

    # --- Snap time stamps to month-start ---
    print("  Snapping time stamps to month-start (MS frequency)…")
    ds = snap_time_to_month_start(ds)

    t0 = pd.Timestamp(ds.time.values[0]).date()
    t1 = pd.Timestamp(ds.time.values[-1]).date()
    print(f"\nFinal variables: {list(ds.data_vars)}")
    print(f"Final dims:      {dict(ds.sizes)}")
    print(f"Time range:      {t0}  →  {t1}")

    # --- Global attributes (CRS carried as a global attr; no crs variable) ---
    global_attrs = nc_io.make_global_attrs(
        title="GRACE gap-filled TWS reconstruction — Germany subset",
        source=("GRACE_REC_VCE_SSLRfull_SRECES_monthly (v1.0.0); "
                "CRC 1502 (SFB 1502) DETECT, project C03; "
                "Hacker et al., University of Bonn"),
        institution="University of Bonn, Institute of Geodesy and Geoinformation",
        references="Hacker et al. (2026), ESSD, doi:10.5194/essd-18-1747-2026",
        crs="EPSG:4326 (WGS84)",
        script=Path(__file__).name,
        extra={
            "method": ("Data combination of a data-driven reconstruction and "
                       "geodetic tracking data (Loecher et al. 2025)"),
            "contact":            "chacker@uni-bonn.de",
            "spatial_coverage":   "Germany and surroundings (5.5–15.5°E, 47.0–55.5°N)",
            "spatial_resolution": "0.5° (~55 km)",
            "time_resolution":    "monthly",
        },
    )

    print(f"\nSaving to: {OUTPUT_FILE}")
    nc_io.save_nc(
        ds,
        OUTPUT_FILE,
        fill_value=np.nan,          # TWS + TWS_variance are continuous floats
        dtype="float32",
        global_attrs=global_attrs,
        compress=True,
        complevel=4,
        encoding_overrides={
            "time": {"dtype":    "float64",
                     "units":    "days since 1970-01-01",
                     "calendar": "standard"},
        },
    )

    ds.close()
    print("\n✓ Done.")
    print(f"  Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

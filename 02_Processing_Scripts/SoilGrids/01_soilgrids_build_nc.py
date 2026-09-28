# -*- coding: utf-8 -*-
"""
01_soilgrids_build_nc.py

Builds the standardized final SoilGrids NetCDF directly from the GeoTIFFs that
were exported with the Google Earth Engine scripts 00a and 00b in this folder.

  9 variables, depth-weighted mean 0–200 cm, native ~250 m, EPSG:4326:
    vwc_10kPa, vwc_33kPa, vwc_1500kPa  [m³/m³]   (×1000 GEE-export correction)
    bulk_density [g/cm³], coarse_fragments [% vol],
    clay/sand/silt [% weight], soc [g/kg]        (already scaled by GEE script)

Orientation, _FillValue (NaN), float32, zlib and CF metadata are all applied
by nc_io.save_nc → descending lat / ascending lon, matching every other dataset.

Input : 01_Input_Global/1_Original/SoilGrids/*.tif
Output: 01_Input_Global/3_Final/SoilGrids/soilgrids.nc

Location: 02_Processing_Scripts/SoilGrids/
Run     : python 01_soilgrids_build_nc.py
"""

import sys
from pathlib import Path

import numpy as np
import rioxarray
import xarray as xr

# ============================================================================
# Shared helper
# ============================================================================
sys.path.append(str(Path(__file__).resolve().parents[1] / "0_Shared"))
import nc_io

# ============================================================================
# PATHS  (relative to the project root; adjust here if the layout changes)
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_GLOBAL = PROJECT_ROOT / "01_Input_Global"
ORIGINAL     = INPUT_GLOBAL / "1_Original"
INTERMEDIATE = INPUT_GLOBAL / "2_Intermediate"
FINAL        = INPUT_GLOBAL / "3_Final"

INPUT_DIR   = ORIGINAL / "SoilGrids"
OUTPUT_FILE = FINAL / "SoilGrids" / "soilgrids.nc"

# ============================================================================
# TIFF configuration
#   var_name → (filename, scale_factor, unit, long_name, source, correction)
#   scale_factor None  → already correctly scaled by the GEE script
#   scale_factor 1000  → VWC correction for the GEE export bug
# ============================================================================
TIFF_CONFIG = {
    "vwc_10kPa": (
        "VWC_10kPa_0-200cm_Mean.tif", 1000.0, "m³/m³",
        "Volumetric Water Content at 10 kPa (field capacity proxy)",
        "ISRIC SoilGrids v2.0 – wv0010, weighted mean 0–200 cm",
        "Multiplied by 1000 to compensate for GEE export error "
        "(TIFF values were 1/1000 of true m³/m³)",
    ),
    "vwc_33kPa": (
        "VWC_33kPa_0-200cm_Mean.tif", 1000.0, "m³/m³",
        "Volumetric Water Content at 33 kPa (field capacity)",
        "ISRIC SoilGrids v2.0 – wv0033, weighted mean 0–200 cm",
        "Multiplied by 1000 to compensate for GEE export error",
    ),
    "vwc_1500kPa": (
        "VWC_1500kPa_0-200cm_Mean.tif", 1000.0, "m³/m³",
        "Volumetric Water Content at 1500 kPa (permanent wilting point)",
        "ISRIC SoilGrids v2.0 – wv1500, weighted mean 0–200 cm",
        "Multiplied by 1000 to compensate for GEE export error",
    ),
    "bulk_density": (
        "BD_0-200cm_Mean.tif", None, "g/cm³",
        "Bulk Density (fine earth fraction), depth-weighted mean 0–200 cm",
        "ISRIC SoilGrids v2.0 – bdod, weighted mean 0–200 cm",
        "No correction applied; GEE script divided raw cg/cm³ by 100.",
    ),
    "coarse_fragments": (
        "CF_0-200cm_Mean.tif", None, "% vol",
        "Coarse Fragments (volumetric), depth-weighted mean 0–200 cm",
        "ISRIC SoilGrids v2.0 – cfvo, weighted mean 0–200 cm",
        "No correction applied; GEE script divided raw cm³/dm³ by 10.",
    ),
    "clay": (
        "Clay_0-200cm_Mean.tif", None, "% weight",
        "Clay Content (mass fraction < 2 µm), depth-weighted mean 0–200 cm",
        "ISRIC SoilGrids v2.0 – clay, weighted mean 0–200 cm",
        "No correction applied; GEE script divided raw g/kg by 10.",
    ),
    "sand": (
        "Sand_0-200cm_Mean.tif", None, "% weight",
        "Sand Content (mass fraction 50–2000 µm), depth-weighted mean 0–200 cm",
        "ISRIC SoilGrids v2.0 – sand, weighted mean 0–200 cm",
        "No correction applied; GEE script divided raw g/kg by 10.",
    ),
    "silt": (
        "Silt_0-200cm_Mean.tif", None, "% weight",
        "Silt Content (mass fraction 2–50 µm), depth-weighted mean 0–200 cm",
        "ISRIC SoilGrids v2.0 – silt, weighted mean 0–200 cm",
        "No correction applied; GEE script divided raw g/kg by 10.",
    ),
    "soc": (
        "SOC_0-200cm_Mean.tif", None, "g/kg",
        "Soil Organic Carbon Content, depth-weighted mean 0–200 cm",
        "ISRIC SoilGrids v2.0 – soc, weighted mean 0–200 cm",
        "No correction applied; GEE script divided raw dg/kg by 10.",
    ),
}

# Plausibility ranges for a quick quality flag
EXPECTED_RANGES = {
    "vwc_10kPa":        (0.10, 0.80),
    "vwc_33kPa":        (0.00, 0.70),
    "vwc_1500kPa":      (0.02, 0.50),
    "bulk_density":     (0.10, 2.00),
    "coarse_fragments": (0.00, 80.0),
    "clay":             (0.00, 80.0),
    "sand":             (0.00, 100.0),
    "silt":             (0.00, 80.0),
    "soc":              (0.00, 600.0),
}


def load_tiff(path: Path, var_name: str, scale_factor: float | None) -> xr.DataArray:
    """Read a GeoTIFF, rename to lon/lat, optionally rescale, drop spatial_ref."""
    da = rioxarray.open_rasterio(path, masked=True).squeeze("band", drop=True)
    da = da.rename({"x": "lon", "y": "lat"})
    da.name = var_name
    if scale_factor is not None:
        da = da * scale_factor
    if "spatial_ref" in da.coords:
        da = da.drop_vars("spatial_ref")
    return da


def check_range(var_name: str, da: xr.DataArray) -> None:
    lo, hi = EXPECTED_RANGES.get(var_name, (-1e9, 1e9))
    vmin, vmax = float(np.nanmin(da.values)), float(np.nanmax(da.values))
    vmean = float(np.nanmean(da.values))
    flag = "  ⚠ OUTSIDE EXPECTED RANGE!" if (vmin < lo or vmax > hi) else ""
    print(f"    min={vmin:.4f}  mean={vmean:.4f}  max={vmax:.4f}"
          f"  (expected {lo}–{hi}){flag}")


# ============================================================================
# Build the dataset
# ============================================================================
print("=" * 65)
print("SoilGrids → final NetCDF")
print("=" * 65)

arrays = {}
for var_name, cfg in TIFF_CONFIG.items():
    filename, scale_factor, unit, long_name, source, correction = cfg
    fpath = INPUT_DIR / filename
    if not fpath.is_file():
        raise FileNotFoundError(f"TIFF not found: {fpath}")

    scale_info = f"× {scale_factor}" if scale_factor is not None else "no factor"
    print(f"\n  Reading {filename}  →  '{var_name}' [{unit}] ({scale_info})")

    da = load_tiff(fpath, var_name, scale_factor)
    check_range(var_name, da)
    da.attrs.update({"long_name": long_name, "units": unit,
                     "source": source, "correction": correction})
    arrays[var_name] = da

ds = xr.Dataset(arrays)

# ============================================================================
# Global attributes
# ============================================================================
attrs = nc_io.make_global_attrs(
    title="SoilGrids v2.0 — depth-weighted mean 0–200 cm — Germany & surroundings",
    source="ISRIC SoilGrids 250 m v2.0, via Google Earth Engine",
    references="Poggio et al. (2021), SOIL, https://doi.org/10.5194/soil-7-217-2021",
    script=Path(__file__).name,
    extra={
        "description": (
            "Depth-weighted mean (0–200 cm) over bbox lon 5.5–15.5°E, "
            "lat 47–55.5°N. Depth intervals 0-5/5-15/15-30/30-60/60-100/"
            "100-200 cm, weights 5/10/15/30/40/100 cm. Native ~250 m. "
            "VWC ×1000 to correct GEE export error; all other variables "
            "already scaled by the GEE script."
        ),
    },
)

# ============================================================================
# Save  (orientation, fill, dtype, encoding, verify — all in nc_io)
# ============================================================================
nc_io.save_nc(
    ds, OUTPUT_FILE,
    fill_value=np.nan,        # all 9 variables are continuous floats
    dtype=np.float32,
    global_attrs=attrs,
)

ds.close()
print("\n✓ SoilGrids complete.")

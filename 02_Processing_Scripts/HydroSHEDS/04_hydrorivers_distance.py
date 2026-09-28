# -*- coding: utf-8 -*-
"""
04_hydrorivers_distance.py

Calculates Euclidean distance to nearest rivers based on HydroRIVERS flow classes.
Subsets to Germany bounds, rasterizes at 15 arc-seconds, computes distance transform.
Skips flow classes that are not present in the AOI.

Input : 01_Input_Global/1_Original/HydroSHEDS_RIVERS_BASINS/HydroRIVERS_v10_eu_shp/HydroRIVERS_v10_eu.shp
Output: 01_Input_Global/3_Final/HydroSHEDS_RIVERS_BASINS/HydroRIVERS_distance.nc

Location: 02_Processing_Scripts/HydroSHEDS/
Run     : python 04_hydrorivers_distance.py
"""

import sys
import gc
from pathlib import Path
import numpy as np
import geopandas as gpd
import xarray as xr
from rasterio import features
from rasterio.transform import from_bounds
from scipy.ndimage import distance_transform_edt

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

# Ensure output directory exists
FINAL.mkdir(parents=True, exist_ok=True)

INPUT_SHP   = ORIGINAL / "HydroRIVERS_v10_eu_shp" / "HydroRIVERS_v10_eu.shp"
OUTPUT_FILE = FINAL / "HydroRIVERS_distance.nc"

# =============================================================================
# CONFIGURATION
# =============================================================================
# Area of Interest (Germany bounds)
MINX, MINY, MAXX, MAXY = 5.5, 47.0, 15.5, 55.5

# 15 arc-seconds resolution in decimal degrees
RES = 15 / 3600.0

# Target ORD_FLOW classes
CLASSES = range(3, 11)


# =============================================================================
# MAIN LOGIC
# =============================================================================
def main():
    print("=" * 65)
    print("  HydroRIVERS → Distance to River classes")
    print("=" * 65)

    if not INPUT_SHP.exists():
        sys.exit(f"[ERROR] Input shapefile not found:\n  {INPUT_SHP}")

    # 1. Define Raster Grid
    print("[1/3] Defining raster grid...")
    width  = int(round((MAXX - MINX) / RES))
    height = int(round((MAXY - MINY) / RES))

    # Create affine transform
    transform = from_bounds(MINX, MINY, MAXX, MAXY, width, height)

    # Generate coordinates directly from transform
    lons = np.array([transform.c + (j + 0.5) * transform.a for j in range(width)], dtype=np.float64)
    lats = np.array([transform.f + (i + 0.5) * transform.e for i in range(height)], dtype=np.float64)

    print(f"      Grid: {height} × {width} (15 arc-seconds)")

    # 2. Load Vector Data
    print("\n[2/3] Loading HydroRIVERS shapefile (AOI subset)...")
    gdf = gpd.read_file(INPUT_SHP, bbox=(MINX, MINY, MAXX, MAXY))
    gdf = gdf[gdf['ORD_FLOW'].isin(CLASSES)]
    print(f"      Found {len(gdf)} river segments in AOI for target classes.")

    # 3. Rasterize and Calculate Distances
    print("\n[3/3] Rasterizing & calculating EDT per class...")
    dataset_dict = {}

    for cls in CLASSES:
        print(f"      Processing ORD_FLOW class {cls}...", end=" ", flush=True)

        gdf_cls = gdf[gdf['ORD_FLOW'] == cls]

        if gdf_cls.empty:
            print("No rivers found -> Skipping class.")
            continue

        geom = [(geom, 1) for geom in gdf_cls.geometry]
        rasterized = features.rasterize(
            geom,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8
        )

        # Distance transform requires True for background, False for features
        inverted_mask = (rasterized == 0)

        # Pixel distance * resolution = degree distance
        pixel_dist = distance_transform_edt(inverted_mask)
        dist_array = (pixel_dist * RES).astype(np.float32)
        dist_array[rasterized == 1] = 0.0
        print("Done.")

        var_name = f"distance_class_{cls}"
        dataset_dict[var_name] = (['lat', 'lon'], dist_array, {
            "long_name": f"Euclidean distance to nearest river of ORD_FLOW class {cls}",
            "units": "degree"
        })

    # Free memory
    del gdf; gc.collect()

    print("\nAssembling Dataset...")
    ds = xr.Dataset(
        dataset_dict,
        coords={
            'lat': (['lat'], lats),
            'lon': (['lon'], lons)
        }
    )

    global_attrs = nc_io.make_global_attrs(
        title="Distance to HydroRIVERS by ORD_FLOW class",
        source="HydroRIVERS v1.0 (HydroSHEDS)",
        references="Lehner B., Grill G. (2013). Global river hydrography and network routing: baseline data and new approaches to study the world's large river systems. Hydrological Processes, 27(15): 2171-2186. doi:10.1002/hyp.9740",
        script=Path(__file__).name,
        extra={"resolution": "15 arc-seconds"}
    )

    nc_io.save_nc(
        ds=ds,
        path=OUTPUT_FILE,
        fill_value=np.nan,
        dtype="float32",
        global_attrs=global_attrs,
        compress=True,
        require_tier=True,
        verify=True
    )

    print("\n✅ Script complete.")

if __name__ == "__main__":
    main()

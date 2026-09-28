# -*- coding: utf-8 -*-
"""
01_glim_glhymps_merge_clip.py

Merges GLiM (lithology) and GLHYMPS (hydraulic properties) geodatabases,
clips them to the Germany AOI, and saves the result as a GeoPackage.

Processing steps:
    1. Read GLHYMPS (permeability, porosity) from ESRI:54034 GDB
    2. Read GLiM   (lithology classes)       from ESRI:54012 GDB
    3. Clip both to Germany bounding box (densified edges for curved CRS)
    4. Join GLiM lithology onto GLHYMPS via shared IDENTITY_ field
    5. Save merged result as GeoPackage in WGS84

Input:
    01_Input_Global/1_Original/GLiM_GLHYMPS/
        GLHYMPS.gdb          (layer: Final_GLHYMPS_Polygon)
        LiMW_GIS 2015.gdb    (layer: GLiM_export)

Output:
    01_Input_Global/2_Intermediate/GLiM_GLHYMPS/
        GLiM_GLHYMPS.gpkg

Variables in output:
    IDENTITY_                       – Polygon ID (join key)
    Porosity                        – Porosity, dimensionless (0–1)
    Permeability_no_permafrost      – log10(k) in m², permafrost = lithology value
    Permeability_permafrost         – log10(k) in m², permafrost = -20
    Permeability_standard_deviation – Std. dev. of log10 permeability
    Litho                           – Full lithology code (e.g. 'vi____', 'pa__mt')
    xx                              – Short lithology class (2-char, e.g. 'su', 'sc')

Location: 02_Processing_Scripts/GLiM_GLHYMPS/
Run     : python 01_glim_glhymps_merge_clip.py

References:
    Gleeson et al. (2014), doi:10.1002/2014gl059856
    Hartmann & Moosdorf (2012), doi:10.1029/2012GC004370
"""

from pathlib import Path
import time

import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from pyproj import Transformer


# =============================================================================
# PATHS  (relative to the project root; adjust here if the layout changes)
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_GLOBAL = PROJECT_ROOT / "01_Input_Global"
ORIGINAL     = INPUT_GLOBAL / "1_Original"
INTERMEDIATE = INPUT_GLOBAL / "2_Intermediate"
FINAL        = INPUT_GLOBAL / "3_Final"

# Input geodatabases (located directly under 1_Original/GLiM_GLHYMPS/)
DATA_DIR    = ORIGINAL / "GLiM_GLHYMPS"
GLHYMPS_GDB = DATA_DIR / "GLHYMPS.gdb"
GLIM_GDB    = DATA_DIR / "LiMW_GIS 2015.gdb"

# Output
OUT_DIR  = INTERMEDIATE / "GLiM_GLHYMPS"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "GLiM_GLHYMPS.gpkg"

# =============================================================================
# CONFIGURATION
# =============================================================================
# Germany bounding box in WGS84 (minx, miny, maxx, maxy)
BBOX_WGS84 = (5.5, 47.0, 15.5, 55.5)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def transform_bbox_dense(bbox_wgs84, target_crs, n_points=64):
    """
    Transform a WGS84 bounding box to target CRS using densified edges.

    Simple 4-corner transforms fail for projections with curved coordinate
    axes (Eckert IV, Cylindrical Equal Area). Instead, we sample many points
    along all edges and take the envelope in the target CRS.

    Parameters
    ----------
    bbox_wgs84 : tuple  (minx, miny, maxx, maxy) in EPSG:4326
    target_crs : CRS    Target coordinate reference system
    n_points   : int    Number of sample points per edge

    Returns
    -------
    tuple : (minx, miny, maxx, maxy) in target CRS
    """
    minx, miny, maxx, maxy = bbox_wgs84
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)

    lons, lats = [], []
    for val in np.linspace(minx, maxx, n_points):
        lons.extend([val, val])
        lats.extend([miny, maxy])
    for val in np.linspace(miny, maxy, n_points):
        lons.extend([minx, maxx])
        lats.extend([val, val])

    xs, ys = transformer.transform(lons, lats)
    return (min(xs), min(ys), max(xs), max(ys))


def create_clip_polygon_dense(bbox_wgs84, target_crs, n_points=128):
    """
    Create a clip polygon by densely sampling WGS84 bbox edges and
    transforming to target CRS.

    Returns
    -------
    Polygon : Shapely polygon in target CRS
    """
    minx, miny, maxx, maxy = bbox_wgs84
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)

    # Walk around the bbox perimeter (W→E bottom, S→N right, E→W top, N→S left)
    edge_points = []
    for lon in np.linspace(minx, maxx, n_points):
        edge_points.append((lon, miny))
    for lat in np.linspace(miny, maxy, n_points):
        edge_points.append((maxx, lat))
    for lon in np.linspace(maxx, minx, n_points):
        edge_points.append((lon, maxy))
    for lat in np.linspace(maxy, miny, n_points):
        edge_points.append((minx, lat))

    lons, lats = zip(*edge_points)
    xs, ys = transformer.transform(list(lons), list(lats))
    return Polygon(zip(xs, ys))


def subset_gdb(gdb_path, layer_name, bbox_wgs84):
    """
    Read a GDB layer and clip to a WGS84 bounding box.

    Handles CRS transformation with densified edges to avoid
    clipping errors in curved projections.

    Returns
    -------
    GeoDataFrame in EPSG:4326
    """
    print(f"\n  Reading {gdb_path.name} / {layer_name} ...")

    # Detect source CRS
    gdf_meta = gpd.read_file(gdb_path, layer=layer_name, rows=0)
    src_crs = gdf_meta.crs
    print(f"    Source CRS: {src_crs}")

    # Spatial filter with densified bbox
    bbox_src = transform_bbox_dense(bbox_wgs84, src_crs)
    print(f"    Bbox in source CRS: {[f'{v:.0f}' for v in bbox_src]}")

    t0 = time.time()
    gdf = gpd.read_file(gdb_path, layer=layer_name, bbox=bbox_src)
    print(f"    Read {len(gdf):,} features in {time.time() - t0:.1f}s")

    if len(gdf) == 0:
        print("    WARNING: No features found in bbox!")
        return gdf

    # Precise clip with densified polygon
    clip_poly = create_clip_polygon_dense(bbox_wgs84, src_crs)
    gdf = gpd.clip(gdf, clip_poly)
    gdf = gdf[~gdf.is_empty].copy()
    print(f"    {len(gdf):,} features after clipping")

    # Reproject to WGS84
    gdf = gdf.to_crs("EPSG:4326")
    return gdf


# =============================================================================
# STEP 1: Subset both GDBs to Germany AOI
# =============================================================================
print("=" * 65)
print("STEP 1: Subset GLiM & GLHYMPS to Germany AOI")
print("=" * 65)
print(f"  AOI (WGS84): W={BBOX_WGS84[0]}°, S={BBOX_WGS84[1]}°, "
      f"E={BBOX_WGS84[2]}°, N={BBOX_WGS84[3]}°")

gdf_glhymps = subset_gdb(GLHYMPS_GDB, "Final_GLHYMPS_Polygon", BBOX_WGS84)
gdf_glim    = subset_gdb(GLIM_GDB,    "GLiM_export",            BBOX_WGS84)

# =============================================================================
# STEP 2: Merge GLiM lithology onto GLHYMPS
# =============================================================================
print(f"\n{'=' * 65}")
print("STEP 2: Merge GLiM lithology onto GLHYMPS via IDENTITY_ field")
print("=" * 65)

# Keep only lithology columns from GLiM (geometry comes from GLHYMPS)
glim_attrs = (gdf_glim[["IDENTITY_", "Litho", "xx"]]
              .drop_duplicates(subset="IDENTITY_"))

gdf_merged = gdf_glhymps.merge(glim_attrs, on="IDENTITY_", how="left")

n_matched = gdf_merged["Litho"].notna().sum()
n_total   = len(gdf_merged)
print(f"  GLHYMPS features:   {n_total:,}")
print(f"  Matched with GLiM:  {n_matched:,}  ({100 * n_matched / n_total:.1f}%)")
print(f"  Unmatched:          {n_total - n_matched:,}")
print(f"  Columns: {list(gdf_merged.columns)}")

# =============================================================================
# STEP 3: Save merged GeoPackage
# =============================================================================
print(f"\n{'=' * 65}")
print("STEP 3: Save merged GeoPackage")
print("=" * 65)

gdf_merged.to_file(OUT_PATH, driver="GPKG")
size_mb = OUT_PATH.stat().st_size / 1024**2
print(f"  Saved: {OUT_PATH}")
print(f"  Size:  {size_mb:.1f} MB")
print("  CRS:   EPSG:4326")

# Print lithology class overview
litho_codes = gdf_merged["xx"].dropna().unique()
print(f"\n  Lithology classes found ({len(litho_codes)}):")
for code in sorted(litho_codes):
    count = (gdf_merged["xx"] == code).sum()
    print(f"    {code}  ({count:,} features)")

print(f"\n{'=' * 65}")
print("Done. Next step: 02_glim_glhymps_rasterize.py")
print(f"{'=' * 65}")

# -*- coding: utf-8 -*-
"""
dataset_manifest.py — single source of truth for HOW each dataset is extracted
==============================================================================

One entry per final NetCDF grid produced by 02_Processing_Scripts, plus the
basin-aggregated HydroBASINS entry. The extraction engine (extract_core.py)
and the store builders read this manifest, so search radius, temporal
coverage and fill handling are configured per dataset in one place instead
of being hardcoded in the scripts.

Location: 02_Dataframes_Processing_Scripts/0_Shared/dataset_manifest.py
          (imported by the scripts in 2_Feature_Stores/)

Run it directly to self-check (writes nothing):
    python dataset_manifest.py                 # project root derived from this file
    python dataset_manifest.py --root "<path to project root>"
    python dataset_manifest.py --fills         # also open each NC and print
                                               # declared _FillValue per variable

The project root can also be set with the environment variable GWL_ROOT.

----------------------------------------------------------------------------
SCHEMA (DatasetSpec)
----------------------------------------------------------------------------
key              short stable token -> used as the column prefix '<key>__var'
dataset          provenance / folder name (3_Final/<dataset>/...)
nc               path to the .nc RELATIVE to 3_Final  (None = not a single NC,
                 e.g. the basin-aggregated source which comes from basin CSVs)
mode             'point'  -> sample the grid cell at the well
                 'basin'  -> polygon join + basin-aggregated series (no cell sampling)
                 'index'  -> a lookup raster (e.g. HYBAS_ID mask), not a feature
temporal         'static' | 'monthly' | 'yearly'
role             'feature' (extracted into the store) | 'index' (support only)
resolution_deg   nominal cell size in degrees (drives search_radius_km='auto')
search_radius_km 'auto'  -> AUTO_RINGS cell-rings, derived from resolution
                 0.0     -> strict nearest cell only (no neighbour search)
                 <float> -> explicit radius in km
on_no_valid      what to do when NO valid cell lies within the radius (the
                 neighbour 'rescue' itself is automatic whenever radius>0):
                 'keep_nan'       -> write NaN, keep the well (== old 02_static)
                 'skip_well'      -> drop this well for this dataset (ERA5/TWSA legacy)
                 'nearest_global' -> nearest valid cell with no distance limit
                                     (the old FALLBACK_ANY_VALID=True)
vars             None = all data_vars minus drop_vars; or an explicit list
drop_vars        variables never extracted (CRS carriers etc.)
categorical      vars that are classes/flags (documentation; fill is still read
                 from the file, these are never interpolated)
coverage         None | (start, end) inclusive, for temporal datasets
fill             'from_nc' -> read the declared _FillValue per variable (default)
transform        None | a named hook for engineered features
                 ('lcc_features', 'basin_aggregate')
notes            human explanation / caveats
----------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ============================================================================
# CONSTANTS
# ============================================================================
KM_PER_DEG = 111.32
AUTO_RINGS = 2          # 'auto' radius ~= this many cell rings outward ...
AUTO_FLOOR_KM = 3.0     # ... but never below this physical floor, so fine
                        # rasters can still escape NoData holes (rivers, lakes,
                        # masked patches) whose size does NOT scale with cell size.

VALID_MODES       = {"point", "basin", "index"}
VALID_TEMPORAL    = {"static", "monthly", "yearly"}
VALID_ON_NO_VALID = {"keep_nan", "skip_well", "nearest_global"}


# ============================================================================
# SCHEMA
# ============================================================================
@dataclass
class DatasetSpec:
    key: str
    dataset: str
    nc: str | None
    mode: str = "point"
    temporal: str = "static"
    role: str = "feature"
    resolution_deg: float | None = None
    search_radius_km: float | str = "auto"
    on_no_valid: str = "keep_nan"
    vars: list | None = None
    drop_vars: list = field(default_factory=lambda: ["crs", "spatial_ref"])
    categorical: list = field(default_factory=list)
    coverage: tuple | None = None
    fill: str = "from_nc"
    transform: str | None = None
    notes: str = ""

    def validate(self) -> None:
        assert self.mode in VALID_MODES, f"{self.key}: bad mode {self.mode!r}"
        assert self.temporal in VALID_TEMPORAL, f"{self.key}: bad temporal {self.temporal!r}"
        assert self.on_no_valid in VALID_ON_NO_VALID, f"{self.key}: bad on_no_valid"
        if isinstance(self.search_radius_km, str):
            assert self.search_radius_km == "auto", f"{self.key}: radius must be float or 'auto'"
        if self.temporal != "static" and self.coverage is None and self.mode != "basin":
            print(f"  ⚠  {self.key}: temporal={self.temporal} but no coverage set")


def auto_radius_km(resolution_deg: float | None,
                   rings: int = AUTO_RINGS,
                   floor_km: float = AUTO_FLOOR_KM) -> float:
    """Resolution-aware default radius: a few cell-rings, but at least
    floor_km so fine rasters can still reach past NoData holes."""
    if not resolution_deg:
        return 0.0
    return round(max(rings * resolution_deg * KM_PER_DEG, floor_km), 3)


def resolved_radius_km(spec: DatasetSpec) -> float:
    """The radius the extractor will actually use (resolves 'auto')."""
    if spec.search_radius_km == "auto":
        return auto_radius_km(spec.resolution_deg)
    return float(spec.search_radius_km)


# ============================================================================
# THE MANIFEST  (resolutions and coverage as documented in the dataset READMEs
#                of 02_Processing_Scripts)
# ============================================================================
MANIFEST: list[DatasetSpec] = [

    # ---- STATIC, point-sampled --------------------------------------------
    DatasetSpec(
        key="era5l_static", dataset="ERA5_Land_static",
        nc="ERA5_Land_static/ERA5_Land_static_Germany.nc",
        mode="point", temporal="static", resolution_deg=0.1,
        search_radius_km="auto", on_no_valid="keep_nan",
        categorical=["slt", "tvh", "tvl"],   # soil type / veg type are class codes
        notes="ERA5-Land static surface fields (lsm, soil/veg type, lake cover...).",
    ),
    DatasetSpec(
        key="soilgrids", dataset="SoilGrids",
        nc="SoilGrids/soilgrids.nc",
        mode="point", temporal="static", resolution_deg=0.00225,
        search_radius_km="auto", on_no_valid="keep_nan",
        notes="9 soil properties, continuous floats. Fine grid -> small auto radius.",
    ),
    DatasetSpec(
        key="gasy", dataset="GASY",
        nc="GASY/GASY.nc",
        mode="point", temporal="static", resolution_deg=0.00833,
        search_radius_km="auto", on_no_valid="keep_nan",
        notes="Has a depth dim (7) -> 'sy' expands to 7 depth-suffixed columns.",
    ),
    DatasetSpec(
        key="glim", dataset="GLiM_GLHYMPS",
        nc="GLiM_GLHYMPS/GLiM_GLHYMPS.nc",
        mode="point", temporal="static", resolution_deg=0.00833,
        search_radius_km="auto", on_no_valid="keep_nan",
        categorical=["litho_class"],
        notes="porosity/permeability are floats; litho_class is categorical.",
    ),
    DatasetSpec(
        key="whymap", dataset="WHYMAP",
        nc="WHYMAP/WHYMAP.nc",
        mode="point", temporal="static", resolution_deg=0.00417,
        search_radius_km="auto", on_no_valid="keep_nan",
        vars=["hygeo2"], categorical=["hygeo2"],
        notes="Hydrogeology class. crs dropped.",
    ),
    DatasetSpec(
        key="hydsheds_terrain", dataset="HydroSHEDS_RIVERS_BASINS",
        nc="HydroSHEDS_RIVERS_BASINS/hydrosheds.nc",
        mode="point", temporal="static", resolution_deg=0.00083,
        search_radius_km="auto", on_no_valid="keep_nan",
        categorical=["dir"],
        notes="dem/slope/aspect/twi/hand/acc + flow dir (class). Very fine -> tiny radius.",
    ),
    DatasetSpec(
        key="hydrivers_dist", dataset="HydroSHEDS_RIVERS_BASINS",
        nc="HydroSHEDS_RIVERS_BASINS/HydroRIVERS_distance.nc",
        mode="point", temporal="static", resolution_deg=0.00417,
        search_radius_km="auto", on_no_valid="keep_nan",
        notes="distance_class_<N> for every HydroRIVERS ORD_FLOW class N present "
              "in the study area (distance in degrees).",
    ),

    # ---- STATIC, support rasters (not direct features) --------------------
    DatasetSpec(
        key="basin_mask_lev06", dataset="HydroSHEDS_RIVERS_BASINS",
        nc="HydroSHEDS_RIVERS_BASINS/basin_mask_lev06.nc",
        mode="index", temporal="static", role="index", resolution_deg=0.1,
        search_radius_km=0.0, on_no_valid="keep_nan",
        vars=["HYBAS_ID"], categorical=["HYBAS_ID"],
        notes="Rasterised HYBAS_ID. Basin assignment uses the polygon join, "
              "not this raster -> role=index, excluded from the feature store.",
    ),
    DatasetSpec(
        key="basin_mask_lev07", dataset="HydroSHEDS_RIVERS_BASINS",
        nc="HydroSHEDS_RIVERS_BASINS/basin_mask_lev07.nc",
        mode="index", temporal="static", role="index", resolution_deg=0.1,
        search_radius_km=0.0, on_no_valid="keep_nan",
        vars=["HYBAS_ID"], categorical=["HYBAS_ID"],
        notes="Rasterised HYBAS_ID lev07. role=index (see lev06).",
    ),

    # ---- DYNAMIC, point-sampled, monthly ----------------------------------
    DatasetSpec(
        key="era5l", dataset="ERA5_Land",
        nc="ERA5_Land/era5_land.nc",
        mode="point", temporal="monthly", resolution_deg=0.1,
        search_radius_km=15.0, on_no_valid="skip_well",
        coverage=("1991-01", "2022-12"),
        notes="Monthly ERA5-Land variables. radius=15 km + skip_well: a well "
              "without a valid cell within 15 km is dropped for this dataset. "
              "Accumulated fluxes are per-day (m/day); x1000 -> mm/day.",
    ),
    DatasetSpec(
        key="terra", dataset="TerraClimate",
        nc="TerraClimate/TerraClimate.nc",
        mode="point", temporal="monthly", resolution_deg=0.04167,
        search_radius_km="auto", on_no_valid="keep_nan",
        coverage=("1991-01", "2022-12"),
        notes="14 vars, monthly. auto radius.",
    ),
    DatasetSpec(
        key="twsa", dataset="TWSA",
        nc="TWSA/TWSA.nc",
        mode="point", temporal="monthly", resolution_deg=0.5,
        search_radius_km=0.0, on_no_valid="skip_well",
        coverage=("1991-01", "2020-12"),
        notes="GRACE TWS + variance, ends 2020-12. radius=0 + skip_well: nearest "
              "cell only, wells whose nearest cell is NaN (ocean) are dropped. "
              "To rescue coastal wells from an adjacent land cell, raise radius to "
              "~60 km (one 0.5 deg cell) -- this is a methodological change, off by default. "
              "STORE the full backbone with NaN after 2020; do NOT trim other datasets.",
    ),

    # ---- DYNAMIC, point-sampled, yearly + engineered transform ------------
    DatasetSpec(
        key="lcc", dataset="C3S_LCC",
        nc="C3S_LCC/c3s_lcc.nc",
        mode="point", temporal="yearly", resolution_deg=0.00278,
        search_radius_km="auto", on_no_valid="keep_nan",
        vars=["lccs_class"],
        categorical=["lccs_class", "processed_flag", "current_pixel_state"],
        coverage=("1992", "2022"),
        transform="lcc_features",
        notes="Yearly land cover (1992-2022). Transform 'lcc_features' builds "
              "LCC, LCC_previous, LCC_changed, "
              "months_since_LCC_change (sentinel -1, anchored to Jan of change year), "
              "LCC_stable; pre-coverage years back-filled (1991<-1992), broadcast "
              "to monthly. lccs_class carries an integer NoData sentinel -> fill read "
              "from the NC so a sentinel is never sampled as a real class.",
    ),

    # ---- DYNAMIC, basin-aggregated (polygon join, not cell sampling) ------
    DatasetSpec(
        key="basin_hydro", dataset="HydroBASINS",
        nc=None,                      # source is basin CSVs + shapefiles, not one NC
        mode="basin", temporal="monthly", resolution_deg=None,
        search_radius_km=0.0, on_no_valid="keep_nan",
        coverage=("1991-01", "2022-12"),
        transform="basin_aggregate",
        notes="ERA5-Land and TerraClimate aggregated per HydroBASINS polygon at "
              "lev06 and lev07 (per-basin CSVs from 02_Processing_Scripts/HydroSHEDS/"
              "05a and 05b). Produces dynamic columns (-> dynamic store) and static "
              "basin attributes HYBAS_ID/UP_AREA/COAST plus coverage_frac/"
              "basin_area_m2/covered_area_m2 per product (-> static store). "
              "No cell rescue. Implemented in basin_core.py.",
    ),
]

# Quick lookup + validation on import
BY_KEY: dict[str, DatasetSpec] = {s.key: s for s in MANIFEST}
assert len(BY_KEY) == len(MANIFEST), "duplicate keys in MANIFEST"
for _s in MANIFEST:
    _s.validate()


# ============================================================================
# PATH RESOLUTION
# ============================================================================
_THIS = Path(__file__).resolve()


def default_project_root() -> Path:
    # <root>/02_Dataframes_Processing_Scripts/0_Shared/dataset_manifest.py -> parents[2]
    try:
        return _THIS.parents[2]
    except IndexError:
        return _THIS.parent


def nc_path(spec: DatasetSpec, root: str | Path | None = None) -> Path | None:
    """Absolute path to the dataset's NetCDF under 3_Final, or None for basin."""
    if spec.nc is None:
        return None
    base = Path(root) if root else PROJECT_ROOT
    return base / FINAL_NC_REL / spec.nc


# PATHS: project root (override with the environment variable GWL_ROOT)
PROJECT_ROOT = Path(os.environ.get("GWL_ROOT", default_project_root()))
# Final NetCDF products of 02_Processing_Scripts, relative to the project root
FINAL_NC_REL = Path("01_Input_Global") / "3_Final"
# Shared NetCDF helper nc_io.py, relative to the project root
NC_IO_DIR_REL = Path("02_Processing_Scripts") / "0_Shared"


# ============================================================================
# CONVENIENCE SELECTORS
# ============================================================================
def features(temporal: str | None = None, mode: str | None = None) -> list[DatasetSpec]:
    """All role='feature' specs, optionally filtered by temporal / mode."""
    out = [s for s in MANIFEST if s.role == "feature"]
    if temporal:
        out = [s for s in out if s.temporal == temporal]
    if mode:
        out = [s for s in out if s.mode == mode]
    return out


def dynamic_features() -> list[DatasetSpec]:
    return [s for s in features() if s.temporal in ("monthly", "yearly")]


def static_features() -> list[DatasetSpec]:
    return [s for s in features() if s.temporal == "static"]


# ============================================================================
# SELF-CHECK / REPORT
# ============================================================================
def _print_table(root: Path) -> None:
    print("=" * 100)
    print(f"  DATASET MANIFEST  ({len(MANIFEST)} entries)   project root: {root}")
    print("=" * 100)
    hdr = (f"  {'key':<17}{'mode':<7}{'temporal':<9}{'res(deg)':<10}"
           f"{'radius(km)':<16}{'on_no_valid':<16}{'transform':<17}role")
    print(hdr)
    print("  " + "-" * 100)
    for s in MANIFEST:
        res = "-" if s.resolution_deg is None else f"{s.resolution_deg:g}"
        rad = resolved_radius_km(s)
        rad_s = f"{rad:g}" + (" (auto)" if s.search_radius_km == "auto" else "")
        tr = s.transform or "-"
        print(f"  {s.key:<17}{s.mode:<7}{s.temporal:<9}{res:<10}"
              f"{rad_s:<16}{s.on_no_valid:<16}{tr:<17}{s.role}")
    print()

    # Coverage span overview for temporal datasets
    print("  Temporal coverage (store backbone is the union of these):")
    for s in MANIFEST:
        if s.coverage:
            print(f"    {s.key:<17} {s.coverage[0]}  ..  {s.coverage[1]}")
    print()

    # Existence check (helps catch wrong paths before extraction)
    print("  NetCDF existence under 3_Final:")
    n_ok = n_missing = 0
    for s in MANIFEST:
        p = nc_path(s, root)
        if p is None:
            print(f"    {s.key:<17} (basin source — no single NC)")
            continue
        if p.exists():
            print(f"    {s.key:<17} OK   {p.relative_to(root)}")
            n_ok += 1
        else:
            print(f"    {s.key:<17} MISSING  {p}")
            n_missing += 1
    print(f"\n  -> {n_ok} found, {n_missing} missing\n")


def _print_fills(root: Path) -> None:
    for cand in (PROJECT_ROOT / NC_IO_DIR_REL, _THIS.parent):
        if (cand / "nc_io.py").exists():
            sys.path.append(str(cand))
            break
    try:
        import nc_io
    except Exception as e:  # noqa: BLE001
        print(f"  (could not import nc_io for fill check: {e})")
        return
    print("=" * 100)
    print("  DECLARED FILL VALUES PER VARIABLE (metadata only, no data load)")
    print("=" * 100)
    for s in MANIFEST:
        p = nc_path(s, root)
        if p is None or not p.exists():
            continue
        print(f"\n  [{s.key}] {s.dataset}")
        try:
            nc_io.report_fills(p)
        except Exception as e:  # noqa: BLE001
            print(f"    ! failed to read {p.name}: {e}")


if __name__ == "__main__":
    argv = sys.argv[1:]
    root = PROJECT_ROOT
    if "--root" in argv:
        root = Path(argv[argv.index("--root") + 1]).resolve()
    _print_table(root)
    if "--fills" in argv:
        _print_fills(root)
    print("  Manifest OK ✅  (all entries validated on import)")

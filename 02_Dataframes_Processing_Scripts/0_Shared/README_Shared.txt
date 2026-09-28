================================================================================
0_Shared  -  Modules imported by the feature store scripts
================================================================================

The files in this folder are not run as part of the pipeline. They are
imported by the scripts in 2_Feature_Stores/. Two of them have an optional
command line self-check (see below). The NetCDF helper nc_io.py is imported
from 02_Processing_Scripts/0_Shared/.


dataset_manifest.py
  Configuration of the extraction: one DatasetSpec entry per dataset.
    key              column prefix (<key>__<variable>)
    nc               NetCDF file relative to 01_Input_Global/3_Final/
    mode             point (sample the cell at the well) | basin (polygon
                     join) | index (lookup raster, not a feature)
    temporal         static | monthly | yearly
    resolution_deg   nominal cell size in degrees
    search_radius_km radius for the search of a valid neighbouring cell:
                     'auto' = max(2 cell widths, 3 km); 0 = nearest cell only
    on_no_valid      keep_nan (write NaN) | skip_well (drop the well for this
                     dataset) | nearest_global (nearest valid cell, no limit)
    categorical      variables that are class codes
    coverage         temporal coverage (start, end)
    transform        named feature transform (lcc_features, basin_aggregate)
  To add or change a dataset, edit the MANIFEST list.
  Self-check: python dataset_manifest.py [--root <path>] [--fills]
              (prints the manifest and checks that all NetCDF files exist)

extract_core.py
  Extraction engine. For every well and dataset:
    1. nearest grid cell to the well;
    2. if that cell holds no valid data (declared _FillValue or NaN) and the
       search radius is > 0, the closest valid cell within the radius;
    3. otherwise the on_no_valid rule.
  The cell is chosen once per dataset from its first variable, and all
  variables are read at that cell. The distance, the rescue flag and the
  outcome are returned as provenance.
  Self-check: python extract_core.py --key <dataset key> [--sample 8]
              (samples one dataset at the real well locations, writes nothing)

transforms.py
  lcc_features: builds the five C3S__ land cover features per well from the
  yearly land cover class (see README.txt in the parent folder, section 6).

basin_core.py
  HydroBASINS features. Joins every well to the basin polygon it lies in
  (levels 6 and 7), reads the per-basin CSVs written by
  02_Processing_Scripts/HydroSHEDS/05a and 05b, and returns
    - static basin columns (HYBAS_ID, UP_AREA, COAST, and per product
      coverage_frac, basin_area_m2, covered_area_m2)
    - monthly basin columns per product (basin_era5l_levX__<var>,
      basin_terra_levX__<var>).
  A product whose CSV folder is missing is skipped. Input paths are defined
  in the PATHS block of the module.

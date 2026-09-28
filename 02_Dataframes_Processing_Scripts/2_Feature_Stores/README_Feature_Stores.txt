================================================================================
2_Feature_Stores  -  Build the static and dynamic feature stores
================================================================================

Run the scripts in this order from this folder. All of them accept
--root <path to project root>. Paths are defined in the PATHS block at the
top of each script (relative to the project root).


01_build_static_store.py
  Samples every static dataset of the manifest (era5l_static, soilgrids, gasy,
  glim, whymap, hydsheds_terrain, hydrivers_dist) at the well locations and
  adds the HydroBASINS attributes (INCLUDE_BASIN = True).
  Input : 03_Dataframes/02_Static_Features/01_GEMS-GER/02_adapted_well_metadata.csv
          (wells with MW_ID, lon, lat; all columns are carried into the store)
          01_Input_Global/3_Final/<Dataset>/<file>.nc
  Output: 03_Dataframes/02_Static_Features/02_global/
            static_features.csv        static store, one row per well
            _provenance.csv            sampling provenance per dataset
            _build_report.csv          number of NaN features per well
            _feature_nan_summary.csv   number and percentage of NaN per feature
  Run   : python 01_build_static_store.py            (--limit 20 for a dry run)

02_build_dynamic_store.py
  Samples every dynamic dataset (era5l, terra, twsa, lcc), builds the land
  cover features and adds the monthly basin series. The monthly groundwater
  level file of each well defines the rows (left join on time).
  Input : 03_Dataframes/02_Static_Features/01_GEMS-GER/02_adapted_well_metadata.csv
          03_Dataframes/01_Dynamic_Features/02_GEMS-GER_monthly/MW_<id>_monthly.csv
          (columns time and GWL are used)
          01_Input_Global/3_Final/<Dataset>/<file>.nc
  Output: 03_Dataframes/01_Dynamic_Features/
            03_global_monthly/MW_<id>.csv   dynamic store, one file per well
            03_global_provenance.csv        sampling provenance per dataset
            03_global_build_report.csv      months, period and status per well
  Run   : python 02_build_dynamic_store.py           (--limit 20 for a dry run)
  Wells without a monthly GWL file are reported with status NO_GWL_CSV.

03_extraction_quality_report.py   (optional)
  Summarises the provenance and NaN files of both stores: how many wells per
  dataset used the home cell, a neighbouring cell ("rescued") or had no valid
  cell, sampling distances, and missing values per dataset and feature.
  Output: 03_Dataframes/00_Extraction_Quality/
            qc_by_dataset.csv, qc_by_feature.csv, qc_rescued_wells.csv,
            qc_completeness_by_dataset.csv, qc_thesis_table.csv, qc_summary.txt
  Run   : python 03_extraction_quality_report.py
          python 03_extraction_quality_report.py --dynamic-scan sample
          (the full scan reads every well CSV and is the slow part)


Settings (CONFIGURATION block in the scripts)
  INCLUDE_BASIN   add HydroBASINS columns (default True; skipped automatically
                  if geopandas or the basin inputs are missing)
  LCC_PREFIX      prefix of the land cover features (default "C3S")
  GWL_KEEP_COLS   columns taken from the monthly GWL files ("time", "GWL")
Search radius and missing-cell policy per dataset are set in
0_Shared/dataset_manifest.py.

If the project lies in a cloud-synchronised folder (e.g. OneDrive), pause the
synchronisation during a full run, because thousands of small files are
written.

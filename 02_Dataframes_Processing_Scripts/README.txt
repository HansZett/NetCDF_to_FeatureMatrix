================================================================================
02_Dataframes_Processing_Scripts  -  Feature stores for machine learning
================================================================================

Thesis: Evaluating Machine Learning-Based Groundwater Level Predictions Using
        Globally Available Earth Observation Data

This folder turns the final NetCDF products of 02_Processing_Scripts into
tables that the machine learning scripts read. For every groundwater well
(well ID MW_<id>) the values of all global datasets are sampled at the well
location and written into two "feature stores":

  DYNAMIC store  one CSV per well, one row per month (target GWL + monthly
                 features)
  STATIC store   one table, one row per well (well metadata + time-invariant
                 features)

A feature metadata table then describes every column of the stores (source,
unit, temporal coverage, categorical or not, plot label).


--------------------------------------------------------------------------------
1. FOLDER STRUCTURE AND RUN ORDER
--------------------------------------------------------------------------------

  0_Shared/            modules that are only imported, never run directly
                       (manifest, extraction engine, basin join, transforms)
  1_GEMS-GER_Monthly/  01_gems_weekly_to_monthly.py
  2_Feature_Stores/    01_build_static_store.py
                       02_build_dynamic_store.py
                       03_extraction_quality_report.py   (optional QC)
  3_Feature_Metadata/  01_build_features_meta.py
                       feature_labels_overrides.csv      (labels, units)

Run order (each command from the folder of the script):

  1. 02_Processing_Scripts (all datasets)       -> 01_Input_Global/3_Final/
  2. python 01_gems_weekly_to_monthly.py        -> monthly GWL per well
  3. python 01_build_static_store.py            -> static store
  4. python 02_build_dynamic_store.py           -> dynamic store
  5. python 03_extraction_quality_report.py     -> quality report (optional)
  6. python 01_build_features_meta.py           -> feature metadata tables

The metadata script reads the column headers of the stores, so it must run
after steps 3 and 4. The store builders accept --limit N for a quick dry run
with the first N wells, and --root <path> to set the project root explicitly.
Each folder contains a README with details on its scripts.


--------------------------------------------------------------------------------
2. PROJECT FOLDER LAYOUT (inputs and outputs)
--------------------------------------------------------------------------------

All paths are relative to the project root, which is derived as the folder
two levels above each script (or set with --root / the environment variable
GWL_ROOT). The paths are defined in the PATHS block at the top of each script.

  <project root>/
    01_Input_Global/3_Final/<Dataset>/<file>.nc     INPUT: gridded datasets
        HydroSHEDS_RIVERS_BASINS/
            basin_era5_lev06/, basin_era5_lev07/    INPUT: per-basin ERA5-Land series
            basin_terraclimate_lev06/, ..._lev07/   INPUT: per-basin TerraClimate series
            metadata_lev06.csv, metadata_lev07.csv  INPUT: HydroBASINS attributes
    01_Input_Global/1_Original/HydroSHEDS_RIVERS_BASINS/
            hybas_eu_lev06_v1c/, hybas_eu_lev07_v1c/  INPUT: basin polygons
    02_Processing_Scripts/0_Shared/nc_io.py         shared NetCDF helper (imported)
    02_Dataframes_Processing_Scripts/               THIS FOLDER
    03_Dataframes/
      00_Extraction_Quality/                        OUTPUT: quality report (qc_*)
      01_Dynamic_Features/
          01_GEMS-GER_weekly/MW_<id>.csv            INPUT: weekly GEMS-GER series
          02_GEMS-GER_monthly/MW_<id>_monthly.csv   monthly GWL per well
                                                    (from 1_GEMS-GER_Monthly)
          03_global_monthly/MW_<id>.csv             OUTPUT: DYNAMIC STORE
          03_global_provenance.csv                  OUTPUT: sampling provenance
          03_global_build_report.csv                OUTPUT: build report
      02_Static_Features/
          01_GEMS-GER/02_adapted_well_metadata.csv  INPUT: well metadata incl. lon/lat
          01_GEMS-GER/01_original_well_metadata.csv INPUT: GEMS-GER static features
          02_global/static_features.csv             OUTPUT: STATIC STORE
          02_global/_provenance.csv                 OUTPUT: sampling provenance
          02_global/_build_report.csv               OUTPUT: NaN count per well
          02_global/_feature_nan_summary.csv        OUTPUT: NaN count per feature
      03_Metadata/
          features_meta_global.csv                  OUTPUT: metadata, own features
          features_meta_GEMS-GER.csv                OUTPUT: metadata, GEMS-GER features

The well inputs (well metadata and weekly groundwater levels, MW_<id>) come
from the GEMS-GER benchmark dataset. The monthly files MW_<id>_monthly.csv
are derived from the weekly files by 1_GEMS-GER_Monthly/: every weekly value
is assigned to the calendar month in which it falls. Flux drivers
(precipitation, evapotranspiration, runoff, snowfall) are summed per month,
all other variables are averaged. The column GWL in these files is therefore
the monthly mean groundwater level (m).


--------------------------------------------------------------------------------
3. CONVENTIONS
--------------------------------------------------------------------------------

Column names   <dataset_key>__<variable>[_<suffix>], e.g. era5l__tp,
               soilgrids__clay, gasy__sy_0p3 (suffix = depth 0.3 m).
               The dataset keys are defined in 0_Shared/dataset_manifest.py.
               Exceptions (engineered or joined groups):
                 C3S__...             land cover features (manifest key lcc)
                 basin_levX__...      shared basin identity (HYBAS_ID,
                                      UP_AREA, COAST), static store only
                 basin_era5l_levX__   ERA5-Land aggregated per basin
                 basin_terra_levX__   TerraClimate aggregated per basin
               Units and long names are NOT part of the column name; they are
               listed in the feature metadata table.
Missing values NaN. The declared _FillValue of every NetCDF variable is read,
               so integer NoData codes (e.g. 255 or 0) are never used as a
               real class.
dtypes         features float32; categorical features as nullable integers
               (class codes, <NA> for missing).
Time           month start, YYYY-MM-01.
Well ID        MW_<id> (column MW_ID in the static store, file name
               MW_<id>.csv in the dynamic store, column well_id in the
               provenance files).


--------------------------------------------------------------------------------
4. STRUCTURE OF THE STORES
--------------------------------------------------------------------------------

4.1 Dynamic store  03_global_monthly/MW_<id>.csv

  One row per month in which the well has a groundwater level value. Columns:
    time                     month start
    GWL                      target: monthly mean groundwater level
    era5l__<var>             ERA5-Land monthly variables
    terra__<var>             TerraClimate monthly variables (14)
    twsa__TWS, twsa__TWS_variance
    C3S__LCC, C3S__LCC_previous, C3S__LCC_changed, C3S__LCC_stable,
    C3S__months_since_LCC_change                    (see section 6)
    basin_era5l_lev06__{tp,ro,ssro,sro}, basin_era5l_lev07__{...}
    basin_terra_lev06__<14 TerraClimate variables>, basin_terra_lev07__{...}
  Values are NaN where a dataset does not cover the month (twsa after
  2020-12) or where the well was dropped for that dataset.

4.2 Static store  02_global/static_features.csv

  One row per well. Columns:
    well metadata (passed through from 02_adapted_well_metadata.csv):
      MW_ID, Proj_ID, Operator, Elevation, Depth, UpFilter, LoFilter,
      ScrLength, AquiferMed, PreState, Easting_3035, Northing_3035, lon, lat
    raster features (point sampling):
      era5l_static__<var>, soilgrids__<var>, gasy__sy_<depth>, glim__<var>,
      whymap__hygeo2, hydsheds_terrain__<var>, hydrivers_dist__distance_class_<n>
    basin attributes:
      basin_levX__HYBAS_ID (identifier, not a model feature),
      basin_levX__UP_AREA, basin_levX__COAST,
      basin_era5l_levX__ / basin_terra_levX__ coverage_frac, basin_area_m2,
      covered_area_m2 (grid-dependent, therefore per product)

4.3 Provenance files

  One row per well. For every point-sampled dataset <key>:
    <key>__sample_dist_km   distance from the well to the sampled cell centre
    <key>__rescued          True if a neighbouring cell was used
    <key>__outcome          home | rescued | nearest_global | no_valid | dropped


--------------------------------------------------------------------------------
5. HOW TO BUILD A MODEL DATASET
--------------------------------------------------------------------------------

  For each well: take 03_global_monthly/MW_<id>.csv (rows = months) and join
  the single row of static_features.csv with the same MW_ID to every month.
  Target = GWL, time key = time, well key = MW_<id> / MW_ID. The provenance
  files can be joined on well_id to filter by sampling quality.

  The stores contain ALL features, including NaN. Which columns a model uses
  is decided by the experiment configuration of the machine learning stage
  (04_Machine_Learning/02_monthly: experiments.yaml, resolve_experiments.py),
  which reads the feature metadata tables of 3_Feature_Metadata. Useful
  filters:
    - experiments with twsa__* are limited to months up to 2020-12
    - wells with twsa__outcome == 'dropped' have no TWSA value
    - basin_*__coverage_frac below a threshold marks poorly covered
      coastal basins
    - _feature_nan_summary.csv shows columns that are (almost) always NaN,
      e.g. glim__permeability_permafrost in Germany


--------------------------------------------------------------------------------
6. LAND COVER FEATURES (engineered, prefix C3S__)
--------------------------------------------------------------------------------

From the yearly class lccs_class (0_Shared/transforms.py):
  C3S__LCC                      class of the current year (categorical)
  C3S__LCC_previous             class of the previous year (categorical)
  C3S__LCC_changed              1 if the class changed against the previous year
  C3S__LCC_stable               1 if only one class occurs in the whole record
  C3S__months_since_LCC_change  months since the last change (counted from
                                January of the change year); -1 = never changed
The record covers 1992-2022; 1991 is back-filled from 1992. Values are
constant within a year.


--------------------------------------------------------------------------------
7. SOFTWARE REQUIREMENTS
--------------------------------------------------------------------------------

Python 3 with numpy, pandas, xarray, netCDF4. The basin join additionally
needs geopandas (without it, or without the basin inputs, the stores are
built without basin columns). tqdm is optional (progress bar).

================================================================================
3_Feature_Metadata  -  Feature metadata tables
================================================================================

01_build_features_meta.py
  Reads the column headers of the feature stores and writes one metadata row
  per feature. Run it after the stores have been built (2_Feature_Stores/).
  Two separate tables are written, one per store profile:
    features_meta_global.csv     own features (global datasets)
    features_meta_GEMS-GER.csv   features of the published GEMS-GER dataset
                                 (described by hand in GEMS_METADATA)
  Input : 03_Dataframes/01_Dynamic_Features/03_global_monthly/  (header of the
          first CSV)
          03_Dataframes/02_Static_Features/02_global/static_features.csv
          03_Dataframes/01_Dynamic_Features/02_GEMS-GER_monthly/
          03_Dataframes/02_Static_Features/01_GEMS-GER/01_original_well_metadata.csv
          feature_labels_overrides.csv  (this folder)
  Output: 03_Dataframes/03_Metadata/  (semicolon-separated CSV)
  Run   : python 01_build_features_meta.py

  Columns of the metadata tables:
    feature_raw_name  column name in the store
    source_system     global | gems
    type              dynamic | static
    group             column prefix before "__" (C3S__ -> lcc)
    var_group         coarse group for SHAP aggregation and plot colours
    processing        lag_roll (lag / rolling features are built later in the
                      ML scripts) | passthrough
    categorical       True for class codes
    temporal, time_start, time_end   temporal coverage
    plot_name         readable label for plots
    unit              unit ('?' if not given in the overrides)

  Settings in the script: DYNAMIC_LAG_GROUPS, GROUP_TEMPORAL,
  CATEGORICAL_GLOBAL, DROP_GLOBAL_META (columns that are not features),
  GEMS_METADATA.

feature_labels_overrides.csv
  Semicolon-separated table with readable labels, var_group and unit per
  feature. Lines starting with # are comments. Columns:
    match_type     exact  = only this column name
                   base   = this column and its later _lag / _rmean children
                   prefix = every column starting with key (longest wins)
    key            column name or prefix
    display_label  plot label
    var_group      group for SHAP aggregation / colours
    unit           unit written to the metadata table
    note           free text
  A new feature needs a line here to get a label and a unit.

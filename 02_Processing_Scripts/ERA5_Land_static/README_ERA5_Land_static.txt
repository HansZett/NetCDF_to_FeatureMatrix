================================================================================
ERA5_Land_static  -  ERA5-Land invariant (static) fields
================================================================================

1. DATA SOURCE
--------------------------------------------------------------------------------
Dataset   : ERA5-Land invariant fields (Copernicus Climate Data Store)
Reference : Munoz Sabater, J. (2019): ERA5-Land hourly data from 1950 to
            present. Copernicus Climate Change Service (C3S) Climate Data
            Store (CDS).
Class tables of slt, tvl and tvh: ECMWF parameter documentation
            (documentation_era5_static.txt).

One NetCDF file per invariant field is stored in
01_Input_Global/1_Original/ERA5_Land_static/.


2. SCRIPTS (run order)
--------------------------------------------------------------------------------
01_era5land_static_merge.py
  1. Reads every *.nc file in 1_Original/ERA5_Land_static/.
  2. Renames coordinates to lat / lon, removes time-like dimensions, converts
     longitudes from 0..360 to -180..180 if needed.
  3. Clips to the study area (47.0-55.5 N, 5.5-15.5 E).
  4. Merges all fields into one dataset (join="override": the coordinates of
     the first file are used for all files; 02b checks that this is safe).
  5. Drops the geopotential z.
  6. Categorical fields slt, tvl, tvh: rounds to integer class codes, keeps
     only documented codes (membership test), stores them as uint8 with
     fill value 255 and flag_values / flag_meanings.
       - code 0 is kept for tvl and tvh (absence of the cover type)
       - code 0 is set to missing for slt (cell without soil)
       - the no-surface-vegetation codes 8, 12, 14, 15 are not accepted
         (switch: KEEP_NO_SURFACE_VEG)
  7. Continuous fields: float32, fill value NaN; units of cl, cvl, cvh, lsm,
     glm (fraction, "1") and dl (m) are set explicitly.

  Input : 01_Input_Global/1_Original/ERA5_Land_static/*.nc
  Output: 01_Input_Global/3_Final/ERA5_Land_static/ERA5_Land_static_Germany.nc
  Run   : python 01_era5land_static_merge.py

02a_check_era5land_static_classes.py   (diagnostic, writes nothing)
  Prints the class histogram of slt, tvl and tvh before the merge (source
  files) and after the merge (final file), and marks which codes are kept by
  the merge script and which codes are not in the documented class tables.
  Run   : python 02a_check_era5land_static_classes.py [--root <project root>]

02b_check_era5land_static_grid_alignment.py   (diagnostic, writes nothing)
  Checks whether all source files share one grid (required for
  join="override" in the merge) and compares the land-sea mask with the
  code 0 of slt, tvl and tvh on the merged grid.
  Run   : python 02b_check_era5land_static_grid_alignment.py [--root <project root>]


3. OUTPUT FILE  ERA5_Land_static_Germany.nc
--------------------------------------------------------------------------------
Grid      : 0.1 deg, study area, lat descending
Variables : all invariant fields of the input files except z
  slt  soil type          uint8, fill 255, classes 1-7
       (coarse, medium, medium_fine, fine, very_fine, organic, tropical_organic)
  tvl  low vegetation     uint8, fill 255, classes 0, 1, 2, 7, 9, 10, 11, 13,
                          16, 17, 20 (0 = no low vegetation)
  tvh  high vegetation    uint8, fill 255, classes 0, 3, 4, 5, 6, 18, 19
                          (0 = no high vegetation)
  other fields            float32, fill NaN
The class names are stored in the flag_meanings attribute of each variable.

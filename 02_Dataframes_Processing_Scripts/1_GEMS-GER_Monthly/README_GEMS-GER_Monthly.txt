================================================================================
1_GEMS-GER_Monthly  -  Weekly GEMS-GER series to monthly values
================================================================================

The GEMS-GER benchmark dataset provides one weekly CSV per well (groundwater
level GWL, quality flag GWL_flag and 15 dynamic drivers). The feature stores
and the monthly ML models need monthly values. This script creates them.

Rule: every weekly value is assigned to the calendar month of its date;
flux variables are then summed per month, all other variables are averaged.


01_gems_weekly_to_monthly.py
  1. Reads each weekly CSV (first column = date YYYY-MM-DD).
  2. Drops GWL_flag (EXCLUDE_COLS).
  3. Assigns every row to the calendar month of its date and aggregates per
     month (NaN ignored):
       sum  : HYRAS_pr, DWD_evapo_p, DWD_evapo_r, DWD_evapo_fao, ERA5_sro,
              ERA5_ssro, ERA5_sf  (SUM_COLS)
       mean : GWL, HYRAS_tas, HYRAS_tasmax, HYRAS_tasmin, HYRAS_hurs,
              DWD_soil_moist, DWD_soil_temp5cm, ERA5_sdwe, ERA5_sm
  4. Removes months without a valid GWL value (DROP_MONTHS_WITHOUT_GWL).
  5. Adds time_days (days since 1970-01-01 of the month start).
  Input : 03_Dataframes/01_Dynamic_Features/01_GEMS-GER_weekly/<well>.csv
  Output: 03_Dataframes/01_Dynamic_Features/02_GEMS-GER_monthly/<well>_monthly.csv
          columns: time (YYYY-MM-01), time_days, GWL, <drivers>
  Run   : python 01_gems_weekly_to_monthly.py
          python 01_gems_weekly_to_monthly.py --out-dir "<folder>"
          Existing files are skipped unless --overwrite is given.

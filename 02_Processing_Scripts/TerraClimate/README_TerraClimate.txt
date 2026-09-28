================================================================================
TerraClimate  -  TerraClimate v1.1 monthly climate and water balance
================================================================================

1. DATA SOURCE
--------------------------------------------------------------------------------
Dataset   : TerraClimate v1.1 (with ERA5 forcing)
Provider  : Climatology Lab, University of California Merced
Reference : Abatzoglou et al. (2018), Scientific Data, doi:10.1038/sdata.2017.191
Access    : THREDDS NetCDF Subset Service (NCSS),
            http://thredds.northwestknowledge.net:8080/thredds/ncss/
Grid      : 1/24 deg (0.04167 deg, ~4.5 km), monthly


2. SCRIPTS (run order)
--------------------------------------------------------------------------------
00_inspect_terraclimate.py   (diagnostic, writes nothing)
  Prints the structure of one downloaded file (set INPUT_FILE in the PATHS
  block to choose the file).
  Run   : python 00_inspect_terraclimate.py

01_terraclimate_download.py
  Downloads the 14 variables for the study area (5.5-15.5 E, 47.0-55.5 N)
  and the period YEAR_START-YEAR_END (1991-2022). The subset is cut on the
  server. Existing files are skipped. The script asks for confirmation
  (y/n) before downloading.
  Output: 01_Input_Global/1_Original/TerraClimate/
          TerraClimate_<variable>_1991-2022_AOI.nc   (one file per variable)
  Run   : python 01_terraclimate_download.py

02_terraclimate_homogenize_merge.py
  1. Reads all TerraClimate_*.nc files.
  2. Decodes the time axis (days since 1900-01-01) to first-of-month dates.
  3. Renames the variables and sets long_name and units.
  4. Merges all variables and writes them with nc_io.save_nc (float32,
     fill NaN, lat descending, time as days since 1970-01-01).
  Input : 01_Input_Global/1_Original/TerraClimate/TerraClimate_*.nc
  Output: 01_Input_Global/3_Final/TerraClimate/TerraClimate.nc
  Run   : python 02_terraclimate_homogenize_merge.py

TerraClimate.nc is also the input of
HydroSHEDS/05b_hydrobasins_terraclimate_basin_means.py.


3. VARIABLES IN TerraClimate.nc
--------------------------------------------------------------------------------
Variable                NCSS name  Description                     Unit
----------------------  ---------  ------------------------------  --------
ET_actual               aet        actual evapotranspiration       mm/month
ET_potential            pet        potential evapotranspiration    mm/month
water_deficit           def        climate water deficit           mm/month
precipitation           ppt        precipitation                   mm/month
runoff                  q          runoff                          mm/month
soil_moisture           soil       soil moisture                   mm
solar_radiation         srad       solar radiation                 W/m2
snow_water              swe        snow water equivalent           mm
temp_max                tmax       maximum temperature             degC
temp_min                tmin       minimum temperature             degC
vapor_pressure          vap        vapor pressure                  kPa
vapor_pressure_deficit  vpd        vapor pressure deficit          kPa
drought_index_pdsi      PDSI       Palmer Drought Severity Index   unitless
wind_speed              ws         wind speed                      m/s

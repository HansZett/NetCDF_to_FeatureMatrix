================================================================================
ERA5_Land  -  ERA5-Land monthly averaged reanalysis (dynamic variables)
================================================================================

1. DATA SOURCE
--------------------------------------------------------------------------------
Dataset    : ERA5-Land monthly averaged data from 1950 to present
Collection : reanalysis-era5-land-monthly-means
Provider   : ECMWF / Copernicus Climate Data Store (CDS)
URL        : https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means
Reference  : Munoz-Sabater et al. (2021), ESSD,
             https://doi.org/10.5194/essd-13-4349-2021
Licence    : CC-BY licence (version 1); Terms of use of the Copernicus Climate
             Data Store (version 11)

CDS request (request ID f1229e5a-d42a-416f-8054-4a105ef7edbf, 2026-05-04):
  product_type : monthly_averaged_reanalysis
  area (N/W/S/E): 55.5 / 5.5 / 47.0 / 15.5
  years        : 1990 - 2022, all 12 months, time 00:00
  variables    : 43 (see section 4)
  format       : NetCDF, unarchived
The request can be repeated in the CDS web form with these settings. The
downloaded file is stored as 01_Input_Global/1_Original/ERA5_Land/era5_1991_2022_big.nc.
The spatial clip was done by the CDS during the download.


2. SCRIPTS
--------------------------------------------------------------------------------
01_era5_land_dynamic_preprocess.py
  Cleans the raw file and writes the final NetCDF.
    1. Renames valid_time -> time, latitude -> lat, longitude -> lon.
    2. Drops the CDS helper coordinates expver and number.
    3. Prints the lat / lon / time extent as a check.
    4. Writes with nc_io.save_nc: float32, _FillValue NaN, zlib,
       lat descending / lon ascending, CF global attributes.

  Input : 01_Input_Global/1_Original/ERA5_Land/era5_1991_2022_big.nc
  Output: 01_Input_Global/3_Final/ERA5_Land/era5_land.nc
  Run   : python 01_era5_land_dynamic_preprocess.py

era5_land.nc is also the input of
HydroSHEDS/05a_hydrobasins_era5_basin_means.py.


3. OUTPUT FILE  era5_land.nc
--------------------------------------------------------------------------------
Dimensions : time (monthly), lat (descending), lon (ascending)
Grid       : regular 0.1 deg x 0.1 deg (ERA5-Land native resolution)
Variables  : 43, all float32, _FillValue NaN (ocean cells are NaN, because
             ERA5-Land is a land-only product)

Units: temperatures are in K. Accumulated variables (precipitation,
evaporation, runoff, radiation and heat fluxes) are, in the monthly averaged
product, daily-mean accumulations expressed per day (for example m of water
equivalent per day), not monthly totals.


4. VARIABLES
--------------------------------------------------------------------------------
Name     Long name                                       Unit
-------  ----------------------------------------------  -----------------
Temperature
d2m      2 metre dewpoint temperature                    K
t2m      2 metre temperature                             K
skt      Skin temperature                                K
stl1-4   Soil temperature level 1-4                      K
Snow
asn      Snow albedo                                     0-1
snowc    Snow cover                                      %
rsn      Snow density                                    kg m-3
sde      Snow depth                                      m
sd       Snow depth water equivalent                     m of water eq.
sf       Snowfall                                        m of water eq.
smlt     Snowmelt                                        m of water eq.
tsn      Temperature of snow layer                       K
es       Snow evaporation                                m of water eq.
Soil / water
src      Skin reservoir content                          m of water eq.
swvl1-4  Volumetric soil water layer 1-4                 m3 m-3
Albedo
al       Forecast albedo                                 0-1
Radiation and heat fluxes
slhf     Surface latent heat flux                        J m-2
ssr      Surface net solar radiation                     J m-2
str      Surface net thermal radiation                   J m-2
sshf     Surface sensible heat flux                      J m-2
ssrd     Surface solar radiation downwards               J m-2
strd     Surface thermal radiation downwards             J m-2
Evaporation
evabs    Evaporation from bare soil                      m of water eq.
evaow    Evaporation from open water (excl. oceans)      m of water eq.
evatc    Evaporation from the top of canopy              m of water eq.
evavt    Evaporation from vegetation transpiration       m of water eq.
pev      Potential evaporation                           m
e        Total evaporation                               m of water eq.
Runoff
ro       Runoff                                          m
ssro     Sub-surface runoff                              m
sro      Surface runoff                                  m
Wind, pressure, precipitation
u10      10 metre U wind component                       m s-1
v10      10 metre V wind component                       m s-1
sp       Surface pressure                                Pa
tp       Total precipitation                             m
Vegetation
lai_hv   Leaf area index, high vegetation                m2 m-2
lai_lv   Leaf area index, low vegetation                 m2 m-2

Soil layers: level 1 = 0-7 cm, level 2 = 7-28 cm, level 3 = 28-100 cm,
level 4 = 100-289 cm (stl1-4 and swvl1-4).

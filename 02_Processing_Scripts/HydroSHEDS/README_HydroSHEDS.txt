================================================================================
HydroSHEDS  -  Terrain, drainage, rivers and HydroBASINS catchments
================================================================================

1. DATA SOURCES
--------------------------------------------------------------------------------
HydroSHEDS v1.1 core layers (GeoTIFF, 3 arc-seconds, Europe)
  eu_dem_3s  void-filled, hydrologically conditioned DEM (SRTM based)
  eu_acc_3s  flow accumulation (number of upstream cells)
  eu_dir_3s  flow direction (D8, ESRI convention)
  Citation : Lehner, B., Verdin, K., Jarvis, A. (2008): New global hydrography
             derived from spaceborne elevation data. Eos, Transactions
             89(10): 93-94.
  Download : https://www.hydrosheds.org
  License  : HydroSHEDS License Agreement (free for non-commercial and
             commercial use)

HydroRIVERS v1.0 (Europe, shapefile)     HydroRIVERS_v10_eu.shp
HydroBASINS v1c (Europe, levels 6 and 7)  hybas_eu_lev06_v1c.shp,
                                          hybas_eu_lev07_v1c.shp
  Citation : Lehner, B., Grill, G. (2013): Global river hydrography and
             network routing: baseline data and new approaches to study the
             world's large river systems. Hydrological Processes, 27(15):
             2171-2186. doi:10.1002/hyp.9740

All files are stored in 01_Input_Global/1_Original/HydroSHEDS_RIVERS_BASINS/
(HydroRIVERS in HydroRIVERS_v10_eu_shp/, HydroBASINS in hybas_eu_lev0X_v1c/).
Outputs are written to .../2_Intermediate/HydroSHEDS_RIVERS_BASINS/ and
.../3_Final/HydroSHEDS_RIVERS_BASINS/.


2. SCRIPTS (run order)
--------------------------------------------------------------------------------
Part A: gridded terrain layers (3 arc-seconds)

01_hydrosheds_raw_to_nc.py
  Reads DEM, ACC and DIR, cuts them to the study area (5.5-15.5 E,
  47.0-55.5 N) on the DEM grid, replaces the nodata values and writes one
  NetCDF file.
  Output: 2_Intermediate/.../eu_hydrosheds_raw.nc   (dem, acc, dir)

02a_hydrosheds_terrain_derivatives.py      (needs 01)
  slope and aspect from the DEM gradient, with the cell width corrected
  for latitude (dy = R * dlat, dx = R * dlon * cos(lat), R = 6 371 000 m).
  TWI = ln(a / tan(beta)), a = upstream area in ha (acc x cell area),
  beta = slope with a minimum of 0.1 deg (SLOPE_MIN_DEG).
  Output: 2_Intermediate/.../eu_hydrosheds_derived.nc   (slope, aspect, twi)

02b_hydrosheds_hand.py                     (needs 01)
  Height Above Nearest Drainage. Drainage cells: acc >= 1000 upstream cells
  (ACC_THRESHOLD). The drainage elevation is propagated upstream along the
  D8 directions until no further cell changes (at most MAX_ITER = 5000
  iterations). HAND = DEM - drainage elevation, negative values set to 0.
  Output: 2_Intermediate/.../eu_hydrosheds_hand.nc   (hand)

03_hydrosheds_merge.py                     (needs 01, 02a, 02b)
  Merges the three intermediate files into one final file.
  Output: 3_Final/.../hydrosheds.nc

04_hydrorivers_distance.py                 (independent)
  Rasterizes the HydroRIVERS segments per flow order class ORD_FLOW 3-10
  on a 15 arc-second grid and computes the Euclidean distance to the
  nearest river of each class. Classes without rivers in the study area
  are skipped.
  Output: 3_Final/.../HydroRIVERS_distance.nc

Part B: HydroBASINS catchment time series (levels 6 and 7)

05a_hydrobasins_era5_basin_means.py
  (needs 3_Final/ERA5_Land/era5_land.nc from ERA5_Land/)
  Rasterizes the basins on the ERA5-Land grid and computes, per basin and
  month, the area-weighted mean of tp, ro, ssro and sro in mm/day. Only
  cells with valid ERA5-Land data are used (coastal basins are partly
  covered).
  Outputs (3_Final/.../):
    basin_era5_lev06/<HYBAS_ID>.csv, basin_era5_lev07/<HYBAS_ID>.csv
    metadata_lev06.csv, metadata_lev07.csv   (HydroBASINS attributes)
    basin_mask_lev06.nc, basin_mask_lev07.nc (basin-ID raster, 0 = none)
    README_units.txt                         (units and coverage columns)

05b_hydrobasins_terraclimate_basin_means.py
  (needs 3_Final/TerraClimate/TerraClimate.nc from TerraClimate/)
  Same as 05a for the 14 TerraClimate variables on the TerraClimate grid,
  in their native units. All columns carry the prefix terra_.
  Outputs (3_Final/.../):
    basin_terraclimate_lev06/<HYBAS_ID>.csv, basin_terraclimate_lev07/...
    metadata_terraclimate_lev0X.csv, basin_mask_terraclimate_lev0X.nc
    README_terraclimate_units.txt

06_hydrobasins_level_statistics.py          (optional, needs 05a and 05b)
  Descriptive statistics that support the choice of HydroBASINS levels 6
  and 7: basin sizes, grid cells per basin, retained share of the spatial
  variance of long-term mean precipitation, and how often both levels give
  the same polygon. The well statistics additionally need
  03_Dataframes/02_Static_Features/02_global/static_features.csv
  (from 02_Dataframes_Processing_Scripts) and are skipped without it.
  Output: 3_Final/.../basin_level_stats/  (CSV tables and report.txt)

All scripts are run with: python <script name>.py


3. VARIABLES IN hydrosheds.nc
--------------------------------------------------------------------------------
Variable  dtype    Unit    Description
--------  -------  ------  ------------------------------------------------------
dem       float32  m       void-filled elevation
acc       float32  count   flow accumulation (number of upstream cells)
dir       uint8    -       D8 flow direction, fill 255:
                           1 E, 2 SE, 4 S, 8 SW, 16 W, 32 NW, 64 N, 128 NE
slope     float32  degree  terrain slope (0 = flat)
aspect    float32  degree  0 = north, clockwise; NaN on flat cells
                           (slope < 0.001 deg)
twi       float32  1       topographic wetness index ln(a / tan(beta))
hand      float32  m       height above nearest drainage (>= 0); NaN = ocean or
                           no drainage outlet inside the extent

HydroRIVERS_distance.nc: distance_class_<N> (float32, degree) for each
ORD_FLOW class N present in the study area.

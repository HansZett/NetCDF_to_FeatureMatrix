================================================================================
02_Processing_Scripts  -  Preprocessing of the global Earth observation datasets
================================================================================

Thesis: Evaluating Machine Learning-Based Groundwater Level Predictions Using
        Globally Available Earth Observation Data

This folder contains the scripts that turn the downloaded global datasets
into standardized NetCDF (and CSV) products for the study area. These
products are the input for the feature extraction and dataframe scripts in
02_Dataframes_Processing_Scripts (separate README there).

Study area (all datasets) : 5.5 - 15.5 deg E, 47.0 - 55.5 deg N
                            (Germany and surroundings)
Coordinate system         : EPSG:4326 (WGS84, lat/lon)


--------------------------------------------------------------------------------
1. EXPECTED PROJECT FOLDER LAYOUT
--------------------------------------------------------------------------------

All paths in the scripts are relative to the project root. The project root
is derived automatically as the folder two levels above each script, so the
scripts work on any computer as long as this layout is kept:

  <project root>/
  |-- 01_Input_Global/
  |   |-- 1_Original/<Dataset>/        downloaded raw data (never modified)
  |   |-- 2_Intermediate/<Dataset>/    intermediate products
  |   `-- 3_Final/<Dataset>/           final products (input for 02_Dataframes...)
  |-- 02_Processing_Scripts/           THIS FOLDER
  |-- 02_Dataframes_Processing_Scripts/
  `-- 03_Dataframes/                   outputs of the dataframe pipeline

The raw data are not part of this folder. They have to be downloaded from the
sources named in the dataset READMEs and placed in 01_Input_Global/1_Original/.


--------------------------------------------------------------------------------
2. FOLDER STRUCTURE OF THIS DIRECTORY
--------------------------------------------------------------------------------

  0_Shared/          shared helper module nc_io.py (NetCDF conventions + writer)
  C3S_LCC/            ESA CCI / C3S land cover, annual, 300 m
  ERA5_Land/          ERA5-Land monthly reanalysis (dynamic variables)
  ERA5_Land_static/   ERA5-Land invariant (static) fields
  GASY/               Global Average Specific Yield (SoilGrids variant)
  GLiM_GLHYMPS/       lithology (GLiM) and permeability/porosity (GLHYMPS)
  HydroSHEDS/         DEM, flow accumulation/direction, terrain indices, HAND,
                      distance to rivers, HydroBASINS basin time series
  SoilGrids/          SoilGrids v2.0 soil properties (via Google Earth Engine)
  TerraClimate/       TerraClimate v1.1 monthly climate and water balance
  TWSA/               GRACE gap-filled terrestrial water storage anomaly
  WHYMAP/             WHYMAP groundwater recharge classes (HYGEO2)

Every dataset folder contains its own README_<Dataset>.txt with the data
source, the scripts in run order, inputs, outputs and variables.


--------------------------------------------------------------------------------
3. CONVENTIONS USED IN ALL SCRIPTS
--------------------------------------------------------------------------------

Script names
  NN_<dataset>_<action>.py. The number NN gives the run order inside a
  folder. Letters (02a, 02b) mark scripts that are independent of each other
  and can run in any order. Scripts with "inspect" or "check" in the name are
  optional diagnostics. They only print information and write no files.

Script header
  Every script starts with a docstring that states its purpose, the input
  and output files, the folder location and the command to run it.

PATHS block
  Directly below the imports, every script has a block titled
  "PATHS (relative to the project root; adjust here if the layout changes)".
  All input and output paths are defined only there. To use another folder
  layout, only this block has to be edited. A few diagnostic scripts also
  accept the project root as a command line argument.

CONFIGURATION block
  Settings such as the bounding box, thresholds, class tables or variable
  lists are collected in a CONFIGURATION block below the PATHS block.

Shared NetCDF convention (0_Shared/nc_io.py)
  All NetCDF outputs are written through nc_io.save_nc. This guarantees for
  every dataset:
    - coordinate names lat / lon / time
    - latitude descending (north to south), longitude ascending
    - an explicit _FillValue per variable (NaN for floats, an explicit
      integer for categorical variables, e.g. 0 or 255)
    - zlib compression (level 4)
    - CF-1.8 global attributes (title, source, references, history, ...)
    - outputs placed in 1_Original / 2_Intermediate / 3_Final tiers
  nc_io.py is also imported by the scripts in 02_Dataframes_Processing_Scripts.
  Its folder name (0_Shared) and file name must therefore not be changed.

How to run
  Open a terminal in the folder of the script and run
      python <script name>.py
  Each script can be run on its own, as long as its inputs exist.


--------------------------------------------------------------------------------
4. RUN ORDER AND DEPENDENCIES BETWEEN DATASETS
--------------------------------------------------------------------------------

Most datasets are independent of each other. Only the HydroBASINS basin
time series depend on other datasets:

  ERA5_Land/01_era5_land_dynamic_preprocess.py
      --> HydroSHEDS/05a_hydrobasins_era5_basin_means.py
  TerraClimate/01_terraclimate_download.py
      --> TerraClimate/02_terraclimate_homogenize_merge.py
      --> HydroSHEDS/05b_hydrobasins_terraclimate_basin_means.py
  HydroSHEDS/05a + 05b
      --> HydroSHEDS/06_hydrobasins_level_statistics.py
          (the well statistics in 06 additionally need
           03_Dataframes/02_Static_Features/02_global/static_features.csv
           from 02_Dataframes_Processing_Scripts; without it they are skipped)

Recommended overall order: first all dataset folders (any order, ERA5_Land
and TerraClimate before HydroSHEDS 05a/05b), then the scripts in
02_Dataframes_Processing_Scripts, then optionally HydroSHEDS 06.


--------------------------------------------------------------------------------
5. FINAL PRODUCTS (01_Input_Global/3_Final/)
--------------------------------------------------------------------------------

  C3S_LCC/c3s_lcc.nc                              annual land cover, 300 m
  ERA5_Land/era5_land.nc                          monthly, 0.1 deg, 43 variables
  ERA5_Land_static/ERA5_Land_static_Germany.nc    static fields, 0.1 deg
  GASY/GASY.nc                                    specific yield, 7 depths, ~1 km
  GLiM_GLHYMPS/GLiM_GLHYMPS.nc                    lithology, porosity,
                                                  permeability, ~1 km
  HydroSHEDS_RIVERS_BASINS/hydrosheds.nc          dem, acc, dir, slope, aspect,
                                                  twi, hand, 3 arc-seconds
  HydroSHEDS_RIVERS_BASINS/HydroRIVERS_distance.nc  distance to rivers,
                                                  15 arc-seconds
  HydroSHEDS_RIVERS_BASINS/basin_*                basin masks and per-basin
                                                  monthly CSV time series
  SoilGrids/soilgrids.nc                          soil properties, ~250 m
  TerraClimate/TerraClimate.nc                    monthly, 1/24 deg, 14 variables
  TWSA/TWSA.nc                                    monthly TWS anomaly, 0.5 deg
  WHYMAP/WHYMAP.nc                                recharge class, 1/240 deg


--------------------------------------------------------------------------------
6. SOFTWARE REQUIREMENTS
--------------------------------------------------------------------------------

The scripts were run with Python 3.12. Required packages:

  numpy, pandas, xarray, netCDF4 (NetCDF backend for xarray), scipy,
  rasterio, rioxarray, geopandas, shapely, pyproj

The two files in SoilGrids/ ending in .js are Google Earth Engine scripts.
They run in the Earth Engine Code Editor, not in Python.

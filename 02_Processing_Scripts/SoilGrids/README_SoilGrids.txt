================================================================================
SoilGrids  -  SoilGrids v2.0 soil properties, depth-weighted mean 0-200 cm
================================================================================

1. DATA SOURCE
--------------------------------------------------------------------------------
Dataset   : SoilGrids 250 m v2.0
Provider  : ISRIC - World Soil Information
Access    : Google Earth Engine (GEE)
Reference : Poggio et al. (2021). SOIL, 7, 217-240.
            https://doi.org/10.5194/soil-7-217-2021


2. SCRIPTS (run order)
--------------------------------------------------------------------------------
00a_GEE_export_soil_properties.js   (Google Earth Engine Code Editor)
  Computes the depth-weighted mean 0-200 cm of bulk density, coarse
  fragments, clay, sand, silt and SOC, converts the SoilGrids integer units
  (see table) and exports one GeoTIFF per variable to Google Drive
  (250 m, EPSG:4326).

00b_GEE_export_water_content.js     (Google Earth Engine Code Editor)
  Same for the volumetric water content at 10 kPa, 33 kPa and 1500 kPa.

  After both exports, the 9 GeoTIFFs are copied to
  01_Input_Global/1_Original/SoilGrids/.

01_soilgrids_build_nc.py
  Reads the 9 GeoTIFFs, multiplies the three VWC variables by 1000 (see
  note below), prints a plausibility check of the value ranges, attaches
  metadata and writes one NetCDF file with nc_io.save_nc (float32,
  fill value NaN).

  Input : 01_Input_Global/1_Original/SoilGrids/*.tif
  Output: 01_Input_Global/3_Final/SoilGrids/soilgrids.nc
  Run   : python 01_soilgrids_build_nc.py


3. DEPTH WEIGHTING
--------------------------------------------------------------------------------
The six SoilGrids depth layers are combined into one 0-200 cm mean:

  weighted_mean = SUM(value_i * thickness_i) / 200 cm

  Depth interval (cm)   0-5   5-15   15-30   30-60   60-100   100-200
  Thickness (cm)          5     10      15      30       40       100


4. VARIABLES IN soilgrids.nc
--------------------------------------------------------------------------------
Variable          Description                                  Unit      Conversion
----------------  -------------------------------------------  --------  ------------------------
vwc_10kPa         volumetric water content at 10 kPa           m3/m3     x1000 in Python
vwc_33kPa         volumetric water content at 33 kPa           m3/m3     x1000 in Python
vwc_1500kPa       volumetric water content at 1500 kPa         m3/m3     x1000 in Python
bulk_density      bulk density, fine earth fraction            g/cm3     /100 in GEE (cg/cm3)
coarse_fragments  coarse fragments, volumetric                 % vol     /10 in GEE (cm3/dm3)
clay              clay content (< 2 um), mass fraction         % weight  /10 in GEE (g/kg)
sand              sand content (50-2000 um), mass fraction     % weight  /10 in GEE (g/kg)
silt              silt content (2-50 um), mass fraction        % weight  /10 in GEE (g/kg)
soc               soil organic carbon content                  g/kg      /10 in GEE (dg/kg)

Grid: native ~250 m, EPSG:4326, 5.5-15.5 E, 47.0-55.5 N, lat descending.

Note on VWC: the values of the three exported VWC GeoTIFFs were 1/1000 of the
true value in m3/m3. The factor 1000 is applied in 01_soilgrids_build_nc.py
(configuration TIFF_CONFIG) and documented in the variable attribute
"correction".

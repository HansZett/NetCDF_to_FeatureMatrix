================================================================================
WHYMAP  -  Groundwater recharge classes (HYGEO2)
================================================================================

1. DATA SOURCE
--------------------------------------------------------------------------------
Dataset   : WHYMAP - Groundwater Resources of the World (aquifer polygon layer)
Reference : BGR & UNESCO (2008): Groundwater Resources of the World - WHYMAP.
Input     : whymap_GW_aquifers_v1_poly.shp, attribute HYGEO2
Stored in : 01_Input_Global/1_Original/WHYMAP/WHYMAP_GWR/shp/
            (folder layout of the download kept)


2. SCRIPT
--------------------------------------------------------------------------------
01_whymap_hygeo2_rasterize.py
  1. Reads the shapefile with a bounding box prefilter, reprojects it to
     EPSG:4326 if needed and clips it to the study area
     (5.5-15.5 E, 47.0-55.5 N).
  2. Builds a cell-centred grid of 1/240 deg (~463 m north-south; chosen as
     half of the 1/120 deg grid of GLiM_GLHYMPS).
  3. Rasterizes HYGEO2 at 5 times finer resolution (OVERSAMPLE_FACTOR).
  4. Aggregates to the target grid with the mode (areas without polygon
     ignored). Cells without any polygon get 255.
  5. Writes the result with nc_io.save_nc and prints the class distribution.

  Input : 01_Input_Global/1_Original/WHYMAP/WHYMAP_GWR/shp/whymap_GW_aquifers_v1_poly.shp
  Output: 01_Input_Global/3_Final/WHYMAP/WHYMAP.nc
  Run   : python 01_whymap_hygeo2_rasterize.py


3. OUTPUT FILE  WHYMAP.nc
--------------------------------------------------------------------------------
Variable hygeo2 (lat, lon), uint8, fill value 255, lat descending.

First digit  = aquifer system type
  1 = major groundwater basins
  2 = complex hydrogeological structures
  3 = local and shallow aquifers
Second digit = groundwater recharge class within the type

Code  Meaning                                    Recharge (mm/yr)
----  -----------------------------------------  ----------------
11    major basin, very low                      < 2
12    major basin, low                           2 - 20
13    major basin, medium                        20 - 100
14    major basin, high                          100 - 300
15    major basin, very high                     > 300
22    complex structures, low to very low        < 20
23    complex structures, medium                 20 - 100
24    complex structures, high                   100 - 300
25    complex structures, very high              > 300
33    local/shallow aquifers, medium to very low < 100
34    local/shallow aquifers, very high to high  > 100
255   no data

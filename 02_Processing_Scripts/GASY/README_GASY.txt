================================================================================
GASY  -  Global Average Specific Yield (SoilGrids variant)
================================================================================

1. DATA SOURCE
--------------------------------------------------------------------------------
Dataset     : GASY - Global gridded dataset of average-state specific yield,
              SoilGrids variant (GASY-SoilGrids)
Authors     : Meizhao Lv, Meixia Lv, Yuanyuan Zha, Lei Wang, Zong-Liang Yang
Publication : Lv et al. (2025). A global dataset of average specific yield
              for soils. Scientific Data, 12, 427.
              https://doi.org/10.1038/s41597-025-04742-1
Data access : Zenodo, https://doi.org/10.5281/zenodo.14216083
License     : Creative Commons Attribution 4.0 International (CC BY 4.0)
Resolution  : 30 arc-seconds (~1 km), global
Depths      : 7 standard SoilGrids depths (0, 5, 15, 30, 60, 100, 200 cm)

Specific yield (Sy) is dimensionless (volume of water drained per unit volume
of soil under gravity). GASY was derived from SoilGrids soil texture fractions
(sand, silt, clay) with the trilinear graph method of Johnson (1967). Further
GASY variants (GSDE, HWSD) exist on Zenodo and are not used here.

The 7 original files soilgrids_sy_1_integrate.nc ... soilgrids_sy_7_integrate.nc
are stored in 01_Input_Global/1_Original/GASY/.


2. SCRIPTS (run order)
--------------------------------------------------------------------------------
00_inspect_gasy_nc.py   (diagnostic, writes nothing)
  Prints global attributes, dimensions, coordinate ranges and spacing,
  variables with basic statistics, and CRS information of every .nc file.
  Run   : python 00_inspect_gasy_nc.py  [optional: "<other folder>"]

01_gasy_merge_depths.py
  1. Checks that all 7 depth files exist.
  2. Selects the grid points inside the study area (SUBSET_BBOX,
     47.0-55.5 N, 5.5-15.5 E; SUBSET_BBOX = None keeps the global grid).
  3. Reads variable sy from each file and stacks the layers along a new
     depth dimension (0.00, 0.05, 0.15, 0.30, 0.60, 1.00, 2.00 m).
  4. Adds CF metadata (sy, depth, WGS84 crs variable) and writes the file
     with nc_io.save_nc.

  Input : 01_Input_Global/1_Original/GASY/soilgrids_sy_{1..7}_integrate.nc
  Output: 01_Input_Global/3_Final/GASY/GASY.nc
  Run   : python 01_gasy_merge_depths.py


3. OUTPUT FILE  GASY.nc
--------------------------------------------------------------------------------
Dimensions : depth (7), lat (descending), lon
Variable   : sy (depth, lat, lon), float32, units "1", fill value NaN,
             valid range 0.0 - 0.5
Resolution : 30 arc-seconds (~1 km), EPSG:4326
Extent     : 5.5 - 15.5 E, 47.0 - 55.5 N

Layer  Depth   Layer  Depth
1        0 cm  5       60 cm
2        5 cm  6      100 cm
3       15 cm  7      200 cm
4       30 cm

Notes
- Sy is static (no time dimension) and continuous.
- Higher Sy indicates coarser (sandy) soils, lower Sy finer (clay-rich) soils.
- Cells without valid source data (e.g. water) are NaN.
- The 2 m depth limit is given by the depth range of the SoilGrids texture data.


4. REFERENCES
--------------------------------------------------------------------------------
Lv, M., Lv, M., Zha, Y., Wang, L., & Yang, Z.-L. (2025). A global dataset of
  average specific yield for soils. Scientific Data, 12, 427.
  https://doi.org/10.1038/s41597-025-04742-1
Poggio, L., de Sousa, L. M., Batjes, N. H., et al. (2021). SoilGrids 2.0:
  producing soil information for the globe with quantified spatial
  uncertainty. SOIL, 7, 217-240. https://doi.org/10.5194/soil-7-217-2021
Hengl, T., et al. (2017). SoilGrids250m: Global gridded soil information based
  on machine learning. PLoS ONE, 12(2), e0169748.
  https://doi.org/10.1371/journal.pone.0169748
Johnson, A. I. (1967). Specific Yield - Compilation of Specific Yields for
  Various Materials. U.S. Geological Survey Water-Supply Paper 1662-D.

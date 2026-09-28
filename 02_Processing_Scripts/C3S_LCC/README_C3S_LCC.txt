================================================================================
C3S_LCC  -  ESA CCI / C3S Land Cover (annual, 300 m)
================================================================================

1. DATA SOURCE
--------------------------------------------------------------------------------
Dataset   : Land cover classification gridded maps (LCCS), annual, 300 m
Provider  : ESA Climate Change Initiative Land Cover (CCI-LC), distributed by
            the Copernicus Climate Change Service (C3S)
Access    : Copernicus Climate Data Store (CDS)
            https://cds.climate.copernicus.eu/datasets/satellite-land-cover
License   : Copernicus License v1.2
Versions  : v2.0.7cds for 1992-2015, v2.1.1 for 2016 onwards.
            Both versions use the same LCCS legend and flag definitions.
Sensors   : 1992-2015 MERIS FR/RR and SPOT VGT (v2.0.7cds),
            2016-2019 PROBA-V, 2020 onwards Sentinel-3 OLCI (v2.1.1)

The annual files were downloaded from the CDS as area subsets of the study
area and stored in 01_Input_Global/1_Original/C3S_LCC/.


2. SCRIPTS
--------------------------------------------------------------------------------
01_c3s_lcc_merge.py
  Merges all annual files into one CF-compliant NetCDF file sorted by year.
    1. Finds all *.nc files and reads the year from the file name.
    2. Renames coordinates to lat / lon, drops singleton time dimensions.
    3. Keeps lccs_class and the auxiliary variables (if present).
    4. Assigns a mid-year time stamp (YYYY-07-02) to each annual map.
    5. Concatenates all years along time.
    6. Writes the file with nc_io.save_nc using per-variable dtypes and
       NoData values (see section 3).

  Input : 01_Input_Global/1_Original/C3S_LCC/*.nc
  Output: 01_Input_Global/3_Final/C3S_LCC/c3s_lcc.nc
  Run   : python 01_c3s_lcc_merge.py


3. OUTPUT FILE  c3s_lcc.nc
--------------------------------------------------------------------------------
Dimensions : time x lat x lon
time       : annual, mid-year date YYYY-07-02, stored as days since 1970-01-01
lat / lon  : ~0.00278 deg (300 m), lat descending

Variable             dtype   NoData  Description
-------------------  ------  ------  ------------------------------------------
lccs_class           uint8   0       LCCS land cover class code (see below)
processed_flag       uint8   255     0 = not processed, 1 = processed
current_pixel_state  uint8   255     0 invalid, 1 clear land, 2 clear water,
                                     3 clear snow/ice, 4 cloud, 5 cloud shadow
observation_count    uint16  65535   number of valid observations (can be > 255)
change_count         uint8   255     number of land cover changes observed

processed_flag and current_pixel_state are float32 in the source files and are
cast to uint8. The NoData values at the top of the dtype range keep a real 0
(for example change_count = 0) distinct from missing data.


4. LCCS CLASS CODES (lccs_class)
--------------------------------------------------------------------------------
  0 NoData                          130 Grassland
 10 Rainfed cropland                140 Lichens and mosses
 11 Herbaceous rainfed cropland     150 Sparse vegetation (<15 %)
 12 Tree or shrub rainfed cropland  151 Sparse tree (<15 %)
 20 Irrigated cropland              152 Sparse shrub (<15 %)
 30 Mosaic cropland (>50 %) /       153 Sparse herbaceous cover (<15 %)
    natural vegetation (<50 %)      160 Tree cover, flooded, fresh or brackish
 40 Mosaic natural vegetation          water
    (>50 %) / cropland (<50 %)      170 Tree cover, flooded, saline water
 50 Tree, broadleaved, evergreen    180 Shrub or herbaceous cover, flooded
 60 Tree, broadleaved, deciduous    190 Urban areas
 61   ... closed (>40 %)            200 Bare areas
 62   ... open (15-40 %)            201 Consolidated bare areas
 70 Tree, needleleaved, evergreen   202 Unconsolidated bare areas
 71   ... closed (>40 %)            210 Water bodies
 72   ... open (15-40 %)            220 Permanent snow and ice
 80 Tree, needleleaved, deciduous
 81   ... closed (>40 %)
 82   ... open (15-40 %)
 90 Tree, mixed leaf type
100 Mosaic tree and shrub (>50 %) / herbaceous cover (<50 %)
110 Mosaic herbaceous cover (>50 %) / tree and shrub (<50 %)
120 Shrubland, 121 Evergreen shrubland, 122 Deciduous shrubland

Aggregation to 6 IPCC classes (stored in the variable comment):
  Agriculture 10-12, 20, 30, 40 | Forest 50-100, 160, 170 |
  Grassland 110, 130 | Wetland 180 | Settlement 190 |
  Other 120-122, 140, 150-153, 200-202, 210, 220

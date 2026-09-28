================================================================================
GLiM_GLHYMPS  -  Lithology (GLiM) and hydraulic properties (GLHYMPS)
================================================================================

1. DATA SOURCES
--------------------------------------------------------------------------------
GLiM - Global Lithological Map
  Hartmann & Moosdorf (2012), doi:10.1029/2012GC004370
  File       : LiMW_GIS 2015.gdb, layer GLiM_export
  Projection : ESRI:54012 (Eckert IV, equal area)
  Used field : xx (level-1 lithology class, 2 characters, always filled)
               Litho (full lithology code = xx + yy + zz, e.g. 'vi____')

GLHYMPS - GLobal HYdrogeology MaPS
  Gleeson et al. (2014), doi:10.1002/2014GL059856
  Huscroft et al. (2018), GLHYMPS 2.0, doi:10.1002/2017GL075860
  File       : GLHYMPS.gdb, layer Final_GLHYMPS_Polygon
  Projection : ESRI:54034 (Cylindrical Equal Area)
  Fields     : Porosity (0-1), Permeability_no_permafrost (log10 m2),
               Permeability_permafrost (log10 m2, permafrost = -20),
               Permeability_standard_deviation (log10 m2)

Download: both geodatabases were obtained from the I-GUIDE platform:
  https://platform.i-guide.io/datasets/fd345695-7dfd-4ba1-97b7-a3cc984b060d
They are stored directly in 01_Input_Global/1_Original/GLiM_GLHYMPS/.


2. SCRIPTS (run order)
--------------------------------------------------------------------------------
01_glim_glhymps_merge_clip.py
  1. Reads both geodatabases with a spatial prefilter and clips them to the
     study area. The bounding box edges are densified before the
     transformation, because the axes of the equal-area projections are
     curved.
  2. Reprojects both layers to EPSG:4326.
  3. Joins the GLiM fields Litho and xx onto the GLHYMPS polygons via the
     shared field IDENTITY_ (left join).
  4. Saves the result as a GeoPackage.

  Input : 01_Input_Global/1_Original/GLiM_GLHYMPS/GLHYMPS.gdb
          01_Input_Global/1_Original/GLiM_GLHYMPS/LiMW_GIS 2015.gdb
  Output: 01_Input_Global/2_Intermediate/GLiM_GLHYMPS/GLiM_GLHYMPS.gpkg
  Run   : python 01_glim_glhymps_merge_clip.py

02_glim_glhymps_rasterize.py
  1. Builds a cell-centred grid of 1/120 deg (~1 km) over the study area.
  2. Rasterizes the polygons at 10 times finer resolution
     (OVERSAMPLE_FACTOR, centre-of-pixel rule).
  3. Aggregates the fine cells to the target grid:
       continuous variables -> mean of the fine cells (area-weighted)
       lithology class      -> mode of the fine cells (no-data ignored)
  4. Writes the result with nc_io.save_nc.

  Input : 01_Input_Global/2_Intermediate/GLiM_GLHYMPS/GLiM_GLHYMPS.gpkg
  Output: 01_Input_Global/3_Final/GLiM_GLHYMPS/GLiM_GLHYMPS.nc
  Run   : python 02_glim_glhymps_rasterize.py


3. OUTPUT FILE  GLiM_GLHYMPS.nc
--------------------------------------------------------------------------------
Grid: 1/120 deg (~1 km), EPSG:4326, study area, lat descending

Variable                     dtype    fill  Unit        Description
---------------------------  -------  ----  ----------  ---------------------------
porosity                     float32  NaN   1           porosity
permeability_no_permafrost   float32  NaN   log10(m2)   log10 permeability
permeability_permafrost      float32  NaN   log10(m2)   log10 permeability,
                                                        permafrost = -20
permeability_std             float32  NaN   log10(m2)   std. dev. of log10 perm.
litho_class                  uint8    255   1           dominant GLiM class

litho_class codes (GLiM level 1, field xx):
   1 su  unconsolidated sediments        9 pb  basic plutonic rocks
   2 sc  carbonate sedimentary rocks    10 pi  intermediate plutonic rocks
   3 ss  siliciclastic sedimentary      11 va  acid volcanic rocks
   4 sm  mixed sedimentary rocks        12 vb  basic volcanic rocks
   5 py  pyroclastic rocks              13 vi  intermediate volcanic rocks
   6 ev  evaporites                     14 wb  water bodies
   7 mt  metamorphic rocks              15 ig  ice and glaciers
   8 pa  acid plutonic rocks           255     no data (GLiM 'nd' or no polygon)


4. REFERENCES
--------------------------------------------------------------------------------
Gleeson, T. et al. (2014): A glimpse beneath Earth's surface: GLobal
  HYdrogeology MaPS (GLHYMPS) of permeability and porosity. Geophys. Res.
  Lett., 41, 3891-3898. doi:10.1002/2014GL059856
Hartmann, J. & Moosdorf, N. (2012): The new global lithological map database
  GLiM: A representation of rock properties at the Earth surface. Geochem.
  Geophys. Geosyst., 13, Q12004. doi:10.1029/2012GC004370
Huscroft, J. et al. (2018): Compiling and mapping global permeability of the
  unconsolidated and consolidated Earth: GLobal HYdrogeology MaPS 2.0
  (GLHYMPS 2.0). Geophys. Res. Lett., 45, 1897-1904. doi:10.1002/2017GL075860

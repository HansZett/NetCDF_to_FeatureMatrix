================================================================================
0_Shared  -  Shared helper module nc_io.py
================================================================================

nc_io.py defines the NetCDF convention of the project and is the single place
where NetCDF output is written. It is not run directly. It is imported by the
dataset scripts in 02_Processing_Scripts and by the extraction scripts in
02_Dataframes_Processing_Scripts:

    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1] / "0_Shared"))
    import nc_io

Do not rename this folder or the file, because the scripts in
02_Dataframes_Processing_Scripts look for 02_Processing_Scripts/0_Shared/nc_io.py.


--------------------------------------------------------------------------------
PROJECT NETCDF CONVENTION
--------------------------------------------------------------------------------

1. Coordinate names    lat / lon / time. The variants latitude, longitude,
                       x, y and valid_time are renamed.
2. Orientation         lat descending (north to south), lon ascending
                       (west to east), as in ERA5-Land.
3. Fill value          every data variable gets an explicit _FillValue:
                       NaN for float variables, an explicit integer for
                       integer / categorical variables. Coordinates get no
                       fill value (CF requirement).
4. Compression         zlib, level 4.
5. Global attributes   title, institution, source, references, Conventions
                       (CF-1.8), history, created, crs, plus the geospatial
                       and time bounds taken from the data.
6. Folder tiers        every output must lie in 1_Original, 2_Intermediate or
                       3_Final. Final files go to 3_Final/<Dataset>/<file>.nc.

The constants at the top of nc_io.py (LAT_ORDER, LON_ORDER, TIER_FOLDERS,
DEFAULT_COMPLEVEL, REQUIRED_GLOBAL_ATTRS) change the convention for all
datasets at once.


--------------------------------------------------------------------------------
MAIN FUNCTIONS
--------------------------------------------------------------------------------

Writing
  save_nc(ds, path, fill_value, dtype, global_attrs, ...)
      Writes a dataset following the convention: tier check, coordinate
      renaming, orientation, coordinate attributes, geospatial bounds,
      encoding (fill value, dtype, compression), metadata check, and an
      optional re-open check of the written file.
  make_global_attrs(title, source, references, script, extra, ...)
      Builds the global attribute block.
  build_encoding(...)
      Builds the per-variable encoding (used by save_nc).

Coordinates
  normalize_coords(ds)       rename coordinates to lat / lon / time
  detect_orientation(obj)    report the lat / lon order of a file or dataset
  enforce_orientation(ds)    sort lat / lon to the project order
  assign_coord_attrs(ds)     add CF attributes to lat / lon / time

Checks
  which_tier(path), check_tier(path)   folder tier checks
  validate_metadata(ds)                warn about missing attributes

Reading (used by the feature extraction)
  read_fill_values(obj)      declared _FillValue of every variable
  valid_mask(values, fill)   True where a cell holds real data
  report_fills(obj)          print dtype and fill value per variable

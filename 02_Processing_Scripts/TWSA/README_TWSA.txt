================================================================================
TWSA  -  GRACE gap-filled terrestrial water storage anomaly (reconstruction)
================================================================================

1. DATA SOURCE
--------------------------------------------------------------------------------
Name        : GRACE_REC_VCE_SSLRfull_SRECES_monthly, version 1.0.0
Institution : University of Bonn, Institute of Geodesy and Geoinformation
Project     : CRC 1502 (SFB 1502) DETECT, project C03
Contact     : chacker@uni-bonn.de
Method      : combination of a data-driven reconstruction and geodetic
              tracking data (Loecher et al. 2025)
Publication : Hacker et al. (2026): Multidecadal reconstruction of terrestrial
              water storage changes by combining pre-GRACE satellite
              observations and climate data. Earth System Science Data, 18,
              1747-. doi:10.5194/essd-18-1747-2026
              https://essd.copernicus.org/articles/18/1747/2026/

Original file: GRACE_1984_2020.nc (NetCDF-4, about 2 GB), global,
  0.5 deg, monthly, 1984-01 to 2020-12, dimensions time=444, lat=360, lon=720
  TWSTORE        (time, lat, lon) float64, mm
  Sigma_TWSTORE  (time, lat, lon) float64, mm^2
  time units "Seconds since 1984-01-01" (steps of about one month that drift
  slightly within the month), fill value -999999.0 (attribute "FillValue")
Stored in: 01_Input_Global/1_Original/TWSA/GRACE_1984_2020.nc


2. SCRIPTS (run order)
--------------------------------------------------------------------------------
00_inspect_twsa.py   (diagnostic, writes nothing)
  Prints the structure of the original file.
  Run   : python 00_inspect_twsa.py

01_twsa_process.py
  1. Decodes the time axis "Seconds since 1984-01-01" manually.
  2. Cuts the study area (5.5-15.5 E, 47.0-55.5 N) and the period from
     1991-01-01 (TIME_START) to the end of the data.
  3. Sets the source fill value -999999.0 to NaN.
  4. Renames TWSTORE -> TWS and Sigma_TWSTORE -> TWS_variance.
  5. Sets all time stamps to the first day of the month and checks that
     every original stamp lies in the assigned month.
  6. Writes with nc_io.save_nc (float32, fill NaN, lat descending, time as
     days since 1970-01-01, CRS as global attribute).
  Input : 01_Input_Global/1_Original/TWSA/GRACE_1984_2020.nc
  Output: 01_Input_Global/3_Final/TWSA/TWSA.nc
  Run   : python 01_twsa_process.py


3. OUTPUT FILE  TWSA.nc
--------------------------------------------------------------------------------
lon  : 5.75 to 15.25 deg E, ascending (0.5 deg cell centres), 20 values
lat  : 55.25 to 47.25 deg N, descending, 17 values
time : 1991-01-01 to 2020-12-01, monthly (first of month), 360 values

Variable      dtype    Unit   Description
------------  -------  -----  -------------------------------------------------
TWS           float32  mm     terrestrial water storage anomaly (equivalent
                              water height), from TWSTORE
TWS_variance  float32  mm^2   variance of the TWS anomaly, from Sigma_TWSTORE


4. NOTES
--------------------------------------------------------------------------------
- TWS_variance is the VARIANCE (sigma^2), not the standard deviation. The
  source name "Sigma_TWSTORE" is misleading; its units (mm^2) and long_name
  ("Variance of the reconstructed GRACE-like TWSA maps") show that it is the
  variance. Standard deviation in mm: sqrt(TWS_variance).
- The values are ANOMALIES relative to a reference period of the source
  dataset (see Hacker et al. 2026), not absolute water storage.
- The month-start time stamps allow a direct join with other monthly
  datasets (e.g. TerraClimate).

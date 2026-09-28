# -*- coding: utf-8 -*-
"""
01_terraclimate_download.py

Downloads 14 TerraClimate v1.1 variables as monthly NetCDF files via the
THREDDS NetCDF Subset Service (NCSS). The spatial subset (study area) and
the temporal subset are applied on the server side. One file per variable is
written. Files that already exist are skipped. The script asks for
confirmation before the download starts.

Output: 01_Input_Global/1_Original/TerraClimate/
        TerraClimate_<variable>_<YEAR_START>-<YEAR_END>_AOI.nc

Location: 02_Processing_Scripts/TerraClimate/
Run     : python 01_terraclimate_download.py
"""

import urllib.request
import urllib.error
from pathlib import Path
import time

# =============================================================================
# PATHS  (relative to the project root; adjust here if the layout changes)
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_GLOBAL = PROJECT_ROOT / "01_Input_Global"
ORIGINAL     = INPUT_GLOBAL / "1_Original"

OUTPUT_DIR = ORIGINAL / "TerraClimate"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# STUDY AREA & TIME PERIOD
# =============================================================================
LAT_MIN, LAT_MAX = 47.0, 55.5
LON_MIN, LON_MAX = 5.5, 15.5

YEAR_START = 1991
YEAR_END   = 2022

# =============================================================================
# PARAMETERS
# Format: (output_filename, url_variable, description)
# =============================================================================
PARAMETERS = [
    ("TerraClimate_ET_actual",              "aet",   "Actual Evapotranspiration"),
    ("TerraClimate_ET_potential",           "pet",   "Potential Evapotranspiration"),
    ("TerraClimate_water_deficit",          "def",   "Climate Water Deficit"),
    ("TerraClimate_precipitation",          "ppt",   "Precipitation"),
    ("TerraClimate_runoff",                 "q",     "Runoff"),
    ("TerraClimate_soil_moisture",          "soil",  "Soil Moisture"),
    ("TerraClimate_solar_radiation",        "srad",  "Solar Radiation"),
    ("TerraClimate_snow_water",             "swe",   "Snow Water Equivalent"),
    ("TerraClimate_temp_max",               "tmax",  "Maximum Temperature"),
    ("TerraClimate_temp_min",               "tmin",  "Minimum Temperature"),
    ("TerraClimate_vapor_pressure",         "vap",   "Vapor Pressure"),
    ("TerraClimate_vapor_pressure_deficit", "vpd",   "Vapor Pressure Deficit"),
    ("TerraClimate_drought_index_pdsi",     "PDSI",  "Palmer Drought Severity Index"),
    ("TerraClimate_wind_speed",             "ws",    "Wind Speed"),
]

# =============================================================================
# FUNCTIONS
# =============================================================================
def create_ncss_url(url_variable):
    """Build the NCSS request URL for one variable (aggregated 1950_CurrentYear file)."""
    base = (
        f"http://thredds.northwestknowledge.net:8080/thredds/ncss/"
        f"agg_terraclimate_{url_variable}_1950_CurrentYear_GLOBE.nc"
    )
    params = (
        f"?var={url_variable}"
        f"&north={LAT_MAX}&west={LON_MIN}&east={LON_MAX}&south={LAT_MIN}"
        f"&disableProjSubset=on&horizStride=1"
        f"&time_start={YEAR_START}-01-01T00%3A00%3A00Z"
        f"&time_end={YEAR_END}-12-01T00%3A00%3A00Z"
        f"&timeStride=1&addLatLon=true&accept=netcdf"
    )
    return base + params


def download_parameter(output_name, url_variable, description):
    """Download one variable. Returns True on success."""
    print(f"\n--- {description} ---")
    url = create_ncss_url(url_variable)
    out_file = OUTPUT_DIR / f"{output_name}_{YEAR_START}-{YEAR_END}_AOI.nc"

    if out_file.exists() and out_file.stat().st_size > 0:
        print(f"  Already exists: {out_file.name} – skipping.")
        return True

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=120) as response:
            print(f"  HTTP Status: {response.getcode()}")

            content_length = response.headers.get('Content-Length')
            if content_length:
                print(f"  Content-Length: {int(content_length) / (1024*1024):.1f} MB")

            with open(out_file, 'wb') as f:
                downloaded = 0
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (5 * 1024 * 1024) < 8192:
                        print(f"    {downloaded/(1024*1024):.1f} MB", end='\r')

        size_mb = out_file.stat().st_size / (1024 * 1024)
        if size_mb > 0:
            print(f"  ✓ Saved: {out_file.name} ({size_mb:.1f} MB)")
            return True
        else:
            print("  ERROR: Downloaded file is empty (0 bytes).")
            out_file.unlink()
            return False

    except urllib.error.HTTPError as e:
        print(f"  HTTP Error {e.code}: {e.reason}")
        return False
    except urllib.error.URLError as e:
        print(f"  URL Error: {e.reason}")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("TERRACLIMATE v1.1 DOWNLOAD")
    print("=" * 70)
    print(f"Area:   Lon {LON_MIN}°–{LON_MAX}°  |  Lat {LAT_MIN}°–{LAT_MAX}°")
    print(f"Period: {YEAR_START}–{YEAR_END}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)
    print(f"\nParameters to download ({len(PARAMETERS)}):")
    for out_name, url_var, desc in PARAMETERS:
        print(f"  • {desc:<35} → {out_name}_{YEAR_START}-{YEAR_END}_AOI.nc")

    response = input("\nStart download? (y/n): ").lower().strip()
    if response != "y":
        print("Download cancelled.")
        return

    successful, failed = [], []

    for i, (output_name, url_variable, description) in enumerate(PARAMETERS, 1):
        print(f"\n[{i}/{len(PARAMETERS)}]")
        if download_parameter(output_name, url_variable, description):
            successful.append(output_name)
        else:
            failed.append(output_name)
        if i < len(PARAMETERS):
            time.sleep(2)

    print("\n" + "=" * 70)
    print(f"✅ Successful: {len(successful)}/{len(PARAMETERS)}")
    if failed:
        print(f"❌ Failed: {len(failed)}")
        for name in failed:
            print(f"  • {name}")
    print("=" * 70)


if __name__ == "__main__":
    main()

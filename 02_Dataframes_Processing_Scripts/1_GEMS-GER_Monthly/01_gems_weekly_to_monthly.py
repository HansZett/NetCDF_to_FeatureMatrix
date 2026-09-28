# -*- coding: utf-8 -*-
"""
01_gems_weekly_to_monthly.py

Aggregates the weekly GEMS-GER well time series to monthly values.

Every weekly record is assigned to the calendar month in which its date
falls. The weekly values are then aggregated per month:
  - flux columns (SUM_COLS: precipitation, evapotranspiration, runoff,
    snowfall) are summed over the weeks of the month,
  - all other numeric columns (groundwater level GWL, temperatures, relative
    humidity, soil moisture, soil temperature, snow depth, snowmelt) are
    averaged with the arithmetic mean.
Missing weekly values (NaN) are ignored; a month in which a column has no
valid value stays NaN. The quality flag column GWL_flag is not aggregated.
Months without any valid GWL value are removed (DROP_MONTHS_WITHOUT_GWL).

Output columns per well:
    time       first day of the month (YYYY-MM-01)
    time_days  days since 1970-01-01 of the month start
    GWL        monthly mean groundwater level (unit as in the weekly data, m)
    <drivers>  monthly sums (SUM_COLS) or means of the GEMS-GER drivers

Input : 03_Dataframes/01_Dynamic_Features/01_GEMS-GER_weekly/<well>.csv
        (first column = date YYYY-MM-DD, one row per week)
Output: 03_Dataframes/01_Dynamic_Features/02_GEMS-GER_monthly/<well>_monthly.csv

The output folder can be changed with --out-dir. Existing files are only
overwritten with --overwrite.

Location: 02_Dataframes_Processing_Scripts/1_GEMS-GER_Monthly/
Run     : python 01_gems_weekly_to_monthly.py
          python 01_gems_weekly_to_monthly.py --out-dir "<folder>"
          python 01_gems_weekly_to_monthly.py --limit 20        (dry run)
          python 01_gems_weekly_to_monthly.py --root "<path to project root>"
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# PATHS  (relative to the project root; adjust here if the layout changes)
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEEKLY_DIR_REL = Path("03_Dataframes") / "01_Dynamic_Features" / "01_GEMS-GER_weekly"
MONTHLY_DIR_REL = Path("03_Dataframes") / "01_Dynamic_Features" / "02_GEMS-GER_monthly"

# =============================================================================
# CONFIGURATION
# =============================================================================
DATE_FORMAT = "%Y-%m-%d"              # date format of the first column (weekly CSVs)
TARGET_COL = "GWL"
EXCLUDE_COLS = ["GWL_flag"]           # columns that are not aggregated
# Columns summed per month; every other numeric column is averaged
SUM_COLS = ["HYRAS_pr", "DWD_evapo_p", "DWD_evapo_r", "DWD_evapo_fao",
            "ERA5_sro", "ERA5_ssro", "ERA5_sf"]
DROP_MONTHS_WITHOUT_GWL = True        # remove months without a valid GWL value
OUTPUT_SUFFIX = "_monthly"            # <well>.csv -> <well>_monthly.csv
EPOCH = pd.Timestamp("1970-01-01")


def weekly_to_monthly(weekly: pd.DataFrame) -> pd.DataFrame:
    """Monthly sums (SUM_COLS) and arithmetic means (all other columns)."""
    df = weekly.drop(columns=[c for c in EXCLUDE_COLS if c in weekly.columns])
    df = df.apply(pd.to_numeric, errors="coerce")
    grouped = df.groupby(df.index.to_period("M"))
    monthly = grouped.mean()                      # NaN values are skipped
    sum_cols = [c for c in SUM_COLS if c in df.columns]
    if sum_cols:
        # min_count=1: a month without any valid value stays NaN (not 0)
        monthly[sum_cols] = grouped[sum_cols].sum(min_count=1)
    monthly.index = monthly.index.to_timestamp()  # first day of the month
    if DROP_MONTHS_WITHOUT_GWL and TARGET_COL in monthly.columns:
        monthly = monthly[monthly[TARGET_COL].notna()]
    monthly.index.name = "time"
    monthly.insert(0, "time_days",
                   (monthly.index - EPOCH).days.astype(np.float64))
    return monthly.reset_index()


def read_weekly(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, sep=",", decimal=".")
    df.index = pd.to_datetime(df.index, format=DATE_FORMAT)
    return df.sort_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="project root (optional)")
    ap.add_argument("--out-dir", default=None,
                    help="output folder (default: <root>/" + str(MONTHLY_DIR_REL) + ")")
    ap.add_argument("--limit", type=int, default=None, help="first N wells (dry run)")
    ap.add_argument("--overwrite", action="store_true",
                    help="overwrite existing monthly files")
    args = ap.parse_args()

    root = Path(args.root).resolve() if args.root else PROJECT_ROOT
    in_dir = root / WEEKLY_DIR_REL
    out_dir = Path(args.out_dir) if args.out_dir else root / MONTHLY_DIR_REL

    files = sorted(in_dir.glob("*.csv"))
    if not files:
        sys.exit(f"No weekly CSV files found in {in_dir}")
    if args.limit:
        files = files[:args.limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Input : {in_dir}  ({len(files)} files)")
    print(f"Output: {out_dir}")

    n_written = n_skipped = 0
    for f in files:
        out = out_dir / f"{f.stem}{OUTPUT_SUFFIX}.csv"
        if out.exists() and not args.overwrite:
            n_skipped += 1
            continue
        monthly = weekly_to_monthly(read_weekly(f))
        monthly.to_csv(out, index=False, date_format=DATE_FORMAT)
        n_written += 1

    print(f"Written: {n_written}   skipped (already present, use --overwrite): {n_skipped}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
config.py  --  shared configuration of the weekly model stack (01_weekly)

Single source of truth for exactly the things on which the four models
(single, dynonly, dynstat, lgbm) and the downstream scripts (shap_lgbm,
evaluation) MUST agree:

  - the input paths and the output root,
  - the train / early-stopping / test date split (incl. the cap at the end
    of 2020),
  - the seed list,
  - the 15 dynamic drivers,
  - the 7 categorical static features,
  - the metadata columns every model drops from the static table.

There is NO model logic here. The LSTM/CNN/LightGBM architecture and the
handling of the static features (dynstat label-encodes them, lgbm uses the
native 'category' dtype, dynonly/single use no static features) stay in the
individual files. These are deliberate, architecture-driven differences.

If the data move or a split date changes: change it HERE, and all four
models follow. This is what keeps the results comparable.

In addition (only used by lgbm.py and shap_lgbm.py): the LightGBM feature
engineering configuration (LAGS / ROLLING_WINDOWS / make_lag_features). It
lives here so that lgbm and shap build the same matrix and cannot drift
apart.

NOTE (public version): the three GEMS-GER reference models single, dynonly
and dynstat are not included in this repository, because the GEMS-GER code is
licensed CC BY-NC-SA 4.0. They are available from the GEMS-GER code
repository (https://github.com/KITHydrogeology/GEMS-GER). Where this file
mentions them, it describes how lgbm.py was aligned with them.
Their output paths (PTH_OUT_SINGLE, PTH_OUT_DYNONLY, PTH_OUT_DYNSTAT) are
kept, so evaluation.py includes their results if they exist.

Location: 04_Machine_Learning/01_weekly/  (imported by all scripts here)
"""

import os
import numpy as np
import pandas as pd


# =====================================================================
# PATHS  (relative to the project root; adjust here if the layout changes)
# =====================================================================
# The project root is the first parent folder of this file that contains
# 03_Dataframes (on the HPC: the workspace $WS). It can be overridden with
# the environment variable ML_PROJECT_ROOT.
def find_project_root(start=os.path.dirname(os.path.abspath(__file__)),
                      marker="03_Dataframes"):
    d = os.path.abspath(start)
    while True:
        if os.path.isdir(os.path.join(d, marker)):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return os.path.abspath(os.path.join(start, "..", ".."))
        d = parent


PROJECT_ROOT = os.environ.get("ML_PROJECT_ROOT") or find_project_root()

# Weekly GEMS-GER data (original resolution).
PTH_DYN = os.path.join(PROJECT_ROOT,
                       "03_Dataframes",
                       "01_Dynamic_Features",
                       "01_GEMS-GER_weekly")
PTH_STAT = os.path.join(PROJECT_ROOT,
                        "03_Dataframes",
                        "02_Static_Features",
                        "01_GEMS-GER",
                        "01_original_well_metadata.csv")

# Output root + one subfolder per model (derived, so evaluation.py stays
# in sync with the training scripts automatically).
PTH_OUT_ROOT    = os.path.join(PROJECT_ROOT, "05_Machine_Learning_Results", "01_weekly")
PTH_OUT_SINGLE  = os.path.join(PTH_OUT_ROOT, "results_single")
PTH_OUT_DYNONLY = os.path.join(PTH_OUT_ROOT, "results_dynonly")
PTH_OUT_DYNSTAT = os.path.join(PTH_OUT_ROOT, "results_dynstat")
PTH_OUT_LGBM    = os.path.join(PTH_OUT_ROOT, "results_lgbm")
PTH_OUT_EVAL    = os.path.join(PTH_OUT_ROOT, "results_evaluation")


# =====================================================================
# Datum-Split
# =====================================================================
# DATE SPLIT
# =====================================================================
# train < STOP <= early-stopping < TEST <= test <= END
#
# END caps the test period at the end of 2020, so that the weekly scores
# match the monthly pipeline (there global_settings.time_end = 2020-12,
# test_start = 2013-01). Same test window 2013-01..2020-12 -> the
# weekly->monthly comparison is like-for-like.
DATE_START_STOP = pd.to_datetime("2008-01-01")
DATE_START_TEST = pd.to_datetime("2013-01-01")
DATE_END        = pd.to_datetime("2020-12-31")

# Sequence length of the LSTM/CNN models (weeks of history) and number of drivers.
N_STEPS_IN = 52
N_INPUT    = 15

# Minimum number of pre-test rows needed to build features and a meaningful scaler.
MIN_PRETEST_ROWS = N_STEPS_IN + 4


# =====================================================================
# SEEDS (10-member ensemble, used by ALL four models)
# =====================================================================
HP_SEEDS = [999, 206, 380, 471, 570, 624, 643, 778, 808, 973]


# =====================================================================
# FEATURES
# =====================================================================
# 15 dynamic drivers, in the column order of the per-well CSVs.
DYNAMIC_COLS = ['HYRAS_pr', 'HYRAS_tas', 'HYRAS_tasmax', 'HYRAS_tasmin',
                'HYRAS_hurs', 'DWD_evapo_p', 'DWD_evapo_r', 'DWD_evapo_fao',
                'DWD_soil_moist', 'DWD_soil_temp5cm', 'ERA5_sro', 'ERA5_ssro',
                'ERA5_sdwe', 'ERA5_sm', 'ERA5_sf']

# Nominal static features that are stored as TEXT in the source CSV. The
# original script of Ohmer et al. label-encodes exactly these seven, because
# they would not fit into a numeric array otherwise. dynstat.py reproduces this
# unchanged, so that the published model is reproduced exactly.
CATEGORICAL_STATICS = ['AquiferMed', 'HUEK250_HU', 'HUEK250_RT',
                       'HUEK250_CT', 'HUEK250_DC', 'HUEK250_GC',
                       'HUMUS1000_OC']

# Nominal static features that are already stored as integer codes in the
# source CSV. The original script therefore does NOT label-encode them and
# passes them on as numbers. They are just as nominal as CATEGORICAL_STATICS;
# the split in the original follows the storage type (text vs. integer) only,
# not the meaning.
INTEGER_CODED_STATICS = ['HYRAUM_HD', 'HYRAUM_MHD', 'BUEK1000_RSA', 'HYSOG_SG',
                         'CLC_90', 'CLC_00', 'CLC_06', 'CLC_12', 'CLC_18',
                         'MUNDIALIS_LU', 'GMK1000_GU', 'DTM20_FD']

# Full set of nominal static features (19). lgbm.py and shap_lgbm.py use THIS
# set as native pandas 'category', so that LightGBM splits on categories
# instead of on an artificial order. dynstat.py keeps CATEGORICAL_STATICS,
# because the network must keep the published encoding. Identical to the
# 'categorical' flags in features_meta_GEMS-GER.csv of the monthly pipeline
# (which additionally contains PreState, dropped here via DROP_STATIC_COLS).
LGBM_CATEGORICAL_STATICS = CATEGORICAL_STATICS + INTEGER_CODED_STATICS

# Metadata columns that every model using static features drops from the table.
DROP_STATIC_COLS = ['Proj_ID', 'Operator', 'Depth', 'UpFilter', 'LoFilter',
                    'ScrLength', 'PreState',
                    'Easting (EPSG:3035)', 'Northing (EPSG:3035)']


# =====================================================================
# SMALL SHARED LOADERS (I/O only, no model logic)
# =====================================================================
def list_wells(pth_dyn=PTH_DYN):
    """Sorted list (file names, well IDs) of all per-well CSVs.
    Sorted, so the well order is identical for all four models."""
    files = sorted(f for f in os.listdir(pth_dyn) if f.endswith('.csv'))
    names = [f[:-4] for f in files]
    return files, names


def load_static(pth_stat=PTH_STAT):
    """Load the static metadata, drop the standard metadata columns, index by
    MW_ID. The 7 categoricals stay raw columns -- every model encodes them
    in its own way."""
    stat = pd.read_csv(pth_stat, sep=',', index_col=[0])
    stat = stat.drop(columns=[c for c in DROP_STATIC_COLS if c in stat.columns])
    stat = stat.set_index('MW_ID')
    return stat


def load_well_dynamic(fname, pth_dyn=PTH_DYN, cap_end=True):
    """Read one per-well weekly CSV, parse the date index, drop GWL_flag and
    (by default) cap at DATE_END -> the test period ends 2020-12."""
    df = pd.read_csv(os.path.join(pth_dyn, fname),
                     index_col=0, dayfirst=True, decimal='.', sep=',')
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
    if 'GWL_flag' in df.columns:
        df = df.drop(columns=['GWL_flag'])
    if cap_end:
        df = df.loc[df.index <= DATE_END]
    return df


# =====================================================================
# LIGHTGBM FEATURE ENGINEERING (lgbm.py + shap_lgbm.py only)
# =====================================================================
# Sparse lag set: short memory (1-2 weeks), monthly (4), seasonal (13),
# half-yearly (26), yearly (52). Rolling windows end at t-1 (no look-ahead).
# lgbm and shap MUST build the same matrix -> defined centrally here.
LAGS = [1, 2, 4, 8, 13, 26, 52]
ROLLING_WINDOWS = [4, 13, 26, 52]


def make_lag_features(df, dynamic_cols=DYNAMIC_COLS,
                      lags=LAGS, rolling_windows=ROLLING_WINDOWS):
    """Lag and rolling-mean features for ONE well.

    The LSTM (dynstat/dynonly) predicts GWL at t from drivers at t-52..t-1
    (not t itself). To match this information window, lags >= 1 and rolling
    windows ending at t-1 (via .shift(1)) are used.
    """
    feats = {}
    for col in dynamic_cols:
        s = df[col]
        for lag in lags:
            feats[f"{col}_lag{lag}"] = s.shift(lag).astype(np.float32)
        for win in rolling_windows:
            feats[f"{col}_rmean{win}"] = (
                s.shift(1).rolling(win, min_periods=1).mean().astype(np.float32)
            )
    return pd.DataFrame(feats, index=df.index)

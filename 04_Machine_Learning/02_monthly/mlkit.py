# -*- coding: utf-8 -*-
"""
mlkit.py  --  shared module of the monthly ML scripts
(lgbm_temporal.py, lgbm_spatial.py, shap_analysis.py). No command line.

ONE source for building the per-well feature matrix, so that training,
spatial cross-validation and SHAP cannot drift apart. Everything is driven by
the feature metadata tables and experiments.yaml.

Contents:
  - _profile_paths(): the store paths per store profile (global, gems)
  - build_dataset(exp): temporal path; per-well z-scored target, date split
  - build_dataset_spatial(exp): spatial path; ONE stacked matrix plus
    fold / well / date arrays; target = groundwater anomaly GWA in cm
    (centred on a reference window, transferable between wells)
  - load_well_folds(): basin fold assignment (well_folds_<fold_dim>.csv,
    written by make_folds.py, cached like twsa_valid_wells.csv)
  - vectorised per-well metrics and spatial_fold_masks for the CV
  - helpers for SHAP (base_column, label_maps)
The GWA conversion sits HERE in the builder (like the StandardScaler in the
temporal path), not in the trainer and not in a second store.

Store layout (global profile, see _profile_paths):
  dynamic: 03_Dataframes/01_Dynamic_Features/03_global_monthly/MW_<id>.csv
           (index = time, target GWL)
  static : 03_Dataframes/02_Static_Features/02_global/static_features.csv
           (index = MW_ID)
"""

import os
import re
import glob
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pandas.api.types import CategoricalDtype

from resolve_experiments import load_experiment, DEFAULT_META_DIR, PROJECT_ROOT, META_FILES  # noqa: F401

# Feature engineering constants
LAGS = [1, 2, 3, 6, 12]
ROLLING_WINDOWS = [3, 6, 12]
MIN_PRETEST_ROWS = 12 + 4          # run-up for max(LAGS) + buffer
MIN_TOTAL_ROWS   = 12 + 4          # spatial: no pre-test split -> total rows


# ---------------------------------------------------------------------
# STORE PROFILES  (paths relative to the project root; adjust here)
# ---------------------------------------------------------------------
def _profile_paths(project_root):
    g = os.path.join(project_root, "03_Dataframes")
    return {
        'global': {
            'dyn_dir':   os.path.join(g, "01_Dynamic_Features", "03_global_monthly"),
            'stat_file': os.path.join(g, "02_Static_Features", "02_global", "static_features.csv"),
            'prov_file': os.path.join(g, "01_Dynamic_Features", "03_global_provenance.csv"),
            'target':    'GWL',
            'stat_index':'MW_ID',
            'well_id':   lambda f: re.sub(r'\.csv$', '', f),
        },
        'gems': {
            'dyn_dir':   os.path.join(g, "01_Dynamic_Features", "02_GEMS-GER_monthly"),
            'stat_file': os.path.join(g, "02_Static_Features", "01_GEMS-GER", "01_original_well_metadata.csv"),
            'prov_file': None,
            'target':    'GWL',
            'stat_index':'MW_ID',
            'well_id':   lambda f: re.sub(r'(_monthly)?\.csv$', '', f),
        },
    }


def sanitize(name):
    return re.sub(r'[^0-9A-Za-z_]', '_', str(name))


# ---------------------------------------------------------------------
# Feature metadata table / roles
# ---------------------------------------------------------------------
def load_meta_for(exp, meta_dir=DEFAULT_META_DIR):
    # File name mapping from resolve_experiments.META_FILES (one source),
    # e.g. profile 'gems' -> features_meta_GEMS-GER.csv.
    fname = META_FILES.get(exp.store_profile, f"features_meta_{exp.store_profile}.csv")
    path = os.path.join(meta_dir, fname)
    meta = pd.read_csv(path, sep=';', dtype=str).fillna('')
    meta['categorical'] = meta['categorical'].str.lower().isin(['true', '1', 'yes'])
    meta = meta[meta['feature_raw_name'].isin(exp.features)].reset_index(drop=True)
    missing = set(exp.features) - set(meta['feature_raw_name'])
    if missing:
        raise RuntimeError(f"Allowlist features missing in the metadata table: {sorted(missing)}")
    return meta


def _split_roles(meta):
    dyn_lag  = meta[(meta.type == 'dynamic') & (meta.processing == 'lag_roll')]['feature_raw_name'].tolist()
    dyn_pass = meta[(meta.type == 'dynamic') & (meta.processing != 'lag_roll')]['feature_raw_name'].tolist()
    static   = meta[meta.type == 'static']['feature_raw_name'].tolist()
    cats     = meta[meta.categorical]['feature_raw_name'].tolist()
    return dyn_lag, dyn_pass, static, cats


# ---------------------------------------------------------------------
# Wells with valid TWSA values
# ---------------------------------------------------------------------
def twsa_valid_wells(project_root, hull, meta_dir=DEFAULT_META_DIR, recompute=False):
    cache = os.path.join(meta_dir, "twsa_valid_wells.csv")
    if os.path.exists(cache) and not recompute:
        return set(pd.read_csv(cache)['well_id'].astype(str))
    paths = _profile_paths(project_root)['global']
    prov = paths['prov_file']
    valid = set()
    if prov and os.path.exists(prov):
        dfp = pd.read_csv(prov)
        idcol = 'well_id' if 'well_id' in dfp.columns else dfp.columns[0]
        outcol = next((c for c in dfp.columns if c.endswith('twsa__outcome')
                       or c == 'twsa__outcome'), None)
        if outcol:
            ok = dfp[~dfp[outcol].isin(['dropped', 'no_valid'])]
            valid = set(ok[idcol].astype(str))
    if not valid:
        t0, t1 = hull['time_start'], hull['time_end']
        for fp in glob.glob(os.path.join(paths['dyn_dir'], "*.csv")):
            f = os.path.basename(fp)
            if f.startswith('_'):
                continue
            try:
                s = pd.read_csv(fp, usecols=['time', 'twsa__TWS'],
                                index_col=0, parse_dates=[0])['twsa__TWS']
            except (ValueError, KeyError):
                continue
            s = s.loc[t0:t1] if t0 else s
            if s.notna().any():
                valid.add(paths['well_id'](f))
    os.makedirs(meta_dir, exist_ok=True)
    pd.DataFrame({'well_id': sorted(valid)}).to_csv(cache, index=False)
    return valid


# ---------------------------------------------------------------------
# Fold lookup (written by make_folds.py, cached per fold_dim)
# ---------------------------------------------------------------------
def load_well_folds(fold_dim, meta_dir=DEFAULT_META_DIR):
    """Read well_folds_<fold_dim>.csv -> dict MW_ID (str) -> fold_id (int)."""
    cache = os.path.join(meta_dir, f"well_folds_{fold_dim}.csv")
    if not os.path.exists(cache):
        raise FileNotFoundError(
            f"{cache} missing -- run make_folds.py --fold-dim {fold_dim} first.")
    df = pd.read_csv(cache)
    df = df.dropna(subset=['fold_id'])
    return {str(w): int(f) for w, f in zip(df['MW_ID'].astype(str), df['fold_id'])}


# ---------------------------------------------------------------------
# Per-well feature engineering
# ---------------------------------------------------------------------
def _lag_features(df, cols):
    feats = {}
    for col in cols:
        s = df[col]
        for lag in LAGS:
            feats[f"{col}_lag{lag}"] = s.shift(lag).astype(np.float32)
        for win in ROLLING_WINDOWS:
            feats[f"{col}_rmean{win}"] = s.shift(1).rolling(win, min_periods=1).mean().astype(np.float32)
    return pd.DataFrame(feats, index=df.index)


def _lcc_to_str(x):
    return 'missing' if pd.isna(x) else str(int(x))


# ---------------------------------------------------------------------
# Shared building blocks (used by temporal + spatial -> no duplication)
# ---------------------------------------------------------------------
def _load_static(paths, static_cols, cat_cols):
    """Load the static table, categoricals as strings ('missing' for NaN), fixed
    category sets -> (stat, keep_static, static_cat, static_cat_dtype, dyn_cat)."""
    stat = pd.read_csv(paths['stat_file'], sep=',')
    if paths['stat_index'] in stat.columns:
        stat = stat.set_index(paths['stat_index'])
    stat.index = stat.index.astype(str)
    keep_static = [c for c in static_cols if c in stat.columns]
    static_cat = [c for c in cat_cols if c in keep_static]
    for c in static_cat:
        stat[c] = stat[c].map(lambda x: 'missing' if pd.isna(x) else str(x))
    static_cat_dtype = {c: CategoricalDtype(categories=sorted(stat[c].dropna().unique()))
                        for c in static_cat}
    dyn_cat = [c for c in cat_cols if c not in keep_static]
    return stat, keep_static, static_cat, static_cat_dtype, dyn_cat


def _well_feature_matrix(df, dyn_lag, dyn_pass, dyn_cat, keep_static, stat_row, lcc_union):
    """Build X for ONE well (lag/rolling + passthrough + static broadcast),
    sanitise the column names and fill lcc_union for dynamic categoricals."""
    blocks = []
    present_lag = [c for c in dyn_lag if c in df.columns]
    if present_lag:
        blocks.append(_lag_features(df, present_lag))
    present_pass = [c for c in dyn_pass if c in df.columns]
    if present_pass:
        fp = df[present_pass].copy()
        for c in present_pass:
            if c in dyn_cat:
                fp[c] = fp[c].map(_lcc_to_str)
                lcc_union[c].update(fp[c].unique())
            else:
                fp[c] = fp[c].astype(np.float32)
        blocks.append(fp)
    blocks.append(pd.DataFrame(
        {c: np.repeat(stat_row[c], len(df)) for c in keep_static}, index=df.index))
    X = pd.concat(blocks, axis=1)
    X.columns = [sanitize(c) for c in X.columns]
    return X


def _finalize_categoricals(frames, static_cat, static_cat_dtype, dyn_cat, lcc_union):
    """Unify the categorical dtypes over ALL frames, everything else -> float32.
    Modifies the frames in place. -> (cat_present, feat_names)."""
    frames = [d for d in frames if len(d)]
    feat_names = list(frames[0].columns) if frames else []
    cat_present = []
    for c in static_cat:
        sc = sanitize(c)
        for X in frames:
            if sc in X.columns:
                X[sc] = X[sc].astype(static_cat_dtype[c])
        if any(sc in X.columns for X in frames):
            cat_present.append(sc)
    for c in dyn_cat:
        sc = sanitize(c)
        dt = CategoricalDtype(categories=sorted(v for v in lcc_union[c] if v == v))
        for X in frames:
            if sc in X.columns:
                X[sc] = X[sc].astype(dt)
        if any(sc in X.columns for X in frames):
            cat_present.append(sc)
    num_cols = [c for c in feat_names if c not in cat_present]
    for X in frames:
        if len(X):
            X[num_cols] = X[num_cols].astype(np.float32)
    return cat_present, feat_names


# ---------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------
@dataclass
class Dataset:
    X_train: pd.DataFrame; y_train: np.ndarray
    X_stop:  pd.DataFrame; y_stop:  np.ndarray
    X_test:  pd.DataFrame; y_test:  np.ndarray
    IDlen: np.ndarray
    kept_names: list
    scalers_y: list
    well_means_pre_test: np.ndarray
    cat_cols: list
    feat_names: list


@dataclass
class SpatialDataset:
    X: pd.DataFrame            # stacked matrix (all wells, all months)
    y: np.ndarray             # GWA in cm
    wid: np.ndarray           # well index per row
    fold: np.ndarray          # fold id per row
    date: np.ndarray          # date per row
    kept_names: list          # well IDs (index = wid)
    well_folds: np.ndarray    # fold per well
    well_std: np.ndarray      # per-well std of the anomaly (cm), diagnostic
    cat_cols: list
    feat_names: list


# ---------------------------------------------------------------------
# TEMPORAL: build_dataset (z-scored target, date split)
# ---------------------------------------------------------------------
def build_dataset(exp, project_root=PROJECT_ROOT, meta_dir=DEFAULT_META_DIR,
                  splits=('train', 'stop', 'test'), verbose=True):
    from sklearn.preprocessing import StandardScaler

    meta = load_meta_for(exp, meta_dir)
    dyn_lag, dyn_pass, static_cols, cat_cols = _split_roles(meta)
    paths = _profile_paths(project_root)[exp.store_profile]

    t0, t1 = exp.time_start, exp.time_end
    stop_start = pd.to_datetime(exp.stop_start + "-01")
    test_start = pd.to_datetime(exp.test_start + "-01")

    well_set = (twsa_valid_wells(project_root, {'time_start': t0, 'time_end': t1}, meta_dir)
                if exp.require_twsa_valid else None)

    stat, keep_static, static_cat, static_cat_dtype, dyn_cat = _load_static(
        paths, static_cols, cat_cols)

    files = sorted(f for f in os.listdir(paths['dyn_dir'])
                   if f.endswith('.csv') and not f.startswith('_') and not f.startswith('merge_report'))

    parts = {s: ([], []) for s in ('train', 'stop', 'test')}
    ID_test_list, scalers_y, well_means, kept_names = [], [], [], []
    lcc_union = {c: set() for c in dyn_cat}

    for f in files:
        wid = paths['well_id'](f)
        if well_set is not None and wid not in well_set:
            continue
        if wid not in stat.index:
            continue
        df = pd.read_csv(os.path.join(paths['dyn_dir'], f), index_col=0, parse_dates=[0])
        df.index = pd.to_datetime(df.index)
        if 'time_days' in df.columns:
            df = df.drop(columns=['time_days'])
        if t0:
            df = df.loc[t0:t1]
        if paths['target'] not in df.columns or df.empty:
            continue
        pre = df.loc[df.index < test_start, paths['target']]
        if len(pre) < MIN_PRETEST_ROWS:
            continue
        scaler = StandardScaler().fit(pre.values.reshape(-1, 1))
        y = pd.Series(scaler.transform(df[[paths['target']]].values).ravel(),
                      index=df.index, dtype=np.float32)

        X = _well_feature_matrix(df, dyn_lag, dyn_pass, dyn_cat, keep_static,
                                 stat.loc[wid], lcc_union)
        m_train = X.index < stop_start
        m_stop  = (X.index >= stop_start) & (X.index < test_start)
        m_test  = X.index >= test_start
        wk = len(kept_names)
        if 'train' in splits:
            parts['train'][0].append(X[m_train]); parts['train'][1].append(y[m_train])
        if 'stop' in splits:
            parts['stop'][0].append(X[m_stop]);   parts['stop'][1].append(y[m_stop])
        if 'test' in splits:
            parts['test'][0].append(X[m_test]);   parts['test'][1].append(y[m_test])
            ID_test_list.append(np.repeat(wk, int(m_test.sum())))
        scalers_y.append(scaler)
        well_means.append(float(pre.mean()))
        kept_names.append(wid)

    if not kept_names:
        ex_dyn  = [paths['well_id'](f) for f in files[:5]]
        ex_stat = list(map(str, list(stat.index)[:5]))
        ex_twsa = (list(well_set)[:5] if well_set is not None
                   else 'n/a (require_twsa_valid=False)')
        raise RuntimeError(
            "No wells left after filtering.\n"
            f"  {len(files)} dynamic files scanned; all removed by well_set/static/"
            f"pre-test length ({MIN_PRETEST_ROWS} months before {test_start}).\n"
            f"  Example IDs dynamic (well_id): {ex_dyn}\n"
            f"  Example IDs static (index):    {ex_stat}\n"
            f"  Example IDs twsa_valid:        {ex_twsa}\n"
            "  -> If dynamic and static IDs are written differently "
            "(e.g. '1' vs 'MW_1'), the static filter removes all wells.")
    if verbose:
        print(f"  Wells kept: {len(kept_names)} (profile={exp.store_profile}, "
              f"twsa_valid={exp.require_twsa_valid})")

    def cat(name):
        if not parts[name][0]:
            return pd.DataFrame(), np.array([], dtype=np.float32)
        X = pd.concat(parts[name][0], axis=0, ignore_index=True)
        y = pd.concat(parts[name][1], axis=0, ignore_index=True).values
        return X, y

    X_train, y_train = cat('train')
    X_stop,  y_stop  = cat('stop')
    X_test,  y_test  = cat('test')
    cat_present, feat_names = _finalize_categoricals(
        [X_train, X_stop, X_test], static_cat, static_cat_dtype, dyn_cat, lcc_union)

    return Dataset(X_train, y_train, X_stop, y_stop, X_test, y_test,
                   np.concatenate(ID_test_list) if ID_test_list else np.array([]),
                   kept_names, scalers_y, np.array(well_means, dtype=np.float32),
                   cat_present, feat_names)


# ---------------------------------------------------------------------
# SPATIAL: build_dataset_spatial (Target = GWA in cm)
# ---------------------------------------------------------------------
def build_dataset_spatial(exp, project_root=PROJECT_ROOT, meta_dir=DEFAULT_META_DIR,
                          verbose=True):
    """Build ONE stacked GWA matrix + fold/wid/date arrays. The GWA conversion
    (centred on the reference window, m -> cm) is done here per well."""
    if exp.target.get('kind') != 'gwa_cm':
        raise RuntimeError(f"[{exp.name}] build_dataset_spatial expects target.kind=gwa_cm, "
                           f"got '{exp.target.get('kind')}'.")
    meta = load_meta_for(exp, meta_dir)
    dyn_lag, dyn_pass, static_cols, cat_cols = _split_roles(meta)
    paths = _profile_paths(project_root)[exp.store_profile]

    t0, t1 = exp.time_start, exp.time_end
    tg = exp.target
    ref_start = pd.to_datetime(tg['reference_start'])
    ref_end   = pd.to_datetime(tg['reference_end'])
    cm        = float(tg['gwl_to_cm'])
    min_ref   = int(tg['min_ref_months'])
    fold_dim  = exp.split['fold_dim']

    well_set = (twsa_valid_wells(project_root, {'time_start': t0, 'time_end': t1}, meta_dir)
                if exp.require_twsa_valid else None)
    folds = load_well_folds(fold_dim, meta_dir)

    stat, keep_static, static_cat, static_cat_dtype, dyn_cat = _load_static(
        paths, static_cols, cat_cols)

    files = sorted(f for f in os.listdir(paths['dyn_dir'])
                   if f.endswith('.csv') and not f.startswith('_') and not f.startswith('merge_report'))

    X_parts, y_parts, wid_parts, fold_parts, date_parts = [], [], [], [], []
    kept_names, well_folds, well_std = [], [], []
    lcc_union = {c: set() for c in dyn_cat}
    n_no_static = n_no_fold = n_short = n_no_ref = 0

    for f in files:
        wid = paths['well_id'](f)
        if well_set is not None and wid not in well_set:
            continue
        if wid not in stat.index:
            n_no_static += 1; continue
        if wid not in folds:
            n_no_fold += 1; continue
        df = pd.read_csv(os.path.join(paths['dyn_dir'], f), index_col=0, parse_dates=[0])
        df.index = pd.to_datetime(df.index)
        if 'time_days' in df.columns:
            df = df.drop(columns=['time_days'])
        if t0:
            df = df.loc[t0:t1]
        if paths['target'] not in df.columns or len(df) < MIN_TOTAL_ROWS:
            n_short += 1; continue

        gwl = df[paths['target']]
        ref = gwl.loc[(gwl.index >= ref_start) & (gwl.index <= ref_end)].dropna()
        if len(ref) < min_ref:
            n_no_ref += 1; continue
        ref_mean = float(ref.mean())
        y = ((gwl.values - ref_mean) * cm).astype(np.float32)        # GWA in cm

        X = _well_feature_matrix(df, dyn_lag, dyn_pass, dyn_cat, keep_static,
                                 stat.loc[wid], lcc_union)
        wk = len(kept_names)
        X_parts.append(X)
        y_parts.append(y)
        wid_parts.append(np.full(len(df), wk, dtype=np.int32))
        fold_parts.append(np.full(len(df), folds[wid], dtype=np.int16))
        date_parts.append(df.index.values)
        kept_names.append(wid)
        well_folds.append(folds[wid])
        well_std.append(float(np.nanstd(y)))

    if not kept_names:
        ex_dyn  = [paths['well_id'](f) for f in files[:5]]
        ex_stat = list(map(str, list(stat.index)[:5]))
        ex_fold = list(folds.keys())[:5]
        ex_twsa = (list(well_set)[:5] if well_set is not None
                   else 'n/a (require_twsa_valid=False)')
        raise RuntimeError(
            "No wells left after filtering.\n"
            f"  Reasons over {len(files)} files: "
            f"no_static={n_no_static}, no_fold={n_no_fold}, "
            f"short/no_target={n_short}, no_baseline={n_no_ref}.\n"
            f"  Example IDs dynamic (well_id): {ex_dyn}\n"
            f"  Example IDs static (index):    {ex_stat}\n"
            f"  Example IDs folds (MW_ID):     {ex_fold}\n"
            f"  Example IDs twsa_valid:        {ex_twsa}\n"
            "  -> no_fold dominates: ID format in well_folds_<dim>.csv does not match\n"
            "     the dynamic file names (e.g. '1' vs 'MW_1').\n"
            "  -> no_static dominates: static index (MW_ID) does not match the dynamic well_id.\n"
            "  -> no_baseline dominates: GWA reference window "
            f"({tg['reference_start']}..{tg['reference_end']}, min {min_ref} months) not covered.\n"
            "     make_folds uses the static index -> static index AND dynamic well_id "
            "must be written the same way.")

    X_all = pd.concat(X_parts, axis=0, ignore_index=True)
    y_all = np.concatenate(y_parts)
    wid_all = np.concatenate(wid_parts)
    fold_all = np.concatenate(fold_parts)
    date_all = np.concatenate(date_parts)
    well_folds = np.array(well_folds); well_std = np.array(well_std)
    del X_parts, y_parts, wid_parts, fold_parts, date_parts

    cat_present, feat_names = _finalize_categoricals(
        [X_all], static_cat, static_cat_dtype, dyn_cat, lcc_union)

    if verbose:
        print(f"  Wells kept: {len(kept_names)} (profile={exp.store_profile}, "
              f"twsa_valid={exp.require_twsa_valid}, fold_dim={fold_dim})")
        print(f"    drops: no_static {n_no_static}, no_fold {n_no_fold}, "
              f"short {n_short}, no_baseline {n_no_ref}")
        nf = exp.split['n_folds']
        for fdx in range(nf):
            print(f"    fold {fdx}: {int((well_folds == fdx).sum())} wells")
        print(f"  X_all: {X_all.shape}  categorical: {cat_present}")
        print(f"  [diag] anomaly std per well (cm): med={np.nanmedian(well_std):.1f}  "
              f"p10={np.nanpercentile(well_std,10):.1f}  p90={np.nanpercentile(well_std,90):.1f}")

    return SpatialDataset(X_all, y_all, wid_all, fold_all, date_all,
                          kept_names, well_folds, well_std, cat_present, feat_names)


def persist_schema(ds, out_dir):
    """Feature schema for raster inference: feature_columns.csv + categories.json."""
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({'feature': list(ds.X.columns)}).to_csv(
        os.path.join(out_dir, "feature_columns.csv"), sep=";", index=False)
    cats = {c: [str(x) for x in ds.X[c].cat.categories] for c in ds.cat_cols}
    with open(os.path.join(out_dir, "categories.json"), "w", encoding='utf-8') as fh:
        json.dump(cats, fh, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------
# Split masks / metrics (spatial)
# ---------------------------------------------------------------------
def spatial_fold_masks(fold_all, test_fold, n_folds, val_fold_offset):
    val_fold = (test_fold + val_fold_offset) % n_folds
    tr  = ~np.isin(fold_all, [test_fold, val_fold])
    val = fold_all == val_fold
    te  = fold_all == test_fold
    return tr, val, te, val_fold


def group_starts(wid):
    if len(wid) == 0:
        return np.array([], dtype=int)
    return np.r_[0, np.where(np.diff(wid) != 0)[0] + 1].astype(int)


def grouped_metrics(pred, obs, starts, total):
    """Per-well NSE/KGE_a/R2/Bias/RMSE/alpha/r (classic NSE: benchmark =
    per-well mean of the observations). Offset-invariant -> directly
    comparable to the temporal model; RMSE/Bias in cm."""
    pred = pred.astype(np.float64); obs = obs.astype(np.float64)
    n = (np.r_[np.diff(starts), total - starts[-1]]).astype(np.float64)
    Sp  = np.add.reduceat(pred,        starts)
    So  = np.add.reduceat(obs,         starts)
    Spp = np.add.reduceat(pred * pred, starts)
    Soo = np.add.reduceat(obs * obs,   starts)
    Spo = np.add.reduceat(pred * obs,  starts)
    mean_p, mean_o = Sp / n, So / n
    var_p = np.maximum(Spp / n - mean_p ** 2, 0.0)
    var_o = np.maximum(Soo / n - mean_o ** 2, 0.0)
    cov   = Spo / n - mean_p * mean_o
    sd_p, sd_o = np.sqrt(var_p), np.sqrt(var_o)
    with np.errstate(invalid='ignore', divide='ignore'):
        r         = np.where((sd_p > 0) & (sd_o > 0), cov / (sd_p * sd_o), np.nan)
        alpha     = np.where(sd_o > 0, sd_p / sd_o, np.nan)
        bias_norm = np.where(sd_o > 0, (mean_p - mean_o) / sd_o, np.nan)
        sse  = Spp - 2 * Spo + Soo
        RMSE = np.sqrt(sse / n)
        NSE  = np.where(var_o > 0, 1 - sse / (n * var_o), np.nan)
    KGE_a = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + bias_norm ** 2)
    return {'NSE': NSE, 'KGE_a': KGE_a, 'R2': r ** 2,
            'Bias': mean_p - mean_o, 'RMSE': RMSE, 'alpha': alpha, 'r': r}


# ---------------------------------------------------------------------
# SHAP helpers
# ---------------------------------------------------------------------
def base_column(feat_name, base_cols):
    cand = re.sub(r'_(lag\d+|rmean\d+)$', '', feat_name)
    if cand in base_cols:
        return cand
    return feat_name


def label_maps(exp, meta_dir=DEFAULT_META_DIR):
    meta = load_meta_for(exp, meta_dir)
    san = {sanitize(r['feature_raw_name']): r for _, r in meta.iterrows()}
    plot = {k: v['plot_name'] for k, v in san.items()}
    vgrp = {k: (v['var_group'] or v['group']) for k, v in san.items()}
    return plot, vgrp, set(san.keys())

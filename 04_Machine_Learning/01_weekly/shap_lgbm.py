# -*- coding: utf-8 -*-
"""
SHAP feature-importance analysis for the LightGBM groundwater-level model
trained by lgbm.py.

For each random seed (or a chosen subset), this script:
  1. rebuilds the same X_test feature matrix that lgbm.py used (same well
     filters, same lag / rolling features, same unified categorical dtypes,
     so the booster accepts the matrix without re-mapping categories);
  2. loads results_lgbm/run<seed>/model.txt;
  3. computes TreeSHAP contributions via LightGBM's native pred_contrib
     (mathematically identical to shap.TreeExplainer but avoids dtype edge
     cases with the 19 nominal categorical statics);
  4. writes a per-feature importance table  (mean |SHAP|, mean SHAP, std);
  5. writes a per-driver aggregation that sums |SHAP| across all lags and
     rolling windows of the same dynamic driver -- easier to read than the
     11 separate rows you would otherwise get per driver;
  6. saves a bar plot of the top features (matplotlib only, no `shap`
     dependency for the bar plot) and, if the `shap` package is installed,
     a beeswarm summary plot too.

SHAP values are in the units of the model's RAW output, i.e. the per-well
standardised GWL (mean 0, std 1 on each well's pre-test window). They are
therefore directly comparable across features / wells / seeds, but not in
metres. This matches how lgbm.py's gain-based importance is reported.

Outputs land in the existing per-seed folder so they sit next to the
feature_importance.csv from training:

    results_lgbm/run<seed>/shap_per_feature.csv
    results_lgbm/run<seed>/shap_per_driver.csv
    results_lgbm/run<seed>/shap_bar.png
    results_lgbm/run<seed>/shap_summary.png         (only if shap installed)
    results_lgbm/run<seed>/shap_values.npz          (only with --save-raw)

Usage:
    python shap_lgbm.py                            # all 10 seeds, 100k-row sample
    python shap_lgbm.py --seeds 999 206 380        # subset of seeds
    python shap_lgbm.py --sample-rows 50000        # cap rows used for SHAP
    python shap_lgbm.py --sample-rows -1           # use the full test set
    python shap_lgbm.py --save-raw                 # also dump raw SHAP arrays
    python shap_lgbm.py --cache-x                  # cache X_test as parquet
"""

#%% packages and paths

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import gc
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime

# Force line-buffered stdout/stderr even when launched without `python -u` or
# PYTHONUNBUFFERED. Without this, prints to the Slurm .out file are block-
# buffered (~4-8 KB) and the log can stay empty for a long time.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
    pass  # Python <3.7

import matplotlib
matplotlib.use('Agg')           # headless: no X display on the cluster
import matplotlib.pyplot as plt

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("WARNING: `shap` package not installed -- beeswarm plots will be skipped. "
          "Install with `pip install shap` for full output. The bar plot and the "
          "CSV tables do NOT require the shap package.", flush=True)


import config

# Paths and feature-engineering config come from config.py -- the SAME source
# lgbm.py reads at training time. That guarantees X_test is reconstructed
# row-for-row (no more "verify the static path" guesswork, no chance of the
# lag/rolling grid drifting out of sync with the trained model).
pth_dt_dyn  = config.PTH_DYN
pth_dt_stat = config.PTH_STAT
pth_out     = config.PTH_OUT_LGBM
pth_cache_x = os.path.join(pth_out, "_cache_X_test.parquet")


#%% feature-engineering config (shared with lgbm.py via config.py)

DYNAMIC_COLS        = config.DYNAMIC_COLS
LAGS                = config.LAGS
ROLLING_WINDOWS     = config.ROLLING_WINDOWS
CATEGORICAL_STATICS = config.LGBM_CATEGORICAL_STATICS   # 19 nominal fields, as in lgbm.py
make_lag_features   = config.make_lag_features

HP_seeds_default = config.HP_SEEDS
date_start_test  = config.DATE_START_TEST


# make_lag_features is imported from config.py above (single source shared with
# lgbm.py). It is NOT redefined here, so the two scripts can never drift apart.


def build_test_matrix():
    """Reconstruct X_test row-for-row as lgbm.py builds it.

    Wells are dropped under the same two conditions as in training:
        - no entry in the static metadata table
        - fewer than 52+4 weekly GWL observations before the test split
    so the resulting X_test contains the same wells and rows the booster
    saw at test time.
    """
    dt_list_files, dt_list_names = config.list_wells()

    dt_static = config.load_static()

    static_cat_dtypes = {}
    for col in CATEGORICAL_STATICS:
        if col in dt_static.columns:
            dt_static[col] = dt_static[col].astype('category')
            static_cat_dtypes[col] = dt_static[col].dtype

    X_test_list = []
    t_build_start = datetime.now()
    n_total = len(dt_list_names)
    for i, name in enumerate(dt_list_names):
        if i % 200 == 0 and i > 0:
            elapsed = (datetime.now() - t_build_start).total_seconds()
            rate = i / elapsed if elapsed > 0 else 0
            eta = (n_total - i) / rate if rate > 0 else float('nan')
            print(f"  well {i}/{n_total}  "
                  f"elapsed={elapsed:6.1f}s  rate={rate:5.1f} wells/s  "
                  f"ETA={eta/60:5.1f} min  "
                  f"kept={len(X_test_list)}", flush=True)
        if name not in dt_static.index:
            continue
        df = config.load_well_dynamic(dt_list_files[i])

        pre_test_gwl = df.loc[df.index < date_start_test, 'GWL']
        if len(pre_test_gwl) < config.MIN_PRETEST_ROWS:
            continue

        feats_dyn = make_lag_features(df)
        static_row = dt_static.loc[name]
        feats_stat = pd.DataFrame(
            {col: np.repeat(static_row[col], len(df))
             for col in dt_static.columns},
            index=df.index,
        )
        X_well = pd.concat([feats_dyn, feats_stat], axis=1)
        mask_test = X_well.index >= date_start_test
        X_test_list.append(X_well[mask_test])

    print(f"  concatenating {len(X_test_list)} per-well frames "
          f"(this can take ~1 min on a wide dataframe)...", flush=True)
    t_concat = datetime.now()
    X_test = pd.concat(X_test_list, axis=0, ignore_index=True)
    print(f"  concat done in {(datetime.now()-t_concat).total_seconds():.1f}s",
          flush=True)

    # Concat can drop the categorical dtype -- restore the unified one
    for col, dtype in static_cat_dtypes.items():
        if col in X_test.columns:
            X_test[col] = X_test[col].astype(dtype)
    num_cols = [c for c in X_test.columns if c not in static_cat_dtypes]
    X_test[num_cols] = X_test[num_cols].astype(np.float32)

    return X_test, static_cat_dtypes


def driver_group(name):
    """Map a feature name to its source variable for per-driver aggregation.

    'HYRAS_pr_lag4'    -> 'HYRAS_pr'
    'ERA5_sm_rmean26'  -> 'ERA5_sm'
    'AquiferMed'       -> 'AquiferMed'   (statics map to themselves)
    """
    for d in DYNAMIC_COLS:
        if name == d or name.startswith(d + '_'):
            return d
    return name


#%% per-seed SHAP computation

def shap_for_seed(seed, X_sample, feat_names, categorical_cols,
                  pth_seed, save_raw=False, max_plot_features=30):
    model_path = os.path.join(pth_seed, 'model.txt')
    if not os.path.isfile(model_path):
        print(f"  [seed {seed}] model.txt not found at {model_path} -- skipping")
        return

    booster = lgb.Booster(model_file=model_path)

    print(f"  [seed {seed}] computing TreeSHAP on {len(X_sample):,} rows x "
          f"{X_sample.shape[1]} features...")
    t0 = datetime.now()
    # pred_contrib returns shape (n_rows, n_features + 1); last col is the
    # base (expected) value. Identical numbers to shap.TreeExplainer.
    contribs    = booster.predict(X_sample, pred_contrib=True).astype(np.float32)
    base_value  = float(contribs[0, -1])
    shap_values = contribs[:, :-1]
    dt = (datetime.now() - t0).total_seconds()
    print(f"  [seed {seed}] TreeSHAP took {dt:.1f}s, base_value = {base_value:.4f}")

    # ---- per-feature summary table ----
    mean_abs  = np.abs(shap_values).mean(axis=0)
    mean_sign = shap_values.mean(axis=0)
    std_shap  = shap_values.std(axis=0)
    per_feat = pd.DataFrame({
        'feature':       feat_names,
        'mean_abs_shap': mean_abs,
        'mean_shap':     mean_sign,
        'std_shap':      std_shap,
        'driver':        [driver_group(f) for f in feat_names],
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    per_feat.to_csv(os.path.join(pth_seed, 'shap_per_feature.csv'),
                    float_format='%.6f', sep=';', index=False)

    # ---- per-driver aggregation ----
    per_driver = (per_feat
                  .groupby('driver', as_index=False)
                  .agg(total_abs_shap=('mean_abs_shap', 'sum'),
                       max_abs_shap=('mean_abs_shap', 'max'),
                       n_features=('feature', 'count'))
                  .sort_values('total_abs_shap', ascending=False)
                  .reset_index(drop=True))
    per_driver.to_csv(os.path.join(pth_seed, 'shap_per_driver.csv'),
                      float_format='%.6f', sep=';', index=False)

    # ---- bar plot of top features (pure matplotlib, no shap pkg needed) ----
    top = per_feat.head(max_plot_features).iloc[::-1]   # reverse so largest on top
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(top) + 1.5))
    ax.barh(top['feature'], top['mean_abs_shap'], color='#3a6ea5')
    ax.set_xlabel('mean |SHAP value|  (impact on standardised GWL prediction)')
    ax.set_title(f'Top {len(top)} features by SHAP -- seed {seed}')
    ax.grid(axis='x', linestyle=':', alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(pth_seed, 'shap_bar.png'), dpi=150)
    plt.close(fig)

    # ---- beeswarm summary plot (only if `shap` is installed) ----
    if HAS_SHAP:
        # shap.summary_plot needs numeric feature values for its colormap.
        # Categoricals -> integer codes (only for the plot).
        X_plot = X_sample.copy()
        for col in categorical_cols:
            if col in X_plot.columns:
                X_plot[col] = X_plot[col].cat.codes.astype(np.float32)

        # Beeswarm gets unreadable beyond ~5k points; subsample for the plot only.
        if len(X_plot) > 5000:
            idx = np.random.default_rng(seed).choice(len(X_plot), 5000, replace=False)
            X_plot_bee = X_plot.iloc[idx]
            sv_bee = shap_values[idx]
        else:
            X_plot_bee = X_plot
            sv_bee = shap_values

        plt.figure(figsize=(9, 0.32 * max_plot_features + 1.5))
        shap.summary_plot(sv_bee, X_plot_bee,
                          max_display=max_plot_features, show=False)
        plt.title(f'SHAP summary -- seed {seed}')
        plt.tight_layout()
        plt.savefig(os.path.join(pth_seed, 'shap_summary.png'), dpi=150)
        plt.close()

    # ---- raw SHAP arrays (optional, can be hundreds of MB per seed) ----
    if save_raw:
        np.savez_compressed(
            os.path.join(pth_seed, 'shap_values.npz'),
            shap_values=shap_values,
            base_value=np.float32(base_value),
            feature_names=np.array(feat_names, dtype=object),
        )

    print(f"  [seed {seed}] wrote shap_per_feature.csv, shap_per_driver.csv, "
          f"shap_bar.png" + (", shap_summary.png" if HAS_SHAP else ""))
    print(f"  [seed {seed}] top-10 by mean |SHAP|:")
    print(per_feat.head(10)[['feature', 'mean_abs_shap', 'mean_shap']]
          .to_string(index=False))

    del contribs, shap_values, booster, per_feat, per_driver
    gc.collect()


#%% main

def main():
    p = argparse.ArgumentParser(
        description="SHAP feature-importance analysis for lgbm.py models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--seeds', type=int, nargs='+', default=HP_seeds_default,
                   help='Seeds to analyse (default: all 10 from lgbm.py).')
    p.add_argument('--sample-rows', type=int, default=100_000,
                   help='Max rows from X_test to use for SHAP. Default 100000; '
                        'pass -1 to use the full test set.')
    p.add_argument('--sample-seed', type=int, default=42,
                   help='RNG seed for the row sample (deterministic across runs).')
    p.add_argument('--save-raw', action='store_true',
                   help='Also save raw SHAP arrays as shap_values.npz '
                        '(can be hundreds of MB per seed).')
    p.add_argument('--cache-x', action='store_true',
                   help='Cache/reuse X_test as parquet to skip preprocessing on re-runs.')
    p.add_argument('--max-plot-features', type=int, default=30,
                   help='Number of top features to show in plots (default 30).')
    args = p.parse_args()

    print("\n=== SHAP analysis for lgbm.py models ===")
    print(f"  seeds:        {args.seeds}")
    print(f"  sample rows:  {args.sample_rows if args.sample_rows > 0 else 'all'}")
    print(f"  save raw:     {args.save_raw}")
    print(f"  shap pkg:     {'available' if HAS_SHAP else 'missing -- beeswarm skipped'}\n")

    # ---- build (or load) X_test ----
    if args.cache_x and os.path.isfile(pth_cache_x):
        print(f"Loading cached X_test from {pth_cache_x}...")
        X_test = pd.read_parquet(pth_cache_x)
        cat_cols = [c for c in CATEGORICAL_STATICS if c in X_test.columns]
        for c in cat_cols:
            X_test[c] = X_test[c].astype('category')
        print(f"  X_test shape: {X_test.shape}")
    else:
        print("Building X_test (this takes a few minutes)...")
        t0 = datetime.now()
        X_test, static_cat_dtypes = build_test_matrix()
        dt = (datetime.now() - t0).total_seconds() / 60
        print(f"  X_test shape: {X_test.shape}  built in {dt:.2f} min")
        cat_cols = list(static_cat_dtypes.keys())
        if args.cache_x:
            try:
                X_test.to_parquet(pth_cache_x, index=False)
                print(f"  cached to {pth_cache_x}")
            except Exception as e:
                print(f"  parquet cache failed ({e}) -- continuing in-memory only")

    # ---- sample rows once, reuse across all seeds ----
    rng = np.random.default_rng(args.sample_seed)
    if args.sample_rows > 0 and len(X_test) > args.sample_rows:
        idx = rng.choice(len(X_test), args.sample_rows, replace=False)
        idx.sort()
        X_sample = X_test.iloc[idx].reset_index(drop=True)
        # restore categorical dtype after iloc (some pandas versions keep it,
        # but be defensive)
        for c in cat_cols:
            if c in X_sample.columns and not pd.api.types.is_categorical_dtype(X_sample[c]):
                X_sample[c] = X_sample[c].astype(X_test[c].dtype)
        print(f"Sampled {len(X_sample):,} rows from {len(X_test):,}.")
    else:
        X_sample = X_test
        print(f"Using full X_test: {len(X_sample):,} rows.")

    feat_names = list(X_sample.columns)
    print(f"Feature count: {len(feat_names)}  "
          f"(categorical: {len(cat_cols)})")

    # ---- per-seed SHAP ----
    for seed in args.seeds:
        print(f"\n--- seed {seed} ---")
        pth_seed = os.path.join(pth_out, f"run{seed}")
        if not os.path.isdir(pth_seed):
            print(f"  {pth_seed} not found -- skipping")
            continue
        shap_for_seed(seed, X_sample, feat_names, cat_cols, pth_seed,
                      save_raw=args.save_raw,
                      max_plot_features=args.max_plot_features)

    print("\nAll seeds complete.")


if __name__ == '__main__':
    main()

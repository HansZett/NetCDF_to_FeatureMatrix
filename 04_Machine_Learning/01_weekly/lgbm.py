# -*- coding: utf-8 -*-
"""
LightGBM baseline for groundwater level prediction.

Designed as a direct comparator for dynstat.py (LSTM + Dense static branch).
Reads from the SAME input directories, uses the SAME train / early-stopping /
test date split, the SAME seed list, and computes the SAME per-well metrics
(NSE, KGE, R^2, Bias, MSE, RMSE), so scores can be compared one-to-one.

Differences from dynstat.py (deliberate, model-architecture driven):
- Tree-based gradient boosting (LightGBM) instead of LSTM + Dense.
- Tabular feature representation: lagged values + rolling-window means of
  the 15 dynamic drivers replace the 52-step sequence input. No lag 0
  (current week's drivers) so the tree model sees the same information
  window as the LSTM, which predicts t from drivers at t-52 ... t-1.
- Native handling of NaN values: lag features carry NaN at the start of
  each well's record and LightGBM learns a default split direction for
  them; no imputation. This is what makes the model gap-tolerant later.
- Native handling of categorical static features: the seven categoricals
  (AquiferMed, HUEK250_*, HUMUS1000_OC) are passed as pandas 'category'
  dtype with a unified category set, NOT label-encoded. LightGBM splits
  on optimal partitions instead of arbitrary integer codes.
- Per-well GWL target standardization is preserved (matches dynstat.py),
  so predictions are inverse-transformed per well before scoring.
- Dynamic features are NOT per-well standardized: trees don't need it,
  and keeping physical units makes feature importance interpretable.

Output files per seed (parallel to dynstat.py):
- model.txt              LightGBM Booster (loadable with lgb.Booster)
- losshistory.csv        train/val RMSE per boosting iteration
- scores.csv             per-well NSE / KGE / R^2 / Bias / MSE / RMSE
- results.csv            per-well sim and obs on the test period
- feature_importance.csv gain- and split-based importance, all features

Requires: lightgbm (>=4.0), pandas, numpy, scipy, scikit-learn.

NOTE (public version): the three GEMS-GER reference models single, dynonly
and dynstat are not included in this repository, because the GEMS-GER code is
licensed CC BY-NC-SA 4.0. They are available from the GEMS-GER code
repository (https://github.com/KITHydrogeology/GEMS-GER). Where this file
mentions them, it describes how lgbm.py was aligned with them.
"""

#%% paths and packages

import warnings
warnings.filterwarnings('ignore')

import os
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from datetime import datetime
from sklearn.preprocessing import StandardScaler

import config

pth_dt_dyn  = config.PTH_DYN
pth_dt_stat = config.PTH_STAT
pth_out     = config.PTH_OUT_LGBM

os.makedirs(pth_out, exist_ok=True)


#%% feature-engineering config

# All shared constants (the 15 drivers, the lag/rolling grid, the 19 nominal
# statics) and make_lag_features now live in config.py, so this script and
# shap_lgbm.py build byte-identical feature matrices.
DYNAMIC_COLS        = config.DYNAMIC_COLS
LAGS                = config.LAGS
ROLLING_WINDOWS     = config.ROLLING_WINDOWS
# The full set of 19 nominal static fields, not only the seven that the
# published scripts had to label-encode. LightGBM can split on a category
# directly, so the twelve integer-coded nominal fields are declared here as
# well. dynstat.py keeps the published seven-field encoding.
CATEGORICAL_STATICS = config.LGBM_CATEGORICAL_STATICS
make_lag_features   = config.make_lag_features


#%% list of wells

dt_list_files, dt_list_names = config.list_wells()

pd.DataFrame({"ID": dt_list_names}).to_csv(pth_out + "/IDremaining.csv", sep=";")
print(f"Found {len(dt_list_files)} wells.")


#%% load static features

dt_static_features = config.load_static()

# Cast to pandas 'category' once, with a fixed category set, so train/stop/test
# splits all share the same encoding (LightGBM requires this).
static_cat_dtypes = {}
for col in CATEGORICAL_STATICS:
    if col in dt_static_features.columns:
        dt_static_features[col] = dt_static_features[col].astype('category')
        static_cat_dtypes[col] = dt_static_features[col].dtype

print(f"Static features loaded: {dt_static_features.shape[1]} columns "
      f"({len(static_cat_dtypes)} categorical).")


#%% Hyperparameters

# Shared 10-seed ensemble (config.HP_SEEDS) -- same list for all four models.
HP_seeds = config.HP_SEEDS

# Train / early-stopping / test split, identical across all weekly models.
# date_start_test .. DATE_END caps the test period at 2020-12 (monthly parity).
date_start_stop = config.DATE_START_STOP
date_start_test = config.DATE_START_TEST

# LightGBM hyperparameters: sensible starting point. Worth a small grid search
# (num_leaves, min_data_in_leaf, learning_rate) once the pipeline is validated.
LGB_PARAMS_BASE = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 127,
    'max_depth': -1,
    'min_data_in_leaf': 200,        # large pooled dataset -> conservative leaf size
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.0,
    'lambda_l2': 0.1,
    'verbosity': -1,
    'force_col_wise': True,         # faster on wide data
}

NUM_BOOST_ROUND = 5000              # ceiling; early stopping decides when to halt
EARLY_STOPPING_ROUNDS = 75


#%% Data preprocessing  (done ONCE, reused across all seeds)

print("\nBuilding feature matrices...")
now0 = datetime.now()

X_train_list, y_train_list = [], []
X_stop_list,  y_stop_list  = [], []
X_test_list,  y_test_list  = [], []
ID_test_list = []
scalers_y = []
well_means_pre_test = []
kept_well_idx = []      # integer index into dt_list_names of wells we actually used

for i, name in enumerate(dt_list_names):
    if i % 250 == 0 and i > 0:
        print(f"  Processing well {i}/{len(dt_list_names)}...")

    # Static metadata: skip wells with no entry (dynstat.py would silently drop these too)
    if name not in dt_static_features.index:
        continue

    # Read one well's weekly series (GWL_flag dropped, capped at DATE_END).
    tempdata = config.load_well_dynamic(dt_list_files[i])

    # ---- per-well GWL standardization (matches dynstat.py) ----
    pre_test_gwl = tempdata.loc[tempdata.index < date_start_test, 'GWL']
    if len(pre_test_gwl) < config.MIN_PRETEST_ROWS:
        # not enough pre-test data to build features and a meaningful scaler
        continue
    scaler_y = StandardScaler().fit(pre_test_gwl.values.reshape(-1, 1))

    y_std = pd.Series(
        scaler_y.transform(tempdata[['GWL']].values).ravel(),
        index=tempdata.index, dtype=np.float32, name='y'
    )

    # ---- dynamic features ----
    feats_dyn = make_lag_features(tempdata)

    # ---- static features broadcast to every row ----
    static_row = dt_static_features.loc[name]
    feats_stat = pd.DataFrame(
        {col: np.repeat(static_row[col], len(tempdata))
         for col in dt_static_features.columns},
        index=tempdata.index
    )

    X_well = pd.concat([feats_dyn, feats_stat], axis=1)

    # ---- split by date (same boundaries as dynstat.py) ----
    mask_train = X_well.index <  date_start_stop
    mask_stop  = (X_well.index >= date_start_stop) & (X_well.index < date_start_test)
    mask_test  =  X_well.index >= date_start_test

    X_train_list.append(X_well[mask_train])
    y_train_list.append(y_std[mask_train])
    X_stop_list.append(X_well[mask_stop])
    y_stop_list.append(y_std[mask_stop])
    X_test_list.append(X_well[mask_test])
    y_test_list.append(y_std[mask_test])
    ID_test_list.append(np.repeat(len(kept_well_idx), mask_test.sum()))

    scalers_y.append(scaler_y)
    well_means_pre_test.append(float(pre_test_gwl.mean()))
    kept_well_idx.append(i)

print(f"  Wells kept: {len(kept_well_idx)} / {len(dt_list_names)}")

# Concatenate splits
X_train = pd.concat(X_train_list, axis=0, ignore_index=True)
y_train = pd.concat(y_train_list, axis=0, ignore_index=True).values
X_stop  = pd.concat(X_stop_list,  axis=0, ignore_index=True)
y_stop  = pd.concat(y_stop_list,  axis=0, ignore_index=True).values
X_test  = pd.concat(X_test_list,  axis=0, ignore_index=True)
y_test  = pd.concat(y_test_list,  axis=0, ignore_index=True).values
IDlen   = np.concatenate(ID_test_list)
well_means_pre_test = np.array(well_means_pre_test, dtype=np.float32)

# Restore unified categorical dtype across all splits (concat can drop it)
for col, dtype in static_cat_dtypes.items():
    if col in X_train.columns:
        X_train[col] = X_train[col].astype(dtype)
        X_stop[col]  = X_stop[col].astype(dtype)
        X_test[col]  = X_test[col].astype(dtype)

# Numeric columns to float32 to halve memory
num_cols = [c for c in X_train.columns if c not in static_cat_dtypes]
X_train[num_cols] = X_train[num_cols].astype(np.float32)
X_stop[num_cols]  = X_stop[num_cols].astype(np.float32)
X_test[num_cols]  = X_test[num_cols].astype(np.float32)

print("\nFeature matrix shapes:")
print(f"  Train: {X_train.shape}")
print(f"  Stop:  {X_stop.shape}")
print(f"  Test:  {X_test.shape}")
print(f"  Categorical features: {[c for c in static_cat_dtypes if c in X_train.columns]}")

# Free intermediate per-well lists
del X_train_list, y_train_list, X_stop_list, y_stop_list, X_test_list, y_test_list
gc.collect()

# Names of the wells that actually made it into modelling (parallel to scalers_y, IDlen)
kept_names = [dt_list_names[i] for i in kept_well_idx]

# Per-well pre-test mean = NSE denominator. Written at the experiment root so the
# output layout matches the monthly pipeline (results_<exp>/well_pretest_mean.csv).
pd.DataFrame({"ID": kept_names, "pretest_mean": well_means_pre_test}).to_csv(
    pth_out + "/well_pretest_mean.csv", sep=";", index=False)

now1 = datetime.now()
print(f"Preprocessing took {(now1-now0).total_seconds()/60:.2f} min")


#%% Modelling  (one run per seed -- preprocessing above is reused)

categorical_cols_present = [c for c in CATEGORICAL_STATICS if c in X_train.columns]

for ii, seed in enumerate(HP_seeds):

    print(f"\n{'='*64}\n  Seed {seed}  ({ii+1}/{len(HP_seeds)})\n{'='*64}")

    pth_out_i = pth_out + '/run' + str(seed)
    os.makedirs(pth_out_i, exist_ok=True)

    now1 = datetime.now()

    params = dict(LGB_PARAMS_BASE)
    params['seed'] = seed
    params['bagging_seed'] = seed
    params['feature_fraction_seed'] = seed
    params['data_random_seed'] = seed

    train_set = lgb.Dataset(
        X_train, label=y_train,
        categorical_feature=categorical_cols_present,
        free_raw_data=False,
    )
    val_set = lgb.Dataset(
        X_stop, label=y_stop,
        categorical_feature=categorical_cols_present,
        reference=train_set,
        free_raw_data=False,
    )

    eval_history = {}

    model = lgb.train(
        params=params,
        train_set=train_set,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[train_set, val_set],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=True),
            lgb.log_evaluation(period=100),
            lgb.record_evaluation(eval_history),
        ],
    )

    now2 = datetime.now()
    timetaken = round((now2 - now1).total_seconds()) / 60
    print(f'\n  timetaken = {timetaken:.2f} min   '
          f'best_iter = {model.best_iteration}')

    # Save the trained model (parallel to dynstat.py's model.keras)
    model.save_model(pth_out_i + '/model.txt', num_iteration=model.best_iteration)

    # ---- Predict on test set, inverse-transform per well ----
    sim_n = model.predict(X_test, num_iteration=model.best_iteration).astype(np.float32)

    results_list = []
    for k, name in enumerate(kept_names):
        idx = (IDlen == k)
        if idx.sum() == 0:
            continue
        sim_raw = scalers_y[k].inverse_transform(sim_n[idx].reshape(-1, 1)).ravel()
        obs_raw = scalers_y[k].inverse_transform(y_test[idx].reshape(-1, 1)).ravel()
        temp = pd.DataFrame({
            f"{name}_ID":  np.repeat(k, len(sim_raw)),
            f"{name}_sim": sim_raw,
            f"{name}_obs": obs_raw,
        })
        results_list.append(temp.reset_index(drop=True))
    results = pd.concat(results_list, axis=1)

    # ---- Evaluate per-well (same metrics, same NSE denominator as dynstat.py) ----
    sim_test = results.filter(like="_sim")
    sim_test.columns = np.arange(0, sim_test.shape[1])
    obs_test = results.filter(like="_obs")
    obs_test.columns = np.arange(0, obs_test.shape[1])

    err_test  = sim_test - obs_test
    err_nash  = obs_test - well_means_pre_test.reshape(1, -1)

    MSE_test  = np.mean(err_test ** 2, axis=0)
    RMSE_test = np.sqrt(MSE_test)

    if sim_test.isnull().any().sum() > 0:
        rr_test = np.full(sim_test.shape[1], np.nan)
    else:
        rr_test = np.zeros(sim_test.shape[1])
        for k in range(sim_test.shape[1]):
            rr_test[k] = stats.pearsonr(sim_test.iloc[:, k], obs_test.iloc[:, k])[0]

    NSE_test   = 1 - (np.sum(err_test ** 2, axis=0) / np.sum(err_nash ** 2, axis=0))
    Bias_test  = np.mean(err_test, axis=0)
    alpha_test = np.std(sim_test, axis=0) / np.std(obs_test, axis=0)
    beta_test  = np.mean(sim_test, axis=0) / np.mean(obs_test, axis=0)
    KGE_test   = 1 - np.sqrt((rr_test - 1) ** 2 + (alpha_test - 1) ** 2 + (beta_test - 1) ** 2)

    scores = pd.DataFrame(
        [NSE_test, KGE_test, rr_test ** 2, Bias_test, MSE_test, RMSE_test]
    ).transpose()
    scores.index = kept_names
    scores.columns = ['NSE', 'KGE', 'R2', 'Bias', 'MSE', 'RMSE']

    # ---- Feature importance ----
    feat_imp = pd.DataFrame({
        'feature': model.feature_name(),
        'gain':    model.feature_importance(importance_type='gain'),
        'split':   model.feature_importance(importance_type='split'),
    }).sort_values('gain', ascending=False)

    # ---- Loss history (mimics dynstat.py losshistory.csv) ----
    n_iter = len(eval_history['train']['rmse'])
    loss_hist = pd.DataFrame({
        'iteration':  np.arange(1, n_iter + 1),
        'train_rmse': eval_history['train']['rmse'],
        'val_rmse':   eval_history['val']['rmse'],
    })

    # ---- Export ----
    loss_hist.to_csv(pth_out_i + '/losshistory.csv',
                     float_format='%.6f', sep=";", index=False)
    scores.to_csv(pth_out_i + '/scores.csv',
                  float_format='%.4f', sep=";")
    results.to_csv(pth_out_i + '/results.csv',
                   float_format='%.4f', sep=";")
    feat_imp.to_csv(pth_out_i + '/feature_importance.csv',
                    float_format='%.4f', sep=";", index=False)

    print(f"  NSE  median = {scores['NSE'].median():.3f}   "
          f"mean = {scores['NSE'].mean():.3f}")
    print(f"  KGE  median = {scores['KGE'].median():.3f}")
    print(f"  RMSE median = {scores['RMSE'].median():.3f}")
    print("  Top-10 features by gain:")
    print(feat_imp.head(10).to_string(index=False))

    # Per-seed cleanup
    del model, train_set, val_set, sim_n, results_list, results
    del sim_test, obs_test, err_test, err_nash
    del MSE_test, RMSE_test, rr_test, NSE_test, KGE_test, Bias_test, alpha_test, beta_test
    del scores, feat_imp, loss_hist, eval_history
    gc.collect()

print("\nAll seeds complete.")

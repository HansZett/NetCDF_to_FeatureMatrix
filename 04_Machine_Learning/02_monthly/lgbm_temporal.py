# -*- coding: utf-8 -*-
"""
lgbm_temporal.py  --  LightGBM training with a temporal (date) split

Generic LightGBM training, controlled by one experiment of the configuration
(experiments.yaml -> resolve_experiments.py -> resolved_allowlists.json).
Target = per-well z-scored GWL (StandardScaler on the pre-test window);
train < stop_start <= early stopping < test_start <= test <= time_end.
One model per seed (HP_SEEDS); predictions are back-transformed to metres.

    python lgbm_temporal.py --exp ERA5_4_twsa
    python lgbm_temporal.py --exp GEMS --seeds 999 206

Feature selection, store profile, output folder and time/well hull come
entirely from the configuration; the matrix is built by mlkit.build_dataset.

Output (<results_root>/<exp.output_path>/):
  run<seed>/model.txt, scores.csv, results.csv, feature_importance.csv,
  losshistory.csv; IDremaining.csv, well_pretest_mean.csv
"""
import warnings; warnings.filterwarnings('ignore')
import os, argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from datetime import datetime

from resolve_experiments import load_experiment, PROJECT_ROOT, DEFAULT_META_DIR
import mlkit

HP_SEEDS = [999, 206, 380, 471, 570, 624, 643, 778, 808, 973]
LGB_PARAMS_BASE = {
    'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.05,
    'num_leaves': 127, 'max_depth': -1, 'min_data_in_leaf': 200,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
    'lambda_l1': 0.0, 'lambda_l2': 0.1, 'verbosity': -1, 'force_col_wise': True,
}
NUM_BOOST_ROUND = 5000
EARLY_STOPPING_ROUNDS = 75


def evaluate(sim_n, ds):
    """Per-well NSE/KGE/R2/Bias/MSE/RMSE on the back-transformed GWL (metres)."""
    rows = []
    for k, name in enumerate(ds.kept_names):
        idx = (ds.IDlen == k)
        if idx.sum() == 0:
            continue
        sim = ds.scalers_y[k].inverse_transform(sim_n[idx].reshape(-1, 1)).ravel()
        obs = ds.scalers_y[k].inverse_transform(ds.y_test[idx].reshape(-1, 1)).ravel()
        rows.append(pd.DataFrame({f"{name}_ID": np.repeat(k, len(sim)),
                                  f"{name}_sim": sim, f"{name}_obs": obs}).reset_index(drop=True))
    results = pd.concat(rows, axis=1)

    sim_t = results.filter(like="_sim"); sim_t.columns = np.arange(sim_t.shape[1])
    obs_t = results.filter(like="_obs"); obs_t.columns = np.arange(obs_t.shape[1])
    err = sim_t - obs_t
    err_nash = obs_t - ds.well_means_pre_test.reshape(1, -1)
    MSE = np.mean(err ** 2, axis=0); RMSE = np.sqrt(MSE)
    if sim_t.isnull().any().sum() > 0:
        rr = np.full(sim_t.shape[1], np.nan)
    else:
        rr = np.array([stats.pearsonr(sim_t.iloc[:, k], obs_t.iloc[:, k])[0]
                       for k in range(sim_t.shape[1])])
    NSE = 1 - (np.sum(err ** 2, axis=0) / np.sum(err_nash ** 2, axis=0))
    Bias = np.mean(err, axis=0)
    alpha = np.std(sim_t, axis=0) / np.std(obs_t, axis=0)
    beta = np.mean(sim_t, axis=0) / np.mean(obs_t, axis=0)
    KGE = 1 - np.sqrt((rr - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    scores = pd.DataFrame([NSE, KGE, rr ** 2, Bias, MSE, RMSE]).T
    scores.index = ds.kept_names
    scores.columns = ['NSE', 'KGE', 'R2', 'Bias', 'MSE', 'RMSE']
    return results, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=HP_SEEDS)
    ap.add_argument('--project-root', default=PROJECT_ROOT)
    ap.add_argument('--meta-dir', default=DEFAULT_META_DIR)
    ap.add_argument('--results-root', default=None,
                    help="overrides results_root from the config (otherwise PROJECT_ROOT/<output_path>)")
    args = ap.parse_args()

    exp = load_experiment(args.exp, meta_dir=args.meta_dir)
    print(f"Experiment {exp.name}  ({exp.display_name})")
    print(f"  profile={exp.store_profile}  n_features={exp.n_features}  "
          f"hull={exp.time_start}..{exp.time_end}  twsa_valid={exp.require_twsa_valid}")
    if exp.hull_note:
        print(f"  hull: {exp.hull_note}")

    root = args.results_root or args.project_root
    out_dir = os.path.join(root, exp.output_path)
    os.makedirs(out_dir, exist_ok=True)

    t0 = datetime.now()
    ds = mlkit.build_dataset(exp, project_root=args.project_root, meta_dir=args.meta_dir)
    print(f"  matrices: train{ds.X_train.shape} stop{ds.X_stop.shape} test{ds.X_test.shape}  "
          f"({(datetime.now()-t0).total_seconds()/60:.1f} min)")
    pd.DataFrame({"ID": ds.kept_names}).to_csv(os.path.join(out_dir, "IDremaining.csv"), sep=";")
    pd.DataFrame({"ID": ds.kept_names, "pretest_mean": ds.well_means_pre_test}).to_csv(
        os.path.join(out_dir, "well_pretest_mean.csv"), sep=";", index=False)

    for ii, seed in enumerate(args.seeds):
        print(f"\n--- seed {seed} ({ii+1}/{len(args.seeds)}) ---")
        seed_dir = os.path.join(out_dir, f"run{seed}"); os.makedirs(seed_dir, exist_ok=True)
        params = dict(LGB_PARAMS_BASE, seed=seed, bagging_seed=seed,
                      feature_fraction_seed=seed, data_random_seed=seed)
        train_set = lgb.Dataset(ds.X_train, label=ds.y_train,
                                categorical_feature=ds.cat_cols, free_raw_data=False)
        val_set = lgb.Dataset(ds.X_stop, label=ds.y_stop, categorical_feature=ds.cat_cols,
                              reference=train_set, free_raw_data=False)
        hist = {}
        model = lgb.train(params, train_set, num_boost_round=NUM_BOOST_ROUND,
                          valid_sets=[train_set, val_set], valid_names=['train', 'val'],
                          callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                                     lgb.log_evaluation(period=200),
                                     lgb.record_evaluation(hist)])
        model.save_model(os.path.join(seed_dir, 'model.txt'), num_iteration=model.best_iteration)

        sim_n = model.predict(ds.X_test, num_iteration=model.best_iteration).astype(np.float32)
        results, scores = evaluate(sim_n, ds)

        feat_imp = pd.DataFrame({
            'feature': model.feature_name(),
            'gain': model.feature_importance('gain'),
            'split': model.feature_importance('split'),
        }).sort_values('gain', ascending=False)
        n = len(hist['train']['rmse'])
        loss = pd.DataFrame({'iteration': np.arange(1, n + 1),
                             'train_rmse': hist['train']['rmse'], 'val_rmse': hist['val']['rmse']})

        loss.to_csv(os.path.join(seed_dir, 'losshistory.csv'), sep=";", index=False, float_format='%.6f')
        scores.to_csv(os.path.join(seed_dir, 'scores.csv'), sep=";", float_format='%.4f')
        results.to_csv(os.path.join(seed_dir, 'results.csv'), sep=";", float_format='%.4f')
        feat_imp.to_csv(os.path.join(seed_dir, 'feature_importance.csv'), sep=";", index=False, float_format='%.4f')
        print(f"  NSE median={scores['NSE'].median():.3f} mean={scores['NSE'].mean():.3f}  "
              f"KGE median={scores['KGE'].median():.3f}  RMSE median={scores['RMSE'].median():.3f}  "
              f"best_iter={model.best_iteration}")

    print("\nAll seeds finished ->", out_dir)


if __name__ == '__main__':
    main()

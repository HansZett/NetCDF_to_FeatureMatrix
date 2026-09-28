# -*- coding: utf-8 -*-
"""
lgbm_spatial.py  --  LightGBM training with a SPATIAL split (basin CV)

Generic LightGBM training with a spatial split (K-fold cross-validation over
whole HydroBASINS basins) and the groundwater anomaly (GWA, cm) as target,
controlled by one experiment of the configuration.

    python lgbm_spatial.py --exp ERA5_4_twsa_spatial
    python lgbm_spatial.py --exp ERA5_6_basin_lev06_spatial --n-seeds-cv 5 --top-k 3

Sister of lgbm_temporal.py. Loss, parameters and feature engineering are
IDENTICAL: the LightGBM parameters are imported directly from
lgbm_temporal.py so they cannot drift apart. Only two things differ:
  - target = GWA in cm (mlkit.build_dataset_spatial; centred, transferable)
  - split  = spatial (K-fold over HYBAS basins; mlkit.spatial_fold_masks)

mlkit.build_dataset_spatial does the preprocessing ONCE; here the rows are
only masked. Model selection / early stopping = pooled validation RMSE (as in
the temporal path), no sample weights, no Huber loss.
Prerequisite: make_folds.py has been run for the fold_dim.

Outputs (under <results_root>/<exp.output_path>):
  fold<k>/model_seed<s>.txt, ensemble_test_scores.csv, ensemble_test_results.csv
  cv_summary.csv, cv_seed_summary.csv
  final/model_seed<s>.txt, final/final_members.csv
  feature_columns.csv, categories.json   (schema for raster inference)
"""
import warnings; warnings.filterwarnings('ignore')
import os, gc, argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime

from resolve_experiments import load_experiment, PROJECT_ROOT, DEFAULT_META_DIR
import mlkit
# Direct import -> guaranteed parity with the temporal training:
from lgbm_temporal import LGB_PARAMS_BASE, NUM_BOOST_ROUND, EARLY_STOPPING_ROUNDS, HP_SEEDS


def make_seeds(n, rng_seed=12345):
    seeds = list(HP_SEEDS)
    rng = np.random.default_rng(rng_seed)
    while len(seeds) < n:
        c = int(rng.integers(1, 10000))
        if c not in seeds:
            seeds.append(c)
    return seeds[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True)
    ap.add_argument('--project-root', default=PROJECT_ROOT)
    ap.add_argument('--meta-dir', default=DEFAULT_META_DIR)
    ap.add_argument('--results-root', default=None,
                    help="overrides results_root from the config")
    ap.add_argument('--n-seeds-cv', type=int, default=5, help="seeds per outer fold")
    ap.add_argument('--top-k', type=int, default=3, help="number of best seeds ensembled per fold")
    ap.add_argument('--final-seeds', type=int, default=10,
                    help="seeds of the deployment ensemble (all wells)")
    args = ap.parse_args()

    exp = load_experiment(args.exp, meta_dir=args.meta_dir)
    if exp.split.get('kind') != 'spatial':
        raise SystemExit(f"[{exp.name}] split.kind != 'spatial' "
                         f"({exp.split.get('kind')}). Use lgbm_temporal.py for temporal experiments.")
    n_folds = int(exp.split['n_folds'])
    offset  = int(exp.split['val_fold_offset'])
    fold_dim = exp.split['fold_dim']

    print(f"Experiment {exp.name}  ({exp.display_name})")
    print(f"  profile={exp.store_profile}  n_features={exp.n_features}  "
          f"hull={exp.time_start}..{exp.time_end}  twsa_valid={exp.require_twsa_valid}")
    print(f"  target={exp.target['kind']} (ref {exp.target['reference_start']}.."
          f"{exp.target['reference_end']}, x{exp.target['gwl_to_cm']})  "
          f"split=spatial({fold_dim}, K={n_folds}, val_offset={offset})")
    if exp.hull_note:
        print(f"  hull: {exp.hull_note}")

    root = args.results_root or args.project_root
    out_dir = os.path.join(root, exp.output_path)
    os.makedirs(out_dir, exist_ok=True)

    # ---- build the data ONCE (the GWA conversion happens in mlkit) ----
    t0 = datetime.now()
    ds = mlkit.build_dataset_spatial(exp, project_root=args.project_root, meta_dir=args.meta_dir)
    mlkit.persist_schema(ds, out_dir)
    print(f"  Preprocessing: {(datetime.now()-t0).total_seconds()/60:.1f} min  X{ds.X.shape}")
    pd.DataFrame({"ID": ds.kept_names, "fold": ds.well_folds, "anom_std_cm": ds.well_std}).to_csv(
        os.path.join(out_dir, "IDremaining.csv"), sep=";", index=False, float_format="%.3f")

    seeds_cv = make_seeds(args.n_seeds_cv)
    print(f"\nSpatial {n_folds}-fold CV; {len(seeds_cv)} Seeds/Fold; TOP_K={args.top_k}. "
          f"L2 / pooled val-RMSE; no sample weights.")

    cv_rows, seed_rows, all_best = [], [], []

    for test_fold in range(n_folds):
        tr, val, te, val_fold = mlkit.spatial_fold_masks(ds.fold, test_fold, n_folds, offset)
        Xtr, ytr   = ds.X[tr],  ds.y[tr]
        Xval, yval = ds.X[val], ds.y[val]
        Xte, yte   = ds.X[te],  ds.y[te]
        wid_te = ds.wid[te]
        starts = mlkit.group_starts(wid_te)

        print(f"\n{'='*60}\n  TEST fold {test_fold} | VAL fold {val_fold} | "
              f"train {tr.sum()} / val {val.sum()} / test {te.sum()}\n{'='*60}")
        fold_dir = os.path.join(out_dir, f"fold{test_fold}"); os.makedirs(fold_dir, exist_ok=True)

        pred_by_seed, val_rmse = {}, {}
        for seed in seeds_cv:
            params = dict(LGB_PARAMS_BASE, seed=seed, bagging_seed=seed,
                          feature_fraction_seed=seed, data_random_seed=seed)
            dtr  = lgb.Dataset(Xtr,  label=ytr,  categorical_feature=ds.cat_cols, free_raw_data=False)
            dval = lgb.Dataset(Xval, label=yval, categorical_feature=ds.cat_cols,
                               reference=dtr, free_raw_data=False)
            hist = {}
            model = lgb.train(params, dtr, num_boost_round=NUM_BOOST_ROUND,
                              valid_sets=[dval], valid_names=['val'],
                              callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                                         lgb.log_evaluation(period=300),
                                         lgb.record_evaluation(hist)])
            bi = model.best_iteration; all_best.append(bi)
            val_rmse[seed] = float(np.nanmin(hist['val']['rmse']))
            pred_by_seed[seed] = model.predict(Xte, num_iteration=bi).astype(np.float32)
            model.save_model(os.path.join(fold_dir, f"model_seed{seed}.txt"), num_iteration=bi)
            seed_rows.append({'test_fold': test_fold, 'val_fold': val_fold, 'seed': seed,
                              'best_iter': bi, 'val_rmse_cm': round(val_rmse[seed], 4)})
            print(f"  seed {seed}: best_iter={bi}  val-RMSE={val_rmse[seed]:.3f} cm")
            del model, dtr, dval, hist; gc.collect()

        best = sorted(val_rmse, key=val_rmse.get)[:args.top_k]
        stacked = np.vstack([pred_by_seed[s] for s in best])
        ens, ens_std = stacked.mean(0).astype(np.float32), stacked.std(0).astype(np.float32)

        m = mlkit.grouped_metrics(ens, yte, starts, len(yte))
        names_te = [ds.kept_names[i] for i in np.unique(wid_te)]
        pd.DataFrame({k: m[k] for k in ['NSE', 'KGE_a', 'R2', 'Bias', 'RMSE', 'alpha', 'r']},
                     index=names_te).to_csv(os.path.join(fold_dir, "ensemble_test_scores.csv"),
                                            sep=";", float_format="%.4f")
        pd.DataFrame({'well': [ds.kept_names[i] for i in wid_te],
                      'time': pd.to_datetime(ds.date[te]),
                      'ens_mean': ens, 'ens_std': ens_std, 'obs': yte}).to_csv(
            os.path.join(fold_dir, "ensemble_test_results.csv"),
            sep=";", float_format="%.4f", index=False)

        cv_rows.append({'test_fold': test_fold, 'val_fold': val_fold,
                        'n_test_wells': len(names_te), 'best_seeds': "|".join(map(str, best)),
                        'KGE_a_median': round(float(np.nanmedian(m['KGE_a'])), 4),
                        'NSE_median':   round(float(np.nanmedian(m['NSE'])), 4),
                        'R2_median':    round(float(np.nanmedian(m['R2'])), 4),
                        'RMSE_cm_median': round(float(np.nanmedian(m['RMSE'])), 4),
                        'alpha_median': round(float(np.nanmedian(m['alpha'])), 4)})
        print(f"  >> fold {test_fold}: KGE_a med={np.nanmedian(m['KGE_a']):.3f}  "
              f"NSE med={np.nanmedian(m['NSE']):.3f}  RMSE med={np.nanmedian(m['RMSE']):.2f} cm")
        del Xtr, Xval, Xte, pred_by_seed, stacked; gc.collect()

    cv = pd.DataFrame(cv_rows)
    cv.to_csv(os.path.join(out_dir, "cv_summary.csv"), sep=";", float_format="%.4f", index=False)
    pd.DataFrame(seed_rows).to_csv(os.path.join(out_dir, "cv_seed_summary.csv"),
                                   sep=";", float_format="%.4f", index=False)
    print(f"\n{'='*60}\nSPATIAL-CV ({exp.name})\n{'='*60}")
    print(cv.to_string(index=False))
    for col in ['KGE_a_median', 'NSE_median', 'RMSE_cm_median']:
        print(f"  {col}: mean={cv[col].mean():.3f}  std={cv[col].std():.3f}  "
              f"min={cv[col].min():.3f}  max={cv[col].max():.3f}")

    # ---- deployment ensemble on ALL wells ----
    final_rounds = int(np.median(all_best))
    final_dir = os.path.join(out_dir, "final"); os.makedirs(final_dir, exist_ok=True)
    seeds_final = make_seeds(args.final_seeds)
    print(f"\n{'='*60}\nFINAL: all {len(ds.kept_names)} wells, {len(ds.X)} rows, "
          f"rounds={final_rounds}, seeds={len(seeds_final)}\n{'='*60}")
    for seed in seeds_final:
        params = dict(LGB_PARAMS_BASE, seed=seed, bagging_seed=seed,
                      feature_fraction_seed=seed, data_random_seed=seed)
        dall = lgb.Dataset(ds.X, label=ds.y, categorical_feature=ds.cat_cols, free_raw_data=False)
        model = lgb.train(params, dall, num_boost_round=final_rounds,
                          callbacks=[lgb.log_evaluation(period=300)])
        model.save_model(os.path.join(final_dir, f"model_seed{seed}.txt"))
        print(f"  final seed {seed} ({final_rounds} rounds).")
        del model, dall; gc.collect()
    pd.DataFrame({'seed': seeds_final, 'rounds': final_rounds}).to_csv(
        os.path.join(final_dir, "final_members.csv"), sep=";", index=False)

    print(f"\nFinished -> {out_dir}")
    print("Raster inference: load final/model_seed*.txt, build the "
          "feature_columns.csv columns per cell with categories.json, predict, "
          "average over the members (GWA cm) + std (uncertainty).")


if __name__ == '__main__':
    main()

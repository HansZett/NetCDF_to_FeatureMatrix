# -*- coding: utf-8 -*-
"""
shap_bundle_export_weekly.py
============================
Weekly counterpart of shap_bundle_export.py (monthly, analysis layer). It
writes a SHAP bundle in exactly the same format, so the downstream figure and
table scripts treat the weekly benchmark like any monthly configuration. The
analysis layer that reads these bundles is not part of this repository; the
script is kept here because it needs the weekly model code (config.py,
shap_lgbm.py).

WHAT IT REUSES. The test matrix is built by build_test_matrix() from
shap_lgbm.py, which in turn uses config.make_lag_features, the same function
lgbm.py trains with. The matrix can therefore not drift from the one the
boosters saw. The ten seed models are averaged per row, as in the monthly
temporal case, which is exact because the decomposition is additive.

WHERE THE GROUPS COME FROM. The weekly pipeline has no feature metatable. Its
feature names, however, are the GEMS-GER predictors with weekly lags and
rolling means, and every base variable of the weekly set also occurs in the
monthly GEMS configuration. The source group and the plot label are therefore
read from features_meta_GEMS-GER.csv, the same table the monthly side uses. No
second, hand-written mapping is introduced that could disagree with it.

Time spans are the weekly ones, lag 1, 2, 4, 8, 13, 26 and 52 weeks and
rolling means over 4, 13, 26 and 52 weeks, plus the static features.

Needs the trained LightGBM seed models results_lgbm/run<seed>/model.txt
(lgbm.py) and features_meta_GEMS-GER.csv in 03_Dataframes/03_Metadata.

Outputs -> <out-dir> (default <project-root>/07_Data_Analysis_Output/
                      shap_analysis/_bundles) :
    GEMS_weekly.npz              top-n SHAP values, feature values, labels
    GEMS_weekly_per_feature.csv  mean |SHAP|, mean SHAP, shap_corr per feature

Usage (on the cluster: sbatch submit/run_bundle_export.sh [args]):
    python shap_bundle_export_weekly.py
    python shap_bundle_export_weekly.py --top-n 25 --sample-rows 60000
    python shap_bundle_export_weekly.py --cache-x        # reuse the parquet cache
"""
import warnings; warnings.filterwarnings('ignore')
import os, sys, gc, re, argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import config
import shap_lgbm
from shap_lgbm import build_test_matrix, driver_group

_WIN_RE = re.compile(r'_(lag|rmean)(\d+)$')
_EID = 'GEMS_weekly'


def window_label(feat):
    m = _WIN_RE.search(feat)
    return f"{m.group(1)}{m.group(2)}" if m else 'static'


def label_maps_from_gems(meta_dir):
    """base variable -> (plot label, source group), from the monthly GEMS table."""
    path = os.path.join(meta_dir, 'features_meta_GEMS-GER.csv')
    if not os.path.isfile(path):
        raise SystemExit(f"Metadata table missing: {path}\n"
                         f"Pass the folder 03_Dataframes/03_Metadata with --meta-dir.")
    meta = pd.read_csv(path, sep=';', dtype=str).fillna('')
    plot, grp = {}, {}
    for _, r in meta.iterrows():
        key = re.sub(r'[^0-9A-Za-z_]', '_', r['feature_raw_name'])   # mlkit.sanitize
        plot[key] = r.get('plot_name') or key
        grp[key] = (r.get('var_group') or r.get('group') or 'other')
    return plot, grp


def value_response(shap_values, X, feat_names, cat_cols):
    """Spearman rank correlation between a feature value and its own SHAP
    contribution, per feature. Identical definition to the monthly side:
    positive means a high value raises the prediction, nominal columns and
    constant columns stay undetermined."""
    cat = set(cat_cols or [])
    out = []
    for j, f in enumerate(feat_names):
        if f in cat:
            out.append(np.nan); continue
        v = pd.to_numeric(X.iloc[:, j], errors='coerce').to_numpy(dtype=np.float64)
        c = np.asarray(shap_values[:, j], dtype=np.float64)
        ok = np.isfinite(v) & np.isfinite(c)
        if ok.sum() < 3:
            out.append(np.nan); continue
        rv = pd.Series(v[ok]).rank().to_numpy()
        rc = pd.Series(c[ok]).rank().to_numpy()
        if rv.std() == 0 or rc.std() == 0:
            out.append(np.nan); continue
        out.append(float(np.corrcoef(rv, rc)[0, 1]))
    return out


def _sample_idx(n, k, seed):
    if 0 < k < n:
        return np.sort(np.random.default_rng(seed).choice(n, k, replace=False))
    return np.arange(n)


def main():
    ap = argparse.ArgumentParser()
    # config.PROJECT_ROOT: ML_PROJECT_ROOT or the first parent folder that
    # contains 03_Dataframes.
    ap.add_argument('--project-root', default=config.PROJECT_ROOT)
    ap.add_argument('--meta-dir', default=None,
                    help="Default: <project-root>/03_Dataframes/03_Metadata")
    ap.add_argument('--out-dir', default=None,
                    help="Default: <project-root>/07_Data_Analysis_Output/"
                         "shap_analysis/_bundles")
    ap.add_argument('--results-dir', default=config.PTH_OUT_LGBM,
                    help="folder with run<seed>/model.txt")
    ap.add_argument('--seeds', type=int, nargs='+', default=config.HP_SEEDS)
    ap.add_argument('--top-n', type=int, default=20)
    ap.add_argument('--sample-rows', type=int, default=40_000)
    ap.add_argument('--sample-seed', type=int, default=42)
    ap.add_argument('--cache-x', action='store_true',
                    help="read X_test from the parquet cache, or write it there")
    args = ap.parse_args()

    meta_dir = args.meta_dir or os.path.join(args.project_root, '03_Dataframes', '03_Metadata')
    out_dir = args.out_dir or os.path.join(args.project_root, '07_Data_Analysis_Output',
                                           'shap_analysis', '_bundles')
    os.makedirs(out_dir, exist_ok=True)
    print(f"project-root : {args.project_root}")
    print(f"meta-dir     : {meta_dir}")
    print(f"out-dir      : {out_dir}")

    plot_map, group_map = label_maps_from_gems(meta_dir)

    # ---- test matrix, identical to the one the boosters were scored on ----
    cache = getattr(shap_lgbm, 'pth_cache_x', None)
    if args.cache_x and cache and os.path.isfile(cache):
        print(f"X_test from cache: {cache}")
        X = pd.read_parquet(cache)
        cat_cols = [c for c in config.CATEGORICAL_STATICS if c in X.columns]
        for c in cat_cols:
            X[c] = X[c].astype('category')
    else:
        print("Building X_test (a few minutes) ...")
        t0 = datetime.now()
        X, static_cat_dtypes = build_test_matrix()
        print(f"  done in {(datetime.now()-t0).total_seconds()/60:.1f} min, shape {X.shape}")
        cat_cols = list(static_cat_dtypes.keys())
        if args.cache_x and cache:
            try:
                X.to_parquet(cache, index=False); print(f"  cache written: {cache}")
            except Exception as e:
                print(f"  caching failed ({e}), continuing in memory")

    # nominal for the booster, and therefore without a value order to colour by
    nominal = set(getattr(config, 'LGBM_CATEGORICAL_STATICS', config.CATEGORICAL_STATICS))
    nominal |= set(cat_cols)
    nominal &= set(X.columns)

    X = X.iloc[_sample_idx(len(X), args.sample_rows, args.sample_seed)].reset_index(drop=True)
    feat_names = list(X.columns)
    print(f"Sample: {X.shape[0]} rows x {X.shape[1]} features")

    # ---- average the seed members per row ----
    acc, n = None, 0
    for s in args.seeds:
        mp = os.path.join(args.results_dir, f"run{s}", 'model.txt')
        if not os.path.isfile(mp):
            print(f"  run{s}/model.txt missing -- skip"); continue
        booster = lgb.Booster(model_file=mp)
        sv = np.asarray(booster.predict(X, pred_contrib=True)[:, :-1], dtype=np.float64)
        acc = sv if acc is None else acc + sv
        n += 1
        print(f"  + run{s}")
        del booster, sv; gc.collect()
    if not n:
        raise SystemExit(f"no run<seed>/model.txt under {args.results_dir}")
    shap_values = (acc / n).astype(np.float32)
    del acc; gc.collect()
    print(f"Contributions averaged over {n} seeds.")

    base = [driver_group(f) for f in feat_names]
    unmapped = sorted({b for b in base if b not in group_map})
    if unmapped:
        print(f"  WARN: {len(unmapped)} base variables not in the GEMS metadata table "
              f"-> 'other': {unmapped[:8]}")

    per_feat = pd.DataFrame({
        'feature': feat_names,
        'plot_name': [plot_map.get(b, b) for b in base],
        'group': [group_map.get(b, 'other') for b in base],
        'window': [window_label(f) for f in feat_names],
        'is_cat': [f in nominal for f in feat_names],
        'mean_abs_shap': np.abs(shap_values).mean(0),
        'mean_shap': shap_values.mean(0),
        'shap_corr': value_response(shap_values, X, feat_names, nominal),
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

    per_feat.to_csv(os.path.join(out_dir, f"{_EID}_per_feature.csv"),
                    sep=';', index=False, float_format='%.6f')

    top = per_feat.head(args.top_n)
    cols = [feat_names.index(f) for f in top['feature']]
    shap_top = shap_values[:, cols].astype(np.float32)

    xv = np.empty_like(shap_top)
    for j, f in enumerate(top['feature']):
        col = X[f]
        if str(col.dtype) == 'category':
            xv[:, j] = col.cat.codes.to_numpy(dtype=np.float32)
        else:
            xv[:, j] = pd.to_numeric(col, errors='coerce').to_numpy(dtype=np.float32)

    info = [
        f"experiment={_EID}",
        "mode=temporal",
        "unit_kind=seeds",
        f"n_units={n}",
        f"n_rows={shap_top.shape[0]}",
        f"n_features_total={len(feat_names)}",
        f"n_features_kept={shap_top.shape[1]}",
        "target_kind=zscore",
        "shap_unit=z",
        "resolution=weekly",
        f"sample_rows={args.sample_rows}",
        f"sample_seed={args.sample_seed}",
    ]

    fp = os.path.join(out_dir, f"{_EID}.npz")
    np.savez_compressed(
        fp,
        shap=shap_top,
        xval=xv,
        feature=np.array(top['feature'], dtype=object).astype('U'),
        plot_name=np.array(top['plot_name'], dtype=object).astype('U'),
        group=np.array(top['group'], dtype=object).astype('U'),
        window=np.array(top['window'], dtype=object).astype('U'),
        is_cat=np.array(top['is_cat'], dtype=bool),
        mean_abs_shap=top['mean_abs_shap'].to_numpy(dtype=np.float64),
        info=np.array(info, dtype=object).astype('U'),
    )
    print(f"-> {fp}  ({shap_top.shape[0]} x {shap_top.shape[1]}, "
          f"{os.path.getsize(fp)/1048576:.1f} MB)")
    print(f"-> {os.path.join(out_dir, _EID + '_per_feature.csv')}")


if __name__ == '__main__':
    main()

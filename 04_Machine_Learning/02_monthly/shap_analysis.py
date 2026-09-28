# -*- coding: utf-8 -*-
"""
shap_analysis.py  --  TreeSHAP analysis of the trained LightGBM models

Generic SHAP analysis controlled by one experiment. Detects temporal vs
spatial itself (via split.kind): one call per experiment, same command.
NOTE: do not rename this file to shap.py (it would shadow the shap library).

TEMPORAL (split.kind=temporal):
    python shap_analysis.py --exp ERA5_4_twsa
  - Test matrix from mlkit.build_dataset(..., splits=('test',)) -> IDENTICAL
    to the training (no duplicated preprocessing).
  - Explains every run<seed>/model.txt; outputs go to run<seed>/.

SPATIAL (split.kind=spatial, target = GWA in cm):
    python shap_analysis.py --exp ERA5_4_twsa_spatial
  - GWA matrix from mlkit.build_dataset_spatial(...) (all wells, all months)
    -> IDENTICAL feature engineering as the spatial training.
  - DEFAULT: for every outer CV fold, explains the evaluated top-k ensemble
    (fold<k>/model_seed*.txt), and ONLY on the rows of the held-out fold.
    Hence (a) the explained model is exactly the model whose scores are
    reported, and (b) the explanation is out-of-sample, just as in the
    temporal case (there: the test period). Result: n_folds attribution
    sets -> mean AND spread across the folds, analogous to the seed spread
    in the temporal case.
    The contributions of the members of one fold are averaged. SHAP is
    additive, so the mean of the member contributions explains the averaged
    ensemble prediction of that fold exactly.
  - Outputs go to fold<k>/ (next to the explained models and scores).
  - OPTIONAL (--final-ensemble): former behaviour, explains the final
    deployment ensemble (final/model_seed*.txt) on ALL rows. The deployment
    ensemble is trained on all wells and is never evaluated, so the
    explanation is in-sample and describes a model without reported skill
    scores. Intended only for later raster inference.

Every shap_per_feature.csv additionally carries the column shap_corr: the
Spearman rank correlation between a feature value and its own SHAP
contribution, i.e. the value response (positive = a high value raises the
prediction). Categorical features stay NaN. The column is evaluated by the
downstream analysis scripts (not part of this repository).

Driver grouping and plot labels come from the feature metadata table
(var_group / plot_name), not from hard-coded prefix lists. Engineered names
(era5l__tp_lag3) are mapped back to their base column.

Outputs: shap_per_feature.csv, shap_per_driver.csv, shap_bar.png,
shap_summary.png (only if the shap package is installed).
"""
import warnings; warnings.filterwarnings('ignore')
import os, gc, glob, argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from resolve_experiments import load_experiment, PROJECT_ROOT, DEFAULT_META_DIR
import mlkit

try:
    import shap; HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("WARNING: shap not installed -- beeswarm plot is skipped.")

# stable colours per var_group (order -> tab10)
_PALETTE = plt.get_cmap('tab10').colors


def _group_colors(groups):
    uniq = list(dict.fromkeys(groups))
    cmap = {g: _PALETTE[i % len(_PALETTE)] for i, g in enumerate(sorted(uniq))}
    return cmap


# ---------------------------------------------------------------------
# Shared output (tables + bar + beeswarm), used by temporal AND spatial so that
# both paths cannot drift apart. Expects precomputed shap_values
# (n_rows x n_features); for an ensemble these are the averaged member contributions.
# ---------------------------------------------------------------------
def _value_response(shap_values, X, feat_names, cat_cols):
    """Spearman rank correlation between a feature's value and its own SHAP
    contribution, one value per feature, computed on the explained rows.

    Positive means a high value of the feature raises the prediction, negative
    means it lowers it, and the magnitude says how monotone the response is.
    Unlike mean_shap the sign is not confounded by where the explained sample
    sat relative to the training mean. Categorical features carry no value
    order and stay undetermined (NaN), as do constant columns and columns with
    fewer than three usable rows. Ties are handled by average ranks, so the
    Pearson correlation of the ranks is exactly Spearman's coefficient.
    """
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


def _emit_shap(shap_values, X, feat_names, cat_cols, plot, vgrp, base_cols,
               out_dir, title_suffix, max_features=30, save_raw=False, sample_seed=42):
    os.makedirs(out_dir, exist_ok=True)

    # per feature -> base column -> label / var_group
    base = [mlkit.base_column(f, base_cols) for f in feat_names]
    labels  = [plot.get(b, b) for b in base]
    drivers = [vgrp.get(b, 'other') for b in base]

    per_feat = pd.DataFrame({
        'feature': feat_names, 'plot_name': labels, 'driver': drivers,
        'mean_abs_shap': np.abs(shap_values).mean(0),
        'mean_shap': shap_values.mean(0),
        'std_shap': shap_values.std(0),
        'shap_corr': _value_response(shap_values, X, feat_names, cat_cols),
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    per_feat.to_csv(os.path.join(out_dir, 'shap_per_feature.csv'), sep=';', index=False, float_format='%.6f')

    per_driver = (per_feat.groupby('driver', as_index=False)
                  .agg(total_abs_shap=('mean_abs_shap', 'sum'),
                       max_abs_shap=('mean_abs_shap', 'max'),
                       n_features=('feature', 'count'))
                  .sort_values('total_abs_shap', ascending=False).reset_index(drop=True))
    per_driver.to_csv(os.path.join(out_dir, 'shap_per_driver.csv'), sep=';', index=False, float_format='%.6f')

    # ---- bar plot (top N, colour = var_group, axis labels = plot_name) ----
    top = per_feat.head(max_features).iloc[::-1]
    cmap = _group_colors(per_feat['driver'])
    colors = [cmap[d] for d in top['driver']]
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(top) + 1.5))
    ax.barh(top['plot_name'], top['mean_abs_shap'], color=colors)
    ax.set_xlabel('mean |SHAP value|')
    ax.set_title(f'Top {len(top)} Features -- {title_suffix}')
    ax.grid(axis='x', linestyle=':', alpha=0.5)
    present = list(dict.fromkeys(top['driver']))
    ax.legend(handles=[Patch(facecolor=cmap[g], label=g) for g in present],
              loc='lower right', fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'shap_bar.png'), dpi=150); plt.close(fig)

    # ---- beeswarm (optional) with plot_name as feature names ----
    if HAS_SHAP:
        Xp = X.copy()
        for c in cat_cols:
            if c in Xp.columns:
                Xp[c] = Xp[c].cat.codes.astype(np.float32)
        Xp.columns = labels
        sv = shap_values
        if len(Xp) > 5000:
            idx = np.random.default_rng(sample_seed).choice(len(Xp), 5000, replace=False)
            Xp, sv = Xp.iloc[idx], shap_values[idx]
        plt.figure(figsize=(9, 0.32 * max_features + 1.5))
        shap.summary_plot(sv, Xp, max_display=max_features, show=False)
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'shap_summary.png'), dpi=150); plt.close()

    if save_raw:
        np.savez_compressed(os.path.join(out_dir, 'shap_values.npz'),
                            shap_values=shap_values, feature_names=np.array(feat_names, dtype=object))


# ---------------------------------------------------------------------
# TEMPORAL: one model per seed (run<seed>/model.txt)
# ---------------------------------------------------------------------
def shap_for_temporal_seed(seed, X, feat_names, cat_cols, plot, vgrp, base_cols,
                           seed_dir, max_features=30, save_raw=False):
    mp = os.path.join(seed_dir, 'model.txt')
    if not os.path.isfile(mp):
        print(f"  [seed {seed}] model.txt missing -- skip"); return
    booster = lgb.Booster(model_file=mp)
    contribs = booster.predict(X, pred_contrib=True).astype(np.float32)
    shap_values = contribs[:, :-1]
    _emit_shap(shap_values, X, feat_names, cat_cols, plot, vgrp, base_cols,
               seed_dir, f'seed {seed}', max_features=max_features,
               save_raw=save_raw, sample_seed=seed)
    del contribs, shap_values, booster; gc.collect()


# ---------------------------------------------------------------------
# SPATIAL: final deployment ensemble (final/model_seed*.txt), contributions averaged
# ---------------------------------------------------------------------
def shap_for_spatial(X, feat_names, cat_cols, plot, vgrp, base_cols,
                     out_dir, seeds=None, max_features=30, save_raw=False, sample_seed=42):
    final_dir = os.path.join(out_dir, 'final')
    if not os.path.isdir(final_dir):
        print(f"  {final_dir} missing -- run lgbm_spatial.py first."); return
    if seeds:
        model_files = [os.path.join(final_dir, f"model_seed{s}.txt") for s in seeds]
        model_files = [m for m in model_files if os.path.isfile(m)]
    else:
        model_files = sorted(glob.glob(os.path.join(final_dir, "model_seed*.txt")))
    if not model_files:
        print(f"  No final/model_seed*.txt in {final_dir} -- skip."); return

    print(f"  SHAP on the final ensemble: {len(model_files)} members, X{X.shape} "
          f"(averaged TreeSHAP contributions).")
    acc, n = None, 0
    for mp in model_files:
        booster = lgb.Booster(model_file=mp)
        contribs = booster.predict(X, pred_contrib=True)        # (n_rows, n_feat+1)
        sv = np.asarray(contribs[:, :-1], dtype=np.float64)     # drop the bias column
        acc = sv if acc is None else acc + sv
        n += 1
        print(f"    + {os.path.basename(mp)}")
        del booster, contribs, sv; gc.collect()
    shap_values = (acc / n).astype(np.float32)
    del acc; gc.collect()

    _emit_shap(shap_values, X, feat_names, cat_cols, plot, vgrp, base_cols,
               final_dir, f'spatial ensemble (n={n})', max_features=max_features,
               save_raw=save_raw, sample_seed=sample_seed)


# ---------------------------------------------------------------------
# SPATIAL (DEFAULT): per outer CV fold the evaluated top-k ensemble,
# explained ONLY on the rows of the held-out fold.
# ---------------------------------------------------------------------
def _fold_members(out_dir, test_fold, top_k=None):
    """Members of one fold. Prefers the top-k models logged as best_seeds in
    cv_summary.csv (i.e. exactly those that formed the fold ensemble);
    otherwise falls back to all model_seed*.txt of the fold."""
    fold_dir = os.path.join(out_dir, f"fold{test_fold}")
    cv_path = os.path.join(out_dir, "cv_summary.csv")
    if os.path.isfile(cv_path):
        try:
            cv = pd.read_csv(cv_path, sep=";")
            row = cv[cv["test_fold"] == test_fold]
            if len(row) and isinstance(row.iloc[0].get("best_seeds"), str):
                seeds = [t for t in str(row.iloc[0]["best_seeds"]).split("|") if t]
                files = [os.path.join(fold_dir, f"model_seed{t}.txt") for t in seeds]
                files = [f for f in files if os.path.isfile(f)]
                if files:
                    return files[:top_k] if top_k else files
        except Exception as e:
            print(f"  cv_summary.csv not readable ({e}) -- using all fold models.")
    files = sorted(glob.glob(os.path.join(fold_dir, "model_seed*.txt")))
    return files[:top_k] if top_k else files


def shap_for_spatial_folds(ds, plot, vgrp, base_cols, out_dir, n_folds, offset,
                           top_k=None, max_features=30, save_raw=False,
                           sample_rows=100_000, sample_seed=42):
    done = 0
    for test_fold in range(n_folds):
        fold_dir = os.path.join(out_dir, f"fold{test_fold}")
        if not os.path.isdir(fold_dir):
            print(f"  {fold_dir} missing -- skip"); continue
        model_files = _fold_members(out_dir, test_fold, top_k)
        if not model_files:
            print(f"  no model_seed*.txt in {fold_dir} -- skip"); continue

        _, _, te, _ = mlkit.spatial_fold_masks(ds.fold, test_fold, n_folds, offset)
        X_te = ds.X[te].reset_index(drop=True)
        X_te = _subsample(X_te, sample_rows, sample_seed)

        print(f"--- fold {test_fold}: {len(model_files)} members, X{X_te.shape} "
              f"(held-out fold only) ---")
        acc, n = None, 0
        for mp in model_files:
            booster = lgb.Booster(model_file=mp)
            contribs = booster.predict(X_te, pred_contrib=True)   # (n_rows, n_feat+1)
            sv = np.asarray(contribs[:, :-1], dtype=np.float64)   # drop the bias column
            acc = sv if acc is None else acc + sv
            n += 1
            print(f"    + {os.path.basename(mp)}")
            del booster, contribs, sv; gc.collect()
        shap_values = (acc / n).astype(np.float32)
        del acc; gc.collect()

        _emit_shap(shap_values, X_te, ds.feat_names, ds.cat_cols, plot, vgrp, base_cols,
                   fold_dir, f"spatial fold {test_fold} (n={n})",
                   max_features=max_features, save_raw=save_raw, sample_seed=sample_seed)
        del shap_values, X_te; gc.collect()
        done += 1
    if not done:
        print("  No folds explained -- run lgbm_spatial.py first.")
    return done


def _subsample(X, sample_rows, sample_seed):
    if 0 < sample_rows < len(X):
        idx = np.sort(np.random.default_rng(sample_seed).choice(len(X), sample_rows, replace=False))
        return X.iloc[idx].reset_index(drop=True)
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=None,
                    help="temporal: seeds = run<seed>/; spatial with --final-ensemble: final members model_seed<s>.txt")
    ap.add_argument('--project-root', default=PROJECT_ROOT)
    ap.add_argument('--meta-dir', default=DEFAULT_META_DIR)
    ap.add_argument('--results-root', default=None)
    ap.add_argument('--sample-rows', type=int, default=100_000)
    ap.add_argument('--sample-seed', type=int, default=42)
    ap.add_argument('--max-features', type=int, default=30)
    ap.add_argument('--save-raw', action='store_true')
    ap.add_argument('--final-ensemble', action='store_true',
                    help="spatial: explain the deployment ensemble in final/ "
                         "instead of the CV folds (in-sample, not evaluated)")
    ap.add_argument('--top-k', type=int, default=None,
                    help="spatial: explain only the first k members per fold "
                         "(default: all logged in cv_summary.csv)")
    args = ap.parse_args()

    exp = load_experiment(args.exp, meta_dir=args.meta_dir)
    out_dir = os.path.join(args.results_root or args.project_root, exp.output_path)
    mode = (exp.split or {}).get('kind', 'temporal')
    print(f"SHAP for {exp.name} (split={mode}) -> {out_dir}")

    plot, vgrp, base_cols = mlkit.label_maps(exp, meta_dir=args.meta_dir)

    # ---- SPATIAL ----
    if mode == 'spatial':
        ds = mlkit.build_dataset_spatial(exp, project_root=args.project_root, meta_dir=args.meta_dir)
        if args.final_ensemble:
            X = _subsample(ds.X, args.sample_rows, args.sample_seed)
            shap_for_spatial(X, ds.feat_names, ds.cat_cols, plot, vgrp, base_cols,
                             out_dir, seeds=args.seeds, max_features=args.max_features,
                             save_raw=args.save_raw, sample_seed=args.sample_seed)
            print(f"\nSHAP finished -> {os.path.join(out_dir, 'final')}")
        else:
            shap_for_spatial_folds(ds, plot, vgrp, base_cols, out_dir,
                                   n_folds=int(exp.split['n_folds']),
                                   offset=int(exp.split['val_fold_offset']),
                                   top_k=args.top_k, max_features=args.max_features,
                                   save_raw=args.save_raw,
                                   sample_rows=args.sample_rows,
                                   sample_seed=args.sample_seed)
            print(f"\nSHAP finished -> {os.path.join(out_dir, 'fold*')}")
        return

    # ---- TEMPORAL ----
    ds = mlkit.build_dataset(exp, project_root=args.project_root,
                             meta_dir=args.meta_dir, splits=('test',))
    X = _subsample(ds.X_test, args.sample_rows, args.sample_seed)

    seeds = args.seeds
    if seeds is None:
        seeds = [int(d[3:]) for d in sorted(os.listdir(out_dir))
                 if d.startswith('run') and d[3:].isdigit()] if os.path.isdir(out_dir) else []
    if not seeds:
        print("No seeds/models found -- run lgbm_temporal.py first."); return

    for seed in seeds:
        seed_dir = os.path.join(out_dir, f"run{seed}")
        if not os.path.isdir(seed_dir):
            print(f"  {seed_dir} missing -- skip"); continue
        print(f"--- seed {seed} ---")
        shap_for_temporal_seed(seed, X, ds.feat_names, ds.cat_cols, plot, vgrp, base_cols,
                               seed_dir, max_features=args.max_features, save_raw=args.save_raw)


if __name__ == '__main__':
    main()

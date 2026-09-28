# -*- coding: utf-8 -*-
"""
evaluation.py  --  comparative evaluation of several experiments

Controlled by resolved_allowlists.json (display_name, color and output_path
of every experiment are defined there). Run one evaluation per mode.

    python evaluation.py --mode temporal --exps GEMS ERA5_0_base ERA5_4_twsa
    python evaluation.py --mode spatial  --exps GEMS_spatial ERA5_0_base_spatial
    python evaluation.py                                  # default: temporal, all experiments

Modes:
- temporal: for every experiment, run*/results.csv are consolidated over the
  seeds into an ensemble-median prediction per well; then NSE/KGE/RMSE/Bias
  per well. NSE denominator = well_pretest_mean.csv (from lgbm_temporal.py).
- spatial:  the per-well scores already exist (fold<k>/ensemble_test_scores.csv,
  computed by lgbm_spatial.py with mlkit.grouped_metrics). Every well is in
  exactly one test fold, so concatenating all folds gives the full per-well
  table. KGE_a is reported as KGE (classic NSE set-up against the mean).

Afterwards (same for both modes): summary, NSE classes, pairwise Wilcoxon
tests (Holm) and Friedman test, boxplots, NSE class plot, significance
heatmap, empirical CDFs.

Well IDs in the stores are canonical (MW_<id>).

Output: <results_root>/Evaluation_<mode>/ (see README.txt, section 11.3)
"""
import os, json, argparse
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon

from resolve_experiments import PROJECT_ROOT, DEFAULT_META_DIR
import mlkit

METRICS = ['NSE', 'KGE', 'RMSE', 'Bias']
PNG_DPI = 200
# The Bias axis has no fixed limit; it is set from the data of each run.
# Units: temporal Bias/RMSE in GWL metres (results.csv is already
# back-transformed with scalers_y, NOT z-scores), spatial in cm.
PLOT_YLIMS = {'NSE': (-1, 1), 'RMSE': (0, None), 'KGE': (-1, 1)}
BIAS_CLIP_PCT = 99          # Bias axis limited to this |Bias| percentile (x1.15)
# NSE classes (label, colour, predicate).
NSE_BINS = [('poor',           '#dc143c', lambda s: s <= 0),
            ('unsatisfactory', '#ff952e', lambda s: (s > 0)    & (s <= 0.5)),
            ('satisfactory',   '#98fb98', lambda s: (s > 0.5)  & (s <= 0.65)),
            ('good',           '#53eca4', lambda s: (s > 0.65) & (s <= 0.75)),
            ('very good',      '#0f8558', lambda s: s > 0.75)]


# ---------------------------------------------------------------------
def _sep(path):
    with open(path) as fh:
        return ';' if ';' in fh.readline() else ','


def consolidate_runs(exp_dir):
    """Merge run*/results.csv into consolidated_obs_sim/<ID>_obs_sim.csv (one sim column per seed)."""
    runs = sorted(d for d in os.listdir(exp_dir)
                  if d.startswith('run') and os.path.isdir(os.path.join(exp_dir, d)))
    out = os.path.join(exp_dir, 'consolidated_obs_sim')
    if not runs:
        return out if os.path.exists(out) else exp_dir
    os.makedirs(out, exist_ok=True)
    dfs = [pd.read_csv(os.path.join(exp_dir, r, 'results.csv'), sep=_sep(os.path.join(exp_dir, r, 'results.csv')))
           for r in runs if os.path.exists(os.path.join(exp_dir, r, 'results.csv'))]
    if not dfs:
        return out
    sim_cols = [c for c in dfs[0].columns if c.endswith('_sim')]
    for sc in sim_cols:
        ID = sc[:-4]                       # 'MW_1_sim' -> 'MW_1'
        oc = f"{ID}_obs"
        obs, sim = None, {}
        for i, df in enumerate(dfs):
            if sc in df.columns and oc in df.columns:
                sim[str(i)] = df[sc]
                if obs is None:
                    obs = df[oc]
        if obs is None:
            continue
        dfo = pd.DataFrame(sim); dfo.insert(0, 'obs', obs)
        dfo.to_csv(os.path.join(out, f"{ID}_obs_sim.csv"), sep=';', index=False)
    return out


def pretest_mean_map(exp_dir, ids, profile, hull, project_root):
    """Read well_pretest_mean.csv (from lgbm_temporal.py); otherwise recompute it from the store."""
    fp = os.path.join(exp_dir, 'well_pretest_mean.csv')
    if os.path.exists(fp):
        m = pd.read_csv(fp, sep=';')
        return dict(zip(m['ID'].astype(str), m['pretest_mean'].astype(float)))
    # fallback
    paths = mlkit._profile_paths(project_root)[profile]
    test_start = pd.to_datetime(hull['test_start'] + '-01')
    out = {}
    for ID in ids:
        for cand in (f"{ID}.csv", f"{ID}_monthly.csv"):
            p = os.path.join(paths['dyn_dir'], cand)
            if os.path.exists(p):
                s = pd.read_csv(p, index_col=0, parse_dates=[0])
                s = s.loc[hull['time_start']:hull['time_end'], paths['target']]
                out[ID] = float(s[s.index < test_start].mean())
                break
    return out


def score_experiment(exp_dir, profile, hull, project_root):
    cons = consolidate_runs(exp_dir)
    files = [f for f in os.listdir(cons) if f.endswith('_obs_sim.csv')] if os.path.isdir(cons) else []
    ids = [f[:-len('_obs_sim.csv')] for f in files]
    means = pretest_mean_map(exp_dir, ids, profile, hull, project_root)
    rows = []
    for ID in ids:
        res = pd.read_csv(os.path.join(cons, f"{ID}_obs_sim.csv"), sep=';')
        obs = res['obs'].to_numpy()
        sim = np.median(res.drop(columns=['obs']).to_numpy(), axis=1)
        err = sim - obs
        mpt = means.get(ID, np.nan)
        denom = np.sum((obs - mpt) ** 2)
        NSE = 1 - np.sum(err ** 2) / denom if denom and not np.isnan(mpt) else np.nan
        if len(sim) > 1 and np.std(obs) > 0 and np.std(sim) > 0:
            r = stats.pearsonr(sim, obs)[0]
            KGE = 1 - np.sqrt((r - 1) ** 2 + (np.std(sim) / np.std(obs) - 1) ** 2 + (np.mean(sim) / np.mean(obs) - 1) ** 2)
        else:
            KGE = np.nan
        rows.append({'ID': ID, 'NSE': NSE, 'KGE': KGE,
                     'RMSE': float(np.sqrt(np.mean(err ** 2))), 'Bias': float(np.mean(err))})
    return pd.DataFrame(rows).set_index('ID') if rows else pd.DataFrame(columns=['ID'] + METRICS).set_index('ID')


def score_experiment_spatial(exp_dir):
    """Spatial counterpart of score_experiment: concatenates all fold*/ensemble_test_scores.csv
    (per-well NSE/KGE_a/Bias/RMSE, from mlkit.grouped_metrics in lgbm_spatial.py).
    Every well is in exactly one test fold -> no duplicates."""
    folds = sorted(d for d in os.listdir(exp_dir)
                   if d.startswith('fold') and os.path.isdir(os.path.join(exp_dir, d)))
    rows = []
    for fd in folds:
        fp = os.path.join(exp_dir, fd, 'ensemble_test_scores.csv')
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp, sep=_sep(fp), index_col=0)
        df.index = df.index.astype(str); df.index.name = 'ID'
        rows.append(df)
    if not rows:
        return pd.DataFrame(columns=METRICS).rename_axis('ID')
    out = pd.concat(rows, axis=0)
    out = out.rename(columns={'KGE_a': 'KGE'})              # unified columns
    keep = [c for c in ('NSE', 'KGE', 'RMSE', 'Bias') if c in out.columns]
    return out[keep]


def holm(pvals):
    p = np.asarray(pvals, float); order = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, min((m - rank) * p[idx], 1.0)); adj[idx] = run
    return adj


# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exps', nargs='+', default=None, help="experiment IDs (default: all of the selected mode)")
    ap.add_argument('--mode', choices=('temporal', 'spatial'), default='temporal',
                    help="temporal reads run*/results.csv; spatial reads fold*/ensemble_test_scores.csv.")
    ap.add_argument('--meta-dir', default=DEFAULT_META_DIR)
    ap.add_argument('--project-root', default=PROJECT_ROOT)
    ap.add_argument('--results-root', default=None)
    ap.add_argument('--out', default=None, help="output subfolder under results-root "
                                                "(default: Evaluation_<mode>)")
    args = ap.parse_args()
    if args.out is None:
        args.out = f"Evaluation_{args.mode}"

    with open(os.path.join(args.meta_dir, 'resolved_allowlists.json'), encoding='utf-8') as fh:
        data = json.load(fh)

    def _kind(eid):
        return data['experiments'][eid].get('split', {}).get('kind', 'temporal')

    # Selection: explicit or all, then filtered to the selected mode.
    if args.exps:
        exps = [e for e in args.exps if _kind(e) == args.mode]
        skipped = [e for e in args.exps if _kind(e) != args.mode]
        if skipped:
            print(f"  Note (--mode={args.mode}): skipped because of a different split.kind: {skipped}")
    else:
        exps = [e for e in data['experiments'] if _kind(e) == args.mode]
    if not exps:
        print(f"No experiments for --mode={args.mode}."); return

    root = args.results_root or args.project_root
    out_dir = os.path.join(root, args.out); os.makedirs(out_dir, exist_ok=True)
    print(f"Evaluation (mode={args.mode}) of {len(exps)} experiments -> {out_dir}")

    models = {}
    for eid in exps:
        e = data['experiments'][eid]
        models[eid] = {
            'dir': os.path.join(root, e['output_path']),
            'display': e['display_name'], 'color': e['color'] or '#1E88E5',
            'profile': e['store_profile'],
            'hull': {'time_start': e['time_start'], 'time_end': e['time_end'], 'test_start': e['test_start']},
        }

    # ---- scores ----
    all_scores = []
    for eid, m in models.items():
        if not os.path.isdir(m['dir']):
            print(f"  {eid}: {m['dir']} missing -- skipped."); continue
        if args.mode == 'spatial':
            sc = score_experiment_spatial(m['dir'])
        else:
            sc = score_experiment(m['dir'], m['profile'], m['hull'], args.project_root)
        sc.to_csv(os.path.join(out_dir, f"scores_{eid}.csv"), sep=';')
        print(f"  {eid:<28} n_wells={len(sc):>4}  NSE median={sc['NSE'].median():.3f}")
        all_scores.append(sc.rename(columns=lambda c: f"{c}_{eid}"))

    if not all_scores:
        print("No scores -- nothing to plot."); return
    merged = all_scores[0]
    for s in all_scores[1:]:
        merged = merged.join(s, how='outer')
    merged.index.name = 'ID'
    merged.to_csv(os.path.join(out_dir, 'merged_scores.csv'), sep=';')
    present = [eid for eid in models if f"NSE_{eid}" in merged.columns]

    # ---- Summary ----
    srows = []
    for metric in METRICS:
        for eid in present:
            col = f"{metric}_{eid}"
            if col in merged:
                v = merged[col].dropna()
                srows.append({'Metric': metric, 'Experiment': eid, 'Min': v.min(),
                              'Median': v.median(), 'Mean': v.mean(), 'Max': v.max(), 'n': len(v)})
    pd.DataFrame(srows).to_csv(os.path.join(out_dir, 'summary_statistics.csv'), sep=';', index=False)

    # ---- NSE classes ----
    binrows = {}
    for eid in present:
        v = merged[f"NSE_{eid}"].dropna()
        binrows[eid] = {lbl: int(cond(v).sum()) for lbl, _, cond in NSE_BINS}
    pd.DataFrame(binrows).T.to_csv(os.path.join(out_dir, 'nse_bin_counts.csv'), sep=';')

    # ---- significance (pairwise Wilcoxon, Holm; Friedman) ----
    sig_rows = []
    if len(present) >= 2:
        for metric in METRICS:
            cols = [f"{metric}_{e}" for e in present]
            d = merged[cols].dropna()
            d.columns = present
            if len(d) == 0:
                continue
            fp = friedmanchisquare(*[d[e].values for e in present]).pvalue if len(present) >= 3 else np.nan
            pairs, raw, md = [], [], []
            for a, b in combinations(present, 2):
                try:
                    raw.append(wilcoxon(d[a].values, d[b].values, zero_method='wilcox').pvalue)
                except ValueError:
                    raw.append(np.nan)
                pairs.append((a, b)); md.append(float(np.median(d[a].values - d[b].values)))
            adj = holm(np.where(np.isnan(raw), 1.0, raw))
            for (a, b), rp, ap_, mdi in zip(pairs, raw, adj, md):
                sig_rows.append({'Metric': metric, 'n_wells': len(d), 'Friedman_p': fp,
                                 'Model_A': a, 'Model_B': b, 'median_diff_A_minus_B': mdi,
                                 'p_raw': rp, 'p_holm': ap_, 'sig_5pct': ap_ < 0.05})
        if sig_rows:
            pd.DataFrame(sig_rows).to_csv(os.path.join(out_dir, 'significance_tests.csv'), sep=';', index=False)

    # ---- Plots ----
    disp = [models[e]['display'] for e in present]
    cols = [models[e]['color'] for e in present]

    # (a) boxplot per metric
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(1.6 * len(present) + 2, 5))
        data = [merged[f"{metric}_{e}"].dropna().values for e in present]
        bp = ax.boxplot(data, positions=np.arange(len(present)), widths=0.55, patch_artist=True,
                        showmeans=True, meanprops=dict(marker='D', markerfacecolor='white',
                        markeredgecolor='#212121', markersize=5),
                        medianprops=dict(color='#212121', linewidth=1.6))
        for box, c in zip(bp['boxes'], cols):
            box.set_facecolor(c); box.set_alpha(0.85); box.set_edgecolor('#333')
        if metric in ('NSE', 'KGE', 'Bias'):
            ax.axhline(0, color='#444', lw=0.7, ls='--', alpha=0.7)
        if metric == 'Bias':
            allv = (np.concatenate([d for d in data if len(d)])
                    if any(len(d) for d in data) else np.array([0.0]))
            L = float(np.nanpercentile(np.abs(allv), BIAS_CLIP_PCT)) if len(allv) else 1.0
            lo, hi = (-max(L * 1.15, 1e-6), max(L * 1.15, 1e-6))
        else:
            lo, hi = PLOT_YLIMS.get(metric, (None, None))
        ax.set_ylim(lo if lo is not None else ax.get_ylim()[0], hi if hi is not None else ax.get_ylim()[1])
        ax.set_xticks(range(len(present))); ax.set_xticklabels(disp, rotation=20, ha='right', fontsize=9)
        ax.set_ylabel(metric); ax.set_title(f'{metric} \u2014 model comparison')
        ax.grid(True, axis='y', ls='--', alpha=0.4); ax.set_axisbelow(True)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, f'comparison_{metric}.png'), dpi=PNG_DPI); plt.close(fig)

    # (b) NSE classes, stacked
    fig, ax = plt.subplots(figsize=(1.6 * len(present) + 2, 5))
    xs = np.arange(len(present)); bottoms = np.zeros(len(present))
    for lbl, color, cond in NSE_BINS:
        h = np.array([int(cond(merged[f"NSE_{e}"].dropna()).sum()) for e in present], float)
        ax.bar(xs, h, bottom=bottoms, color=color, edgecolor='white', label=lbl)
        for i, hi in enumerate(h):
            if hi > 0:
                ax.text(xs[i], bottoms[i] + hi / 2, str(int(hi)), ha='center', va='center', fontsize=8)
        bottoms += h
    ax.set_xticks(xs); ax.set_xticklabels(disp, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Number of wells'); ax.set_title('NSE bin counts')
    h_, l_ = ax.get_legend_handles_labels()
    ax.legend(h_[::-1], l_[::-1], fontsize=8, loc='upper right', frameon=True)
    ax.grid(True, axis='y', ls='--', alpha=0.4); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'nse_bin_counts.png'), dpi=PNG_DPI); plt.close(fig)

    # (c) significance heatmap
    if sig_rows:
        sig = pd.DataFrame(sig_rows)
        fig, axes = plt.subplots(1, len(METRICS), figsize=(3.3 * len(METRICS), 4.2))
        axes = np.atleast_1d(axes); im = None
        for ax, metric in zip(axes, METRICS):
            sub = sig[sig['Metric'] == metric]
            mat = pd.DataFrame(np.nan, index=present, columns=present)
            for _, r in sub.iterrows():
                mat.loc[r['Model_A'], r['Model_B']] = r['p_holm']
                mat.loc[r['Model_B'], r['Model_A']] = r['p_holm']
            im = ax.imshow(mat.values, vmin=0, vmax=0.1, cmap='RdYlGn_r')
            ax.set_xticks(range(len(present))); ax.set_yticks(range(len(present)))
            ax.set_xticklabels([models[e]['display'] for e in present], rotation=45, ha='right', fontsize=7)
            ax.set_yticklabels([models[e]['display'] for e in present], fontsize=7)
            ax.set_title(metric, fontsize=9)
            for i in range(len(present)):
                for j in range(len(present)):
                    v = mat.values[i, j]
                    if not np.isnan(v):
                        ax.text(j, i, f'{v:.2g}', ha='center', va='center', fontsize=6,
                                color='white' if v < 0.01 else 'black')
        if im is not None:
            fig.colorbar(im, ax=list(axes), shrink=0.7, label='Holm p (clipped at 0.1)')
        fig.suptitle('Pairwise significance (Wilcoxon, Holm)')
        fig.savefig(os.path.join(out_dir, 'significance_heatmap.png'), dpi=PNG_DPI, bbox_inches='tight'); plt.close(fig)

    # (d) empirical CDFs of NSE and KGE (one file per metric and mode)
    def _draw_cdf(metric, xlim):
        fig, ax = plt.subplots(figsize=(8, 5.5))
        any_line = False
        for e in present:
            col = f"{metric}_{e}"
            if col not in merged:
                continue
            v = np.sort(merged[col].dropna().values)
            if len(v) == 0:
                continue
            cdf = np.arange(1, len(v) + 1) / len(v)
            ax.plot(v, cdf, label=models[e]['display'], color=models[e]['color'],
                    linewidth=2.0, alpha=0.9)
            any_line = True
        if not any_line:
            plt.close(fig); return
        ax.set_xlim(*xlim); ax.set_ylim(0, 1.02)
        ax.axvline(0, color='#444', lw=0.7, ls='--', alpha=0.5)
        ax.set_xlabel(metric); ax.set_ylabel('Cumulative fraction of wells')
        ax.set_title(f'Empirical CDF of {metric}')
        ax.grid(True, ls='--', alpha=0.4); ax.set_axisbelow(True)
        ax.legend(loc='upper left', fontsize=8, frameon=True)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'cdf_{metric}.png'), dpi=PNG_DPI)
        plt.close(fig)

    _draw_cdf('NSE', (-1, 1))
    _draw_cdf('KGE', (-1, 1))

    print(f"\nEvaluation finished -> {out_dir}")


if __name__ == '__main__':
    main()

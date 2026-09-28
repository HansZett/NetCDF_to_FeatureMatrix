# -*- coding: utf-8 -*-
"""
evaluation.py  --  comparative evaluation of the four weekly models
(single / dynonly / dynstat / lgbm) on the same weekly GEMS-GER dataset.

Deliberately built as a structural TWIN of the monthly evaluation.py (same
metrics NSE/KGE/RMSE/Bias, same NSE classes, same output files and plots), so
that the (LightGBM) results can easily be compared with the monthly setup.
Difference: the compared units here are the four MODEL architectures (not
feature experiments), and the model registry is defined below (MODELS)
instead of in resolved_allowlists.json.

    python evaluation.py
    python evaluation.py --models lgbm dynstat
    python evaluation.py --restrict-to <path>/IDremaining.csv

For every model and well, the ensemble median over the seeds is taken as the
prediction, and NSE/KGE/RMSE/Bias are computed from it. NSE denominator =
pre-test mean per well, computed ONCE from the raw series -> identical for
all models.

With --restrict-to <IDremaining.csv> the evaluation is limited to the well
set of another pipeline (e.g. the monthly one) -> like-for-like for the
weekly->monthly comparison (no re-training needed).

NOTE (public version): the three GEMS-GER reference models single, dynonly
and dynstat are not included in this repository, because the GEMS-GER code is
licensed CC BY-NC-SA 4.0. They are available from the GEMS-GER code
repository (https://github.com/KITHydrogeology/GEMS-GER). Where this file
mentions them, it describes how lgbm.py was aligned with them.
Models without results are skipped, so with this repository alone only
lgbm is evaluated.

Location: 04_Machine_Learning/01_weekly/
Output  : config.PTH_OUT_EVAL (05_Machine_Learning_Results/01_weekly/results_evaluation)
"""
import os, argparse
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon

import config

# --- identical to the monthly evaluation.py -------------------------------
METRICS = ['NSE', 'KGE', 'RMSE', 'Bias']
PNG_DPI = 200
PLOT_YLIMS = {'NSE': (-1, 1), 'RMSE': (0, None), 'KGE': (-1, 1)}
BIAS_CLIP_PCT = 99          # Bias axis limited to this |Bias| percentile (x1.15)
NSE_BINS = [('poor',           '#dc143c', lambda s: s <= 0),
            ('unsatisfactory', '#ff952e', lambda s: (s > 0)    & (s <= 0.5)),
            ('satisfactory',   '#98fb98', lambda s: (s > 0.5)  & (s <= 0.65)),
            ('good',           '#53eca4', lambda s: (s > 0.65) & (s <= 0.75)),
            ('very good',      '#0f8558', lambda s: s > 0.75)]

# --- the four weekly models (key -> output folder, display name, colour, type) ---
# kind='single': reads results_single/<ID>_obs_sim.csv directly (already an ensemble).
# kind='runs'  : consolidates run*/results.csv into <ID>_obs_sim.csv.
MODELS = {
    'single':  {'dir': config.PTH_OUT_SINGLE,  'display': 'Single (CNN)',           'color': '#E07A5F', 'kind': 'single'},
    'dynonly': {'dir': config.PTH_OUT_DYNONLY,  'display': 'Global Dyn-Only (LSTM)', 'color': '#3D9970', 'kind': 'runs'},
    'dynstat': {'dir': config.PTH_OUT_DYNSTAT,  'display': 'Global Dyn+Stat (LSTM)', 'color': '#0074D9', 'kind': 'runs'},
    'lgbm':    {'dir': config.PTH_OUT_LGBM,     'display': 'Global LGBM',            'color': '#001f3f', 'kind': 'runs'},
}
MODEL_ORDER = ['single', 'dynonly', 'dynstat', 'lgbm']


# ---------------------------------------------------------------------
def _sep(path):
    with open(path) as fh:
        return ';' if ';' in fh.readline() else ','


def consolidate_runs(exp_dir):
    """Merge run*/results.csv into consolidated_obs_sim/<ID>_obs_sim.csv
    (columns: 'obs' + one sim column per seed). Same logic as monthly."""
    runs = sorted(d for d in os.listdir(exp_dir)
                  if d.startswith('run') and os.path.isdir(os.path.join(exp_dir, d)))
    out = os.path.join(exp_dir, 'consolidated_obs_sim')
    if not runs:
        return out if os.path.exists(out) else exp_dir
    os.makedirs(out, exist_ok=True)
    dfs = []
    for r in runs:
        fp = os.path.join(exp_dir, r, 'results.csv')
        if os.path.exists(fp):
            dfs.append(pd.read_csv(fp, sep=_sep(fp)))
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


def _load_obs_sim(path):
    """obs vector + ensemble-median sim from an obs_sim file (sim columns =
    seeds). Accepts 'obs' (consolidated) as well as 'GWL' (single) as the obs
    column and drops a possible date column (object dtype)."""
    df = pd.read_csv(path, sep=_sep(path))
    obs_col = 'obs' if 'obs' in df.columns else ('GWL' if 'GWL' in df.columns else None)
    if obs_col is None:
        return None, None
    obs = df[obs_col].to_numpy(dtype=float)
    sims = df.drop(columns=[obs_col]).select_dtypes(include=[np.number])
    if sims.shape[1] == 0:
        return None, None
    sim = np.nanmedian(sims.to_numpy(dtype=float), axis=1)
    return obs, sim


def pretest_mean_map():
    """Pre-test mean per well from the raw series (< test_start), computed ONCE
    and identical for all models (NSE denominator). The DATE_END cap does not
    matter here, because only rows before the test period are used."""
    files, names = config.list_wells()
    out = {}
    for f, name in zip(files, names):
        try:
            df = config.load_well_dynamic(f)
        except Exception:
            continue
        s = df.loc[df.index < config.DATE_START_TEST, 'GWL']
        if len(s):
            out[name] = float(s.mean())
    return out


def _metrics(obs, sim, mpt):
    """NSE/KGE/RMSE/Bias -- same formulas as the monthly evaluation.py."""
    m = ~(np.isnan(obs) | np.isnan(sim))
    obs, sim = obs[m], sim[m]
    if len(obs) < 2:
        return np.nan, np.nan, np.nan, np.nan
    err = sim - obs
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    denom = np.sum((obs - mpt) ** 2) if mpt == mpt else np.nan      # mpt == mpt is False for NaN
    nse = float(1 - np.sum(err ** 2) / denom) if (denom and denom > 0) else np.nan
    so, sp = np.std(obs), np.std(sim)
    if so > 0 and sp > 0 and np.mean(obs) != 0:
        r = stats.pearsonr(sim, obs)[0]
        kge = float(1 - np.sqrt((r - 1) ** 2 + (sp / so - 1) ** 2 + (np.mean(sim) / np.mean(obs) - 1) ** 2))
    else:
        kge = np.nan
    return nse, kge, rmse, bias


def score_model(key, pmean):
    """Per-well NSE/KGE/RMSE/Bias for one model (ensemble median over seeds)."""
    m = MODELS[key]
    d = m['dir']
    if not os.path.isdir(d):
        return None
    if m['kind'] == 'single':
        src = d
    else:
        src = consolidate_runs(d)
    if not os.path.isdir(src):
        return None
    files = sorted(f for f in os.listdir(src) if f.endswith('_obs_sim.csv'))
    rows = []
    for f in files:
        ID = f[:-len('_obs_sim.csv')]
        obs, sim = _load_obs_sim(os.path.join(src, f))
        if obs is None:
            continue
        nse, kge, rmse, bias = _metrics(obs, sim, pmean.get(ID, np.nan))
        rows.append({'ID': ID, 'NSE': nse, 'KGE': kge, 'RMSE': rmse, 'Bias': bias})
    if not rows:
        return None
    return pd.DataFrame(rows).set_index('ID')


def holm(pvals):
    p = np.asarray(pvals, float); order = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, min((m - rank) * p[idx], 1.0)); adj[idx] = run
    return adj


def _restrict_ids(path):
    """Read an IDremaining.csv (column 'ID' or the last column) -> set of MW IDs."""
    df = pd.read_csv(path, sep=_sep(path))
    col = 'ID' if 'ID' in df.columns else df.columns[-1]
    return set(df[col].astype(str))


# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=None,
                    help=f"subset of {MODEL_ORDER} (default: all available)")
    ap.add_argument('--out', default=None,
                    help="output folder (default: config.PTH_OUT_EVAL)")
    ap.add_argument('--restrict-to', default=None,
                    help="IDremaining.csv of another setup -> evaluate only its wells")
    args = ap.parse_args()

    chosen = [k for k in MODEL_ORDER if (args.models is None or k in args.models)]
    out_dir = args.out or config.PTH_OUT_EVAL
    os.makedirs(out_dir, exist_ok=True)

    keep_ids = _restrict_ids(args.restrict_to) if args.restrict_to else None
    if keep_ids is not None:
        print(f"Restricting to {len(keep_ids)} wells from {args.restrict_to}")

    print("Pre-test means from the raw series ...")
    pmean = pretest_mean_map()

    # ---- scores per model ----
    scores, present = {}, []
    for key in chosen:
        sc = score_model(key, pmean)
        if sc is None or sc.empty:
            print(f"  {key:<8} no results in {MODELS[key]['dir']} -- skip")
            continue
        if keep_ids is not None:
            sc = sc.loc[sc.index.isin(keep_ids)]
            if sc.empty:
                print(f"  {key:<8} empty after --restrict-to -- skip"); continue
        scores[key] = sc; present.append(key)
        sc.to_csv(os.path.join(out_dir, f"scores_{key}.csv"), sep=';', float_format='%.4f')
        print(f"  {key:<8} n_wells={len(sc):>4}  NSE median={sc['NSE'].median():.3f}")

    if not present:
        print("No scores -- nothing to plot."); return

    merged = scores[present[0]].rename(columns=lambda c: f"{c}_{present[0]}")
    for key in present[1:]:
        merged = merged.join(scores[key].rename(columns=lambda c: f"{c}_{key}"), how='outer')
    merged.index.name = 'ID'
    merged.to_csv(os.path.join(out_dir, 'merged_scores.csv'), sep=';', float_format='%.4f')

    # ---- Summary ----
    srows = []
    for metric in METRICS:
        for key in present:
            v = merged[f"{metric}_{key}"].dropna()
            srows.append({'Metric': metric, 'Model': MODELS[key]['display'],
                          'Min': v.min(), 'Median': v.median(), 'Mean': v.mean(),
                          'Max': v.max(), 'n': len(v)})
    pd.DataFrame(srows).to_csv(os.path.join(out_dir, 'summary_statistics.csv'),
                               sep=';', index=False, float_format='%.4f')
    print("\n" + pd.DataFrame(srows).pivot_table(index='Model', columns='Metric',
                                                  values='Median').to_string())

    # ---- NSE classes ----
    binrows = {MODELS[k]['display']: {lbl: int(cond(merged[f"NSE_{k}"].dropna()).sum())
                                      for lbl, _, cond in NSE_BINS} for k in present}
    pd.DataFrame(binrows).T.to_csv(os.path.join(out_dir, 'nse_bin_counts.csv'), sep=';')

    # ---- significance (Friedman + pairwise Wilcoxon, Holm) ----
    sig_rows = []
    if len(present) >= 2:
        disp_of = {k: MODELS[k]['display'] for k in present}
        for metric in METRICS:
            cols = [f"{metric}_{k}" for k in present]
            d = merged[cols].dropna(); d.columns = present
            if len(d) == 0:
                continue
            fp = friedmanchisquare(*[d[k].values for k in present]).pvalue if len(present) >= 3 else np.nan
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
                                 'Model_A': disp_of[a], 'Model_B': disp_of[b],
                                 'median_diff_A_minus_B': mdi,
                                 'p_raw': rp, 'p_holm': ap_, 'sig_5pct': ap_ < 0.05})
        if sig_rows:
            pd.DataFrame(sig_rows).to_csv(os.path.join(out_dir, 'significance_tests.csv'),
                                          sep=';', index=False, float_format='%.4g')

    # ---- plots (same style as monthly) ----
    disp = [MODELS[k]['display'] for k in present]
    cols = [MODELS[k]['color'] for k in present]

    # (a) boxplot per metric
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(1.6 * len(present) + 2, 5))
        data = [merged[f"{metric}_{k}"].dropna().values for k in present]
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
        h = np.array([int(cond(merged[f"NSE_{k}"].dropna()).sum()) for k in present], float)
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
        sig = pd.DataFrame(sig_rows); disp_idx = [MODELS[k]['display'] for k in present]
        fig, axes = plt.subplots(1, len(METRICS), figsize=(3.3 * len(METRICS), 4.2))
        axes = np.atleast_1d(axes); im = None
        for ax, metric in zip(axes, METRICS):
            sub = sig[sig['Metric'] == metric]
            mat = pd.DataFrame(np.nan, index=disp_idx, columns=disp_idx)
            for _, r in sub.iterrows():
                mat.loc[r['Model_A'], r['Model_B']] = r['p_holm']
                mat.loc[r['Model_B'], r['Model_A']] = r['p_holm']
            im = ax.imshow(mat.values, vmin=0, vmax=0.1, cmap='RdYlGn_r')
            ax.set_xticks(range(len(disp_idx))); ax.set_yticks(range(len(disp_idx)))
            ax.set_xticklabels(disp_idx, rotation=45, ha='right', fontsize=7)
            ax.set_yticklabels(disp_idx, fontsize=7); ax.set_title(metric, fontsize=9)
            for i in range(len(disp_idx)):
                for j in range(len(disp_idx)):
                    v = mat.values[i, j]
                    if not np.isnan(v):
                        ax.text(j, i, f'{v:.2g}', ha='center', va='center', fontsize=6,
                                color='white' if v < 0.01 else 'black')
        if im is not None:
            fig.colorbar(im, ax=list(axes), shrink=0.7, label='Holm p (clipped at 0.1)')
        fig.suptitle('Pairwise significance (Wilcoxon, Holm)')
        fig.savefig(os.path.join(out_dir, 'significance_heatmap.png'), dpi=PNG_DPI, bbox_inches='tight'); plt.close(fig)

    # (d) empirical CDFs of NSE and KGE
    def _draw_cdf(metric, xlim):
        fig, ax = plt.subplots(figsize=(8, 5.5)); any_line = False
        for k in present:
            v = np.sort(merged[f"{metric}_{k}"].dropna().values)
            if len(v) == 0:
                continue
            cdf = np.arange(1, len(v) + 1) / len(v)
            ax.plot(v, cdf, label=MODELS[k]['display'], color=MODELS[k]['color'], linewidth=2.0, alpha=0.9)
            any_line = True
        if not any_line:
            plt.close(fig); return
        ax.set_xlim(*xlim); ax.set_ylim(0, 1.02)
        ax.axvline(0, color='#444', lw=0.7, ls='--', alpha=0.5)
        ax.set_xlabel(metric); ax.set_ylabel('Cumulative fraction of wells')
        ax.set_title(f'Empirical CDF of {metric}')
        ax.grid(True, ls='--', alpha=0.4); ax.set_axisbelow(True)
        ax.legend(loc='upper left', fontsize=8, frameon=True)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, f'cdf_{metric}.png'), dpi=PNG_DPI); plt.close(fig)

    _draw_cdf('NSE', (-1, 1))
    _draw_cdf('KGE', (-1, 1))

    print(f"\nEvaluation finished -> {out_dir}")


if __name__ == '__main__':
    main()

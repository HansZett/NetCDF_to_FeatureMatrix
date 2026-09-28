===============================================================================
01_weekly  --  LightGBM on the weekly GEMS-GER resolution
===============================================================================

PUBLIC VERSION: the three GEMS-GER reference models single (CNN per well),
dynonly (global LSTM) and dynstat (global LSTM + static features) are NOT
included in this repository, because the GEMS-GER code is licensed
CC BY-NC-SA 4.0. They are available from the GEMS-GER code repository
(https://github.com/KITHydrogeology/GEMS-GER). This README still describes
them where lgbm.py was aligned with them, and evaluation.py includes their
results if they exist in the output folders listed in section 5.

Folder contents:
  config.py        shared settings (paths, split dates, seeds, drivers)
  lgbm.py          global LightGBM
  shap_lgbm.py     SHAP for the lgbm.py models
  evaluation.py    evaluation of lgbm (and comparison with the reference
                   models, if their results exist)
  shap_bundle_export_weekly.py
                   SHAP bundle of the lgbm.py models in the format of the
                   monthly analysis layer (not part of this repository);
                   optional, run via submit/run_bundle_export.sh
  submit/          SLURM scripts (env.sh, run_*.sh, submit_all.sh)

Standalone stack that models the WEEKLY native resolution of the GEMS-GER
dataset. Four models on the SAME data, the SAME temporal hull and the SAME
output schema, so they are cleanly comparable -- and so the LightGBM run can
serve as the reference for the manual weekly->monthly comparison.

Unlike the monthly pipeline (experiments.yaml -> resolve_experiments.py ->
mlkit.py) there is NO config resolving here: the weekly feature set is fixed
(15 dynamic drivers + static metadata). Instead of a shared model library there
is only a slim config.py holding exactly the things the four models MUST agree
on. Each model's logic stays in its own file.

This file is meant to be self-contained: someone analysing the results can
work with the outputs using only this README (see section 6, which fully
specifies every file's columns, units and time axis, plus a ready-to-use Python
loader for plotting hydrographs).


-------------------------------------------------------------------------------
1. THE MODELS
-------------------------------------------------------------------------------
single       CNN, ONE model per well        no static   52-week sequence
             SLURM array (1 task/well), 10-seed ensemble   [not included]
dynonly      global LSTM, dynamic-only       no static   52-week sequence
             1 job, loops over seeds                       [not included]
dynstat      global LSTM + dense static arm  label-encode + MinMax
             1 job, loops over seeds                       [not included]
lgbm.py      global LightGBM                  native 'category'  lag/rolling table
             1 job, loops over seeds
shap_lgbm.py TreeSHAP for the lgbm.py models  -> runs after lgbm.py
evaluation.py comparison + significance       -> runs last, reads run<seed>/

single, dynonly, dynstat are the GEMS reference models (not included in this
repository); lgbm is the new comparison partner.


-------------------------------------------------------------------------------
2. SHARED vs DELIBERATELY DIFFERENT
-------------------------------------------------------------------------------
SHARED (config.py -- one source of truth; change here -> all models follow):
input/output paths, the sorted well list, the 9-column static drop, the
nominal static features (CATEGORICAL_STATICS, 7 text-coded columns;
LGBM_CATEGORICAL_STATICS, all 19 nominal columns), split dates (stop=2008-01, test=2013-01, END=2020-12),
N_STEPS_IN=52, HP_SEEDS, the 15 dynamic drivers, the NSE denominator (per-well
pre-test mean), the metric formulas, the results.csv column schema and the
run<seed>/ file names. Plus -- used only by lgbm + shap -- the lag/rolling
configuration and make_lag_features (so both build a byte-identical matrix).

MODEL-SPECIFIC (allowed to differ, by design -- intent, not drift):
- Static encoding: dynstat label-encodes the 7 text-coded nominal columns
  (CATEGORICAL_STATICS) + MinMax (for the dense arm), exactly as in the
  published scripts; the 12 integer-coded nominal columns are passed on as
  numbers there. lgbm and shap_lgbm keep all 19 nominal columns
  (LGBM_CATEGORICAL_STATICS) as native 'category' (optimal splits).
  dynonly/single use no static features.
- Input representation: 52-step sequence (NN) vs lag/rolling table (LGBM) --
  both represent the same t-52..t-1 information window.
- single stays the per-well array job with its own obs/sim output.
- Hyperparameters (LSTM size, LGB params, ...) stay local to each model.


-------------------------------------------------------------------------------
3. TEST-PERIOD CAP (end of 2020)
-------------------------------------------------------------------------------
All four models cap the series at DATE_END = 2020-12-31. The test window is thus
2013-01..2020-12 -- identical to the monthly pipeline (time_end = 2020-12,
test_start = 2013-01). The cap only affects the test tail; train/stop (< 2013)
and the NSE denominator (pre-test mean) are untouched. This makes the
weekly->monthly comparison apples-to-apples.


-------------------------------------------------------------------------------
4. WELL SET (no TWSA filter)
-------------------------------------------------------------------------------
The monthly pipeline drops wells without TWSA coverage (require_twsa_valid). The
weekly data contain NO TWSA at all, so there is no such filter here, and wells
the monthly setup dropped REMAIN in the weekly runs. The set is determined only
by: single/dynonly/dynstat take every well in the data folder; lgbm/shap
additionally only those with a static-metadata row and >= MIN_PRETEST_ROWS
pre-test weeks.

Consequence for the weekly->monthly comparison: the two score distributions run
over DIFFERENT well populations, so an aggregate difference mixes a resolution
effect with a population effect. For goal 1 (LGBM vs GEMS models) this is
harmless: the significance test in evaluation.py uses dropna(), i.e. only wells
present in all models. For a clean weekly->monthly comparison (goal 3) restrict
the weekly scores to the monthly-kept MW_IDs at comparison time
(evaluation.py --restrict-to <IDremaining.csv>; no re-training needed).


===============================================================================
5. WHERE OUTPUTS GO
===============================================================================
Output root (config.PTH_OUT_ROOT):
  <project root>/05_Machine_Learning_Results/01_weekly/
  (project root = the first parent folder of 01_weekly that contains
   03_Dataframes; on the HPC this is the workspace $WS)

  results_single/           (single, reference model, not included)
  results_dynonly/          (dynonly, reference model, not included)
  results_dynstat/          (dynstat, reference model, not included)
  results_lgbm/             (lgbm.py, + shap_lgbm.py artefacts)
  results_evaluation/       (evaluation.py)
Logs go to a separate tree (mirrors the monthly setup):
  $WS/06_Machine_Learning_Logs/01_weekly/<model>/

Directory contents:

  results_<global>/                 (global = dynonly | dynstat | lgbm)
    IDremaining.csv                  wells actually used
    well_pretest_mean.csv            NSE denominator (same as monthly)
    run<seed>/                       one folder per HP seed (HP_SEEDS)
      scores.csv                     per-well NSE/KGE/R2/Bias/MSE/RMSE
      results.csv                    per-well sim & obs on the test period
      losshistory.csv
      model.txt (lgbm) | model.keras (dynonly/dynstat)
      feature_importance.csv         (lgbm only)
      shap_per_feature.csv           (lgbm only, written by shap_lgbm.py)
      shap_per_driver.csv
      shap_bar.png  shap_summary.png
    consolidated_obs_sim/            created by evaluation.py (see 6.3)

  results_single/
    IDremaining.csv
    <ID>_obs_sim.csv                 native per-well ensemble output (see 6.2)

  results_evaluation/                structural twin of the monthly evaluation
    scores_<model>.csv               per-well NSE/KGE/RMSE/Bias, one file/model
    merged_scores.csv                all models side by side (see 6.5)
    summary_statistics.csv
    nse_bin_counts.csv
    significance_tests.csv           Friedman + pairwise Wilcoxon, Holm-adjusted
    comparison_{NSE,KGE,RMSE,Bias}.png
    nse_bin_counts.png  significance_heatmap.png  cdf_NSE.png  cdf_KGE.png


===============================================================================
6. OUTPUT FILE SPECIFICATION  (for downstream data analysis)
===============================================================================
All CSVs use ';' as the separator and '.' as the decimal point. Well IDs are
canonical (MW_<n>). sim and obs are in GWL units (metres), per-well
z-standardised during training and inverse-transformed back to metres before
writing -- so sim and obs are directly comparable to the observed groundwater
level. Every model uses a 10-seed ensemble; the point prediction is the
ROW-WISE MEDIAN across the seed columns.

KEY TIME-AXIS DIFFERENCE (read this before plotting hydrographs):
- single writes a real Date index in its obs_sim file.
- the global models (dynonly/dynstat/lgbm) write their test rows POSITIONALLY,
  WITHOUT dates (the date index was dropped at write time). The rows are the
  weekly test timesteps in chronological order on the continuous weekly grid,
  i.e. exactly the weeks in [test_start, DATE_END]. To get dates, reattach the
  raw weekly index sliced to that window (the loader in 6.6 does this for you).


6.1  scores.csv  (results_<global>/run<seed>/scores.csv)
     Per-seed, per-well metrics. Index = well ID. Columns:
       NSE, KGE, R2, Bias, MSE, RMSE
     Note: evaluation.py does NOT use these; it recomputes metrics from the
     ensemble-median sim (see 6.5). Use these only if you want per-seed spread.


6.2  <ID>_obs_sim.csv  (results_single/)   -- single, HAS dates
     One file per well. Read with index_col=0, parse_dates=[0].
       column 'Date' (index)  weekly timestamp
       column 'GWL'           observed GWL (metres)
       columns '0' .. '9'     simulated GWL per seed (metres)
     Ensemble simulation = median over columns '0'..'9'.


6.3  <ID>_obs_sim.csv  (results_<global>/consolidated_obs_sim/)  -- NO dates
     Created by evaluation.py from run<seed>/results.csv. One file per well.
       column 'obs'           observed GWL (metres)
       columns '0','1',...     simulated GWL per seed (metres)
     Ensemble simulation = median over the seed columns.
     Row i corresponds to the i-th weekly test timestamp (see time-axis note).
     If this folder is missing, run evaluation.py once (it creates it), or build
     it yourself from 6.4.


6.4  results.csv  (results_<global>/run<seed>/results.csv)  -- raw per-seed
     Wide table, one block of three columns per well, plus a leading unnamed
     integer row index. For well <name>:
       <name>_ID      seed/run id (constant within the file)
       <name>_sim     simulated GWL (metres), this seed
       <name>_obs     observed GWL (metres)
     Rows are positional weekly test timesteps (no dates). consolidated_obs_sim
     (6.3) is the convenient consolidation of these across seeds.


6.5  results_evaluation/ files  (model comparison; twin of monthly evaluation)
     scores_<model>.csv   index = ID; columns NSE, KGE, RMSE, Bias
                          (ensemble-median metrics, one row per well)
     merged_scores.csv    index = ID; columns "<metric>_<model>" for every
                          metric x model, e.g. NSE_lgbm, KGE_single, ...
     summary_statistics.csv  long form: Metric, Model, Min, Median, Mean, Max, n
     nse_bin_counts.csv   rows = model display name; columns = NSE classes
                          (poor / unsatisfactory / satisfactory / good / very good
                           at thresholds 0, 0.5, 0.65, 0.75)
     significance_tests.csv  Metric, n_wells, Friedman_p, Model_A, Model_B,
                          median_diff_A_minus_B, p_raw, p_holm, sig_5pct
     Model keys: single, dynonly, dynstat, lgbm.
     Metrics here are NSE, KGE, RMSE, Bias (same set as the monthly evaluation,
     so weekly-lgbm scores line up directly with the monthly ones).


6.6  READY-TO-USE LOADER  (obs + ensemble-median sim, with dates, per well)
     Drop this next to config.py (it imports config) and call load_hydrograph().
     Returns a DataFrame indexed by Date with columns 'obs' and 'sim' in metres.

     ------------------------------------------------------------------
     import os
     import numpy as np
     import pandas as pd
     import config

     TEST_START = pd.to_datetime(config.DATE_START_TEST)
     DATE_END   = pd.to_datetime(config.DATE_END)

     _files, _names = config.list_wells()
     _FILE_OF = dict(zip(_names, _files))     # well ID -> raw weekly file path

     def _ensemble_median(df, obs_col):
         sims = df.drop(columns=[obs_col]).select_dtypes(include=[np.number])
         return sims.median(axis=1).to_numpy()

     def load_hydrograph(model, well_id):
         """obs + ensemble-median sim (GWL metres) for one well, with a Date index.
            model in {'single','dynonly','dynstat','lgbm'}."""
         if model == 'single':
             p = os.path.join(config.PTH_OUT_SINGLE, f'{well_id}_obs_sim.csv')
             df = pd.read_csv(p, sep=';', index_col=0, parse_dates=[0])
             obs = df['GWL'].to_numpy()
             sim = _ensemble_median(df, 'GWL')
             idx = df.index
         else:
             root = {'dynonly': config.PTH_OUT_DYNONLY,
                     'dynstat': config.PTH_OUT_DYNSTAT,
                     'lgbm':    config.PTH_OUT_LGBM}[model]
             p = os.path.join(root, 'consolidated_obs_sim', f'{well_id}_obs_sim.csv')
             df = pd.read_csv(p, sep=';')                 # 'obs' + seed cols, no dates
             obs = df['obs'].to_numpy()
             sim = _ensemble_median(df, 'obs')
             # reattach weekly test dates from the raw series (continuous grid)
             raw = config.load_well_dynamic(_FILE_OF[well_id])
             idx = raw.index[(raw.index >= TEST_START) & (raw.index <= DATE_END)]
             idx = idx[:len(obs)]                         # length safety
         return pd.DataFrame({'obs': obs, 'sim': sim},
                             index=pd.Index(idx, name='Date'))
     ------------------------------------------------------------------

     EXAMPLE -- "plot the simulated GWL hydrographs of models a, b, c for well
     MW_1234 over 2015-2018":

     ------------------------------------------------------------------
     import matplotlib.pyplot as plt
     well = 'MW_1234'
     lo, hi = '2015-01-01', '2018-12-31'
     fig, ax = plt.subplots(figsize=(11, 4))
     first = True
     for model in ['single', 'dynstat', 'lgbm']:
         h = load_hydrograph(model, well).loc[lo:hi]
         if first:                                        # observed once
             ax.plot(h.index, h['obs'], color='k', lw=1.3, label='observed')
             first = False
         ax.plot(h.index, h['sim'], lw=1.1, label=f'{model} (sim)')
     ax.set_xlabel('Date'); ax.set_ylabel('GWL [m]')
     ax.set_title(f'{well} -- simulated vs observed GWL')
     ax.legend(); fig.tight_layout(); fig.savefig(f'{well}_hydrograph.png', dpi=200)
     ------------------------------------------------------------------

     Notes:
     - For the global models this needs results_<global>/consolidated_obs_sim/
       to exist (evaluation.py creates it). single needs nothing extra.
     - To compare against the MONTHLY pipeline, remember the well populations
       differ (section 4) and that both are in metres (section 7).


-------------------------------------------------------------------------------
7. THREE GOALS -> where they are served
-------------------------------------------------------------------------------
1. LGBM ~ GEMS models: evaluation.py compares Single / Dyn-Only / Dyn+Stat /
   LGBM per well (median over seeds), incl. Friedman/Wilcoxon/Holm.
2. Important features: shap_lgbm.py -> per-feature and per-driver SHAP + bar /
   beeswarm plots, per seed, next to feature_importance.csv.
3. weekly->monthly reference: the results_lgbm/ tree mirrors the monthly
   pipeline (same test window, same NSE denominator, same scores.csv). The
   actual comparison against the monthly GEMS-GER LGBM is manual.
   UNITS: RMSE/Bias are in GWL units (metres, per-well z back-transformed) --
   make sure both sides use the same unit when comparing.


-------------------------------------------------------------------------------
8. RUN ORDER (HPC)
-------------------------------------------------------------------------------
Layout: the .py (incl. config.py) live in 04_Machine_Learning/01_weekly/, the
SLURM scripts in the subfolder 01_weekly/submit/ (same split as the monthly
setup). Submit from inside 01_weekly/. Shared environment in submit/env.sh:
the HPC workspace WS, the module devel/python/3.12_gnu_11.4 and the venv
$WS/envs/ml_hydro. All jobs run on CPU. Each run script resolves the code home robustly
(ML_DIR -> SLURM_SUBMIT_DIR -> BASH_SOURCE) and cds there so 'import config'
works.

Everything at once, with dependencies:
    cd 04_Machine_Learning/01_weekly
    ./submit/submit_all.sh

This launches: lgbm; shap with afterok on lgbm; evaluation with afterok on
lgbm.
Paths (out-root, data folder) are read from config.py. Watch with
'squeue -u $USER'.

Skip parts / restrict the well set:
    SKIP_LGBM=1 ./submit/submit_all.sh          # lgbm already finished
    ./submit/submit_all.sh --restrict-to <path>/02_monthly/<exp>/IDremaining.csv
(extra args after submit_all.sh are forwarded to evaluation.py.)

Individually (e.g. to re-run one piece):
    sbatch submit/run_lgbm.sh
    sbatch --dependency=afterok:<lgbm_jid> submit/run_shap.sh
    sbatch submit/run_bundle_export.sh --top-n 25 --cache-x   # optional

Logs: $WS/06_Machine_Learning_Logs/01_weekly/<model>/ (submit_all.sh creates the
subfolders; override with LOGS_ROOT=...). For
standalone submits without the orchestrator, SLURM writes slurm-%j.out in the
code folder.


-------------------------------------------------------------------------------
9. IF SOMETHING CHANGES
-------------------------------------------------------------------------------
- Data moves / split changes: only config.py.
- Different fold / different drivers: fixed here by design; for other feature
  sets use the monthly, config-driven pipeline.
- Paths: config.py finds the project root automatically (first parent folder
  that contains 03_Dataframes; override with the environment variable
  ML_PROJECT_ROOT). The HPC workspace WS, the Python module and the venv are
  set only in submit/env.sh.
- SLURM settings (--partition, --gres, --time, --mem, --mail-user) are in the
  #SBATCH header of each submit/run_*.sh. Change --mail-user to your own
  address.

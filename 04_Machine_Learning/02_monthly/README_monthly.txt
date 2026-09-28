=====================================================================
 02_monthly  -  MONTHLY ML PIPELINE (LightGBM) : OVERVIEW & USAGE
=====================================================================

Modelling groundwater levels from GLOBALLY AVAILABLE data, benchmarked
against the published national dataset GEMS-GER. The pipeline is
CONFIG-DRIVEN: an experiment is one block in experiments.yaml, not its
own code. A single "--exp <ID>" call covers training, SHAP and
evaluation; feature selection, the time/well hull, the target, the
split and the output folder all come from the table.

The pipeline runs in two modes that share the same feature engineering
and the same LightGBM parameters:

  TEMPORAL  - per-well z-scored GWL, date split (train / early-stop /
              test). The original / published setup.
  SPATIAL   - groundwater anomaly in cm (GWA, centred, transferable),
              K-fold cross-validation over whole HYBAS basins. Built
              for spatial transfer and raster inference.

Default = zscore + temporal, so the original experiments stay
bit-identical; the spatial variants are purely additive.


---------------------------------------------------------------------
 1. CORE IDEA
---------------------------------------------------------------------

Three principles carry the structure:

1. The metatable is the truth about features. Whether a column is
   dynamic (lag/rolling), passthrough or static, whether it is
   categorical, its plot label and its group - all of that lives in
   features_meta_global.csv / features_meta_GEMS-GER.csv. No hard-coded
   DROP_* lists in the ML code.

2. experiments.yaml is the truth about experiments. An experiment is a
   feature allowlist (via inheritance + group-include + fine-tuning)
   plus an output folder name, a display name, a colour, and - new -
   an optional target block and split block.

3. One code-home instead of one copy per experiment. The same handful
   of scripts serve all experiments, temporal and spatial alike.


---------------------------------------------------------------------
 2. DATA FLOW
---------------------------------------------------------------------

  feature metatables (built upstream)        experiments.yaml (you edit)
          |                                          |
          v                                          v
  features_meta_global.csv  ------------>  resolve_experiments.py
  features_meta_GEMS-GER.csv                  | (resolve inheritance,
          |                                   |  validate vs metatable,
          |                                   |  attach target + split,
          |                                   |  spatial leakage check)
          |                                   v
          |                            resolved_allowlists.json
          |                            resolved_features_*.csv (TRUE/FALSE)
          |                            resolved_summary.csv
          |                                   |
          +-----------------+-----------------+
                            v
                       mlkit.py   (builds the per-well feature matrix ONCE)
              build_dataset (temporal)   build_dataset_spatial (GWA, K-fold)
                  |        |                       |        |
                  v        v                       v        v
         lgbm_temporal.py  shap_analysis.py   lgbm_spatial.py  shap_analysis.py
                  |                                |
                  v                                v
   results_<exp>/run<seed>/              results_<exp>/fold<k>/  + final/
   (model.txt, scores, shap_*)           (models, ensemble scores, shap_*;
                                          final/ = deployment ensemble)
                  |                                |
                  +---------------+----------------+
                                  v
                            evaluation.py   --mode temporal | spatial
                                  v
              Evaluation_<mode>/  (merged_scores, summary, sig tests, plots)

mlkit.py is used by lgbm_temporal.py, lgbm_spatial.py AND
shap_analysis.py - so preprocessing is guaranteed identical across
training, spatial CV and SHAP (it used to be duplicated and could
drift). The spatial trainer additionally imports LGB_PARAMS_BASE,
NUM_BOOST_ROUND, EARLY_STOPPING_ROUNDS and HP_SEEDS directly from
lgbm_temporal.py, so the two trainers cannot use diverging parameters.


---------------------------------------------------------------------
 3. DIRECTORY STRUCTURE (real paths on the HPC)
---------------------------------------------------------------------

Workspace root on the HPC ($WS, set in submit/env.sh), e.g.
  /path/to/your/workspace/

CODE-HOME (all scripts live together; they import each other):
  04_Machine_Learning/02_monthly/
        experiments.yaml            # single source of truth (experiments)
        resolve_experiments.py      # config -> allowlists + loader
        mlkit.py                    # shared matrix builder (temporal + spatial)
        lgbm_temporal.py            # temporal training (z-score GWL, date split)
        lgbm_spatial.py             # spatial training (GWA cm, basin K-fold CV)
        make_folds.py               # one-off: assign basins to CV folds
        shap_analysis.py            # TreeSHAP (temporal seeds OR spatial CV folds)
        evaluation.py               # cross-experiment comparison
        submit/
            submit_all.sh           # submit everything with one command
            submit_eval_only.sh     # re-run only the evaluation jobs
            submit_shap_only.sh     # re-run only the SHAP jobs (no training)
            run_array.sh            # one train/shap job (dispatches temporal/spatial)
            run_shap.sh             # standalone SHAP job for one experiment
            run_eval.sh             # one evaluation job per mode
            run_bundle_export.sh    # SHAP bundle export of the analysis layer
                                    # (needs the analysis folder, which is
                                    #  not part of this repository)
            env.sh                  # HPC workspace (WS), module + venv activation

  The submit scripts write temporary experiment lists into submit/
  (exp_list*.txt, exp_kind.txt, shap_exp_*.txt). They are regenerated on
  every call and need not be kept.

DATA HUB:
  03_Dataframes/
      01_Dynamic_Features/
          02_GEMS-GER_monthly/MW_<id>_monthly.csv   # GEMS-GER dynamic features
          03_global_monthly/MW_<id>.csv             # global dynamic features
          03_global_provenance.csv                  # TWSA coverage provenance
      02_Static_Features/
          01_GEMS-GER/01_original_well_metadata.csv # GEMS-GER static table
          02_global/static_features.csv             # global static table
      03_Metadata/                                  # <-- the bridge / cache
          features_meta_global.csv
          features_meta_GEMS-GER.csv
          resolved_allowlists.json                  # ML scripts read this
          resolved_features_global.csv / _gems.csv
          resolved_summary.csv
          twsa_valid_wells.csv                      # computed once, cached
          well_folds_basin_lev06.csv                # from make_folds.py (spatial CV)

OUTPUTS (large, on the HPC):
  05_Machine_Learning_Results/02_monthly/
      results_<exp>/...
      Evaluation_temporal/  Evaluation_spatial/

LOGS (same hierarchy, names mirror the outputs):
  06_Machine_Learning_Logs/02_monthly/
      <EXP>/{lgbm,shap}.{out,err}
      Evaluation_{temporal,spatial}/eval.{out,err}

Project-root detection (find_project_root in resolve_experiments.py)
walks upwards looking for the marker directory "03_Dataframes", so the
exact placement of the code-home is uncritical as long as it sits
somewhere below the workspace root that contains 03_Dataframes. It can be
overridden with the environment variable ML_PROJECT_ROOT.

The input tables (feature stores and feature metadata tables) are built by
02_Dataframes_Processing_Scripts (see its README.txt).


---------------------------------------------------------------------
 4. THE BUILDING BLOCKS
---------------------------------------------------------------------

resolve_experiments.py - resolves, per experiment:
    resolved = (base U include_groups U include_features)
               \ (exclude_groups U exclude_features)
  With a cycle check and hard validation against the metatable (a typo
  in a feature/group name aborts immediately). Derives the time hull and
  reports when the global cap differs from a feature's natural end.
  Attaches the resolved target block (zscore | gwa_cm) and split block
  (temporal | spatial). For a spatial split it checks for leakage: the
  fold key <fold_dim>__HYBAS_ID must NOT be in the allowlist. Provides
  load_experiment() for the ML scripts.
  -> Output: resolved_allowlists.json, resolved_features_*.csv,
     resolved_summary.csv
  -> Run: python resolve_experiments.py

mlkit.py - no CLI. Builds the per-well matrix, metatable-driven (lag/
  rolling for dynamic drivers, passthrough for LCC, broadcast for static,
  categorical dtypes unified). Determines the uniform twsa_valid well set
  (computed once from the global store, cached). Holds _profile_paths() -
  THE store paths. Two builders:
    build_dataset(exp, ..., splits) -> Dataset
        temporal; per-well StandardScaler on the pre-test window.
    build_dataset_spatial(exp, ...) -> SpatialDataset
        one stacked matrix + per-row fold/well/date arrays; target = GWA
        in cm (centred on the reference window, m -> cm). Also provides
        spatial_fold_masks(), grouped_metrics() and persist_schema().

make_folds.py - one-off prerequisite for spatial CV. Assigns whole
  HYBAS basins to K balanced folds (greedy longest-processing-time bin
  packing - no basin split across folds, so no spatial leakage between
  folds) and caches the mapping as well_folds_<fold_dim>.csv in
  03_Metadata. mlkit.load_well_folds() reads it.
  -> Run once: python make_folds.py --fold-dim basin_lev06 --n-folds 5

lgbm_temporal.py - training over the seeds, per-well evaluation, saving.
  -> python lgbm_temporal.py --exp ERA5_4_twsa [--seeds 999 206] [--results-root <abs>]
  -> Output: results_<exp>/run<seed>/{model.txt, scores.csv, results.csv,
     feature_importance.csv, losshistory.csv} + IDremaining.csv,
     well_pretest_mean.csv

lgbm_spatial.py - spatial training with basin K-fold CV and GWA target.
  Same loss / params / feature engineering as lgbm_temporal.py (params
  are imported from it). Selection / early stopping = pooled val-RMSE.
  Per outer fold it trains N seeds, ensembles the TOP_K by val-RMSE, and
  scores the held-out fold with mlkit.grouped_metrics (offset-invariant,
  so directly comparable to the temporal model; RMSE/Bias in cm). A
  final deployment ensemble is trained on ALL wells for raster inference.
  Requires make_folds.py to have run for the fold_dim.
  -> python lgbm_spatial.py --exp ERA5_4_twsa_spatial [--n-seeds-cv 5] [--top-k 3] [--final-seeds 10]
  -> Output: results_<exp>/fold<k>/{model_seed<s>.txt, ensemble_test_scores.csv,
     ensemble_test_results.csv}, cv_summary.csv, cv_seed_summary.csv,
     final/{model_seed<s>.txt, final_members.csv}, feature_columns.csv,
     categories.json, IDremaining.csv

shap_analysis.py - TreeSHAP on the trained models, grouped by var_group,
  labels from plot_name; engineered names (era5l__tp_lag3) are folded
  back to the base column. Detects temporal vs spatial itself (via
  split.kind):
    temporal -> explains each run<seed>/model.txt on the test matrix;
                outputs land in run<seed>/.
    spatial  -> DEFAULT: explains, for every outer CV fold, the evaluated
                top-k ensemble (fold<k>/model_seed*.txt, members taken
                from best_seeds in cv_summary.csv) ONLY on the rows of the
                held-out fold. The explained model is therefore exactly
                the scored model, and the explanation is out-of-sample,
                as in the temporal case. TreeSHAP contributions are
                averaged across the members of a fold (SHAP is additive,
                so the mean of the member contributions explains the
                mean-of-predictions ensemble exactly); outputs land in
                fold<k>/. With --final-ensemble the former behaviour is
                used instead: the deployment ensemble in final/ is
                explained in-sample on all rows (outputs in final/).
  -> python shap_analysis.py --exp ERA5_4_twsa
  -> python shap_analysis.py --exp ERA5_4_twsa_spatial [--top-k 3]
  -> python shap_analysis.py --exp ERA5_4_twsa_spatial --final-ensemble
  -> Output: shap_per_feature.csv, shap_per_driver.csv, shap_bar.png,
     shap_summary.png (in run<seed>/ for temporal, in fold<k>/ for spatial).
     shap_per_feature.csv also carries shap_corr, the Spearman rank
     correlation between a feature value and its own SHAP contribution
     (positive = a high value raises the prediction; NaN for categorical
     features).
  NOTE: the file is deliberately named shap_analysis.py - a file named
  shap.py would shadow the SHAP library.

evaluation.py - compares experiments WITHIN a mode (run one job per
  mode). NSE/KGE/RMSE/Bias per well, summary, NSE classes, pairwise
  Wilcoxon tests (Holm) + Friedman, boxplots, NSE-bin plot, significance
  heatmap. Display names / colours / output paths come from
  resolved_allowlists.json.
    --mode temporal : consolidates run*/results.csv to an ensemble-MEDIAN
                      prediction per well, NSE denominator from
                      well_pretest_mean.csv. Bias/RMSE in metres (GWL,
                      back-transformed with scalers_y).
    --mode spatial  : concatenates fold*/ensemble_test_scores.csv (per-well
                      scores already computed by grouped_metrics; KGE_a is
                      reported as KGE). Bias/RMSE in cm.
  -> python evaluation.py --mode temporal --exps ERA5_0_base ERA5_4_twsa
  -> python evaluation.py --mode spatial  --exps ERA5_0_base_spatial ERA5_4_twsa_spatial
  -> Output under Evaluation_<mode>/ (--out, default): merged_scores.csv,
     summary_statistics.csv, significance_tests.csv, comparison_<metric>.png,
     nse_bin_counts.png, significance_heatmap.png


---------------------------------------------------------------------
 5. OUTPUT FOLDERS
---------------------------------------------------------------------

results_root (in global_settings) + output_subdir (per experiment) form
the path. On the HPC --results-root points at the absolute
05_Machine_Learning_Results/02_monthly path.

Temporal: lgbm_temporal.py and shap_analysis.py write into the SAME
run<seed>/ folder, so SHAP artefacts sit next to the model they came from.

Spatial: lgbm_spatial.py writes fold<k>/ (CV) and final/ (deployment),
plus the inference schema (feature_columns.csv, categories.json) at the
experiment root; shap_analysis.py writes its artefacts into fold<k>/
(default) or into final/ (with --final-ensemble).


---------------------------------------------------------------------
 6. RUNNING IT
---------------------------------------------------------------------

Prerequisite for spatial experiments (one-off):
    python make_folds.py --fold-dim basin_lev06 --n-folds 5

Local / single:
    python resolve_experiments.py
    python lgbm_temporal.py --exp ERA5_4_twsa
    python lgbm_spatial.py  --exp ERA5_4_twsa_spatial
    python shap_analysis.py --exp ERA5_4_twsa
    python shap_analysis.py --exp ERA5_4_twsa_spatial
    python evaluation.py --mode temporal --exps ERA5_0_base ERA5_4_twsa
    python evaluation.py --mode spatial  --exps ERA5_0_base_spatial ERA5_4_twsa_spatial

HPC - everything with one command (from the code-home 02_monthly):
    bash submit/submit_all.sh                                  # ALL experiments
    bash submit/submit_all.sh ERA5_4_twsa ERA5_4_twsa_spatial   # only these
    STEP=train bash submit/submit_all.sh                       # training only (no SHAP)
    RESULTS_ROOT=<path> bash submit/submit_all.sh
    bash submit/submit_eval_only.sh                            # evaluation only
    bash submit/submit_shap_only.sh [EXP ...]                  # SHAP only (models exist)
    MODE=spatial DRY_RUN=1 bash submit/submit_shap_only.sh     # list, submit nothing

submit_all.sh refreshes the allowlists, then submits one train job per
experiment (dispatching lgbm_temporal.py vs lgbm_spatial.py by
split.kind), one SHAP job per experiment (afterok the train job), and
finally TWO evaluation jobs - one per mode - each hanging via afterok on
the SHAP jobs of its mode. Jobs are submitted per experiment via
--export (EXP / KIND), not as a SLURM array.

Local plotting of HPC results: sync 03_Metadata/ plus, from results_*/,
the score files only (temporal: run*/results.csv, scores.csv,
well_pretest_mean.csv - no model.txt needed; spatial:
fold*/ensemble_test_scores.csv), then run evaluation.py with
--results-root <local_path>. The JSON carries names/colours, so local
and HPC plots are identical.


---------------------------------------------------------------------
 7. ADDING AN EXPERIMENT
---------------------------------------------------------------------

1. Add a block in experiments.yaml - usually as a delta on an existing one:

     Exp09_MyTest:
       description: "..."
       base_experiment: ERA5_4_twsa      # inherits its features
       output_subdir: results_exp09_mytest
       display_name: "+ Mine"
       color: "#FF7043"
       include_groups: [hydrivers_dist]     # whole group in
       exclude_features: [era5l__sf]        # single column out

   store_profile is inherited from base_experiment (default: global).

   A spatial twin just flips target + split on top of a temporal base:

     Exp09_MyTest_Spatial:
       base_experiment: Exp09_MyTest        # inherits features/hull/params
       output_subdir: results_exp09_mytest_spatial
       display_name: "+ Mine (spatial)"
       target: {kind: gwa_cm}
       split:  {kind: spatial, fold_dim: basin_lev06}

   IMPORTANT: with a spatial split, <fold_dim>__HYBAS_ID must not be a
   feature (resolve_experiments.py aborts otherwise - remove it via
   exclude_features, as the basin leaves (ERA5/Terra _6_basin_lev0X) already do).

2. python resolve_experiments.py - validates; a typo in a feature/group
   name aborts here.
3. Check resolved_summary.csv (feature count, target_kind, split_kind,
   hull_note).
4. Submit: bash submit/submit_all.sh Exp09_MyTest Exp09_MyTest_Spatial
5. Add it to the comparison: include the ID in the matching evaluation
   mode (evaluation.py --mode temporal|spatial --exps ...).

Feature-subset experiment (e.g. reduced ERA5): instead of include_groups
use include_features: [era5l__tp, era5l__sro, ...] - the explicit list
beats the group.


---------------------------------------------------------------------
 8. WHAT TO SET MANUALLY (CHECKLIST)
---------------------------------------------------------------------

[ ] mlkit._profile_paths() - store paths, target column and the
    file->well-ID rule per profile (global, gems) against the real paths.
[ ] Column-name convention - all store columns follow <dataset_key>__<var>.
    Basin static attributes share the prefix basin_lev06__ / basin_lev07__
    with the dynamic basin rivers; they are separated via type (static vs
    dynamic) in the metatable, not via the group. LCC sits under C3S__.
    (A misnamed column shows up as "unknown feature" on the next
    resolve_experiments.py run.)
[ ] feature metatables - curate plot names and var_group, especially for
    the cryptic ERA5 variables.
[ ] experiments.yaml -> global_settings - hull (time_end,
    require_twsa_valid), splits (stop_start, test_start), results_root,
    and the target / split defaults.
[ ] make_folds.py - run once per fold_dim BEFORE any spatial training,
    so well_folds_<fold_dim>.csv exists.
[ ] submit/env.sh - HPC workspace (WS), module and venv path (PY_MODULE, VENV).
[ ] SLURM header of submit/run_array.sh, run_eval.sh, run_shap.sh,
    run_bundle_export.sh -
    --partition, optionally --account, resources (--time, --mem,
    --cpus-per-task) and --mail-user (replace with your own address).
[ ] LAGS / ROLLING_WINDOWS in mlkit.py, if other time scales are wanted.


---------------------------------------------------------------------
 9. DATA CONTRACTS & EXTENSION POINTS
---------------------------------------------------------------------

This section describes the interfaces between the layers - useful when an
alternative ML path should sit on the same data and swap out only part of
it while reusing the rest unchanged.

The temporal contract: mlkit.build_dataset(exp, ...).
  Signature: build_dataset(exp, project_root, meta_dir,
             splits=('train','stop','test'), verbose=True) -> Dataset.
  The returned Dataset (dataclass) is what lgbm_temporal.py,
  shap_analysis.py and evaluation.py rely on:
    X_train,y_train, X_stop,y_stop, X_test,y_test - matrices + target per
       partition (empty if not requested; SHAP uses only ('test',)).
    IDlen              - well index per test row (links rows to kept_names).
    kept_names         - kept well IDs (order = index 0..N-1).
    scalers_y          - per-well StandardScaler (back-transform to GWL).
    well_means_pre_test- per-well pre-test mean (NSE denominator).
    cat_cols           - categorical column names (for lgb.Dataset).
    feat_names         - final (sanitised) matrix column names.

The spatial contract: mlkit.build_dataset_spatial(exp, ...).
  Returns a SpatialDataset (dataclass):
    X        - the stacked matrix (all wells, all months).
    y        - GWA in cm.
    wid      - per-row well index; fold - per-row fold id; date - per-row date.
    kept_names - well IDs (index = wid); well_folds - fold per well;
    well_std - per-well anomaly std in cm (diagnostic).
    cat_cols, feat_names - as above.
  Helpers: spatial_fold_masks(fold, test_fold, n_folds, val_offset) ->
  (train, val, test, val_fold) masks; group_starts() + grouped_metrics()
  for vectorised per-well NSE/KGE_a/R2/Bias/RMSE/alpha/r; persist_schema()
  writes feature_columns.csv + categories.json for raster inference.

Whoever satisfies a contract can reuse the matching scripts unchanged.

The partition is decided in exactly ONE place - in the builder. For the
temporal path: time_start/time_end (hull), stop_start/test_start (the two
date cut points), require_twsa_valid (well filter), plus per-well
standardisation on the pre-test part and the MIN_PRETEST_ROWS minimum
run-up. For the spatial path: the GWA centring window
(reference_start/reference_end, gwl_to_cm) and the basin fold assignment.
Feature engineering (lag/rolling, static broadcast, categorical dtypes)
is partition-independent.

The ensemble convention evaluation.py builds on:
  temporal - per experiment, run<seed>/results.csv with columns
             <ID>_sim / <ID>_obs; evaluation takes the MEDIAN of sim over
             the seeds per well and uses well_pretest_mean.csv for the NSE
             denominator.
  spatial  - per experiment, fold<k>/ensemble_test_scores.csv already
             holds the per-well scores; each well is in exactly one test
             fold, so concatenating the folds gives the full per-well
             table.

Loader contract: load_experiment(name) -> ResolvedExperiment with, among
others, features, store_profile, output_subdir/output_path, display_name,
color, time_start/time_end, stop_start/test_start, require_twsa_valid,
target (dict) and split (dict).


---------------------------------------------------------------------
 10. VALIDATION & PITFALLS
---------------------------------------------------------------------

- Uniform hull: require_twsa_valid: true + time_end: 2020-12 in
  global_settings means all experiments run on the SAME wells and the
  same period -> paired, fair comparisons. twsa_valid_wells.csv is
  computed once and cached; delete the file to recompute.
- Spatial prerequisite: run make_folds.py for the fold_dim before any
  spatial training, otherwise build_dataset_spatial raises a clear
  FileNotFoundError pointing at the missing well_folds_<fold_dim>.csv.
- Spatial leakage guard: <fold_dim>__HYBAS_ID must not be a feature in a
  spatial experiment; resolve_experiments.py aborts if it is.
- LCC sits under prefix C3S__; basin static (basin_lev0X__UP_AREA, ...)
  shares its prefix with the dynamic basin rivers - the split runs via
  type (static vs dynamic), not via the group.
- Resolver errors are features: an unknown feature/group, a cycle in the
  base_experiment chain, or an empty allowlist abort with a clear message,
  so store drift surfaces early.
- Units differ by mode: temporal sim/obs/Bias/RMSE are in GWL METRES
  (lgbm_temporal.evaluate() inverse-transforms via scalers_y before writing
  results.csv), spatial sim/obs/Bias/RMSE are in cm (GWA). NSE/KGE/R2 are
  dimensionless. Compare within a mode, not across (this is why evaluation
  runs one job per mode). See Section 11.4 for the full unit table.
- Do not rename shap_analysis.py to shap.py (name clash with the library).


---------------------------------------------------------------------
 11. RESULT FILES - FULL REFERENCE  (what landed where, columns, units)
---------------------------------------------------------------------

Paths are relative to  <results_root>/<exp.output_path>/  (output_path from
resolved_allowlists.json; on the cluster results_root =
05_Machine_Learning_Results/<code-home>). Separator ';' throughout. Mode =
exp.split.kind.

11.1 TEMPORAL experiment  (split.kind = temporal)   [lgbm_temporal.py]
  run<seed>/model.txt        LightGBM booster (best_iteration).
  run<seed>/results.csv      WIDE per-well test series. Per kept well <ID>:
        <ID>_ID    integer well index (= position in kept_names)
        <ID>_sim   simulated GWL [m]   (inverse-transformed via scalers_y)
        <ID>_obs   observed  GWL [m]
      Row index = plain 0..n-1 counter (NO dates). Wells with fewer test
      rows are NaN-padded at the tail. The data are gap-free monthly, so the
      date axis is simply [test_start .. time_end] (month start); rebuild it
      from the hull in resolved_allowlists.json when plotting.
  run<seed>/scores.csv       per-well NSE,KGE,R2,Bias,MSE,RMSE (Bias/MSE/RMSE
                             in [m]); index = well ID.
  run<seed>/losshistory.csv  iteration, train_rmse, val_rmse (on the z-scored
                             target -> NOT comparable to scores.csv RMSE).
  run<seed>/feature_importance.csv  feature, gain, split.
  IDremaining.csv            kept well IDs (experiment root).
  well_pretest_mean.csv      ID, pretest_mean [m] -> NSE denominator reused
                             by evaluation.py.

11.2 SPATIAL experiment  (split.kind = spatial)     [lgbm_spatial.py]
  Target = GWA in cm.
  fold<k>/model_seed<s>.txt           CV member boosters for test-fold k.
  fold<k>/ensemble_test_scores.csv    per-well scores on fold-k wells (each
        well is in exactly ONE test fold). Index = well ID. Columns:
        NSE, KGE_a, R2, Bias, RMSE, alpha, r  (Bias/RMSE in cm; evaluation.py
        renames KGE_a -> KGE).
  fold<k>/ensemble_test_results.csv   LONG per-well series (spatial time-
        series source). Columns:
        well; time (real monthly date); ens_mean = sim GWA [cm];
        ens_std = ensemble std [cm] (uncertainty, see 11.5); obs GWA [cm].
  cv_summary.csv / cv_seed_summary.csv   per outer fold / per fold x seed.
  final/model_seed<s>.txt             deployment ensemble (ALL wells).
  final/final_members.csv             seed, rounds.
  fold<k>/<shap files>                spatial SHAP (default) is computed per
        outer fold on the averaged top-k fold ensemble, on the held-out fold
        only; shap_per_feature/driver.csv therefore live in fold<k>/.
  final/<shap files>                  only with shap_analysis.py
        --final-ensemble (in-sample explanation of the deployment ensemble).
  feature_columns.csv, categories.json   raster-inference schema.
  IDremaining.csv                     ID; fold; anom_std_cm.

11.3 EVALUATION outputs   [evaluation.py, one run per mode]
  -> <results_root>/Evaluation_<mode>/ :
     evaluation.py's default --out is Evaluation_<mode>, matching what
     run_eval.sh writes on the HPC.
  scores_<eid>.csv      per-experiment per-well scores (NSE/KGE/RMSE/Bias).
  merged_scores.csv     ALL experiments joined on well ID (outer). Column
        convention  <metric>_<eid>  (e.g. NSE_ERA5_4_twsa). The main table
        for further analysis. No skill columns (see 11.6).
  summary_statistics.csv  min/median/mean/max per (metric, experiment).
  nse_bin_counts.csv    number of wells per NSE class and experiment.
  significance_tests.csv  pairwise Wilcoxon (Holm) + Friedman.
  comparison_<metric>.png, nse_bin_counts.png, cdf_<metric>.png,
  significance_heatmap.png   built-in plots (ALL experiments of the mode).
  consolidated_obs_sim/<ID>_obs_sim.csv  (temporal only) obs + one column per
        seed; median over seed columns = ensemble sim [m].

11.4 UNITS - quick table
  quantity            temporal        spatial
  sim / obs           GWL [m]         GWA [cm]
  RMSE / Bias / MSE   [m]             [cm]
  NSE/KGE/R2/r/alpha  dimensionless   dimensionless
  Compare within a mode only.

11.5 UNCERTAINTY - where a spread can come from
  spatial: ens_std in ensemble_test_results.csv = std ACROSS the top-k CV
      ensemble members (seeds) per month (lgbm_spatial.py: stacked.std(0),
      default top_k=3). It is seed/member disagreement, NOT a calibrated
      predictive interval.
  temporal: results.csv stores no std; the analogous spread is the std
      across the per-seed sims of the run<seed>/results.csv files.

11.6 KGE SKILL SCORE - not stored in the scores
  KGEskill = (KGE_model - KGE_bench) / (1 - KGE_bench); >0 better than the
  benchmark, =0 parity, <0 worse. It is a RELATIVE metric and is therefore
  not written by evaluation.py; it can be derived from the KGE_<eid> columns
  of merged_scores.csv, with GEMS (temporal) or GEMS_spatial (spatial) as the
  benchmark. The benchmark must be among the evaluated experiments.

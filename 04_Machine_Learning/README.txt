================================================================================
04_Machine_Learning  -  Groundwater level models
================================================================================

Thesis: Evaluating Machine Learning-Based Groundwater Level Predictions Using
        Globally Available Earth Observation Data

This folder contains the machine learning code. It reads the feature stores
and feature metadata tables built by 02_Dataframes_Processing_Scripts and the
GEMS-GER benchmark data, trains the models on the HPC cluster (SLURM) and
evaluates them.


--------------------------------------------------------------------------------
1. FOLDER STRUCTURE
--------------------------------------------------------------------------------

  01_weekly/    Weekly GEMS-GER resolution. A global LightGBM model, plus
                SHAP and an evaluation. The GEMS-GER reference models (single,
                dynonly, dynstat) are not included (GEMS-GER code licence
                CC BY-NC-SA 4.0); evaluation.py compares against them if
                their results exist.
                -> README_weekly.txt

  02_monthly/   Monthly resolution, LightGBM only, config-driven. Every
                experiment (feature set, target, temporal or spatial split)
                is one block in experiments.yaml. Compares the global
                Earth observation feature sets (ERA5-Land and TerraClimate
                "ladders") with the GEMS-GER benchmark, in a temporal and a
                spatial (basin cross-validation) mode.
                -> README_monthly.txt

  Each code folder has a subfolder submit/ with the SLURM scripts.

  The analysis and plotting layer (SHAP aggregation, maps, figures of the
  thesis) is not part of this repository. The two optional bundle export
  scripts (run_bundle_export.sh) only prepare input for that layer.


--------------------------------------------------------------------------------
2. EXPECTED PROJECT LAYOUT
--------------------------------------------------------------------------------

  <project root>/                      (on the HPC: the workspace $WS)
    03_Dataframes/                     input tables (feature stores, metadata)
    04_Machine_Learning/               THIS FOLDER
    05_Machine_Learning_Results/       model outputs (created by the scripts)
      01_weekly/  02_monthly/
    06_Machine_Learning_Logs/          SLURM logs (created by submit_all.sh)
    envs/ml_hydro/                     Python virtual environment (HPC)

The scripts find the project root themselves: it is the first parent folder
of the code that contains 03_Dataframes. It can be set explicitly with the
environment variable ML_PROJECT_ROOT.


--------------------------------------------------------------------------------
3. WHAT A NEW USER HAS TO ADJUST
--------------------------------------------------------------------------------

  - <folder>/submit/env.sh : WS (HPC workspace), PY_MODULE (Python module),
    VENV (virtual environment). This is the only place with an absolute path.
  - #SBATCH header of every <folder>/submit/run_*.sh : --partition, --gres,
    --time, --mem, --cpus-per-task and --mail-user (own e-mail address).
  - 02_monthly/experiments.yaml : experiments, time window, split dates.
  - 01_weekly/config.py : split dates, seeds, drivers of the weekly models.


--------------------------------------------------------------------------------
4. SOFTWARE
--------------------------------------------------------------------------------

Python 3.12 with numpy, pandas, scipy, scikit-learn, matplotlib, pyyaml,
lightgbm (>= 4.0) and optionally shap (beeswarm plots).

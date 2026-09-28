# Groundwater level prediction with globally available Earth observation data

Code accompanying the Master's thesis **"Evaluating Machine Learning-Based
Groundwater Level Predictions Using Globally Available Earth Observation Data"**.

The repository contains the full processing chain, from the downloaded global
datasets to the trained and evaluated machine learning models:

| Stage | Folder | What it does |
|---|---|---|
| 1 | `02_Processing_Scripts/` | Preprocesses the global Earth observation datasets (ERA5-Land, TerraClimate, GRACE TWSA, SoilGrids, GASY, GLiM/GLHYMPS, HydroSHEDS, C3S land cover, WHYMAP) into standardized NetCDF products for the study area (5.5 to 15.5 deg E, 47.0 to 55.5 deg N). |
| 2 | `02_Dataframes_Processing_Scripts/` | Samples all products at the groundwater well locations and builds the static and dynamic feature stores plus a feature metadata table. Also aggregates the weekly GEMS-GER series to monthly values. |
| 3 | `04_Machine_Learning/` | Trains and evaluates the models on an HPC cluster (SLURM): a weekly LightGBM model on the GEMS-GER data and the config-driven monthly LightGBM experiments (temporal and spatial cross-validation) with SHAP analysis. |

Every folder has its own plain-text README (`README.txt`, `README_<Dataset>.txt`,
`README_weekly.txt`, `README_monthly.txt`) with inputs, outputs, run order and
settings. Start there before running a script.

## Expected project layout

The data are **not** part of this repository. All scripts use paths relative to
a project root, which is the root of this repository. Place the data folders
next to the code folders:

```
<project root>/                         (= this repository)
├── 01_Input_Global/                    raw, intermediate and final gridded data   [not tracked]
│   ├── 1_Original/<Dataset>/           downloaded raw data
│   ├── 2_Intermediate/<Dataset>/
│   └── 3_Final/<Dataset>/              output of stage 1
├── 02_Processing_Scripts/              stage 1
├── 02_Dataframes_Processing_Scripts/   stage 2
├── 03_Dataframes/                      feature stores and metadata             [not tracked]
│   ├── 01_Dynamic_Features/01_GEMS-GER_weekly/          GEMS-GER input (weekly)
│   └── 02_Static_Features/01_GEMS-GER/                  GEMS-GER well metadata
├── 04_Machine_Learning/                stage 3
├── 05_Machine_Learning_Results/        model outputs (created by the scripts)  [not tracked]
└── 06_Machine_Learning_Logs/           SLURM logs (created by the scripts)     [not tracked]
```

The data sources and download instructions are listed in the dataset READMEs in
`02_Processing_Scripts/<Dataset>/`. The GEMS-GER benchmark dataset
([Zenodo record](https://zenodo.org/records/16736908), licence CC BY-NC-ND 4.0)
has to be downloaded separately and placed in `03_Dataframes/` as shown above.
It is not redistributed here.

## Setting the paths

No absolute paths are hard-coded, except the HPC workspace placeholder.

* Stages 1 and 2 derive the project root from the location of each script
  (two folder levels up). Many scripts also accept `--root <path>`; in stage 2
  the root can also be set with the environment variable `GWL_ROOT`.
* Stage 3 uses the first parent folder that contains `03_Dataframes`, or the
  environment variable `ML_PROJECT_ROOT`.
* Each script defines its paths in a `PATHS` block at the top.

Before running stage 3 on a cluster, adjust:

* `04_Machine_Learning/<folder>/submit/env.sh`: `WS` (HPC workspace, currently
  the placeholder `/path/to/your/workspace`), `PY_MODULE`, `VENV`.
* The `#SBATCH` header of every `submit/run_*.sh`: `--partition`, `--gres`,
  `--time`, `--mem`, `--cpus-per-task` and `--mail-user`
  (currently the placeholder `your.email@example.org`).

## Installation

Python 3.12 was used.

```
python -m venv envs/ml_hydro
source envs/ml_hydro/bin/activate
pip install -r requirements.txt
```

The two `.js`
files in `02_Processing_Scripts/SoilGrids/` are Google Earth Engine scripts and
run in the Earth Engine Code Editor.

## Run order

1. `02_Processing_Scripts/`: one folder per dataset, scripts in the order of
   their number prefix (`01_`, `02a_`, `02b_`, ...). Scripts named `inspect` or
   `check` are optional diagnostics.
2. `02_Dataframes_Processing_Scripts/`: GEMS-GER monthly aggregation, static
   store, dynamic store, optional quality report, feature metadata.
3. `04_Machine_Learning/`: `01_weekly/submit/submit_all.sh` and
   `02_monthly/submit/submit_all.sh`.

The detailed commands are in the folder READMEs.

## Not included

The analysis and plotting layer (SHAP aggregation, maps, figures) is not part of
this repository. The optional `run_bundle_export.sh` jobs only prepare input for
that layer.

## GEMS-GER reference models

The weekly GEMS-GER reference models (`single`, `dynonly`, `dynstat`) are not
included, because the GEMS-GER code is licensed CC BY-NC-SA 4.0. They are
available from the [GEMS-GER code repository](https://github.com/KITHydrogeology/GEMS-GER).
The weekly LightGBM model uses the same data, split dates, seeds and metrics,
and `04_Machine_Learning/01_weekly/evaluation.py` compares against the
reference models if their results exist in the expected output folders.

## License

MIT, see [LICENSE](LICENSE). The licenses of the input datasets are defined by
their providers; see the dataset READMEs.

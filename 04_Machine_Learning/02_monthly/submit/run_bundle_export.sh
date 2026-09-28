#!/bin/bash
# =====================================================================
# run_bundle_export.sh
# One SLURM job for shap_bundle_export.py. The export rebuilds the full
# feature matrix and runs TreeSHAP over several ensembles, so it belongs
# on a compute node, not on the login node.
#
# The script itself lives in the ANALYSIS folder, not in the ML code-home,
# but it needs the ML code-home on sys.path (mlkit, resolve_experiments,
# shap_analysis). analysis_common resolves that by itself, so the job only
# has to load the environment and change into the analysis folder.
# NOTE: the analysis folder is not part of this repository. The job is kept
# for completeness; it only works if $ANALYSIS_DIR exists on the cluster.
#
# Extra arguments are passed straight through to shap_bundle_export.py:
#
#   sbatch submit/run_bundle_export.sh --exps GEMS GEMS_spatial
#   sbatch submit/run_bundle_export.sh --exps Terra_6_basin_lev06 --top-n 25
#   ANALYSIS_DIR=$WS/06_Data_Analysis_Scripts_Cluster/shap_analysis \
#       sbatch --export=ALL submit/run_bundle_export.sh --exps GEMS
# =====================================================================
#SBATCH --job-name=shap_bundle
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128000MB
#SBATCH --time=08:00:00
#SBATCH --export=ALL
#SBATCH --mail-type=FAIL,REQUEUE
#SBATCH --mail-user=your.email@example.org

set -euo pipefail
unset LANG
unset LC_CTYPE
ulimit -s unlimited

echo "START_TIME    = $(date +'%y-%m-%d %H:%M:%S')"
echo "HOSTNAME      = ${HOSTNAME}"
echo "SLURM_JOBID   = ${SLURM_JOB_ID:-n/a}"
echo "CPUS_PER_TASK = ${SLURM_CPUS_PER_TASK:-n/a}"

# Code-home robustly, same order as run_array.sh: exported ML_DIR, then the
# submit directory of the sbatch call, then this script's own location. Under
# SLURM ${BASH_SOURCE[0]} points at the spool copy, so it is the last resort.
if [ -z "${ML_DIR:-}" ]; then
    if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
        ML_DIR="$SLURM_SUBMIT_DIR"
    else
        ML_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    fi
fi
cd "$ML_DIR"
source "$ML_DIR/submit/env.sh"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export LIGHTGBM_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1          # live progress in the .out file

# WS comes from submit/env.sh (the single place for the HPC path).
ANALYSIS_DIR="${ANALYSIS_DIR:-$WS/06_Data_Analysis_Scripts_Cluster/shap_analysis}"
[ -f "$ANALYSIS_DIR/shap_bundle_export.py" ] || {
    echo "shap_bundle_export.py missing in $ANALYSIS_DIR" >&2; exit 2; }

echo "ML_DIR        = $ML_DIR"
echo "ANALYSIS_DIR  = $ANALYSIS_DIR"
echo "ARGS          = $*"

cd "$ANALYSIS_DIR"
python -u shap_bundle_export.py "$@"

echo "END_TIME      = $(date +'%y-%m-%d %H:%M:%S')"

#!/bin/bash
#SBATCH --job-name=w_shap
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128000MB
#SBATCH --time=08:00:00
#SBATCH --export=ALL
#SBATCH --mail-type=FAIL,REQUEUE
#SBATCH --mail-user=your.email@example.org
# Needs results_lgbm/run<seed>/model.txt -> runs after lgbm. Extra args -> shap_lgbm.py.
#   sbatch submit/run_shap.sh --seeds 999 --sample-rows 200000

set -euo pipefail
unset LANG
unset LC_CTYPE
ulimit -s unlimited

echo "START_TIME    = $(date +'%y-%m-%d %H:%M:%S')"
echo "HOSTNAME      = ${HOSTNAME}"
echo "SLURM_JOBID   = ${SLURM_JOB_ID:-n/a}"

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
export PYTHONUNBUFFERED=1     # live progress in the .out file instead of 4 KB blocks

python -u shap_lgbm.py "$@"

echo "END_TIME      = $(date +'%y-%m-%d %H:%M:%S')"

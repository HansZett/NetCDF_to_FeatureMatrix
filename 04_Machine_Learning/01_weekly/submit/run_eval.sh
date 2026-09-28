#!/bin/bash
#SBATCH --job-name=w_eval
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16000MB
#SBATCH --time=00:30:00
#SBATCH --export=ALL
#SBATCH --mail-type=FAIL,REQUEUE
#SBATCH --mail-user=your.email@example.org
# Reads the results of lgbm (and of single/dynonly/dynstat if they exist). Extra args are
# passed to evaluation.py, e.g.:  sbatch submit/run_eval.sh --restrict-to <path>/IDremaining.csv

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

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

python evaluation.py "$@"

echo "END_TIME      = $(date +'%y-%m-%d %H:%M:%S')"

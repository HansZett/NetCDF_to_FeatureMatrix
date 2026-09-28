#!/bin/bash
#SBATCH --job-name=w_lgbm
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128000MB
#SBATCH --time=24:00:00
#SBATCH --export=ALL
#SBATCH --mail-type=FAIL,REQUEUE
#SBATCH --mail-user=your.email@example.org
# submit_all.sh sets --output/--error per job. Standalone: slurm-%j.out in the cwd.

set -euo pipefail
unset LANG
unset LC_CTYPE
ulimit -s unlimited

echo "START_TIME    = $(date +'%y-%m-%d %H:%M:%S')"
echo "HOSTNAME      = ${HOSTNAME}"
echo "SLURM_JOBID   = ${SLURM_JOB_ID:-n/a}"

# Code home: ML_DIR (exported by submit_all.sh) -> SLURM_SUBMIT_DIR (sbatch is
# called from the code home) -> BASH_SOURCE (local run). Under SLURM,
# ${BASH_SOURCE[0]} points to the spool copy and cannot be used.
if [ -z "${ML_DIR:-}" ]; then
    if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
        ML_DIR="$SLURM_SUBMIT_DIR"
    else
        ML_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    fi
fi
cd "$ML_DIR"
source "$ML_DIR/submit/env.sh"

# LightGBM is OpenMP-parallel -> bind the threads to the allocated cores.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export LIGHTGBM_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python lgbm.py

echo "END_TIME      = $(date +'%y-%m-%d %H:%M:%S')"

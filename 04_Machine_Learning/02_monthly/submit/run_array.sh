#!/bin/bash
# =====================================================================
# submit/run_array.sh  --  one training or SHAP job (one STEP per call).
# Normally submitted by submit_all.sh with EXP (experiment ID), KIND
# (temporal|spatial) and STEP (train|shap) as exported variables.
# Alternatively as an array job: the experiment ID is read from EXP_LIST
# (line = ARRAY_TASK_ID + 1) and the split kind from EXP_KIND.
# train dispatches to lgbm_temporal.py or lgbm_spatial.py;
# shap_analysis.py detects temporal/spatial itself.
# =====================================================================
#SBATCH --job-name=gwl_step
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128000MB
#SBATCH --time=24:00:00
#SBATCH --export=ALL
#SBATCH --mail-type=FAIL,REQUEUE
#SBATCH --mail-user=your.email@example.org
# submit_all.sh sets --output/--error per experiment.

set -euo pipefail
unset LANG
unset LC_CTYPE
ulimit -s unlimited

echo "START_TIME       = $(date +'%y-%m-%d %H:%M:%S')"
echo "HOSTNAME         = ${HOSTNAME}"
echo "SLURM_JOBID      = ${SLURM_JOB_ID}  ARRAY_TASK = ${SLURM_ARRAY_TASK_ID:-n/a}"
echo "CPUS_PER_TASK    = ${SLURM_CPUS_PER_TASK}"

# Determine the code home. Under SLURM a spool COPY of the script runs
# (/var/spool/slurmd/job*/slurm_script), so ${BASH_SOURCE[0]} does NOT point
# to submit/. Order: ML_DIR exported by submit_all.sh -> SLURM_SUBMIT_DIR
# (sbatch is called from the code home) -> BASH_SOURCE (local call without SLURM).
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

EXP="${EXP:-}"
KIND="${KIND:-}"
if [ -z "$EXP" ]; then
    EXP_LIST="${EXP_LIST:?EXP_LIST or EXP not set}"
    EXP_KIND="${EXP_KIND:?EXP_KIND or KIND not set}"
    IDX="${SLURM_ARRAY_TASK_ID:?neither EXP nor SLURM_ARRAY_TASK_ID set}"
    EXP="$(sed -n "$((IDX + 1))p" "$EXP_LIST")"
    KIND="$(sed -n "$((IDX + 1))p" "$EXP_KIND")"
fi
STEP="${STEP:?STEP not set (train|shap)}"
[ -z "$EXP" ] && { echo "No experiment ID"; exit 1; }
KIND="${KIND:-temporal}"

RESULTS_ARG=""
[ -n "${RESULTS_ROOT:-}" ] && RESULTS_ARG="--results-root $RESULTS_ROOT"

echo "Task ${SLURM_ARRAY_TASK_ID:-n/a}  ->  ${EXP}  (kind=${KIND}, step=${STEP})"

case "$STEP" in
    train)
        if [ "$KIND" = "spatial" ]; then
            python lgbm_spatial.py --exp "$EXP" $RESULTS_ARG
        else
            python lgbm_temporal.py --exp "$EXP" $RESULTS_ARG
        fi
        ;;
    shap)
        # shap_analysis.py detects temporal (run<seed>/) vs spatial (fold<k>/) itself.
        python shap_analysis.py --exp "$EXP" $RESULTS_ARG
        ;;
    *)
        echo "Unknown STEP '$STEP' (expected: train|shap)"; exit 2 ;;
esac

echo "END_TIME         = $(date +'%y-%m-%d %H:%M:%S')"

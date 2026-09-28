#!/bin/bash
# =====================================================================
# submit/run_eval.sh  --  evaluation as ONE job per mode (temporal|spatial).
# Submitted by submit_all.sh (afterok on the SHAP jobs of that mode) or by
# submit_eval_only.sh. Experiment list and output folder come as variables
# (EVAL_MODE, EXP_LIST, RESULTS_ROOT, EVAL_OUT).
# =====================================================================
#SBATCH --job-name=gwl_eval
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16000MB
#SBATCH --time=01:00:00
#SBATCH --export=ALL
#SBATCH --mail-type=FAIL,REQUEUE
#SBATCH --mail-user=your.email@example.org
# submit_all.sh sets --output/--error.

set -euo pipefail
unset LANG
unset LC_CTYPE
ulimit -s unlimited

echo "START_TIME       = $(date +'%y-%m-%d %H:%M:%S')"
echo "HOSTNAME         = ${HOSTNAME}"
echo "SLURM_JOBID      = ${SLURM_JOB_ID}"

# Determine the code home (see run_array.sh): ML_DIR -> SLURM_SUBMIT_DIR
# -> BASH_SOURCE. Under SLURM ${BASH_SOURCE[0]} points to the spool copy.
if [ -z "${ML_DIR:-}" ]; then
    if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
        ML_DIR="$SLURM_SUBMIT_DIR"
    else
        ML_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    fi
fi
cd "$ML_DIR"

source "$ML_DIR/submit/env.sh"

EVAL_MODE="${EVAL_MODE:?EVAL_MODE not set (temporal|spatial)}"
EXP_LIST="${EXP_LIST:?EXP_LIST not set}"
mapfile -t EXPS < "$EXP_LIST"
RESULTS_ARG=""
[ -n "${RESULTS_ROOT:-}" ] && RESULTS_ARG="--results-root $RESULTS_ROOT"
OUT="${EVAL_OUT:-Evaluation_${EVAL_MODE}}"

echo "Evaluation (mode=$EVAL_MODE) of ${#EXPS[@]} experiments -> $OUT"
python evaluation.py --mode "$EVAL_MODE" --exps "${EXPS[@]}" $RESULTS_ARG --out "$OUT"
echo "END_TIME         = $(date +'%y-%m-%d %H:%M:%S')"

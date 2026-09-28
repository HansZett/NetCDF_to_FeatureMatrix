#!/bin/bash
# =====================================================================
# submit/submit_all.sh  --  submit the whole weekly stack with ONE command.
#
#   cd 04_Machine_Learning/01_weekly
#   ./submit/submit_all.sh
#
# Pipeline:
#   lgbm    (LightGBM)
#   shap    -> afterok lgbm
#   eval    -> afterok lgbm
#
# The GEMS-GER reference models (single, dynonly, dynstat) are not part of
# this public version (GEMS-GER code licence CC BY-NC-SA 4.0). evaluation.py
# includes their results if they exist in the output folders.
#
# Switches (environment variables):
#   SKIP_LGBM=1  SKIP_SHAP=1  SKIP_EVAL=1   leave out parts
#   LOGS_ROOT=...   log root (default: $WS/06_Machine_Learning_Logs/01_weekly)
# Extra arguments are passed to evaluation.py, e.g.:
#   ./submit/submit_all.sh --restrict-to <path>/02_monthly/<exp>/IDremaining.csv
# =====================================================================
set -euo pipefail

# Code home = parent folder of submit/. Exported so that --export=ALL carries
# it into the jobs (run_*.sh use ML_DIR; in a job, BASH_SOURCE points to the
# SLURM spool copy and cannot be used).
ML_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_HOME="$(basename "$ML_DIR")"
cd "$ML_DIR"
export ML_DIR

# Environment (module + venv), so that python below can read config.py.
source "$ML_DIR/submit/env.sh"

# Paths from config.py (single source of truth) ---------------------------
OUT_ROOT="$(python -c 'import config; print(config.PTH_OUT_ROOT)')"
# Logs separate from the results (as in the monthly setup):
#   $WS/06_Machine_Learning_Logs/<code-home>/<model>/
LOGS_ROOT="${LOGS_ROOT:-$WS/06_Machine_Learning_Logs/$CODE_HOME}"

# --parsable may append ";cluster" -> cut it off.
jid() { cut -d';' -f1; }

mkdir -p "$LOGS_ROOT"/{lgbm,shap,evaluation}

echo "=== weekly submit ==="
echo "code-home : $ML_DIR  ($CODE_HOME)"
echo "out-root  : $OUT_ROOT"
echo "logs-root : $LOGS_ROOT"

# ---------------------------------------------------------------------
# lgbm (can be skipped with SKIP_LGBM=1, e.g. if already finished)
# ---------------------------------------------------------------------
sub() {  # sub <name> <script>  -> echo jid
    sbatch --parsable \
        --output="$LOGS_ROOT/$1/$1_%j.out" \
        --error="$LOGS_ROOT/$1/$1_%j.err" \
        "$2" | jid
}
if [ "${SKIP_LGBM:-0}" != "1" ]; then
    JID_LG=$(sub lgbm submit/run_lgbm.sh); echo "  lgbm    -> $JID_LG"
else
    JID_LG=""; echo "  lgbm    : SKIPPED (using existing results)"
fi

# ---------------------------------------------------------------------
# shap afterok lgbm -- or immediately if lgbm was skipped (already finished)
# ---------------------------------------------------------------------
if [ "${SKIP_SHAP:-0}" != "1" ]; then
    DEP_SH=""; [ -n "$JID_LG" ] && DEP_SH="--dependency=afterok:$JID_LG"
    JID_SH=$(sbatch --parsable $DEP_SH \
        --output="$LOGS_ROOT/shap/shap_%j.out" \
        --error="$LOGS_ROOT/shap/shap_%j.err" \
        submit/run_shap.sh | jid)
    echo "  shap    -> $JID_SH   ${JID_LG:+(afterok $JID_LG)}"
else
    echo "  shap    : SKIPPED"
fi

# ---------------------------------------------------------------------
# eval afterok lgbm if it was submitted; otherwise results come from disk.
# ---------------------------------------------------------------------
if [ "${SKIP_EVAL:-0}" != "1" ]; then
    DEP=""; [ -n "$JID_LG" ] && DEP="--dependency=afterok:$JID_LG"
    JID_EV=$(sbatch --parsable $DEP \
        --output="$LOGS_ROOT/evaluation/eval_%j.out" \
        --error="$LOGS_ROOT/evaluation/eval_%j.err" \
        --export=ALL,ML_DIR="$ML_DIR" \
        submit/run_eval.sh "$@" | jid)
    echo "  eval    -> $JID_EV   (${DEP:-no dependency})"
else
    echo "  eval    : SKIPPED"
fi

echo "=== submitted. squeue -u \$USER ==="

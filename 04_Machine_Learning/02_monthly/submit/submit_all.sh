#!/bin/bash
# =====================================================================
# submit/submit_all.sh  --  submit all (or selected) experiments with ONE
# command. Run from the code home 04_Machine_Learning/02_monthly:
#
#   bash submit/submit_all.sh                       # ALL experiments of the JSON
#   bash submit/submit_all.sh ERA5_4_twsa ERA5_4_twsa_spatial   # only these
#   STEP=train bash submit/submit_all.sh            # training only, no SHAP
#                                                   # (evaluation still follows)
#   RUN_EVAL=0 bash submit/submit_all.sh            # no evaluation jobs
#   RESULTS_ROOT=<path> bash submit/submit_all.sh   # other output root
#
# Pipeline:
#   resolve_experiments.py (refreshes the allowlists)
#   -> one train job per experiment (run_array.sh; lgbm_temporal / lgbm_spatial)
#   -> one SHAP job per experiment, afterok its train job
#   -> Evaluation_temporal: afterok all temporal SHAP jobs (if any)
#   -> Evaluation_spatial:  afterok all spatial SHAP jobs (if any)
#
# Output layout:
#   $WS/05_Machine_Learning_Results/<code-home>/<output folder per experiment>/...
#   $WS/05_Machine_Learning_Results/<code-home>/Evaluation_{temporal,spatial}/
#
# Log layout (same hierarchy, names match the outputs):
#   $WS/06_Machine_Learning_Logs/<code-home>/<EXP>/{lgbm,shap}.{out,err}
#   $WS/06_Machine_Learning_Logs/<code-home>/Evaluation_{temporal,spatial}/eval.{out,err}
#
# $WS (HPC workspace) is set in submit/env.sh.
# =====================================================================
set -euo pipefail

ML_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_HOME="$(basename "$ML_DIR")"          # e.g. "02_monthly"
cd "$ML_DIR"
# IMPORTANT: export ML_DIR so that --export=ALL carries it into the jobs. In
# the jobs ${BASH_SOURCE[0]} points to the SLURM spool copy (/var/spool/slurmd/...)
# and cannot be used to find the code home. run_array.sh/run_eval.sh use this
# ML_DIR (fallback there: SLURM_SUBMIT_DIR, then BASH_SOURCE).
export ML_DIR

# --- load the environment (module + venv, sets $WS) -- no manual activate needed.
source "$ML_DIR/submit/env.sh"

# --- configuration (can be overridden as environment variables) ----------
RESULTS_ROOT="${RESULTS_ROOT:-$WS/05_Machine_Learning_Results/$CODE_HOME}"
LOGS_ROOT="${LOGS_ROOT:-$WS/06_Machine_Learning_Logs/$CODE_HOME}"
STEP="${STEP:-all}"                       # all | train  (shap and eval depend on train)
RUN_EVAL="${RUN_EVAL:-1}"

mkdir -p submit "$RESULTS_ROOT" "$LOGS_ROOT"

echo "Code-Home    : $ML_DIR"
echo "RESULTS_ROOT : $RESULTS_ROOT"
echo "LOGS_ROOT    : $LOGS_ROOT"
echo "STEP=$STEP  RUN_EVAL=$RUN_EVAL"

# 1) refresh the allowlists
echo ">> resolve_experiments.py"
python resolve_experiments.py

# 2) experiment list: arguments or all experiments of the JSON
if [ "$#" -gt 0 ]; then
    printf "%s\n" "$@" > submit/exp_list.txt
else
    python - <<'PY' > submit/exp_list.txt
import json, os
from resolve_experiments import DEFAULT_META_DIR
d = json.load(open(os.path.join(DEFAULT_META_DIR, "resolved_allowlists.json")))
print("\n".join(d["experiments"]))
PY
fi
N=$(grep -c . submit/exp_list.txt)
echo ">> $N experiments:"; sed 's/^/   - /' submit/exp_list.txt

# 2b) split kind per experiment + separate lists for temporal/spatial
python - <<'PY'
import json, os
from resolve_experiments import DEFAULT_META_DIR
d = json.load(open(os.path.join(DEFAULT_META_DIR, "resolved_allowlists.json")))
names = open("submit/exp_list.txt").read().split()
def kind(n): return d["experiments"][n].get("split", {}).get("kind", "temporal")
def write(p, lines): open(p, "w").write("\n".join(lines) + ("\n" if lines else ""))
write("submit/exp_kind.txt",      [kind(n) for n in names])
write("submit/exp_list_temporal.txt", [n for n in names if kind(n) == "temporal"])
write("submit/exp_list_spatial.txt",  [n for n in names if kind(n) == "spatial"])
PY
N_TEMP=$(grep -c . submit/exp_list_temporal.txt || true)
N_SPAT=$(grep -c . submit/exp_list_spatial.txt  || true)
echo ">> temporal: $N_TEMP  |  spatial: $N_SPAT"

# 3) one log folder per experiment (SLURM does NOT create it).
while IFS= read -r exp; do
    [ -n "$exp" ] && mkdir -p "$LOGS_ROOT/$exp"
done < submit/exp_list.txt
mkdir -p "$LOGS_ROOT/Evaluation_temporal" "$LOGS_ROOT/Evaluation_spatial"

# 4) one train (+ SHAP) job per experiment. EXP/KIND are passed directly via
#    --export to run_array.sh -- no array index mapping needed.
echo ">> submitting train and SHAP jobs (one job pair per experiment)..."
TRAIN_JIDS=()
SHAP_JIDS_TEMP=()
SHAP_JIDS_SPAT=()
i=0
while IFS= read -r exp; do
    [ -z "$exp" ] && continue
    kind="$(sed -n "$((i + 1))p" submit/exp_kind.txt)"; kind="${kind:-temporal}"

    JT=$(sbatch --parsable \
        --job-name="train_${exp}" \
        --output="$LOGS_ROOT/$exp/lgbm.out" \
        --error="$LOGS_ROOT/$exp/lgbm.err" \
        --export=ALL,EXP="$exp",KIND="$kind",STEP=train,RESULTS_ROOT="$RESULTS_ROOT" \
        submit/run_array.sh)
    TRAIN_JIDS+=("$JT")

    JS=""
    if [ "$STEP" != "train" ]; then
        JS=$(sbatch --parsable \
            --job-name="shap_${exp}" \
            --output="$LOGS_ROOT/$exp/shap.out" \
            --error="$LOGS_ROOT/$exp/shap.err" \
            --dependency="afterok:$JT" \
            --export=ALL,EXP="$exp",KIND="$kind",STEP=shap,RESULTS_ROOT="$RESULTS_ROOT" \
            submit/run_array.sh)
        if [ "$kind" = "spatial" ]; then SHAP_JIDS_SPAT+=("$JS"); else SHAP_JIDS_TEMP+=("$JS"); fi
    fi
    echo "   $exp  train=$JT  shap=${JS:-skipped}  ($kind)"
    i=$((i + 1))
done < submit/exp_list.txt

# 5) evaluation: two jobs, one per mode. Each depends (afterok) on the SHAP
#    jobs of its mode (or on the train jobs if STEP=train).
submit_eval() {
    local mode="$1"; local n="$2"; local list="$3"; shift 3
    local deps=("$@")
    [ "$n" -lt 1 ] && { echo ">> no $mode experiments -> eval skipped."; return 0; }
    [ "$RUN_EVAL" = "1" ] || { echo ">> RUN_EVAL=0 -> eval ($mode) skipped."; return 0; }
    local dep=""
    if [ "${#deps[@]}" -gt 0 ]; then
        dep="--dependency=afterok:$(IFS=: ; echo "${deps[*]}")"
    fi
    local OUT="Evaluation_${mode}"
    local JE
    JE=$(sbatch --parsable \
        --job-name="eval_${mode}" \
        --output="$LOGS_ROOT/$OUT/eval.out" \
        --error="$LOGS_ROOT/$OUT/eval.err" \
        $dep \
        --export=ALL,EVAL_MODE="$mode",EXP_LIST="$list",RESULTS_ROOT="$RESULTS_ROOT",EVAL_OUT="$OUT" \
        submit/run_eval.sh)
    echo ">> eval $mode: $JE  ($n experiments)"
}

if [ "$STEP" = "train" ]; then
    submit_eval temporal "$N_TEMP" submit/exp_list_temporal.txt "${TRAIN_JIDS[@]}"
    submit_eval spatial  "$N_SPAT" submit/exp_list_spatial.txt  "${TRAIN_JIDS[@]}"
else
    submit_eval temporal "$N_TEMP" submit/exp_list_temporal.txt "${SHAP_JIDS_TEMP[@]}"
    submit_eval spatial  "$N_SPAT" submit/exp_list_spatial.txt  "${SHAP_JIDS_SPAT[@]}"
fi

echo ">> submission finished."
echo "   Logs:    $LOGS_ROOT"
echo "   Results: $RESULTS_ROOT"

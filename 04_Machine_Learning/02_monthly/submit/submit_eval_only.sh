#!/bin/bash
# =====================================================================
# submit/submit_eval_only.sh
# Resubmit ONLY the evaluation step (one SLURM job per mode), with NO
# train/shap dependency. Use this when training and SHAP are already
# finished and only fresh Evaluation_<mode>/ outputs are needed -- for
# example after changing evaluation.py.
#
# It mirrors the evaluation part of submit_all.sh (same environment, same
# RESULTS_ROOT/LOGS_ROOT defaults, same Evaluation_<mode> layout), but
# submits run_eval.sh directly instead of chaining it behind the SHAP jobs.
#
# Usage:
#   bash submit/submit_eval_only.sh                    # all experiments from the JSON
#   bash submit/submit_eval_only.sh ERA5_4_twsa GEMS   # only these (include the benchmark!)
#   RESULTS_ROOT=<path> bash submit/submit_eval_only.sh
#   SKIP_RESOLVE=1 bash submit/submit_eval_only.sh     # do not rerun resolve_experiments.py
#
# Note: a KGE skill score against the GEMS benchmark can only be derived
# later if the benchmark's KGE is in merged_scores.csv, so keep GEMS
# (temporal) / GEMS_spatial (spatial) in the evaluated set. With no arguments
# (all experiments) they are included automatically; with explicit
# arguments, add them yourself.
# =====================================================================
set -euo pipefail

# Code-home (parent of submit/), same detection as submit_all.sh.
ML_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_HOME="$(basename "$ML_DIR")"
cd "$ML_DIR"
# Export so --export=ALL carries it into the compute-node job (run_eval.sh
# resolves the code-home from ML_DIR first).
export ML_DIR

# Module + venv (login node, sets $WS), identical to submit_all.sh.
source "$ML_DIR/submit/env.sh"

RESULTS_ROOT="${RESULTS_ROOT:-$WS/05_Machine_Learning_Results/$CODE_HOME}"
LOGS_ROOT="${LOGS_ROOT:-$WS/06_Machine_Learning_Logs/$CODE_HOME}"
SKIP_RESOLVE="${SKIP_RESOLVE:-0}"

mkdir -p submit "$RESULTS_ROOT" "$LOGS_ROOT" \
         "$LOGS_ROOT/Evaluation_temporal" "$LOGS_ROOT/Evaluation_spatial"

echo "Code-Home    : $ML_DIR"
echo "RESULTS_ROOT : $RESULTS_ROOT"
echo "LOGS_ROOT    : $LOGS_ROOT"

# 1) Refresh allowlists (cheap, idempotent) unless told to skip.
if [ "$SKIP_RESOLVE" != "1" ]; then
    echo ">> resolve_experiments.py"
    python resolve_experiments.py
fi

# 2) Experiment list: CLI args, or all experiments from the JSON.
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
N=$(grep -c . submit/exp_list.txt || true)
echo ">> $N experiments:"; sed 's/^/   - /' submit/exp_list.txt

# 2b) Split into temporal/spatial lists (same logic as submit_all.sh).
python - <<'PY'
import json, os
from resolve_experiments import DEFAULT_META_DIR
d = json.load(open(os.path.join(DEFAULT_META_DIR, "resolved_allowlists.json")))
names = open("submit/exp_list.txt").read().split()
def kind(n): return d["experiments"][n].get("split", {}).get("kind", "temporal")
def write(p, lines): open(p, "w").write("\n".join(lines) + ("\n" if lines else ""))
write("submit/exp_list_temporal.txt", [n for n in names if kind(n) == "temporal"])
write("submit/exp_list_spatial.txt",  [n for n in names if kind(n) == "spatial"])
PY
N_TEMP=$(grep -c . submit/exp_list_temporal.txt || true)
N_SPAT=$(grep -c . submit/exp_list_spatial.txt  || true)
echo ">> temporal: $N_TEMP  |  spatial: $N_SPAT"

# Benchmark presence warning: a skill score is only possible later if the
# benchmark KGE column is in merged_scores.csv -> keep GEMS in the set.
check_bench() {
    local list="$1" bench="$2" mode="$3" n="$4"
    [ "$n" -lt 1 ] && return 0
    grep -qx "$bench" "$list" \
        || echo "   WARN: '$bench' not in $mode list -> no KGE_$bench in merged_scores -> no KGE skill score vs $bench possible."
}
check_bench submit/exp_list_temporal.txt GEMS         temporal "$N_TEMP"
check_bench submit/exp_list_spatial.txt  GEMS_spatial spatial  "$N_SPAT"

# 3) One eval job per mode, submitted immediately (no afterok dependency).
submit_eval() {
    local mode="$1" n="$2" list="$3"
    [ "$n" -lt 1 ] && { echo ">> no $mode experiments -> eval skipped."; return 0; }
    local OUT="Evaluation_${mode}"
    local JE
    JE=$(sbatch --parsable \
        --job-name="eval_${mode}" \
        --output="$LOGS_ROOT/$OUT/eval.out" \
        --error="$LOGS_ROOT/$OUT/eval.err" \
        --export=ALL,EVAL_MODE="$mode",EXP_LIST="$list",RESULTS_ROOT="$RESULTS_ROOT",EVAL_OUT="$OUT" \
        submit/run_eval.sh)
    echo ">> eval $mode: job $JE  ($n experiments)  -> $RESULTS_ROOT/$OUT/"
}
submit_eval temporal "$N_TEMP" submit/exp_list_temporal.txt
submit_eval spatial  "$N_SPAT" submit/exp_list_spatial.txt

echo ">> submitted. Outputs -> $RESULTS_ROOT/Evaluation_{temporal,spatial}/"
echo "   Logs    -> $LOGS_ROOT/Evaluation_{temporal,spatial}/eval.{out,err}"

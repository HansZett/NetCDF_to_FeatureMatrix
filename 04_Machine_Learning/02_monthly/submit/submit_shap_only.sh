#!/bin/bash
# =====================================================================
# submit/submit_shap_only.sh
# Resubmit ONLY the SHAP step (one SLURM job per experiment), with NO
# train dependency. Use this when the models are already trained and
# only shap_analysis.py has changed -- for example after switching the
# spatial explanation from the final deployment ensemble to the scored
# per-fold ensembles, or after adding the shap_corr value-response
# column that the analysis layer consumes.
#
# It mirrors the SHAP part of submit_all.sh exactly (same env, same
# RESULTS_ROOT/LOGS_ROOT defaults, same log names), but submits
# run_array.sh with STEP=shap directly instead of chaining it behind
# the train jobs. Going through run_array.sh keeps environment,
# temporal/spatial dispatch and resources in ONE place, so a SHAP run
# from here is identical to one from the full pipeline.
#
# Usage:
#   bash submit/submit_shap_only.sh                     # all experiments from the JSON
#   bash submit/submit_shap_only.sh ERA5_4_twsa GEMS    # only these
#   MODE=spatial  bash submit/submit_shap_only.sh       # only spatial experiments
#   MODE=temporal bash submit/submit_shap_only.sh       # only temporal experiments
#   RESULTS_ROOT=<path> bash submit/submit_shap_only.sh
#   SKIP_RESOLVE=1 bash submit/submit_shap_only.sh      # do not rerun resolve_experiments.py
#   DRY_RUN=1 bash submit/submit_shap_only.sh           # print the list, submit nothing
#
# Note: this overwrites <EXP>/shap.{out,err} of the previous SHAP run.
# The evaluation does not read SHAP outputs, so it never needs to be
# resubmitted because of this script. The analysis layer (not part of this
# repository) has to be rerun on the new SHAP outputs.
#
# The generated lists submit/shap_exp_request.txt and submit/shap_exp_kind.txt
# are rewritten on every call and need not be kept.
# =====================================================================
set -euo pipefail

# Code-home (parent of submit/), same detection as submit_all.sh.
ML_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_HOME="$(basename "$ML_DIR")"
cd "$ML_DIR"
# Export so --export=ALL carries it into the compute-node job (run_array.sh
# resolves the code-home from ML_DIR first).
export ML_DIR

# Module + venv (login node, sets $WS), identical to submit_all.sh.
source "$ML_DIR/submit/env.sh"

RESULTS_ROOT="${RESULTS_ROOT:-$WS/05_Machine_Learning_Results/$CODE_HOME}"
LOGS_ROOT="${LOGS_ROOT:-$WS/06_Machine_Learning_Logs/$CODE_HOME}"
SKIP_RESOLVE="${SKIP_RESOLVE:-0}"
MODE="${MODE:-all}"                  # all | temporal | spatial
DRY_RUN="${DRY_RUN:-0}"

case "$MODE" in
    all|temporal|spatial) ;;
    *) echo "MODE must be all, temporal or spatial (got: '$MODE')" >&2; exit 2 ;;
esac

mkdir -p submit "$RESULTS_ROOT" "$LOGS_ROOT"

echo "Code home    : $ML_DIR"
echo "RESULTS_ROOT : $RESULTS_ROOT"
echo "LOGS_ROOT    : $LOGS_ROOT"
echo "MODE=$MODE  DRY_RUN=$DRY_RUN"

# 1) Refresh allowlists (cheap, idempotent) unless told to skip.
if [ "$SKIP_RESOLVE" != "1" ]; then
    echo ">> resolve_experiments.py"
    python resolve_experiments.py >/dev/null
fi

# 2) Experiment list + split kind: CLI args, or all experiments from the JSON.
#    Written as "<name> <kind>" per line; the kind drives KIND= for run_array.sh.
if [ "$#" -gt 0 ]; then
    printf "%s\n" "$@" > submit/shap_exp_request.txt
else
    : > submit/shap_exp_request.txt
fi

MODE="$MODE" python - <<'PY' > submit/shap_exp_kind.txt
import json, os
from resolve_experiments import DEFAULT_META_DIR
d = json.load(open(os.path.join(DEFAULT_META_DIR, "resolved_allowlists.json")))["experiments"]
req = [l.strip() for l in open("submit/shap_exp_request.txt") if l.strip()]
mode = os.environ.get("MODE", "all")
names = req or list(d)
missing = [n for n in names if n not in d]
if missing:
    raise SystemExit(f"Not in resolved_allowlists.json: {missing}")
def kind(n):
    return d[n].get("split", {}).get("kind", "temporal")
sel = [n for n in names if mode == "all" or kind(n) == mode]
print("\n".join(f"{n} {kind(n)}" for n in sel))
PY

N=$(grep -c . submit/shap_exp_kind.txt || true)
if [ "$N" -lt 1 ]; then
    echo ">> No matching experiments -> nothing to do."; exit 0
fi
echo ">> $N experiment(s):"; sed 's/^/   - /' submit/shap_exp_kind.txt

if [ "$DRY_RUN" = "1" ]; then
    echo ">> DRY_RUN=1 -> nothing submitted."; exit 0
fi

# 3) One SHAP job per experiment. Log dir per experiment (SLURM does not
#    create it), names 1:1 like submit_all.sh so the tree stays consistent.
JIDS=()
while read -r exp kind; do
    [ -z "$exp" ] && continue
    mkdir -p "$LOGS_ROOT/$exp"
    JS=$(sbatch --parsable \
        --job-name="shap_${exp}" \
        --output="$LOGS_ROOT/$exp/shap.out" \
        --error="$LOGS_ROOT/$exp/shap.err" \
        --export=ALL,EXP="$exp",KIND="$kind",STEP=shap,RESULTS_ROOT="$RESULTS_ROOT" \
        submit/run_array.sh)
    JIDS+=("$JS")
    echo "   $exp  shap=$JS  ($kind)"
done < submit/shap_exp_kind.txt

echo ">> ${#JIDS[@]} SHAP jobs submitted."
echo "   Logs:    $LOGS_ROOT/<EXP>/shap.{out,err}"
echo "   Results: $RESULTS_ROOT/<results_dir>/{run<seed>,fold<k>}/"
echo "   Next:    rerun the analysis layer on the new SHAP outputs"

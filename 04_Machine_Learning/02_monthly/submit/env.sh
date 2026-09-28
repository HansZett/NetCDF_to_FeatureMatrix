#!/bin/bash
# =====================================================================
# submit/env.sh  --  shared environment of all monthly submit scripts.
# ONE place for the HPC workspace, the Python module and the venv.
# Sourced by submit_all.sh / submit_eval_only.sh (login node) and by
# run_array.sh / run_eval.sh / run_shap.sh (compute nodes).
# Idempotent: sourcing it several times does no harm.
#
# HPC SETTINGS (adjust for another user / cluster, or override them as
# environment variables before submitting):
#   WS         HPC workspace = project root on the cluster (contains
#              03_Dataframes, 04_Machine_Learning, 05_..._Results, ...)
#   PY_MODULE  Lmod module that provides Python
#   VENV       Python virtual environment with the packages
# =====================================================================
# Replace the placeholder below with your own HPC workspace path.
export WS="${WS:-/path/to/your/workspace}"
PY_MODULE="${PY_MODULE:-devel/python/3.12_gnu_11.4}"
VENV="${VENV:-$WS/envs/ml_hydro}"
[ -d "$WS" ] || echo "WARN: workspace WS=$WS does not exist; set WS in submit/env.sh" >&2

# Lmod ('module') and some venv activate scripts are not 'set -u' safe.
# Remember the caller's strict flags, relax them here, restore them afterwards.
_env_old_opts="$-"
set +u

# In non-interactive batch shells (compute node) 'module' is often not
# defined, because /etc/profile.d is not sourced -> 'module load' would be
# skipped silently and the runtime libraries (gcc/OpenMP for numpy/lightgbm)
# would be missing. Therefore initialise Lmod here if needed.
if ! command -v module >/dev/null 2>&1; then
    [ -f /etc/profile.d/00-modulepath.sh ] && source /etc/profile.d/00-modulepath.sh
    for _init in "${LMOD_PKG:-/usr/share/lmod/lmod}/init/bash" \
                 /etc/profile.d/lmod.sh /etc/profile.d/modules.sh; do
        [ -f "$_init" ] && { source "$_init"; break; }
    done
    unset _init
fi

if command -v module >/dev/null 2>&1; then
    module load "$PY_MODULE" 2>/dev/null || true
fi

# ALWAYS (re-)activate the venv, do not rely on VIRTUAL_ENV:
# submit_all.sh sources env.sh on the login node, so --export=ALL carries a set
# VIRTUAL_ENV into the job. The compute node rewrites PATH (system Python
# first) but keeps VIRTUAL_ENV -> an equality check would skip the activation
# and 'python' would point to the wrong interpreter (-> no pandas).
# The venv activation is idempotent (it resets PATH itself first).
if [ -f "$VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
else
    echo "WARN: venv activate script missing: $VENV/bin/activate" >&2
fi

case "$_env_old_opts" in
    *u*) set -u ;;
esac
unset _env_old_opts

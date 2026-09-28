#!/bin/bash
# =====================================================================
# submit/env.sh  --  shared environment of the weekly stack.
# ONE place for the HPC workspace, the Python module and the venv.
# Sourced by submit/submit_all.sh (login node) and submit/run_*.sh
# (compute nodes). Idempotent: sourcing it several times does no harm.
#
# HPC SETTINGS (adjust for another user / cluster, or override them as
# environment variables before submitting):
#   WS         HPC workspace = project root on the cluster (contains
#              03_Dataframes, 04_Machine_Learning, 05_..._Results, ...)
#   PY_MODULE  Lmod module that provides Python
#   VENV       Python virtual environment with the packages
# =====================================================================
#
# Mirrors the monthly submit/env.sh (same venv). All jobs of this public
# version (lgbm, shap, eval) run on CPU.
# Replace the placeholder below with your own HPC workspace path.
export WS="${WS:-/path/to/your/workspace}"
PY_MODULE="${PY_MODULE:-devel/python/3.12_gnu_11.4}"
VENV="${VENV:-$WS/envs/ml_hydro}"
[ -d "$WS" ] || echo "WARN: workspace WS=$WS does not exist; set WS in submit/env.sh" >&2

# Lmod ('module') and venv activation are not 'set -u' safe.
_env_old_opts="$-"
set +u

# In non-interactive batch shells (compute node) 'module' is often not
# defined (no /etc/profile.d sourcing) -> initialise Lmod here.
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

# ALWAYS (re-)activate the venv, do not rely on VIRTUAL_ENV: submit_all.sh
# sources env.sh on the login node and --export=ALL carries VIRTUAL_ENV into
# the job, but the compute node rewrites PATH. Activation is idempotent.
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

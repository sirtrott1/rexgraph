#!/bin/sh

# run_core_tests.sh - run the rexgraph CORE test suite against the compiled core.

# sh run_core_tests.sh            # whole core suite (~40s, 1700+ tests)
# sh run_core_tests.sh -k boundary   # extra args pass straight to pytest
# sh run_core_tests.sh -x -q         # (any pytest flags)

# The package is installed EDITABLE (meson-python): `rexgraph` is served from the
# source tree and its compiled extensions from build/, both via the meson-python
# import finder, which rebuilds changed .pyx on import. The repo-root conftest.py
# points rexgraph.__path__ at the source dir so `rexgraph.tests` (not a shipped
# package) resolves too. So we just run pytest against the tree - no assembling a
# throwaway package, no source-vs-installed juggling.
#   install editable:  pip install --no-build-isolation -e ".[io]"

set -eu

ENV_NAME="${ENV_NAME:-rexgraph}"

# Pick a test runner DYNAMICALLY - never hard-fail just because conda is absent:
#   1. a conda frontend (micromamba/mamba/conda) with an env named $ENV_NAME  -> "conda run -n"
#   2. else the active interpreter (a venv/uv/poetry/pdm/system python) that can import the core
# Override the interpreter for path 2 via PYTHON=... (default: python3, then python).
CONDA=""
if command -v micromamba >/dev/null 2>&1; then CONDA="micromamba"
elif [ -x "$HOME/.local/bin/micromamba" ]; then CONDA="$HOME/.local/bin/micromamba"
elif command -v mamba >/dev/null 2>&1; then CONDA="mamba"
elif command -v conda >/dev/null 2>&1; then CONDA="conda"
fi

# does the conda frontend actually have the target env? (empty -> no)
ENV_PRESENT=""
if [ -n "$CONDA" ] && "$CONDA" env list 2>/dev/null | grep -qw "$ENV_NAME"; then ENV_PRESENT=1; fi

if [ -n "$ENV_PRESENT" ]; then
    RUNNER="conda"
    INENV() { "$CONDA" run -n "$ENV_NAME" "$@"; }   # existing conda path, unchanged
else
    # fall back to a plain interpreter (venv/uv/poetry/pdm/system)
    PYBIN=""
    for c in "${PYTHON:-}" python3 python; do
        [ -n "$c" ] && command -v "$c" >/dev/null 2>&1 && { PYBIN="$c"; break; }
    done
    [ -n "$PYBIN" ] || { echo "ERROR: no conda env '$ENV_NAME' and no python on PATH." >&2; exit 1; }
    RUNNER="python"
    # call sites use `INENV python ...` (the conda convention); drop that leading token here.
    INENV() { case "${1:-}" in python|python3) shift ;; esac; "$PYBIN" "$@"; }
    if [ -z "$CONDA" ]; then echo "==> no conda frontend; using '$PYBIN' directly"; \
    else echo "==> conda frontend '$CONDA' has no env '$ENV_NAME'; using '$PYBIN' directly"; fi
fi

REPO="$(cd "$(dirname "$0")" && pwd)"
[ -d "$REPO/rexgraph/tests" ] || { echo "ERROR: run from the repo root (needs rexgraph/tests/)." >&2; exit 1; }

# sanity: the compiled core must import (catches a missing/broken editable build)
INENV python -c "from rexgraph.core import _boundary" 2>/dev/null || {
    echo "ERROR: rexgraph core not importable. Install it editable first:" >&2
    if [ "$RUNNER" = conda ]; then
        echo "         $CONDA run -n $ENV_NAME pip install --no-build-isolation -e \".[io]\"" >&2
    else
        echo "         pip install --no-build-isolation -e \".[io]\"   (in your active venv)" >&2
    fi
    exit 1; }

echo "==> core suite (editable install)"
( cd "$REPO" && INENV python -m pytest -q -p no:cacheprovider rexgraph/tests "$@" )

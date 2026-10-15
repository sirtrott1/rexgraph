#!/bin/sh
# Run Core tests against the selected installation.
set -eu
REPO="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
VENV_DIR="${VENV_DIR:-$REPO/.venv}"
if [ -n "${PYTHON:-}" ]; then
    INENV() { "$PYTHON" "$@"; }
elif [ -x "$VENV_DIR/bin/python" ]; then
    INENV() { "$VENV_DIR/bin/python" "$@"; }
else
    CONDA=""
    for candidate in micromamba mamba conda; do
        if command -v "$candidate" >/dev/null 2>&1; then CONDA="$candidate"; break; fi
    done
    if [ -z "$CONDA" ] && [ -x "$HOME/.local/bin/micromamba" ]; then
        CONDA="$HOME/.local/bin/micromamba"
    fi
    if [ -n "$CONDA" ] && "$CONDA" env list 2>/dev/null | grep -qw "${ENV_NAME:-rexgraph}"; then
        INENV() { "$CONDA" run -n "${ENV_NAME:-rexgraph}" python "$@"; }
    else
        PYTHON=python3
        INENV() { "$PYTHON" "$@"; }
    fi
fi
cd "$REPO"
INENV -I "$REPO/scripts/test_installed.py" --allow-editable --package rexgraph -- "$REPO/rexgraph/tests" "$@"

#!/usr/bin/env bash
# setup_hpc.sh -- set up rexgraph on a cluster.
#
#   bash setup_hpc.sh              # full install
#   bash setup_hpc.sh --minimal    # core only
#   bash setup_hpc.sh --test       # install + run tests
#   bash setup_hpc.sh --clean      # remove environment

set -euo pipefail

ENV_NAME="rcf"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINIMAL=false
RUN_TESTS=false
CLEAN=false

for arg in "$@"; do
    case "$arg" in
        --minimal) MINIMAL=true ;;
        --test)    RUN_TESTS=true ;;
        --clean)   CLEAN=true ;;
        --help|-h)
            echo "Usage: bash setup_hpc.sh [--minimal] [--test] [--clean]"
            exit 0
            ;;
        *) echo "Unknown: $arg"; exit 1 ;;
    esac
done

if $CLEAN; then
    conda env remove -n "$ENV_NAME" -y 2>/dev/null || true
    exit 0
fi

# Try to load gcc from the module system if available.
# Some clusters require this before conda can link against libgomp.
for mod in gcc gnu compiler; do
    module load "$mod" 2>/dev/null && break || true
done

# Pick conda or mamba.
if command -v mamba &>/dev/null; then
    CONDA=mamba
elif command -v conda &>/dev/null; then
    CONDA=conda
else
    echo "No conda or mamba found."
    echo "Load a module (module load anaconda / miniconda / miniforge)"
    echo "or install miniforge:"
    echo "  wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    echo "  bash Miniforge3-Linux-x86_64.sh -b -p \$HOME/miniforge3"
    echo "  eval \"\$(\$HOME/miniforge3/bin/conda shell.bash hook)\""
    exit 1
fi

# Unload modules that conflict with conda-provided compilers and BLAS.
for mod in openblas mkl lapack blas intel; do
    module unload "$mod" 2>/dev/null || true
done

# Pick environment file.
if $MINIMAL; then
    ENV_FILE="$REPO_DIR/environment-minimal.yml"
else
    ENV_FILE="$REPO_DIR/environment.yml"
fi

[ -f "$ENV_FILE" ] || { echo "Missing: $ENV_FILE"; exit 1; }

# Create or update.
if conda env list | grep -q "^${ENV_NAME} "; then
    $CONDA env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
    $CONDA env create -f "$ENV_FILE"
fi

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# Build in scratch or tmpdir if available (faster than NFS home).
BUILD_DIR="builddir"
if [ -n "${SCRATCH:-}" ] && [ -d "$SCRATCH" ]; then
    BUILD_DIR="$SCRATCH/rexgraph_build"
elif [ -n "${TMPDIR:-}" ] && [ -d "$TMPDIR" ]; then
    BUILD_DIR="$TMPDIR/rexgraph_build"
fi
rm -rf "$BUILD_DIR"

cd "$REPO_DIR"
pip install -e . --no-build-isolation --no-cache-dir \
    -Cbuilddir="$BUILD_DIR" 2>&1 | tail -5

# Smoke test.
python -c "
from rexgraph.graph import RexGraph
import numpy as np
rex = RexGraph.from_graph([0,1,0],[1,2,2])
assert rex.betti == (1,1,0)
print(f'OK: {rex.nV}V {rex.nE}E betti={rex.betti}')
" || { echo "Smoke test failed."; exit 1; }

if $RUN_TESTS; then
    python -m pytest tests/ -x -q --tb=short
fi

echo ""
echo "Done. Activate with: conda activate $ENV_NAME"
echo "Rebuild with: pip install -e . --no-build-isolation -Cbuilddir=$BUILD_DIR"

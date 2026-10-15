#!/bin/sh
# Install with the shared repository installer and generate SLURM templates.
set -eu
REPO_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"
case "$VENV_DIR" in /*) ;; *) VENV_DIR="$PWD/$VENV_DIR" ;; esac
GPU_TYPE=cpu
case "${1:-}" in
    --help|-h)
        printf 'Usage: sh agent/setup_hpc.sh [--cpu-only | --gpu cpu|cuda|rocm]\n'
        printf 'Load site modules first for INSTALLER=pip. GPU installs require TORCH_INDEX_URL.\n'
        exit 0 ;;
    --cpu-only|'') ;;
    --gpu) [ "$#" -eq 2 ] || { echo '--gpu requires cpu, cuda, or rocm' >&2; exit 1; }; GPU_TYPE="$2" ;;
    *) echo 'Use --help for HPC setup options.' >&2; exit 1 ;;
esac
case "$GPU_TYPE" in
    cpu) PROFILE=standard ;;
    cuda|rocm)
        [ -n "${TORCH_INDEX_URL:-}" ] || { echo 'Set TORCH_INDEX_URL for the cluster GPU.' >&2; exit 1; }
        PROFILE=ml ;;
    *) echo 'GPU must be cpu, cuda, or rocm.' >&2; exit 1 ;;
esac
VENV_DIR="$VENV_DIR" sh "$REPO_DIR/install.sh" --profile "$PROFILE"
cd "$REPO_DIR"
mkdir -p slurm
"$VENV_DIR/bin/python" -I - <<'PY'
from agent.cli.hpc import write_template
from pathlib import Path
import sys
for name in ('build', 'serve', 'ocr_batch', 'corpus', 'training', 'array'):
    path = Path(f'slurm/{name}.sbatch')
    if path.exists():
        print(f'Keeping existing template: {path}')
    else:
        write_template(name, output=str(path), conda_env=sys.prefix)
PY
printf '\nTemplates written to %s/slurm. Set the account, partition and paths before submitting.\n' "$REPO_DIR"

#!/bin/bash
# setup_hpc.sh - Set up rexgraph on an HPC cluster

# Usage:
# bash setup_hpc.sh # interactive
# bash setup_hpc.sh --gpu rocm # ROCm GPU (AMD)
# bash setup_hpc.sh --gpu cuda # CUDA GPU (NVIDIA)
# bash setup_hpc.sh --cpu-only # CPU only (no GPU)

# Prerequisites:
# - conda or module system
# - C compiler (gcc/clang)
# - BLAS/LAPACK (openblas or mkl)

set -euo pipefail

GPU_TYPE="${1:-auto}"
CONDA_ENV="${CONDA_ENV:-rexgraph}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

echo "============================================"
echo "rexgraph HPC setup"
echo "============================================"
echo "GPU type:    $GPU_TYPE"
echo "Conda env:   $CONDA_ENV"
echo "Python:      $PYTHON_VERSION"
echo ""

# Detect GPU if auto
if [ "$GPU_TYPE" = "auto" ] || [ "$GPU_TYPE" = "--gpu" ]; then
    shift 2>/dev/null || true
    GPU_TYPE="${1:-auto}"
fi

if [ "$GPU_TYPE" = "auto" ]; then
    if command -v rocminfo &>/dev/null; then
        GPU_TYPE="rocm"
        echo "Detected: ROCm (AMD GPU)"
    elif command -v nvidia-smi &>/dev/null; then
        GPU_TYPE="cuda"
        echo "Detected: CUDA (NVIDIA GPU)"
    else
        GPU_TYPE="cpu"
        echo "Detected: CPU only"
    fi
fi

if [ "$GPU_TYPE" = "--cpu-only" ]; then
    GPU_TYPE="cpu"
fi

# Load modules if available
if command -v module &>/dev/null; then
    echo ""
    echo "Loading modules..."
    module load python 2>/dev/null || true
    module load openblas 2>/dev/null || module load mkl 2>/dev/null || true
    module load gcc 2>/dev/null || true
    if [ "$GPU_TYPE" = "rocm" ]; then
        module load rocm 2>/dev/null || true
    elif [ "$GPU_TYPE" = "cuda" ]; then
        module load cuda 2>/dev/null || module load cudatoolkit 2>/dev/null || true
    fi
fi

# Create conda environment
echo ""
echo "Setting up conda environment: $CONDA_ENV"
if conda info --envs | grep -q "$CONDA_ENV"; then
    echo "Environment exists, activating..."
    source activate "$CONDA_ENV" 2>/dev/null || conda activate "$CONDA_ENV"
else
    echo "Creating new environment..."
    conda create -n "$CONDA_ENV" python="$PYTHON_VERSION" -y
    source activate "$CONDA_ENV" 2>/dev/null || conda activate "$CONDA_ENV"
fi

# Install BLAS
echo ""
echo "Installing BLAS/LAPACK..."
conda install -y numpy scipy openblas 2>/dev/null || pip install numpy scipy

# Install build deps
echo ""
echo "Installing build dependencies..."
pip install meson-python meson cython ninja

# Build rexgraph
echo ""
echo "Building rexgraph..."
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

if [ -f "meson.build" ] || [ -f "pyproject.toml" ]; then
    pip install -e . --no-build-isolation
    echo "rexgraph built successfully"
else
    echo "ERROR: Run this script from the rexgraph repo root"
    exit 1
fi

# Install agent with server deps
echo ""
echo "Installing agent..."
pip install -e ./agent[server]

# Install GPU-specific packages
echo ""
if [ "$GPU_TYPE" = "rocm" ]; then
    echo "Installing ROCm PyTorch..."
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/rocm6.2
    pip install -e ./agent[ocr]
    echo "ROCm packages installed"

elif [ "$GPU_TYPE" = "cuda" ]; then
    echo "Installing CUDA PyTorch..."
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124
    pip install -e ./agent[ocr]
    echo "CUDA packages installed"

else
    echo "CPU-only mode - skipping GPU packages"
    echo "OCR will use tesseract (install: apt install tesseract-ocr)"
fi

# Install optional packages
echo ""
echo "Installing optional packages..."
pip install pyyaml 2>/dev/null || true       # YAML agent configs
pip install safetensors 2>/dev/null || true   # training export
pip install sqlalchemy 2>/dev/null || true    # persistence

# Set up shared model cache
echo ""
echo "Setting up model cache..."
python -c "
from agent.cli.hpc import resolve_model_cache
cache = resolve_model_cache()
print('Model cache: %s' % cache)
"

# Generate SLURM templates
echo ""
echo "Generating SLURM templates..."
mkdir -p slurm/
python -c "
from agent.cli.hpc import write_template, list_templates
print('Available templates:')
list_templates()
print()
write_template('build', output='slurm/build.sbatch')
write_template('serve', output='slurm/serve.sbatch')
write_template('ocr_batch', output='slurm/ocr_batch.sbatch')
write_template('corpus', output='slurm/corpus.sbatch')
write_template('training', output='slurm/training.sbatch')
write_template('array', output='slurm/array.sbatch')
"

# Verify
echo ""
echo "Verifying installation..."
python -c "
from rexgraph.graph import RexGraph
import numpy as np
rex = RexGraph.from_graph([0,1,0],[1,2,2])
print('rexgraph:  %dV %dE betti=%s' % (rex.nV, rex.nE, rex.betti))
from agent.adapters.text import TextAdapter
from agent.corpus import CorpusBuilder
from agent.builder import AgentBuilder
print('agent:     %d builder steps, %d modules' % (len(AgentBuilder.available_steps()), 30))
print('templates: %s' % ', '.join(sorted(AgentBuilder.template('default').keys())))
"

echo ""
echo "============================================"
echo "Setup complete!"
echo ""
echo "Quick start:"
echo "  sbatch slurm/build.sbatch      # build on compute node"
echo "  sbatch slurm/serve.sbatch      # start GPU server"
echo "  sbatch slurm/ocr_batch.sbatch  # batch OCR"
echo "  sbatch slurm/corpus.sbatch     # build corpus"
echo "  sbatch slurm/training.sbatch   # export training data"
echo "  sbatch --array=0-9 slurm/array.sbatch  # parallel processing"
echo ""
echo "Edit the .sbatch files to set your paths, partitions, and accounts."
echo "============================================"

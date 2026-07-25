"""
agent.cli.hpc - SLURM job templates and HPC deployment.

Generates job scripts for all rexgraph HPC workloads:
    ocr_batch   - batch OCR processing with GPU
    serve       - persistent GPU server for interactive use
    build       - build rexgraph from source on compute nodes
    corpus      - build corpus from large document collections
    training    - export training data as safetensors
    array       - parallel document processing (SLURM array job)

Usage:
    from agent.cli.hpc import write_template
    write_template("ocr_batch", output="ocr_batch.sbatch",
                   input_dir="/data/pdfs", partition="gpu-a100")

    # Or from CLI:
    rexgraph-hpc template ocr_batch --partition gpu-a100
    rexgraph-hpc template array --input-dir /data/docs --files-per-task 50
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import Optional


# Shared model cache

def resolve_model_cache(platform_info=None) -> Path:
    """Best location for model weights on HPC.

    Checks $PROJECT, $WORK, $SCRATCH for shared storage.
    Falls back to ~/.cache/rexgraph/models/.
    """
    for env_var in ("PROJECT", "WORK", "SCRATCH"):
        shared = os.environ.get(env_var, "")
        if shared and os.path.isdir(shared):
            cache = Path(shared) / "rexgraph_models"
            cache.mkdir(parents=True, exist_ok=True)
            return cache

    cache = Path.home() / ".cache" / "rexgraph" / "models"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


# Common header

def _header(job_name, partition, time_limit, mem, cpus, account, gpus=0, array_spec=""):
    lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=%s" % job_name,
        "#SBATCH --partition=%s" % partition,
    ]
    if gpus:
        lines.append("#SBATCH --gres=gpu:%d" % gpus)
    if array_spec:
        lines.append("#SBATCH --array=%s" % array_spec)
    lines.extend([
        "#SBATCH --time=%s" % time_limit,
        "#SBATCH --mem=%s" % mem,
        "#SBATCH --cpus-per-task=%d" % cpus,
        "#SBATCH --output=%s-%%j.out" % job_name,
        "#SBATCH --error=%s-%%j.err" % job_name,
    ])
    if account:
        lines.append("#SBATCH --account=%s" % account)
    lines.extend([
        "",
        "set -euo pipefail",
        'echo "Job $SLURM_JOB_ID on $(hostname) - $(date)"',
        "",
    ])
    return "\n".join(lines)


def _activate(conda_env):
    return (
        "source activate %s 2>/dev/null || conda activate %s 2>/dev/null || true\n"
        'export REXGRAPH_MODELS=$(python -c "from agent.cli.hpc import resolve_model_cache; print(resolve_model_cache())")\n'
    ) % (conda_env, conda_env)


# Templates

def generate_slurm_ocr_batch(
    input_dir="/path/to/pdfs", output_dir="/path/to/output",
    partition="gpu", gpus=1, time_limit="04:00:00", mem="32G",
    account="", conda_env="rexgraph",
):
    """Batch OCR: process a directory of PDFs with GPU OCR.

    Starts a GPU OCR server, processes every PDF in input_dir,
    saves each document's RexGraph as a .rex bundle.
    """
    return _header("rexgraph-ocr", partition, time_limit, mem, 4, account, gpus) + "\n" + _activate(conda_env) + textwrap.dedent("""\

        # Start GPU OCR server
        rexgraph-ocr serve --port 10000 &
        OCR_PID=$!
        sleep 30

        INPUT_DIR="%s"
        OUTPUT_DIR="%s"
        mkdir -p "$OUTPUT_DIR"

        for pdf in "$INPUT_DIR"/*.pdf; do
            [ -f "$pdf" ] || continue
            name=$(basename "$pdf" .pdf)
            echo "Processing: $name"
            python -c "
from agent.adapters.ocr import OCRAdapter
from agent.server.persistence import save_document_rex
from rexgraph.graph import RexGraph
import numpy as np

adapter = OCRAdapter()
ec = adapter.build('$pdf', strategy='layout')
if ec.nE > 0:
    rex = RexGraph(sources=ec.sources, targets=ec.targets)
    if ec.n_types > 1:
        rex = rex.typed_face_selection(ec.type_labels)
    save_document_rex('batch', '$name', rex)
    print(f'  $name: {rex.nV}V {rex.nE}E')
else:
    print(f'  $name: no edges')
" 2>&1 | tee -a "$OUTPUT_DIR/$name.log"
        done

        kill $OCR_PID 2>/dev/null
        echo "Done - $(date)"
    """ % (input_dir, output_dir))


def generate_slurm_serve(
    model_id="deepseek-ai/DeepSeek-OCR-2",
    partition="gpu", gpus=1, time_limit="24:00:00", mem="48G",
    port=10000, account="", conda_env="rexgraph",
):
    """Persistent GPU server for interactive use.

    Starts the vLLM model server and the rexgraph agent server.
    Connect from your laptop with RexClient.
    """
    return _header("rexgraph-serve", partition, time_limit, mem, 8, account, gpus) + "\n" + _activate(conda_env) + textwrap.dedent("""\

        PORT=%d
        echo "GPU server on port $PORT, agent on port 8000"

        rexgraph-ocr serve --model %s --port $PORT &
        GPU_PID=$!
        sleep 30

        export CHAT_MODEL_URL="http://localhost:$PORT"
        export UNLIMITED_OCR_URL="http://localhost:$PORT"
        python agent/run.py --host 0.0.0.0 --port 8000 --no-browser

        kill $GPU_PID 2>/dev/null
    """ % (port, model_id))


def generate_slurm_build(
    partition="batch", time_limit="01:00:00", mem="16G",
    account="", conda_env="rexgraph",
):
    """Build rexgraph from source on a compute node."""
    return _header("rexgraph-build", partition, time_limit, mem, 8, account) + "\n" + _activate(conda_env) + textwrap.dedent("""\

        cd ${SLURM_SUBMIT_DIR:-$(pwd)}
        pip install -e . --no-build-isolation
        pip install -e ./agent[server,ocr]

        python -c "
from rexgraph.graph import RexGraph
rex = RexGraph.from_graph([0,1,0],[1,2,2])
print(f'Build OK: {rex.nV}V {rex.nE}E betti={rex.betti}')
"
        echo "Build complete - $(date)"
    """)


def generate_slurm_corpus(
    input_dir="/path/to/documents", workspace="batch",
    depth="standard", partition="batch", time_limit="08:00:00",
    mem="64G", account="", conda_env="rexgraph",
):
    """Build a corpus from a large document collection.

    Reads all files from input_dir, builds relational complexes,
    runs structural analysis, and persists to workspace.
    """
    return _header("rexgraph-corpus", partition, time_limit, mem, 8, account) + "\n" + _activate(conda_env) + textwrap.dedent("""\

        python << 'PYSCRIPT'
from agent.corpus import CorpusBuilder
from agent.server.persistence import save_document_rex, save_analysis_sql
import os, time

corpus = CorpusBuilder()
input_dir = "%s"
n = 0
for f in sorted(os.listdir(input_dir)):
    path = os.path.join(input_dir, f)
    if os.path.isfile(path):
        try:
            corpus.add_document(source=path, doc_id=f)
            n += 1
        except Exception as e:
            print("Skip %%s: %%s" %% (f, e))

print("Added %%d documents" %% n)
t0 = time.time()
corpus.build(depth="%s")
print("Built in %%.1fs" %% (time.time() - t0))

for doc in corpus.documents:
    if doc.rex:
        save_document_rex("%s", doc.doc_id, doc.rex)
        save_analysis_sql("%s", doc.doc_id, doc.rex, doc.analysis)
        print("  %%s: %%dV %%dE" %% (doc.doc_id, doc.rex.nV, doc.rex.nE))

print("Saved to workspace: %s")
PYSCRIPT
        echo "Done - $(date)"
    """ % (input_dir, depth, workspace, workspace, workspace))


def generate_slurm_training(
    workspace="batch", output="training_data.safetensors",
    target="channel", partition="batch", time_limit="04:00:00",
    mem="64G", account="", conda_env="rexgraph",
):
    """Export structural features as safetensors for model training.

    Reads a built corpus from workspace, extracts per-chunk features,
    and saves as safetensors for PyTorch/JAX/HuggingFace.
    """
    return _header("rexgraph-training", partition, time_limit, mem, 4, account) + "\n" + _activate(conda_env) + textwrap.dedent("""\

        python << 'PYSCRIPT'
from agent.training import TrainingExporter
from agent.server.persistence import list_document_bundles

docs = list_document_bundles("%s")
print("Found %%d documents" %% len(docs))

te = TrainingExporter.from_files([])  # will need corpus rebuild
te.export_features("%s")
print("Features exported: %%s" %% str(te.feature_matrix().shape))
te.export_training_pairs("%s".replace(".safetensors", "_pairs.safetensors"), target="%s")
print("%%d training examples exported" %% len(te.examples))
PYSCRIPT
        echo "Done - $(date)"
    """ % (workspace, output, output, target))


def generate_slurm_array(
    input_dir="/path/to/documents", files_per_task=10,
    partition="batch", time_limit="02:00:00", mem="32G",
    account="", conda_env="rexgraph",
):
    """SLURM array job: parallel document processing.

    Each array task handles a batch of files. Submit with:
        sbatch --array=0-N array.sbatch
    where N = ceil(total_files / files_per_task) - 1
    """
    return _header("rexgraph-array", partition, time_limit, mem, 4, account, array_spec="0-99") + "\n" + _activate(conda_env) + textwrap.dedent("""\

        python << 'PYSCRIPT'
import os, sys
from agent.adapters.text import TextAdapter
from agent.server.persistence import save_document_rex
from rexgraph.graph import RexGraph
import numpy as np

task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
files_per = %d
input_dir = "%s"

all_files = sorted(f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)))
start = task_id * files_per
batch = all_files[start:start + files_per]

if not batch:
    print("Task %%d: no files in range" %% task_id)
    sys.exit(0)

print("Task %%d: processing %%d files (%%d-%%d)" %% (task_id, len(batch), start, start + len(batch) - 1))

ta = TextAdapter()
for fname in batch:
    path = os.path.join(input_dir, fname)
    try:
        with open(path) as f:
            text = f.read()
        ec = ta.build(text, min_count=1, max_vocab=500)
        if ec.nE == 0:
            print("  %%s: no edges" %% fname)
            continue
        rex = RexGraph(sources=ec.sources, targets=ec.targets)
        if ec.n_types > 1:
            rex = rex.typed_face_selection(ec.type_labels)
        save_document_rex("array-batch", fname, rex)
        print("  %%s: %%dV %%dE kappa=%%.3f" %% (fname, rex.nV, rex.nE, rex.coherence.mean()))
    except Exception as e:
        print("  %%s: ERROR %%s" %% (fname, e))
PYSCRIPT
        echo "Task $SLURM_ARRAY_TASK_ID done - $(date)"
    """ % (files_per_task, input_dir))


# Template registry

TEMPLATES = {
    "ocr_batch": ("Batch OCR processing with GPU", generate_slurm_ocr_batch),
    "serve": ("Persistent GPU server for interactive use", generate_slurm_serve),
    "build": ("Build rexgraph from source on compute node", generate_slurm_build),
    "corpus": ("Build corpus from large document collection", generate_slurm_corpus),
    "training": ("Export training data as safetensors", generate_slurm_training),
    "array": ("Parallel document processing (array job)", generate_slurm_array),
}


def write_template(template_name, output=None, **kwargs):
    """Generate and optionally write a job template."""
    if template_name not in TEMPLATES:
        available = ", ".join(sorted(TEMPLATES.keys()))
        raise ValueError("Unknown template: %s. Available: %s" % (template_name, available))

    description, generator = TEMPLATES[template_name]
    script = generator(**kwargs)

    if output:
        output_path = Path(output)
        output_path.write_text(script)
        os.chmod(str(output_path), 0o755)
        print("Wrote %s template to %s" % (template_name, output_path))
        return str(output_path)

    return script


def list_templates():
    """List all available templates with descriptions."""
    for name, (desc, _) in sorted(TEMPLATES.items()):
        print("  %-12s %s" % (name, desc))

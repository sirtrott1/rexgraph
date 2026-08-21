# RexGraph - build, install, serve

# Auto-detects GPU (ROCm or CUDA). Just run `make install`.

# Install targets map to the agent profiles (see the README install table):
# make install           core + agent[server] (portable, runtime only)
# make install-standard  + full local deployment (agent[standard])
# make install-ml        + torch fine-tuning stack (agent[ml])
# make install-gpu       + torch built for the detected GPU
# make install-all       + everything on CPU (agent[all])

# Or use pip directly for precise control:
# pip install .                       rexgraph core
# pip install -e "./agent[server]"    server only
# pip install -e "./agent[standard]"  full local deployment
# pip install -e "./agent[all]"       everything on CPU

# See `make help` for all targets.

PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
PYTEST ?= $(PYTHON) -m pytest

# GPU auto-detection
HAS_ROCM := $(shell command -v rocminfo >/dev/null 2>&1 && echo 1 || echo 0)
HAS_CUDA := $(shell command -v nvidia-smi >/dev/null 2>&1 && echo 1 || echo 0)

ifeq ($(HAS_ROCM),1)
  GPU := rocm
  TORCH_URL := https://download.pytorch.org/whl/rocm6.2
else ifeq ($(HAS_CUDA),1)
  GPU := cuda
  TORCH_URL := https://download.pytorch.org/whl/cu124
else
  GPU := cpu
  TORCH_URL :=
endif

# Install

# Build backend deps (needed for editable meson-python installs, which use
# --no-build-isolation). Kept as a prerequisite so a fresh venv just works.
.PHONY: _build-deps
_build-deps:
	$(PIP) install "meson-python>=0.16" "meson>=1.3" "cython>=3.0" "numpy>=1.24"

.PHONY: install
install: _build-deps
	$(PIP) install -e . --no-build-isolation
	$(PIP) install -e "./agent[server]"
	@echo ""
	@echo "  Installed rexgraph + agent (runtime, GPU: $(GPU))"
	@echo "  Run: make serve"
	@echo ""
	@echo "  Optional profiles:"
	@echo "    make install-standard  full local deployment (schema, connectors, OCR, YAML)"
	@echo "    make install-ml        torch fine-tuning + HuggingFace stack"
	@echo "    make install-dev       everything needed to run the test suite"
	@echo "    make install-native    machine-optimized build (-march=native, not portable)"
	@echo "    make install-gpu       torch built for the detected GPU ($(GPU))"
	@echo "    make install-all       everything on CPU"

# Full local deployment: the standard agent profile (UI, schema, connectors,
# OCR, training export, YAML). No torch, no cloud-warehouse drivers.
.PHONY: install-standard
install-standard: _build-deps
	$(PIP) install -e . --no-build-isolation
	$(PIP) install -e "./agent[standard]"
	@echo "  Installed rexgraph + agent[standard] (full local deployment, no torch)"

# The torch ML stack: rexgraph.nn substrate + the agent ml profile (LoRA
# fine-tuning, HuggingFace analysis). Use install-gpu for a GPU torch build.
.PHONY: install-ml
install-ml: _build-deps
	$(PIP) install -e ".[nn]" --no-build-isolation
	$(PIP) install -e "./agent[ml]"
	@echo "  Installed the torch ML stack (rexgraph.nn + agent[ml])."
	@echo "  For a GPU torch build: make install-gpu"

# Full development/test environment: every optional runtime dep plus the test
# tooling for BOTH packages. This is the target that pulls in pytest, pandas,
# h5py, zarr, pyarrow, sqlalchemy, fastapi, httpx, etc.
.PHONY: install-dev
install-dev: _build-deps
	$(PIP) install -e ".[all]" --no-build-isolation
	$(PIP) install -e "./agent[server,dev]"
	@echo ""
	@echo "  Dev environment ready. Run: make test"

# Machine-optimized build: -march=native + -ffast-math (faster, NOT portable).
.PHONY: install-native
install-native: _build-deps
	$(PIP) install -e . --no-build-isolation -Csetup-args=-Dnative=true
	$(PIP) install -e "./agent[server]"
	@echo "  Installed rexgraph (native/-march build) + agent"

# Compile/refresh the Cython extensions in place (portable flags).
.PHONY: build
build: _build-deps
	$(PIP) install -e . --no-build-isolation
	@echo "  Extensions built (portable). Use 'make install-native' for -march=native."

.PHONY: install-gpu
install-gpu: install
ifneq ($(GPU),cpu)
	$(PIP) install torch torchvision torchaudio --index-url $(TORCH_URL)
	@echo "  PyTorch installed for $(GPU)"
else
	@echo "  No GPU detected. Skipping torch."
	@echo "  OCR uses tesseract (CPU). Install GPU packages manually if needed."
endif

.PHONY: install-train
install-train:
	$(PIP) install -e ./agent[training]
	@echo "  safetensors installed for training export"

.PHONY: install-all
install-all: install
	$(PIP) install -e ".[all]" --no-build-isolation
	$(PIP) install -e "./agent[all]"
ifneq ($(GPU),cpu)
	$(PIP) install torch torchvision torchaudio --index-url $(TORCH_URL)
endif
	@echo "  All CPU packages installed (GPU: $(GPU))"

# OCR Server (Unlimited-OCR / DeepSeek-OCR-2)

.PHONY: install-ocr-server
install-ocr-server:
	@echo ""
	@echo "  Installing OCR server for $(GPU)..."
	@echo ""
ifeq ($(GPU),cpu)
	@echo "ERROR: No GPU detected. vLLM requires ROCm or CUDA."
	@echo "  If you have a GPU, make sure rocminfo or nvidia-smi is in PATH."
	@echo "  For CPU-only OCR, install tesseract: sudo apt install tesseract-ocr"
	@exit 1
endif
	$(PIP) install torch torchvision torchaudio --index-url $(TORCH_URL)
	$(PIP) install vllm huggingface-hub
	@echo ""
	@echo "  Downloading DeepSeek-OCR-2 (~8GB)..."
	$(PYTHON) -c "from huggingface_hub import snapshot_download; from pathlib import Path; snapshot_download('deepseek-ai/DeepSeek-OCR-2', local_dir=str(Path.home() / '.cache/rexgraph/models/deepseek-ai--DeepSeek-OCR-2'), local_dir_use_symlinks=False)"
	@echo ""
	@echo "  Done. OCR server ready."
	@echo "  Start:  make ocr-serve"
	@echo "  Then:   make serve  (in another terminal)"

.PHONY: install-got-ocr
install-got-ocr:
	@echo "  Installing GOT-OCR2.0 dependencies..."
ifneq ($(GPU),cpu)
	$(PIP) install torch torchvision torchaudio --index-url $(TORCH_URL)
endif
	$(PIP) install transformers accelerate tiktoken verovio
	@echo "  Downloading GOT-OCR2.0 model (~4GB)..."
	$(PYTHON) -c "from transformers import AutoProcessor, GotOcr2ForConditionalGeneration; print('  Downloading processor...'); AutoProcessor.from_pretrained('stepfun-ai/GOT-OCR-2.0-hf', trust_remote_code=True); print('  Downloading model weights...'); GotOcr2ForConditionalGeneration.from_pretrained('stepfun-ai/GOT-OCR-2.0-hf', trust_remote_code=True)"
	@echo ""
	@echo "  GOT-OCR2.0 ready. Model cached in ~/.cache/huggingface/hub/"
	@echo "  No server needed - runs directly on GPU during pipeline."

.PHONY: ocr-serve
ocr-serve:
	@echo ""
	@echo "  Starting DeepSeek-OCR-2 on port 10000..."
	@echo "  (This loads ~8GB into VRAM. Leave running while using the web UI.)"
	@echo ""
	cd agent && $(PYTHON) -c "from agent.cli.serve import serve; serve(foreground=True)"

.PHONY: ocr-stop
ocr-stop:
	cd agent && $(PYTHON) -c "from agent.cli.serve import stop; stop()"

.PHONY: ocr-status
ocr-status:
	@cd agent && $(PYTHON) -c "\
import json, shutil, os; \
from pathlib import Path; \
s = {}; \
s['gpu'] = 'rocm' if shutil.which('rocminfo') else ('cuda' if shutil.which('nvidia-smi') else 'cpu'); \
try: import vllm; s['vllm'] = True \
except: s['vllm'] = False; \
try: import torch; s['torch'] = torch.__version__; s['torch_cuda'] = torch.cuda.is_available() \
except: s['torch'] = False; \
m = Path.home() / '.cache/rexgraph/models/deepseek-ai--DeepSeek-OCR-2'; \
s['deepseek_ocr2'] = m.exists() and any(m.iterdir()) if m.exists() else False; \
s['tesseract'] = shutil.which('tesseract') is not None; \
try: import transformers; s['got_ocr'] = True \
except: s['got_ocr'] = False; \
print(json.dumps(s, indent=2))"

# Serve

.PHONY: serve
serve:
	@echo ""
	@echo "  http://127.0.0.1:8000"
	@echo ""
	cd agent && $(PYTHON) run.py --no-browser $(if $(SSL_KEY),--ssl-key $(SSL_KEY)) $(if $(SSL_CERT),--ssl-cert $(SSL_CERT))

.PHONY: serve-https
serve-https:
	@test -f certs/server.key || $(MAKE) gen-cert
	@echo ""
	@echo "  https://127.0.0.1:8000"
	@echo ""
	cd agent && $(PYTHON) run.py --no-browser --ssl-key ../certs/server.key --ssl-cert ../certs/server.crt

.PHONY: gen-cert
gen-cert:
	cd agent && $(PYTHON) -m agent.cli.auth gen-cert --out ../certs

.PHONY: serve-dev
serve-dev:
	cd agent && $(PYTHON) run.py --reload --no-browser

.PHONY: gpu-serve
gpu-serve:
	rexgraph-ocr serve

# Test


.PHONY: test-agent
test-agent:
# agent/pyproject.toml sets testpaths = ["tests"], and five modules import fixtures
# from a sibling as `tests.test_x`, so the suite resolves from agent/ and not from
# the repo root. Same reason test-agent-cli below changes directory.
	cd agent && $(PYTEST) tests/ -q

.PHONY: test-agent-cli
test-agent-cli:
	cd agent && $(PYTHON) -m agent.cli.test_all --verbose

.PHONY: test-agent-ocr
test-agent-ocr:
	cd agent && $(PYTHON) -m agent.cli.test_all --only ocr

.PHONY: test-agent-pipeline
test-agent-pipeline:
	cd agent && $(PYTHON) -m agent.cli.test_all --only pipeline

# CLI Pipeline (headless, no web UI)

.PHONY: run
run:
	@echo "Usage: make run FILES='paper.pdf' [QUERY='what is...'] [DEPTH=standard] [BACKEND=tesseract]"
	cd agent && $(PYTHON) -m agent.cli.run_pipeline $(FILES) \
		$(if $(QUERY),--query "$(QUERY)") \
		$(if $(DEPTH),--depth $(DEPTH)) \
		$(if $(BACKEND),--backend $(BACKEND))

# HPC (Cheaha / SLURM)


.PHONY: hpc-job-ocr
hpc-job-ocr:
	@echo "Generating SLURM job: ocr_batch.sbatch"
	cd agent && $(PYTHON) -c "from agent.cli.hpc import write_template; write_template('ocr_batch', partition='$(or $(PARTITION),pascalnodes)')"

.PHONY: hpc-job-serve
hpc-job-serve:
	@echo "Generating SLURM job: serve.sbatch"
	cd agent && $(PYTHON) -c "from agent.cli.hpc import write_template; write_template('serve', partition='$(or $(PARTITION),pascalnodes)')"

.PHONY: hpc-job-pipeline
hpc-job-pipeline:
	@echo "Generating SLURM job: pipeline.sbatch"
	cd agent && $(PYTHON) -c "from agent.cli.hpc import write_template; write_template('pipeline_batch', partition='$(or $(PARTITION),pascalnodes)')"

.PHONY: test
test: test-rexgraph test-agent

.PHONY: test-rexgraph
test-rexgraph:
	$(PYTEST) rexgraph/tests/ -q

.PHONY: test-quick
test-quick:
	@$(PYTHON) -c "\
from rexgraph.graph import RexGraph; \
rex = RexGraph.from_graph([0,1,0],[1,2,2]); \
print('rexgraph: %dV %dE betti=%s kappa=%.3f' % (rex.nV, rex.nE, rex.betti, rex.coherence.mean())); \
from agent.builder import AgentBuilder; \
print('agent:    %d steps, %d templates' % (len(AgentBuilder.available_steps()), 5)); \
print('GPU:      $(GPU)'); \
print('OK'); \
"

# GPU

.PHONY: gpu-status
gpu-status:
	@echo "Detected: $(GPU)"
ifeq ($(HAS_ROCM),1)
	@rocminfo 2>/dev/null | grep -E 'Name:|Marketing' | head -4
endif
ifeq ($(HAS_CUDA),1)
	@nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
endif
ifeq ($(GPU),cpu)
	@echo "No GPU found. OCR uses tesseract. Models need a GPU."
endif

# HPC

.PHONY: hpc-setup
hpc-setup:
	bash agent/setup_hpc.sh --gpu $(GPU)

.PHONY: hpc-templates
hpc-templates:
	@mkdir -p slurm/
	@$(PYTHON) -c "\
from agent.cli.hpc import write_template; \
[write_template(t, output='slurm/%s.sbatch' % t) for t in \
 ['build','serve','ocr_batch','corpus','training','array']]"
	@echo "  6 templates in slurm/"

# Clean

.PHONY: clean
clean:
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete 2>/dev/null || true
	find . -name '*.egg-info' -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name '.pytest_cache' -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ 2>/dev/null || true
	rm -f 'file::memory:' 2>/dev/null || true
	@echo "Clean"

# Help

.PHONY: help
help:
	@echo ""
	@echo "  rexgraph"
	@echo ""
	@echo "  Install:"
	@echo "    make install          rexgraph + agent server (portable, runtime only)"
	@echo "    make install-standard full local deployment (schema, connectors, OCR, YAML)"
	@echo "    make install-ml       torch fine-tuning + HuggingFace stack"
	@echo "    make install-dev      + all test deps for both packages (pytest, pandas, fastapi...)"
	@echo "    make install-native   machine-optimized build (-march=native, NOT portable)"
	@echo "    make install-gpu      + PyTorch for the detected GPU"
	@echo "    make install-all      everything on CPU"
	@echo ""
	@echo "  Run:"
	@echo "    make serve          start server on localhost:8000"
	@echo "    make serve-dev      start with auto-reload"
	@echo "    make serve-https    start with HTTPS"
	@echo "    make gpu-serve      start GPU model server (vLLM)"
	@echo ""
	@echo "  Test:"
	@echo "    make test           full pytest suite (rexgraph/tests + agent/tests)"
	@echo "    make test-rexgraph  rexgraph kernels only"
	@echo "    make test-agent     agent unit tests (pytest)"
	@echo "    make test-agent-cli agent integration runner (agent.cli.test_all)"
	@echo "    make test-quick     fast smoke test"
	@echo ""
	@echo "  Info:"
	@echo "    make gpu-status     show GPU detection"
	@echo "    make help           this message"
	@echo ""
	@echo "  HPC:"
	@echo "    make hpc-setup      full SLURM environment setup"
	@echo "    make hpc-templates  generate .sbatch scripts"
	@echo ""
	@echo "  Or use pip directly (see the README install table):"
	@echo "    pip install .                             rexgraph core"
	@echo "    pip install -e './agent[server]'          server only"
	@echo "    pip install -e './agent[standard]'        full local deployment"
	@echo "    pip install -e './agent[all]'             everything on CPU"
	@echo ""
# Versioning
#
# Five files declare the version and each serves something that cannot read the others,
# so they are set together or they drift. They already did: meson.build sat at 1.0.1
# through two 1.0.6 releases. test_version_consistency.py fails if they disagree.
.PHONY: version
version:
ifndef VERSION
	@python scripts/set_version.py --show
	@echo ""
	@echo "to change it:  make version VERSION=1.0.7"
else
	@python scripts/set_version.py $(VERSION)
endif

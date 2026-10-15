# RexGraph installation and development commands.
.DEFAULT_GOAL := help

VENV_DIR ?= $(CURDIR)/.venv
PYTHON ?= $(VENV_DIR)/bin/python
BASE_PYTHON ?= python3
INSTALLER ?= auto
BUILD_JOBS ?= 2
GPU_BACKEND ?= auto
NATIVE_HIP ?= auto
TEST_ARGS ?=
PIP = "$(PYTHON)" -I -m pip
INSTALL = VENV_DIR="$(VENV_DIR)" INSTALLER="$(INSTALLER)" PYTHON="$(BASE_PYTHON)" BUILD_JOBS="$(BUILD_JOBS)" sh install.sh

# GPU information. Select the PyTorch wheel index explicitly for GPU installs.
HAS_ROCM := $(shell command -v rocminfo >/dev/null 2>&1 && echo 1 || echo 0)
HAS_CUDA := $(shell command -v nvidia-smi >/dev/null 2>&1 && echo 1 || echo 0)
GPU := $(if $(filter 1,$(HAS_ROCM)),rocm,$(if $(filter 1,$(HAS_CUDA)),cuda,cpu))
TORCH_URL = $(TORCH_INDEX_URL)

.PHONY: install install-core install-server install-standard install-ml install-dev install-dev-full install-native install-gpu install-train install-all build verify python system
install install-standard:
	$(INSTALL) --profile standard

install-core:
	$(INSTALL) --profile core

install-server:
	$(INSTALL) --profile server

install-ml:
	$(INSTALL) --profile ml

install-dev:
	$(INSTALL) --profile dev

install-dev-full:
	$(INSTALL) --profile dev-full

install-native:
	$(INSTALL) --profile standard --native

install-gpu:
	GPU_BACKEND="$(GPU_BACKEND)" NATIVE_HIP="$(NATIVE_HIP)" REQUIRE_GPU=1 TORCH_INDEX_URL="$(TORCH_INDEX_URL)" $(INSTALL) --profile ml --gpu "$(GPU_BACKEND)"

install-train:
	$(INSTALL) --profile standard --extras training

install-all:
	$(INSTALL) --profile all

build:
	$(INSTALL) --profile core --editable

verify:
	"$(PYTHON)" -I scripts/verify_install.py --profile $(or $(PROFILE),auto)
	$(PIP) check

.PHONY: doctor
doctor:
	@"$(PYTHON)" -I scripts/check_setup.py $(ARGS)

python:
	"$(PYTHON)" -I

system:
	"$(PYTHON)" -I -c "from system.server.launch import main; main()" $(ARGS)

# OCR server and model downloads

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
	@test -n "$(TORCH_URL)" || { echo "Set TORCH_INDEX_URL for this GPU."; exit 1; }
ifeq ($(GPU),rocm)
	@test -n "$(VLLM_WHEEL_URL)" || { echo "Set VLLM_WHEEL_URL to a compatible ROCm wheel from https://docs.vllm.ai/en/latest/getting_started/installation/gpu/"; exit 1; }
endif
	$(PIP) install --upgrade --force-reinstall torch torchvision torchaudio --index-url "$(TORCH_URL)"
ifeq ($(GPU),rocm)
	$(PIP) install "$(VLLM_WHEEL_URL)" huggingface-hub
else
	$(PIP) install vllm huggingface-hub
endif
	@echo ""
	@echo "  Downloading DeepSeek-OCR-2 (~8GB)..."
	"$(PYTHON)" -I -c "from huggingface_hub import snapshot_download; from agent.data_paths import models_root; snapshot_download('deepseek-ai/DeepSeek-OCR-2', local_dir=str(models_root() / 'deepseek-ai--DeepSeek-OCR-2'))"
	@echo ""
	@echo "  Done. OCR server ready."
	@echo "  Start:  make ocr-serve"
	@echo "  Then:   make serve  (in another terminal)"

.PHONY: install-got-ocr
install-got-ocr:
	@echo "  Installing GOT-OCR2.0 dependencies..."
ifneq ($(GPU),cpu)
	@test -n "$(TORCH_URL)" || { echo "Set TORCH_INDEX_URL for this GPU."; exit 1; }
	$(PIP) install --upgrade --force-reinstall torch torchvision torchaudio --index-url "$(TORCH_URL)"
endif
	$(PIP) install transformers accelerate tiktoken verovio
	@echo "  Downloading GOT-OCR2.0 model (~4GB)..."
	"$(PYTHON)" -I -c "from transformers import AutoProcessor, GotOcr2ForConditionalGeneration; print('  Downloading processor...'); AutoProcessor.from_pretrained('stepfun-ai/GOT-OCR-2.0-hf', trust_remote_code=True); print('  Downloading model weights...'); GotOcr2ForConditionalGeneration.from_pretrained('stepfun-ai/GOT-OCR-2.0-hf', trust_remote_code=True)"
	@echo ""
	@echo "  GOT-OCR2.0 ready in the configured Hugging Face cache."
	@echo "  Runs directly during the pipeline."

.PHONY: ocr-serve
ocr-serve:
	@echo ""
	@echo "  Starting DeepSeek-OCR-2 on port 10000..."
	@echo "  (This loads ~8GB into VRAM. Leave running while using the web UI.)"
	@echo ""
	"$(PYTHON)" -I -c "from agent.cli.serve import serve; serve(foreground=True)"

.PHONY: ocr-stop
ocr-stop:
	"$(PYTHON)" -I -c "from agent.cli.serve import stop; stop()"

.PHONY: ocr-status
ocr-status:
	@"$(PYTHON)" -I -m agent.cli.ocr status

# Serve

.PHONY: serve
serve:
	@echo ""
	@echo "  http://127.0.0.1:8000"
	@echo ""
	"$(PYTHON)" -I agent/run.py --no-browser $(ARGS) $(if $(SSL_KEY),--ssl-key "$(SSL_KEY)") $(if $(SSL_CERT),--ssl-cert "$(SSL_CERT)")

.PHONY: serve-https
serve-https:
	@test -f certs/server.key || $(MAKE) gen-cert
	@echo ""
	@echo "  https://127.0.0.1:8000"
	@echo ""
	"$(PYTHON)" -I agent/run.py --no-browser --ssl-key certs/server.key --ssl-cert certs/server.crt $(ARGS)

.PHONY: gen-cert
gen-cert:
	"$(PYTHON)" -I -m agent.cli.auth gen-cert --out certs

.PHONY: serve-dev
serve-dev:
	"$(PYTHON)" -I agent/run.py --reload --no-browser $(ARGS)

.PHONY: gpu-serve
gpu-serve:
	"$(VENV_DIR)/bin/rexgraph-ocr" serve

# Test


.PHONY: test-agent
test-agent:
# Run against the selected wheel or declared editable installation.
	"$(PYTHON)" -I scripts/test_installed.py --allow-editable --package rexgraph --package rcdb --package agent -- agent/tests -q $(TEST_ARGS)

.PHONY: test-agent-cli
test-agent-cli:
	"$(PYTHON)" -I -m agent.cli.test_all --verbose

.PHONY: test-agent-ocr
test-agent-ocr:
	"$(PYTHON)" -I -m agent.cli.test_all --only ocr

.PHONY: test-agent-pipeline
test-agent-pipeline:
	"$(PYTHON)" -I -m agent.cli.test_all --only pipeline

# CLI Pipeline (headless, no web UI)

.PHONY: run
run:
	@echo "Usage: make run FILES='document.pdf' [QUERY='what is...'] [DEPTH=standard] [BACKEND=tesseract]"
	"$(PYTHON)" -I -m agent.cli.run_pipeline $(FILES) \
		$(if $(QUERY),--query "$(QUERY)") \
		$(if $(DEPTH),--depth $(DEPTH)) \
		$(if $(BACKEND),--backend $(BACKEND))

# SLURM job templates


.PHONY: hpc-job-ocr
hpc-job-ocr:
	@echo "Generating SLURM job: ocr_batch.sbatch"
	"$(PYTHON)" -I -c "import sys; from agent.cli.hpc import write_template; write_template('ocr_batch', partition='$(or $(PARTITION),batch)', conda_env=sys.prefix)"

.PHONY: hpc-job-serve
hpc-job-serve:
	@echo "Generating SLURM job: serve.sbatch"
	"$(PYTHON)" -I -c "import sys; from agent.cli.hpc import write_template; write_template('serve', partition='$(or $(PARTITION),batch)', conda_env=sys.prefix)"

.PHONY: hpc-job-pipeline
hpc-job-pipeline:
	@echo "Generating SLURM job: pipeline.sbatch"
	"$(PYTHON)" -I -c "import sys; from agent.cli.hpc import write_template; write_template('corpus', output='pipeline.sbatch', partition='$(or $(PARTITION),batch)', conda_env=sys.prefix)"

.PHONY: test
test: test-rexgraph test-agent test-rcdb test-rcql test-system

.PHONY: test-rexgraph
test-rexgraph:
	"$(PYTHON)" -I scripts/test_installed.py --allow-editable --package rexgraph -- rexgraph/tests -q $(TEST_ARGS)

.PHONY: test-quick test-rcdb test-rcql test-system
test-quick: verify

test-rcdb:
	"$(PYTHON)" -I scripts/test_installed.py --allow-editable --package rexgraph --package rcdb -- rcdb/tests -q $(TEST_ARGS)

test-rcql:
	"$(PYTHON)" -I scripts/test_installed.py --allow-editable --package rexgraph --package rcql -- rcql/tests -q $(TEST_ARGS)

test-system:
	"$(PYTHON)" -I scripts/test_installed.py --allow-editable --package rexgraph --package rcql --package system -- system/tests -q $(TEST_ARGS)

# GPU

.PHONY: gpu-status verify-gpu
gpu-status:
	"$(PYTHON)" -I -m rexgraph.gpu_preflight --diagnose --json

verify-gpu:
	"$(PYTHON)" -I scripts/setup_gpu.py --verify --require-gpu --backend "$(GPU_BACKEND)" $(ARGS)

# HPC

.PHONY: hpc-setup
hpc-setup:
	VENV_DIR="$(VENV_DIR)" sh agent/setup_hpc.sh

.PHONY: hpc-templates
hpc-templates:
	@mkdir -p slurm/
	@"$(PYTHON)" -I -c "\
import sys; from agent.cli.hpc import write_template; \
[write_template(t, conda_env=sys.prefix, output='slurm/%s.sbatch' % t) for t in \
 ['build','serve','ocr_batch','corpus','training','array']]"
	@echo "  6 templates in slurm/"

# Clean

.PHONY: clean
clean:
	"$(BASE_PYTHON)" -I scripts/clean_cache.py --environment "$(VENV_DIR)"
	@echo "Clean"

# Help

.PHONY: help
help:
	@echo "RexGraph"
	@echo "  make install           Install all five packages and the standard Agent profile"
	@echo "  make install-core      Install Core only"
	@echo "  make install-server    Install all five packages with the Agent server profile"
	@echo "  make install-dev       Install editable packages and test dependencies"
	@echo "  make install-dev-full  Add ML, TensorFlow, molecular, analytics and integration tests"
	@echo "  make install-ml        Add the CPU PyTorch and Hugging Face stack"
	@echo "  make install-gpu       Detect the GPU and install ML with its compute runtime"
	@echo "  make verify-gpu        Check GPU calculations and required native kernels"
	@echo "  make install-all       Install all declared Agent profile dependencies"
	@echo "  make install-native    Compile for this CPU with relaxed floating point rules"
	@echo "  make serve             Start Agent at http://127.0.0.1:8000"
	@echo "  make system            Start System at http://127.0.0.1:8010"
	@echo "  make python            Open Python with the installed packages"
	@echo "  make verify            Check compiled operators, UI assets and dependencies"
	@echo "  make doctor            List optional dependencies, tools and data paths"
	@echo "  make test              Test Core, Agent, RCDB, RCQL and System"
	@echo "  make hpc-setup         Install and generate SLURM templates"
	@echo "Settings: VENV_DIR, INSTALLER, BASE_PYTHON, BUILD_JOBS, TORCH_INDEX_URL"
	@echo "Installer options: sh install.sh --help"

# Versioning
#
# Update package and build versions together; the consistency test checks agreement.
.PHONY: version
version:
ifndef VERSION
	@"$(PYTHON)" scripts/set_version.py --show
	@echo ""
	@echo "Set a version: make version VERSION=x.y.z"
else
	@"$(PYTHON)" scripts/set_version.py $(VERSION)
endif

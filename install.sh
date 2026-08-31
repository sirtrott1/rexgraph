#!/bin/sh

# install.sh - OS-agnostic installer for the RexGraph monorepo.

# Run from the REPO ROOT (the dir with meson.build + agent/):
# sh install.sh

# Autodetects, like enterprise software should:
# * OS + CPU arch (Linux/macOS, x86_64/arm64/aarch64/ppc64le)
# * package manager (apt / dnf / yum / pacman / zypper / apk / brew)
# * installer FRONTEND:
#     - conda: reuses mamba/micromamba/conda if present, else bootstraps
#       micromamba (no miniforge). Hermetic toolchain + OpenBLAS from
#       conda-forge. This is the default and the recommended path.
#     - pip/venv: fallback when no conda frontend is wanted/available - creates
#       a plain virtualenv (uv if present, else python -m venv) and builds with
#       pip's isolated build (build deps come from pyproject). Uses the SYSTEM
#       compiler + BLAS, so ensure a C/C++ toolchain is installed.
#   Choose with INSTALLER=auto|conda|pip (default auto). NO_CONDA=1 forces pip
#   when no conda frontend exists instead of bootstrapping micromamba.
# * GPU (nvidia -> CUDA, amd -> ROCm/Vulkan incl. integrated/APU, else CPU) - reported
# * shell (bash/zsh/fish) for the activation hook (conda path)

# The heavy toolchain (C/C++ compilers, OpenBLAS) comes from the conda-forge
# env, so the build is hermetic and does NOT depend on system BLAS packages
# whose names differ per distro. System deps are just git + curl.

# Idempotent and safe to re-run. Override any of these via env:
# ENV_NAME=rexgraph PY_SPEC= NATIVE=0 EXTRAS=server,schema,training
# EXTRA_CONNECT=0 EXTRA_WAREHOUSE=0 MAMBA_ROOT_PREFIX=$HOME/micromamba
# INSTALLER=auto NO_CONDA=0 VENV_DIR=<repo>/.venv
# EXTRAS is any agent profile or comma list (server / standard / ml /
# integrations / all, or granular extras). See the README install table.

set -eu

ENV_NAME="${ENV_NAME:-rexgraph}"
PY_SPEC="${PY_SPEC:-}"                 # e.g. "python=3.12" to pin; empty = env file default
NATIVE="${NATIVE:-0}"                  # 1 = -march=native core build
EXTRAS="${EXTRAS:-server,schema,training}"
EXTRA_CONNECT="${EXTRA_CONNECT:-0}"
EXTRA_WAREHOUSE="${EXTRA_WAREHOUSE:-0}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
export MAMBA_ROOT_PREFIX
INSTALLER="${INSTALLER:-auto}"         # auto | conda | pip
NO_CONDA="${NO_CONDA:-0}"              # 1 = prefer pip/venv over bootstrapping micromamba
VENV_DIR="${VENV_DIR:-$PWD/.venv}"     # used only on the pip/venv path

say()  { printf '\n\033[36m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
die()  { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# 0. sanity
[ -f meson.build ] && [ -d agent ] || die "Run from the repo root (needs meson.build + agent/)."
[ -f environment.yml ] || die "environment.yml not found in the repo root."

# 1. detect OS / arch
OS="$(uname -s)"; ARCH="$(uname -m)"
case "$OS" in
    Linux)  PLAT_OS=linux ;;
    Darwin) PLAT_OS=osx ;;
    *) die "Unsupported OS: $OS (Linux/macOS only)." ;;
esac
case "$ARCH" in
    x86_64|amd64)   MM_ARCH="${PLAT_OS}-64" ;;
    aarch64|arm64)  [ "$PLAT_OS" = osx ] && MM_ARCH=osx-arm64 || MM_ARCH=linux-aarch64 ;;
    ppc64le)        MM_ARCH=linux-ppc64le ;;
    *) die "Unsupported CPU arch: $ARCH." ;;
esac
say "Detected $OS / $ARCH  (micromamba platform: $MM_ARCH)"

# 2. detect package manager + ensure git/curl
PM=""; for c in apt-get dnf yum pacman zypper apk brew; do have "$c" && { PM="$c"; break; }; done
SUDO=""; [ "$(id -u)" -ne 0 ] && have sudo && SUDO="sudo"

pm_install() {  # pm_install <pkgs...>
    case "$PM" in
        apt-get) $SUDO apt-get update && $SUDO apt-get install -y "$@" ;;
        dnf)     $SUDO dnf install -y "$@" ;;
        yum)     $SUDO yum install -y "$@" ;;
        pacman)  $SUDO pacman -S --needed --noconfirm "$@" ;;
        zypper)  $SUDO zypper install -y "$@" ;;
        apk)     $SUDO apk add "$@" ;;
        brew)    brew install "$@" ;;
        *) info "No known package manager - ensure git + curl are installed yourself." ;;
    esac
}

say "Ensuring git + curl (package manager: ${PM:-none})"
NEED=""
have git  || NEED="$NEED git"
have curl || NEED="$NEED curl"
if [ -n "$NEED" ]; then
    info "installing:$NEED"
 # shellcheck disable=SC2086
    pm_install $NEED || info "package install failed - continuing if git/curl already usable"
else
    info "git + curl present."
fi

# 3. choose an installer frontend (conda by default; pip/venv as a fallback)
say "Selecting an installer frontend (INSTALLER=$INSTALLER)"
CONDA=""
if have micromamba; then CONDA="micromamba"
elif [ -x "$HOME/.local/bin/micromamba" ]; then CONDA="$HOME/.local/bin/micromamba"
elif have mamba; then CONDA="mamba"
elif have conda; then CONDA="conda"
fi

FRONTEND=""
case "$INSTALLER" in
    pip)  FRONTEND="pip" ;;
    conda) FRONTEND="conda" ;;
    auto|*)
        if [ -n "$CONDA" ]; then FRONTEND="conda"
        elif [ "$NO_CONDA" = 1 ]; then FRONTEND="pip"
        else FRONTEND="conda"; fi   # default: bootstrap micromamba (existing behavior)
        ;;
esac

if [ "$FRONTEND" = conda ]; then
    # -------- conda path (unchanged behavior) --------
    if [ -z "$CONDA" ]; then
        info "No conda frontend found - bootstrapping micromamba (standalone, not miniforge)…"
        mkdir -p "$HOME/.local/bin"
        curl -Ls "https://micro.mamba.pm/api/micromamba/${MM_ARCH}/latest" \
            | tar -xj -C "$HOME/.local" bin/micromamba || die "micromamba download failed."
        CONDA="$HOME/.local/bin/micromamba"
    fi
    info "using: $CONDA ($($CONDA --version 2>/dev/null | head -1))"

    # helper: create/update env and run inside it, tool-agnostically
    CREATE_YML="environment.yml"
    case "$CONDA" in
        *micromamba)
            ENV_EXISTS() { "$CONDA" env list 2>/dev/null | grep -qw "$ENV_NAME"; }
            ENV_CREATE() { "$CONDA" create -y -n "$ENV_NAME" -f "$CREATE_YML"; }
            ENV_UPDATE() { "$CONDA" env update -n "$ENV_NAME" -f "$CREATE_YML"; }
            INENV()      { "$CONDA" run -n "$ENV_NAME" "$@"; }
            ;;
        *)
            ENV_EXISTS() { "$CONDA" env list 2>/dev/null | grep -qw "$ENV_NAME"; }
            ENV_CREATE() { "$CONDA" env create -n "$ENV_NAME" -f "$CREATE_YML"; }
            ENV_UPDATE() { "$CONDA" env update -n "$ENV_NAME" -f "$CREATE_YML"; }
            INENV()      { "$CONDA" run -n "$ENV_NAME" "$@"; }
            ;;
    esac

    # 4. create the environment
    say "Creating/updating the '$ENV_NAME' environment (conda-forge)"
    if ENV_EXISTS; then info "exists - updating…"; ENV_UPDATE; else ENV_CREATE; fi
    [ -n "$PY_SPEC" ] && { info "pinning $PY_SPEC as requested"; INENV "${CONDA##*/}" install -y "$PY_SPEC" 2>/dev/null || true; }
    info "python in env: $(INENV python --version 2>&1)"
    BUILD_ISOLATION="--no-build-isolation"   # build deps come from the conda env
else
    # -------- pip / venv path (fallback: conda/mamba/micromamba absent or unwanted) --------
    CONDA=""                                   # signal to the rest of the script: no conda
    info "Using a plain virtualenv - no conda. Build uses the SYSTEM compiler + BLAS."
    # pick a base python and a venv creator (uv if available, else python -m venv)
    BASEPY=""; for c in python3 python; do have "$c" && { BASEPY="$c"; break; }; done
    [ -n "$BASEPY" ] || die "No python3/python on PATH for the pip/venv path."
    have cc || have gcc || have clang || \
        info "WARNING: no C/C++ compiler (cc/gcc/clang) found - the core build will fail; \
install a toolchain (build-essential / base-devel / xcode-select --install)."
    say "Creating the virtualenv at $VENV_DIR"
    if [ ! -x "$VENV_DIR/bin/python" ]; then
        if have uv; then info "using uv venv"; uv venv "$VENV_DIR" || die "uv venv failed."
        else info "using $BASEPY -m venv"; "$BASEPY" -m venv "$VENV_DIR" || die "venv creation failed."; fi
    else
        info "reusing existing venv."
    fi
    VENV_PY="$VENV_DIR/bin/python"
    [ -x "$VENV_PY" ] || VENV_PY="$VENV_DIR/Scripts/python.exe"   # (msys, best-effort)
    # INENV maps the conda call convention (`INENV python ...`, `INENV pip ...`) onto the venv.
    INENV() {
        _c="$1"; shift
        case "$_c" in
            python|python3) "$VENV_PY" "$@" ;;
            pip)            "$VENV_PY" -m pip "$@" ;;
            *)              "$VENV_PY" -m "$_c" "$@" ;;
        esac
    }
    info "python in venv: $(INENV python --version 2>&1)"
    INENV pip install --upgrade pip wheel >/dev/null 2>&1 || info "pip upgrade skipped"
    BUILD_ISOLATION=""                         # let pip fetch build deps from pyproject (isolated)
fi

# 5. GPU detection (informational)
say "Detecting GPU"
if have nvidia-smi; then info "NVIDIA GPU -> CUDA. For OCR/HF extras: torch CUDA wheels."
elif have rocminfo; then info "AMD GPU -> ROCm. For OCR/HF extras: torch ROCm wheels."
else info "No GPU detected -> CPU. (OCR falls back to tesseract.)"; fi

# 6. shell activation hook
case "$CONDA" in *micromamba)
    USER_SHELL="$(basename "${SHELL:-sh}")"
    say "Wiring $USER_SHELL activation hook"
    "$CONDA" shell init -s "$USER_SHELL" -r "$MAMBA_ROOT_PREFIX" >/dev/null 2>&1 \
        && info "hook added (new shells): '$CONDA activate $ENV_NAME'" \
        || info "could not auto-init $USER_SHELL - activate with: eval \"\$($CONDA shell hook -s $USER_SHELL)\"; $CONDA activate $ENV_NAME"
    ;;
esac

# 7. build the core, install the siblings, then the agent
say "Building the rexgraph core from source (compiles Cython - ~3-4 min)"
NATIVE_ARG=""; [ "$NATIVE" = 1 ] && { NATIVE_ARG="-Csetup-args=-Dnative=true"; info "(native/-march=native enabled)"; }
# The buildable core package is the REPO ROOT: the root meson.build holds the
# project() call and descends into rexgraph/ via subdir('rexgraph'). The nested
# rexgraph/meson.build is only a subdir include (no project()), so
# `pip install ./rexgraph` fails with "Not the project root". Build from '.'.
# [io] and [security] are declared in the root pyproject, which is the only one the
# core has: rexgraph/meson.build is a subdir include with no project(). [security] brings
# cryptography, which the AES-GCM envelopes and Ed25519 signatures need: without it the
# modules still import and fail only when a sealing or signing call is made, which is a
# worse way to find out than at install.
# shellcheck disable=SC2086
INENV pip install $BUILD_ISOLATION $NATIVE_ARG ".[io,security]" || die "core build failed."

# The store, the query language and the observatory are distributions of their own,
# sitting beside the core rather than inside the agent. They are installed from THIS
# repo, in dependency order, and they have to be installed before the agent: the agent
# requires rexgraph-rcdb, and without a local install pip would go looking for it on an
# index where it does not exist, and the install would fail there rather than here.
# Editable for the same reason the agent is: this repo stays the source of truth.
say "Installing the sibling distributions (rcdb, rcql, system)"
# Every rcdb extra, because this is the full-repo installer: the agent uses SQL and
# object stores and record encryption, and each of those is an optional dependency of the
# store rather than a base one. The protected search index is NOT among them: safetensors
# is a base dependency, because every backend's put writes safetensors bytes.
INENV pip install -e "./rcdb[sql,objectstore,crypto]" || die "rcdb install failed."
INENV pip install -e "./rcql" || die "rcql install failed."
INENV pip install -e "./system" || die "system install failed."

say "Installing the agent (extras: $EXTRAS)"
X="$EXTRAS"
[ "$EXTRA_CONNECT" = 1 ]   && X="$X,connectors"
[ "$EXTRA_WAREHOUSE" = 1 ] && X="$X,warehouse"
# EDITABLE (-e) is required: the web UI is served from agent/frontend/, which the
# server locates relative to the package source tree. A non-editable install
# copies the package into site-packages, where that sibling frontend/ dir does
# NOT exist, so the browser app silently 404s (API + CLI still work). Editable
# keeps the package pointing at this repo - so DO NOT move/delete this repo dir
# after install.
INENV pip install -e "./agent[$X]" || die "agent install failed."

# 8. vendor the offline UI assets
if [ ! -f agent/frontend/react.production.min.js ]; then
    say "Vendoring React for offline UI (not present)"
    RV=18.2.0
    curl -Ls "https://cdnjs.cloudflare.com/ajax/libs/react/$RV/umd/react.production.min.js"        -o agent/frontend/react.production.min.js || true
    curl -Ls "https://cdnjs.cloudflare.com/ajax/libs/react-dom/$RV/umd/react-dom.production.min.js" -o agent/frontend/react-dom.production.min.js || true
    if [ -s agent/frontend/react.production.min.js ]; then
        sed -i.bak 's#https://cdnjs.cloudflare.com/ajax/libs/react/[^"]*#/static/react.production.min.js#;s#https://cdnjs.cloudflare.com/ajax/libs/react-dom/[^"]*#/static/react-dom.production.min.js#' agent/frontend/index.html && rm -f agent/frontend/index.html.bak
    fi
else
    info "React already vendored - UI is offline-capable."
fi

# 9. verify
# The core is installed NON-editable (compiled .so live in the env's
# site-packages). Run the smoke tests from a NEUTRAL directory: from the repo
# root the source rexgraph/ dir (which has the .pyx but no .so) would shadow the
# compiled package and every core import would fail (this is the real cause of
# the "_laplacians is None" symptom - it's CWD shadowing, not a cache warmup).
say "Verifying"
SMOKE_DIR="$(mktemp -d)"
( cd "$SMOKE_DIR" && INENV python -c "from rexgraph.graph import RexGraph; r=RexGraph.from_graph([0,1,0],[1,2,2]); print('  core OK: betti', r.betti)" ) || { rmdir "$SMOKE_DIR" 2>/dev/null; die "core smoke test failed."; }
( cd "$SMOKE_DIR" && INENV python -c "import agent.server.app; print('  server app imports OK')" ) || { rmdir "$SMOKE_DIR" 2>/dev/null; die "server import failed."; }
rmdir "$SMOKE_DIR" 2>/dev/null || true
info "running the agent test suite…"
( cd agent && INENV python -m pytest -q 2>&1 | tail -3 ) || true
if [ "${RUN_CORE_TESTS:-0}" = 1 ]; then
    info "running the core test suite (RUN_CORE_TESTS=1)…"
    # run_core_tests.sh auto-detects the runner: conda env if present, else PYTHON.
    if [ "$FRONTEND" = conda ]; then
        ENV_NAME="$ENV_NAME" sh run_core_tests.sh 2>&1 | tail -3 || true
    else
        ENV_NAME="$ENV_NAME" PYTHON="$VENV_PY" sh run_core_tests.sh 2>&1 | tail -3 || true
    fi
else
    info "core suite: run  'sh run_core_tests.sh'  (compiled core; ~30s, 1700+ tests)."
fi

# 10. next steps
printf '\n\033[32m============================================================\n'
printf ' RexGraph is installed in the "%s" environment.\n' "$ENV_NAME"
printf '============================================================\033[0m\n\n'
if [ "$FRONTEND" = conda ]; then
    ACT="$CONDA activate $ENV_NAME"; case "$CONDA" in */*) ACT="$(basename "$CONDA") activate $ENV_NAME";; esac
else
    ACT=". $VENV_DIR/bin/activate"
fi
info "Activate:   $ACT"
info "Web app:    python agent/run.py            # http://127.0.0.1:8000"
info "            rcf-server                     # env-driven console script"
info "Agent test: (cd agent && python -m pytest -q)"
info "Core test:  sh run_core_tests.sh           # compiled core, from repo root"
info "Connectors: rexgraph-connect list"
info "HTTPS:      python agent/run.py --https    # opt-in self-signed"
printf '\n'

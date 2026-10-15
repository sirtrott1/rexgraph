#!/bin/sh
# Install RexGraph in a dedicated environment. See sh install.sh --help.
set -eu

REPO_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROFILE="${PROFILE:-standard}"
INSTALLER="${INSTALLER:-auto}"
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"
PYTHON="${PYTHON:-python3}"
PY_SPEC="${PY_SPEC:-python=3.13}"
NATIVE="${NATIVE:-0}"
EDITABLE="${EDITABLE:-0}"
BUILD_JOBS="${BUILD_JOBS:-2}"
EXTRAS="${EXTRAS:-}"
RUN_AGENT_TESTS="${RUN_AGENT_TESTS:-0}"
RUN_CORE_TESTS="${RUN_CORE_TESTS:-0}"
GPU_BACKEND="${GPU_BACKEND:-auto}"
NATIVE_HIP="${NATIVE_HIP:-auto}"
GPU_VERIFY="${GPU_VERIFY:-1}"
REQUIRE_GPU="${REQUIRE_GPU:-0}"
SETUP_GPU=0

say() { printf '\n%s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }
usage() {
    cat <<'HELP'
Usage: sh install.sh [options]

  --profile NAME     core, server, standard (default), ml, all, dev, or dev-full
  --installer NAME   auto (default), conda, or pip
  --venv PATH        Environment location (default: <checkout>/.venv)
  --python PATH      Base Python for the pip installer
  --extras LIST      Additional Agent extras, separated by commas
  --gpu BACKEND      auto, cuda, rocm, vulkan, or cpu; enables GPU dependency setup
  --editable         Install packages for source development
  --native           Compile for this CPU with relaxed floating point rules
  --test             Install test dependencies and run the selected suites
  --help             Show this help

The default installer uses conda, mamba, or micromamba for Python, a C compiler
and OpenBLAS. It downloads micromamba if needed. The pip installer uses Python's
venv module and requires a system C compiler, Python headers and BLAS/LAPACK.

Environment settings: PROFILE, INSTALLER, VENV_DIR, PYTHON, PY_SPEC, EXTRAS,
EDITABLE, NATIVE, BUILD_JOBS, TORCH_INDEX_URL, RUN_CORE_TESTS, RUN_AGENT_TESTS.
GPU_BACKEND, NATIVE_HIP (auto, enabled, disabled), HIP_ARCH, ROCM_INDEX_URL.
GPU_VERIFY=0 defers GPU execution checks when building without device access.
NO_CONDA=1 selects pip when INSTALLER=auto.
HELP
}
while [ "$#" -gt 0 ]; do
    case "$1" in
        --profile|--installer|--venv|--python|--extras|--gpu)
            [ "$#" -ge 2 ] && [ -n "$2" ] || die "$1 requires a value."
            case "$1" in
                --profile) PROFILE="$2" ;;
                --installer) INSTALLER="$2" ;;
                --venv) VENV_DIR="$2" ;;
                --python) PYTHON="$2" ;;
                --extras) EXTRAS="$2" ;;
                --gpu) GPU_BACKEND="$2"; SETUP_GPU=1 ;;
            esac
            shift 2 ;;
        --editable) EDITABLE=1; shift ;;
        --native) NATIVE=1; shift ;;
        --test) RUN_CORE_TESTS=1; RUN_AGENT_TESTS=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) die "Unknown option: $1. Use --help." ;;
    esac
done
case "$PROFILE" in core|server|standard|ml|all|dev|dev-full) ;; *) die "Unknown profile: $PROFILE." ;; esac
case "$INSTALLER" in auto|conda|pip) ;; *) die "Unknown installer: $INSTALLER." ;; esac
case "$GPU_BACKEND" in auto|cpu|cuda|rocm|vulkan) ;; *) die "Unknown GPU backend: $GPU_BACKEND." ;; esac
case "$NATIVE_HIP" in auto|enabled|disabled) ;; *) die "NATIVE_HIP must be auto, enabled, or disabled." ;; esac
case "$BUILD_JOBS" in ''|*[!0-9]*|0) die "BUILD_JOBS must be a positive integer." ;; esac
for setting in "$NATIVE" "$EDITABLE" "$RUN_CORE_TESTS" "$RUN_AGENT_TESTS" "$GPU_VERIFY" "$REQUIRE_GPU"; do
    case "$setting" in 0|1) ;; *) die "Boolean settings must be 0 or 1." ;; esac
done
case "$EXTRAS" in *[!a-zA-Z0-9_,-]*) die "EXTRAS must be comma separated extra names." ;; esac
[ "$PROFILE" != core ] || [ -z "$EXTRAS" ] || die "Agent extras require a platform profile."
case "$VENV_DIR" in /*) ;; *) VENV_DIR="$PWD/$VENV_DIR" ;; esac
cd "$REPO_DIR"
OS="$(uname -s)"
case "$OS" in Linux|Darwin) ;; *) die "Use Linux, macOS, or Windows with WSL for this installer." ;; esac

if [ "$INSTALLER" = auto ]; then
    if [ -f "$VENV_DIR/pyvenv.cfg" ] || [ "${NO_CONDA:-0}" = 1 ]; then
        INSTALLER=pip
    else
        INSTALLER=conda
    fi
fi
say "Installing profile '$PROFILE' at $VENV_DIR ($INSTALLER)"

if [ "$INSTALLER" = conda ]; then
    [ ! -f "$VENV_DIR/pyvenv.cfg" ] || die "This is a Python venv. Use --installer pip or another --venv path."
    CONDA=""
    for candidate in micromamba mamba conda; do
        if have "$candidate"; then CONDA="$candidate"; break; fi
    done
    if [ -z "$CONDA" ] && [ -x "$HOME/.local/bin/micromamba" ]; then
        CONDA="$HOME/.local/bin/micromamba"
    fi
    if [ -z "$CONDA" ]; then
        have curl && have tar || die "Install curl and tar, or use --installer pip."
        case "$(uname -m)" in
            x86_64|amd64) MM_ARCH=64 ;;
            aarch64|arm64) [ "$OS" = Darwin ] && MM_ARCH=arm64 || MM_ARCH=aarch64 ;;
            ppc64le) MM_ARCH=ppc64le ;;
            *) die "No micromamba download configured for this CPU. Use --installer pip." ;;
        esac
        [ "$OS" = Darwin ] && MM_OS=osx || MM_OS=linux
        DOWNLOAD_DIR="$(mktemp -d)"
        trap 'rm -rf "$DOWNLOAD_DIR"' 0
        say "Downloading micromamba"
        curl --fail --location --retry 3 "https://micro.mamba.pm/api/micromamba/$MM_OS-$MM_ARCH/latest" \
            -o "$DOWNLOAD_DIR/micromamba.tar.bz2"
        tar -xjf "$DOWNLOAD_DIR/micromamba.tar.bz2" -C "$DOWNLOAD_DIR" bin/micromamba
        mkdir -p "$HOME/.local/bin"
        cp "$DOWNLOAD_DIR/bin/micromamba" "$HOME/.local/bin/micromamba"
        CONDA="$HOME/.local/bin/micromamba"
    fi
    INENV() { "$CONDA" run -p "$VENV_DIR" "$@"; }
    say "Installing Python and the build toolchain from conda-forge"
    if [ -d "$VENV_DIR/conda-meta" ]; then
        "$CONDA" install -y -p "$VENV_DIR" --override-channels -c conda-forge \
            "$PY_SPEC" pip c-compiler openblas pkg-config
    else
        [ ! -e "$VENV_DIR" ] || die "Environment path already exists and is not a conda environment."
        "$CONDA" create -y -p "$VENV_DIR" --override-channels -c conda-forge \
            "$PY_SPEC" pip c-compiler openblas pkg-config
    fi
else
    [ ! -d "$VENV_DIR/conda-meta" ] || die "This is a conda environment. Use --installer conda or another --venv path."
    have "$PYTHON" || die "Python not found: $PYTHON."
    have "${CC:-cc}" || die "Install a C compiler or use --installer conda."
    if [ ! -f "$VENV_DIR/pyvenv.cfg" ]; then
        [ ! -e "$VENV_DIR" ] || die "Environment path already exists and is not a Python venv."
        "$PYTHON" -m venv "$VENV_DIR" || die "Install Python's venv package or use --installer conda."
    fi
    INENV() {
        case "$1" in python|python3) shift ;; *) die "Expected a Python command." ;; esac
        PATH="$VENV_DIR/bin:$PATH" "$VENV_DIR/bin/python" "$@"
    }
fi
INENV python -c 'import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 13), "Use Python 3.10 through 3.13."'
INENV python -m pip install --upgrade pip

# Editable builds retain their build dependencies in the environment.
if [ "$PROFILE" = dev ] || [ "$PROFILE" = dev-full ]; then EDITABLE=1; fi
if [ "$EDITABLE" = 1 ]; then
    INENV python -m pip install 'meson-python>=0.18' 'meson>=1.3' 'cython>=3.0' 'numpy>=2.0' ninja
fi
CORE_EXTRAS=""
AGENT_EXTRAS="server,rcql"
case "$PROFILE" in
    core) RUN_AGENT_TESTS=0 ;;
    server) CORE_EXTRAS=all ;;
    standard) CORE_EXTRAS=all; AGENT_EXTRAS=standard,rcql ;;
    ml) CORE_EXTRAS=all,nn; AGENT_EXTRAS=standard,ml,rcql ;;
    all) CORE_EXTRAS=all,nn; AGENT_EXTRAS=all,rcql ;;
    dev) CORE_EXTRAS=all,dev; AGENT_EXTRAS=standard,rcql,dev ;;
    dev-full) CORE_EXTRAS=all,dev,nn,tf,molecules; AGENT_EXTRAS=test-full,rcql ;;
esac
if [ "$RUN_CORE_TESTS" = 1 ]; then CORE_EXTRAS="${CORE_EXTRAS:+$CORE_EXTRAS,}dev"; fi
if [ "$RUN_AGENT_TESTS" = 1 ]; then AGENT_EXTRAS="$AGENT_EXTRAS,dev"; fi
[ -z "$EXTRAS" ] || AGENT_EXTRAS="$AGENT_EXTRAS,$EXTRAS"
[ "${EXTRA_CONNECT:-0}" != 1 ] || AGENT_EXTRAS="$AGENT_EXTRAS,connectors"
[ "${EXTRA_WAREHOUSE:-0}" != 1 ] || AGENT_EXTRAS="$AGENT_EXTRAS,warehouse"

# Configure device dependencies before building Core so native kernels use the
# selected SDK. CPU installations remain available on machines without a GPU.
if [ "$PROFILE" = ml ] || [ "$PROFILE" = all ] || [ "$PROFILE" = dev-full ] || [ -n "${TORCH_INDEX_URL:-}" ]; then
    SETUP_GPU=1
fi
HIP_BUILD="$NATIVE_HIP"
HIP_TARGETS="${HIP_ARCH:-}"
if [ "$SETUP_GPU" = 1 ]; then
    say "Configuring compute dependencies ($GPU_BACKEND)"
    set -- --backend "$GPU_BACKEND" --native-hip "$NATIVE_HIP"
    [ "$REQUIRE_GPU" != 1 ] || set -- "$@" --require-gpu
    INENV python -I "$REPO_DIR/scripts/setup_gpu.py" "$@"
    HIP_BUILD="$(INENV python -I "$REPO_DIR/scripts/setup_gpu.py" --field native_hip)"
    HIP_TARGETS="$(INENV python -I "$REPO_DIR/scripts/setup_gpu.py" --field targets)"
    HIP_BUILD="${HIP_BUILD:-auto}"
    case ",$CORE_EXTRAS," in *,nn,*) ;; *) CORE_EXTRAS="${CORE_EXTRAS:+$CORE_EXTRAS,}nn" ;; esac
fi
install_package() {
    if [ "$EDITABLE" = 1 ]; then
        case "$1" in
            .|.\[*\]) INENV python -m pip install --no-build-isolation -e "$@" ;;
            *) INENV python -m pip install -e "$@" ;;
        esac
    else
        INENV python -m pip install "$@"
    fi
}
say "Building the compiled Core"
set -- .
[ -z "$CORE_EXTRAS" ] || set -- ".[${CORE_EXTRAS}]"
set -- "$@" "-Ccompile-args=-j$BUILD_JOBS"
set -- "$@" "-Csetup-args=-Dhip=$HIP_BUILD"
[ -z "$HIP_TARGETS" ] || set -- "$@" "-Csetup-args=-Dhip_arch=$HIP_TARGETS"
[ "$NATIVE" != 1 ] || set -- "$@" -Csetup-args=-Dnative=true
if [ "$OS" = Darwin ] && [ "$INSTALLER" = pip ]; then
    set -- "$@" -Csetup-args=-Dblas_provider=accelerate -Csetup-args=-Dopenmp=disabled
fi
install_package "$@"

if [ "$PROFILE" != core ]; then
    say "Installing RCDB, RCQL, System and Agent"
    RCDB_EXTRAS=sql,objectstore,crypto
    RCQL_EXTRAS=""
    SYSTEM_EXTRAS=""
    if [ "$PROFILE" = dev ] || [ "$PROFILE" = dev-full ]; then
        RCDB_EXTRAS="$RCDB_EXTRAS,dev"
        RCQL_EXTRAS=dev
        SYSTEM_EXTRAS=dev
    fi
    install_package "./rcdb[$RCDB_EXTRAS]"
    if [ -n "$RCQL_EXTRAS" ]; then install_package "./rcql[$RCQL_EXTRAS]"; else install_package ./rcql; fi
    if [ -n "$SYSTEM_EXTRAS" ]; then install_package "./system[$SYSTEM_EXTRAS]"; else install_package ./system; fi
    install_package "./agent[$AGENT_EXTRAS]"
fi

# Verify installed packages with the source checkout excluded from imports.
say "Verifying the installation"
CHECK_OPTION=""
[ "$EDITABLE" != 1 ] || CHECK_OPTION=--allow-editable
INENV python -I "$REPO_DIR/scripts/verify_install.py" --profile "$PROFILE"
INENV python -m pip check
if [ "$SETUP_GPU" = 1 ]; then
    if [ "$GPU_VERIFY" = 1 ]; then
        say "Checking the selected compute backend"
        INENV python -I "$REPO_DIR/scripts/setup_gpu.py" --verify
    else
        say "GPU execution checks deferred. Run the following in a terminal with device access:"
        printf 'make -C "%s" verify-gpu VENV_DIR="%s"\n' "$REPO_DIR" "$VENV_DIR"
    fi
fi
if [ "$RUN_CORE_TESTS" = 1 ]; then
    INENV python -I "$REPO_DIR/scripts/test_installed.py" ${CHECK_OPTION:+"$CHECK_OPTION"} --package rexgraph -- "$REPO_DIR/rexgraph/tests" -q
fi
if [ "$RUN_AGENT_TESTS" = 1 ]; then
    INENV python -I "$REPO_DIR/scripts/test_installed.py" ${CHECK_OPTION:+"$CHECK_OPTION"} \
        --package rexgraph --package agent --package rcdb -- "$REPO_DIR/agent/tests" -q
fi
say "Installation complete."
printf 'Python: %s/bin/python\n' "$VENV_DIR"
printf 'Verify: %s/bin/python -I %s/scripts/verify_install.py --profile %s\n' "$VENV_DIR" "$REPO_DIR" "$PROFILE"
if [ "$PROFILE" != core ]; then
    printf 'Agent: make -C "%s" serve VENV_DIR="%s"\n' "$REPO_DIR" "$VENV_DIR"
    printf 'System: make -C "%s" system VENV_DIR="%s"\n' "$REPO_DIR" "$VENV_DIR"
fi

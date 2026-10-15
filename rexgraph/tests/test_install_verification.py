"""Exercise installer routing and failures without modifying real environments."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_installed_runner_isolates_subprocess_imports_when_started_from_checkout(tmp_path):
    probe = tmp_path / "test_child_import.py"
    probe.write_text('''import subprocess, sys
from pathlib import Path

def test_child_import():
    assert not (Path.cwd() / "meson.build").exists()
    subprocess.run([sys.executable, "-c",
        "import rexgraph; assert not rexgraph.core._failed; print(rexgraph.__file__)"],
        check=True, capture_output=True, text=True, timeout=30)
''')
    report = tmp_path / "report.xml"
    command = [sys.executable, "-I", str(ROOT / "scripts/test_installed.py"),
               "--allow-editable", "--package", "rexgraph", "--",
               os.path.relpath(probe, ROOT), "-q", "--junitxml=" + os.path.relpath(report, ROOT)]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=60,
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert report.is_file()


def test_wheel_conftest_does_not_import_a_missing_sibling_from_source(tmp_path):
    checkout = tmp_path / "checkout"
    tests = checkout / "rexgraph" / "tests"
    tests.mkdir(parents=True)
    (tests / "__init__.py").touch()
    (checkout / "scripts").mkdir()
    shutil.copy2(ROOT / "conftest.py", checkout / "conftest.py")
    shutil.copy2(ROOT / "scripts/_rexgraph_test_state.py", checkout / "scripts/_rexgraph_test_state.py")
    sibling = checkout / "rcql" / "rcql"
    sibling.mkdir(parents=True)
    marker = tmp_path / "source-was-imported"
    (sibling / "__init__.py").write_text(
        "from pathlib import Path\nPath(" + repr(str(marker)) + ").touch()\n")
    probe = tests / "test_missing_sibling.py"
    probe.write_text("from pathlib import Path\ndef test_source_is_not_loaded():\n    assert not Path(" +
                     repr(str(marker)) + ").exists()\n")
    result = subprocess.run([sys.executable, "-I", str(ROOT / "scripts/test_installed.py"),
                             "--allow-editable", "--package", "rexgraph", "--", str(probe), "-q"],
                            cwd=checkout, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not marker.exists()


def test_fixture_state_replaces_inherited_user_paths_before_import(tmp_path):
    settings = {"REXGRAPH_CONFIG_DIR": str(tmp_path / "user-config"),
                "REXGRAPH_CACHE_DIR": str(tmp_path / "user-cache"),
                "REXGRAPH_SESSION_DIR": str(tmp_path / "user-sessions"),
                "HF_HUB_CACHE": str(tmp_path / "user-models"),
                "REXGRAPH_WORDNET": str(tmp_path / "user-wordnet")}
    code = '''import os, sys, tempfile
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from _rexgraph_test_state import prepare_test_state
root = prepare_test_state()
assert root == prepare_test_state()
from agent.cli.config import CONFIG_FILE
from agent.data_paths import cache_root, sessions_root, huggingface_cache_root, lexical_path
for path in (CONFIG_FILE, cache_root(), sessions_root(), huggingface_cache_root(), lexical_path("wordnet")):
    assert Path(path).is_relative_to(root), path
assert Path(tempfile.gettempdir()).is_relative_to(root)
'''
    pytest.importorskip("agent.cli.config", reason="Agent is required to check import time configuration")
    result = subprocess.run([sys.executable, "-I", "-c", code, str(ROOT / "scripts")],
                            cwd=tmp_path, env=dict(os.environ, **settings), text=True,
                            capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not any(Path(path).exists() for path in settings.values())


@pytest.fixture
def installation(tmp_path):
    checkout = tmp_path / "checkout with spaces"
    checkout.mkdir()
    shutil.copy2(ROOT / "install.sh", checkout / "install.sh")
    shutil.copy2(ROOT / "Makefile", checkout / "Makefile")
    (checkout / "scripts").mkdir()
    shutil.copy2(ROOT / "scripts/setup_gpu.py", checkout / "scripts/setup_gpu.py")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    log = tmp_path / "commands.jsonl"
    python = binaries / "mock-python"
    python.write_text(f"#!{sys.executable}\n" + r'''
import json, os, sys, shutil
from pathlib import Path
args = sys.argv[1:]
with open(os.environ["COMMAND_LOG"], "a") as stream:
    stream.write(json.dumps({"args": args, "cwd": os.getcwd(), "python": sys.argv[0],
                             "path_python": shutil.which("python")}) + "\n")
if "verify_install.py" in " ".join(args):
    stage = "verify"
elif "test_installed.py" in " ".join(args):
    stage = "agent_suite" if any("agent/tests" in arg for arg in args) else "core_suite"
elif args == ["-m", "pip", "check"]:
    stage = "pip_check"
elif any(arg.startswith("./rcdb") for arg in args):
    stage = "store_install"
elif any(arg == "." or arg.startswith(".[") for arg in args):
    stage = "core_build"
else:
    stage = "other"
sys.exit(1 if stage == os.environ.get("FAIL_STAGE") else 0)
''')
    python.chmod(0o755)
    conda = binaries / "micromamba"
    conda.write_text('''#!/bin/sh
set -eu
case "$1" in
    create|install)
        shift
        while [ "$1" != -p ]; do shift; done
        shift
        mkdir -p "$1/conda-meta" "$1/bin"
        cp "$MOCK_PYTHON" "$1/bin/python" ;;
    run) shift 4; exec "$TARGET_ENV/bin/python" "$@" ;;
    *) exit 7 ;;
esac
''')
    conda.chmod(0o755)
    env = tmp_path / "environment with spaces"
    env.mkdir()
    (env / "bin").mkdir()
    shutil.copy2(python, env / "bin" / "python")
    settings = dict(os.environ, PATH=str(binaries) + os.pathsep + os.environ["PATH"],
                    VENV_DIR=str(env), COMMAND_LOG=str(log), MOCK_PYTHON=str(python),
                    TARGET_ENV=str(env), RUN_AGENT_TESTS="0", RUN_CORE_TESTS="0",
                    PROFILE="standard", INSTALLER="pip", NATIVE="0", EDITABLE="0", EXTRAS="",
                    BUILD_JOBS="2", TORCH_INDEX_URL="", EXTRA_CONNECT="0", EXTRA_WAREHOUSE="0",
                    GPU_BACKEND="auto", NATIVE_HIP="auto", GPU_VERIFY="1", REQUIRE_GPU="0")
    # Parent Make command variables must not override this fixture's mock environment.
    for name in ("MAKEFLAGS", "MFLAGS", "MAKELEVEL", "MAKEOVERRIDES", "GNUMAKEFLAGS"):
        settings.pop(name, None)

    def run(*args, frontend="pip", failed="", make=False):
        marker = env / ("pyvenv.cfg" if frontend == "pip" else "conda-meta")
        if frontend == "pip":
            marker.touch()
        else:
            marker.mkdir(exist_ok=True)
        settings.update(INSTALLER=frontend, FAIL_STAGE=failed)
        command = ["make", *args] if make else ["sh", str(checkout / "install.sh"), *args]
        # Invoke outside the checkout to verify script location handling.
        cwd = checkout if make else tmp_path
        result = subprocess.run(command, cwd=cwd, env=settings, text=True, capture_output=True)
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        return result, calls

    return run


@pytest.mark.parametrize("frontend", ["conda", "pip"])
@pytest.mark.parametrize("failed", ["", "core_build", "store_install", "verify", "pip_check", "core_suite", "agent_suite"])
def test_install_failures_are_reported_and_checks_use_selected_python(installation, frontend, failed):
    result, calls = installation("--test", frontend=frontend, failed=failed)
    assert result.returncode == (1 if failed else 0), result.stdout + result.stderr
    assert all("environment with spaces/bin/python" in call["python"] for call in calls)
    if frontend == "pip":
        assert all(call["path_python"] == call["python"] for call in calls)
    checks = [call["args"] for call in calls if any("verify_install.py" in arg or "test_installed.py" in arg
                                                 for arg in call["args"])]
    assert all(args[0] == "-I" for args in checks)
    if not failed:
        assert len(checks) == 3
        assert calls[-2]["args"][-1] == "-q"
        assert "Installation complete." in result.stdout
    else:
        assert "Installation complete." not in result.stdout


@pytest.mark.parametrize("profile", ["core", "server", "standard", "ml", "all", "dev", "dev-full"])
def test_profile_installs_local_dependencies_before_agent(installation, profile):
    result, calls = installation("--profile", profile)
    assert result.returncode == 0, result.stdout + result.stderr
    packages = [call["args"] for call in calls if "install" in call["args"]]
    core = next(i for i, args in enumerate(packages) if any(arg == "." or arg.startswith(".[") for arg in args))
    if profile == "core":
        assert not any(arg.startswith("./agent") or arg.startswith("./rcdb") for args in packages for arg in args)
    else:
        store = next(i for i, args in enumerate(packages) if any(arg.startswith("./rcdb") for arg in args))
        query = next(i for i, args in enumerate(packages) if any(arg.startswith("./rcql") for arg in args))
        system = next(i for i, args in enumerate(packages) if any(arg.startswith("./system") for arg in args))
        agent = next(i for i, args in enumerate(packages) if any(arg.startswith("./agent") for arg in args))
        assert core < store < query < system < agent
    assert any("-e" in args for args in packages) == (profile in {"dev", "dev-full"})
    assert not any("test_installed.py" in arg for call in calls for arg in call["args"])


@pytest.mark.parametrize("args", [("--profile", "unknown"), ("--installer", "unknown"),
                                 ("--venv",), ("--extras", "invalid;value"), ("--unknown",)])
def test_invalid_options_do_not_install(installation, args):
    result, calls = installation(*args)
    assert result.returncode == 1
    assert not calls


def test_make_and_script_share_the_standard_profile(installation):
    result, calls = installation("install", "INSTALLER=pip", make=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert any("./agent[standard,rcql]" in call["args"] for call in calls)
    assert not any("test_installed.py" in arg for call in calls for arg in call["args"])


def test_make_without_a_target_prints_help_without_installing(installation):
    result, calls = installation(make=True)
    assert result.returncode == 0
    assert "make install" in result.stdout
    assert not calls


def test_gpu_install_automatically_configures_and_verifies_selected_environment(installation):
    result, calls = installation("install-gpu", make=True)
    assert result.returncode == 0, result.stdout + result.stderr
    setup = [call["args"] for call in calls if any("setup_gpu.py" in arg for arg in call["args"])]
    assert setup[0][-5:] == ["--backend", "auto", "--native-hip", "auto", "--require-gpu"]
    assert setup[-1][-1] == "--verify"
    assert any("./agent[standard,ml,rcql]" in call["args"] for call in calls)


def test_make_ml_preserves_an_explicit_cpu_selection(installation):
    result, calls = installation("install-ml", "GPU_BACKEND=cpu", make=True)
    assert result.returncode == 0, result.stdout + result.stderr
    setup = next(call["args"] for call in calls if any("setup_gpu.py" in arg for arg in call["args"]))
    assert setup[-4:] == ["--backend", "cpu", "--native-hip", "auto"]


def test_amd_ocr_install_requires_a_platform_wheel_before_pip(installation):
    result, calls = installation("install-ocr-server", "GPU=rocm",
                                 "TORCH_INDEX_URL=https://example.invalid/selected-rocm",
                                 "VLLM_WHEEL_URL=", make=True)
    assert result.returncode != 0
    assert "VLLM_WHEEL_URL" in result.stdout
    assert not calls


def test_clean_preserves_environments_builds_metadata_and_symbolic_links(tmp_path):
    checkout = tmp_path / "checkout with spaces"
    checkout.mkdir()
    (checkout / "scripts").mkdir()
    shutil.copy2(ROOT / "scripts" / "clean_cache.py", checkout / "scripts" / "clean_cache.py")
    shutil.copy2(ROOT / "Makefile", checkout / "Makefile")
    package = checkout / "package"
    (package / "__pycache__").mkdir(parents=True)
    (package / "__pycache__" / "module.pyc").write_bytes(b"synthetic")
    (checkout / ".pytest_cache").mkdir()
    (package / "loose.pyc").write_bytes(b"synthetic")
    metadata = package / "package.egg-info"
    metadata.mkdir()
    (metadata / "PKG-INFO").write_text("Name: synthetic\n")
    protected = [checkout / "build", checkout / "custom environment", tmp_path / "outside"]
    for directory in protected:
        cache = directory / "__pycache__"
        cache.mkdir(parents=True)
        (cache / "keep.pyc").write_bytes(b"synthetic")
    (checkout / "linked directory").symlink_to(protected[-1], target_is_directory=True)
    (checkout / "linked.pyc").symlink_to(protected[-1] / "__pycache__" / "keep.pyc")
    result = subprocess.run(["make", "clean", f"BASE_PYTHON={sys.executable}",
                             f"VENV_DIR={protected[1]}"], cwd=checkout, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (package / "__pycache__").exists()
    assert not (checkout / ".pytest_cache").exists()
    assert not (package / "loose.pyc").exists()
    assert (metadata / "PKG-INFO").exists()
    assert all((directory / "__pycache__" / "keep.pyc").exists() for directory in protected)
    assert (checkout / "linked.pyc").is_symlink()


def test_clean_discovers_local_virtual_environments(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("clean_cache", ROOT / "scripts" / "clean_cache.py")
    cleaner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cleaner)
    for name, marker in (("python env", "pyvenv.cfg"), ("conda env", "conda-meta")):
        env = tmp_path / name
        cache = env / "__pycache__"
        cache.mkdir(parents=True)
        (cache / "keep.pyc").write_bytes(b"synthetic")
        if marker == "pyvenv.cfg":
            (env / marker).write_text("home = /synthetic\n")
        else:
            (env / marker).mkdir()
    assert cleaner.clean(tmp_path) == 0
    assert len(list(tmp_path.rglob("*.pyc"))) == 2

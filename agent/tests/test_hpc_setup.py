"""Generated jobs select their environment and reuse the repository installer."""
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from agent.cli.hpc import TEMPLATES, _activate, generate_slurm_build


def _environment(tmp_path):
    prefix = tmp_path / "environment with 'quotes'"
    (prefix / "bin").mkdir(parents=True)
    python = prefix / "bin" / "python"
    python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
    python.chmod(0o755)
    cache = tmp_path / "cache"
    cache.mkdir()
    env = dict(os.environ, PROJECT=str(cache), SLURM_JOB_ID="test")
    for name in ("REXGRAPH_MODELS_DIR", "REXGRAPH_CACHE_DIR"):
        env.pop(name, None)
    return prefix, env


@pytest.mark.parametrize("name", TEMPLATES)
def test_templates_are_valid_shell_scripts(tmp_path, name):
    script = tmp_path / f"{name}.sbatch"
    script.write_text(TEMPLATES[name][1](conda_env=str(tmp_path / "environment with spaces")))
    result = subprocess.run(["bash", "-n", str(script)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_prefix_selection_handles_spaces_and_quotes(tmp_path):
    prefix, env = _environment(tmp_path)
    script = _activate(str(prefix)) + 'python -I -c "import shutil; print(shutil.which(\'python\'))"\n'
    result = subprocess.run(["bash", "-ec", script], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert str(prefix / "bin" / "python") in result.stdout
    assert (tmp_path / "cache" / "rexgraph_models").is_dir()


def test_missing_environment_stops_before_running_work(tmp_path):
    script = _activate(str(tmp_path / "missing")) + "echo work-started\n"
    result = subprocess.run(["/bin/bash", "-ec", script], env={"PATH": str(tmp_path)},
                            text=True, capture_output=True)
    assert result.returncode == 1
    assert "Environment unavailable" in result.stderr
    assert "work-started" not in result.stdout


def test_build_job_calls_shared_installer_before_smoke_test(tmp_path):
    prefix, env = _environment(tmp_path)
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "install.sh").write_text('#!/bin/sh\nprintf "%s\\n" "$VENV_DIR" "$@" > install-call\n')
    env["SLURM_SUBMIT_DIR"] = str(checkout)
    result = subprocess.run(["bash", "-ec", generate_slurm_build(conda_env=str(prefix))],
                            cwd=tmp_path, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert (checkout / "install-call").read_text().splitlines() == [
        str(Path(sys.prefix)), "--profile", "standard", "--editable"]
    assert "Build OK:" in result.stdout


def test_setup_preserves_edited_templates_and_selects_the_installed_prefix(tmp_path):
    checkout = tmp_path / "checkout"
    (checkout / "agent").mkdir(parents=True)
    source = Path(__file__).resolve().parents[2] / "agent" / "setup_hpc.sh"
    shutil.copy2(source, checkout / "agent" / "setup_hpc.sh")
    (checkout / "install.sh").write_text(
        '#!/bin/sh\ncd "$(dirname "$0")"\nprintf "%s\\n" "$@" > install-call\n')
    (checkout / "slurm").mkdir()
    customized = checkout / "slurm" / "build.sbatch"
    customized.write_text("# local cluster settings\n")
    result = subprocess.run(["sh", str(checkout / "agent" / "setup_hpc.sh")],
                            cwd=tmp_path, env=dict(os.environ, VENV_DIR=sys.prefix),
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert customized.read_text() == "# local cluster settings\n"
    assert (checkout / "install-call").read_text().splitlines() == ["--profile", "standard"]
    assert f"REX_ENV={shlex.quote(sys.prefix)}" in (checkout / "slurm" / "corpus.sbatch").read_text()

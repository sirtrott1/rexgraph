"""Installer verification must preserve failures and its selected interpreter."""
from pathlib import Path
import subprocess

import pytest


INSTALLER = Path(__file__).resolve().parents[2] / "install.sh"


@pytest.mark.parametrize("frontend", ["conda", "pip"])
@pytest.mark.parametrize("failed", ["none", "core_smoke", "server_smoke", "agent_suite", "core_suite"])
def test_installer_checks_fail_without_running_any_install(tmp_path, frontend, failed):
    source = INSTALLER.read_text()
    verification = source.split("# 9. verify\n", 1)[1].split("# 10. next steps\n", 1)[0]
    checkout = tmp_path / "checkout with spaces"
    checkout.mkdir()
    stubs = r'''
set -eu
say() { printf '%s\n' "$*"; }
info() { printf '%s\n' "$*"; }
die() { printf '%s\n' "$*" >&2; exit 1; }
INENV() {
    case "$*" in
        *'from rexgraph.graph'*) stage=core_smoke ;;
        *'import agent.server.app'*) stage=server_smoke ;;
        *'/agent/tests'*) stage=agent_suite ;;
        *'/rexgraph/tests'*) stage=core_suite ;;
        *) printf 'unexpected command: %s\n' "$*" >&2; exit 2 ;;
    esac
    [ "$PWD" != "$REPO_DIR" ] || exit 3
    [ "$1" = python ] && [ "$2" = -I ] || exit 4
    printf 'checked %s\n' "$stage"
    [ "$stage" != "$failed" ]
}
RUN_AGENT_TESTS=1
RUN_CORE_TESTS=1
'''
    script = f"FRONTEND={frontend}\nfailed={failed}\n" + stubs + verification
    result = subprocess.run(["sh", "-c", script], cwd=checkout, text=True, capture_output=True)
    assert result.returncode == (0 if failed == "none" else 1), result.stdout + result.stderr
    stages = ["core_smoke", "server_smoke", "agent_suite", "core_suite"]
    count = len(stages) if failed == "none" else stages.index(failed) + 1
    assert [line for line in result.stdout.splitlines() if line.startswith("checked ")] == [
        f"checked {stage}" for stage in stages[:count]]


def test_installer_installs_dependencies_before_requested_checks():
    source = INSTALLER.read_text()
    dependency_setup = source.index('X="${X:+$X,}dev"')
    install = source.index('INENV pip install -e "./agent[$X]"')
    verify = source.index("# 9. verify\n")
    assert dependency_setup < install < verify
    assert '"$CONDA" install -y -n "$ENV_NAME" "$PY_SPEC" || die' in source

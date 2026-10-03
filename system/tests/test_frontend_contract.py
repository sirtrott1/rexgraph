"""Run async display contracts against the actual packaged JavaScript asset."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest
import system


def test_frontend_discards_stale_responses_and_forwards_exact_policy():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is unavailable for frontend request-contract verification")
    asset = Path(system.__file__).parent/"frontend/app.jsx"
    runner = Path(__file__).with_name("frontend_contract.js")
    result = subprocess.run([node, str(runner), str(asset)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == {key:"passed" for key in (
        "get_stale_response", "get_error_recovery", "source_query_and_exact_policy",
        "query_stale_response", "policy_change_response", "empty_selection", "source_navigation", "rcdb_panel_wired")}

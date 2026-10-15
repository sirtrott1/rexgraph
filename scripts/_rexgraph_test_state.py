"""Place process defaults in temporary storage before importing tested packages."""
from __future__ import annotations

import atexit
import os
from pathlib import Path
import shutil
import tempfile

_root = None
_previous = {}
_previous_tempdir = None


def prepare_test_state() -> Path:
    """Isolate user files and inherited path overrides once per test process."""
    global _root, _previous_tempdir
    if _root is not None:
        return _root
    _previous_tempdir = tempfile.tempdir
    _root = Path(tempfile.mkdtemp(prefix="rexgraph_test_state_"))
    temporary = _root / "tmp"
    temporary.mkdir()
    settings = {
        "REXGRAPH_CONFIG_DIR": str(_root / "config"),
        "REXGRAPH_CACHE_DIR": str(_root / "cache"),
        "REXGRAPH_DATA_DIR": str(_root / "data"),
        "REXGRAPH_SESSION_DIR": str(_root / "sessions"),
        "REXGRAPH_MODELS_DIR": str(_root / "models"),
        "REXGRAPH_MODEL_SCAN_DIRS": os.pathsep.join((str(_root / "models"),
                                                     str(_root / "huggingface" / "hub"))),
        "REXGRAPH_RCDB_URI": "local://" + str(_root / "records"),
        "REXGRAPH_ACTIVITY_JOURNAL": str(_root / "activity.jsonl"),
        "REXGRAPH_AUDIT_JOURNAL": str(_root / "audit.jsonl"),
        "REXGRAPH_AUTH_MODE": "bearer", "RCF_RATE_LIMIT": "0",
        "REXGRAPH_DEBUG_ERRORS": "0",
        "HF_HOME": str(_root / "huggingface"),
        "HF_DATASETS_CACHE": str(_root / "huggingface" / "datasets"),
        "HF_MODULES_CACHE": str(_root / "huggingface" / "modules"),
        "TORCH_HOME": str(_root / "torch"),
        "XDG_CONFIG_HOME": str(_root / "xdg" / "config"),
        "XDG_CACHE_HOME": str(_root / "xdg" / "cache"),
        "XDG_DATA_HOME": str(_root / "xdg" / "data"),
        "HF_HUB_CACHE": str(_root / "huggingface" / "hub"),
        "HUGGINGFACE_HUB_CACHE": str(_root / "huggingface" / "hub"),
        "TRANSFORMERS_CACHE": str(_root / "huggingface" / "hub"),
        "TMPDIR": str(temporary), "TEMP": str(temporary), "TMP": str(temporary),
    }
    cleared = ("REXGRAPH_LEXICAL_DIR", "REXGRAPH_WORDNET", "REXGRAPH_ROGET",
               "REXGRAPH_WIKTIONARY", "REXGRAPH_SECRETS_URI", "REXGRAPH_NO_CACHE",
               "REXGRAPH_NO_JOURNAL", "REXGRAPH_CACHE_MAX_ENTRIES",
               "REXGRAPH_TLS_CERT", "REXGRAPH_TLS_KEY", "REXGRAPH_ALLOWED_DIRS",
               "REXGRAPH_MODEL_DIRS", "REXGRAPH_ONTOLOGY")
    _previous.update({key: os.environ.get(key) for key in (*settings, *cleared)})
    for key in cleared:
        os.environ.pop(key, None)
    os.environ.update(settings)
    tempfile.tempdir = str(temporary)
    atexit.register(_cleanup)
    return _root


def _cleanup():
    global _root
    if _root is None:
        return
    root, _root = _root, None
    for key, value in _previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    tempfile.tempdir = _previous_tempdir
    shutil.rmtree(root, ignore_errors=True)

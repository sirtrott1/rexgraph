"""Resolve user data, configuration and cache paths without creating resources."""
from __future__ import annotations

import os
import platform
from pathlib import Path


def cache_root() -> Path:
    """Return the Agent cache root, with explicit settings taking precedence."""
    if os.environ.get("REXGRAPH_CACHE_DIR"):
        return Path(os.environ["REXGRAPH_CACHE_DIR"]).expanduser()
    if platform.system() == "Windows":
        base = os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local"
        return Path(base).expanduser() / "rexgraph" / "cache"
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base).expanduser() / "rexgraph"


def config_root() -> Path:
    """Return the Agent configuration root."""
    if os.environ.get("REXGRAPH_CONFIG_DIR"):
        return Path(os.environ["REXGRAPH_CONFIG_DIR"]).expanduser()
    if platform.system() == "Windows":
        base = os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming"
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base).expanduser() / "rexgraph"


def models_root() -> Path:
    """Return the directory for explicitly downloaded RexGraph model files."""
    setting = os.environ.get("REXGRAPH_MODELS_DIR")
    return Path(setting).expanduser() if setting else cache_root() / "models"


def sessions_root() -> Path:
    """Resolve session storage, retaining existing legacy sessions without overrides."""
    setting = os.environ.get("REXGRAPH_SESSION_DIR")
    if setting:
        return Path(setting).expanduser()
    root = config_root() / "sessions"
    if (root.is_dir() or os.environ.get("REXGRAPH_CONFIG_DIR")
            or os.environ.get("XDG_CONFIG_HOME")
            or (platform.system() == "Windows" and os.environ.get("APPDATA"))):
        return root
    legacy = Path.home() / ".rexgraph-agent" / "sessions"
    return legacy if legacy.is_dir() else root


def huggingface_cache_root() -> Path:
    """Resolve the Hub cache using Hugging Face's directory settings."""
    setting = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if setting:
        return Path(setting).expanduser()
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home).expanduser() / "hub"
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base).expanduser() / "huggingface" / "hub"


def lexical_root() -> Path:
    if os.environ.get("REXGRAPH_LEXICAL_DIR"):
        return Path(os.environ["REXGRAPH_LEXICAL_DIR"]).expanduser()
    if os.environ.get("REXGRAPH_DATA_DIR"):
        return Path(os.environ["REXGRAPH_DATA_DIR"]).expanduser() / "lexical"
    data_home = (Path(os.environ["XDG_DATA_HOME"]).expanduser() if os.environ.get("XDG_DATA_HOME")
                 else Path.home() / ".local" / "share")
    root = data_home / "rexgraph" / "lexical"
    if root.is_dir() or os.environ.get("XDG_DATA_HOME"):
        return root
    legacy = Path.home() / "projects" / "rexgraph" / "data" / "lexical"
    return legacy if legacy.is_dir() else root


def lexical_path(resource: str) -> Path:
    paths = {
        "wordnet": ("REXGRAPH_WORDNET", "wordnet/english-wordnet-2024.xml.gz"),
        "roget": ("REXGRAPH_ROGET", "thesaurus/rogets-pg22.txt"),
        "wiktionary": ("REXGRAPH_WIKTIONARY", "wiktionary/wiktionary-en.rexidx"),
    }
    variable, relative = paths[resource]
    return Path(os.environ[variable]).expanduser() if os.environ.get(variable) else lexical_root() / relative


def lexical_status() -> dict:
    result = {}
    for name in ("wordnet", "roget", "wiktionary"):
        path = lexical_path(name)
        result[name] = {"path": str(path), "present": path.is_file()}
    return result

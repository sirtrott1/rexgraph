"""Resolve user data, configuration and cache paths without creating resources."""
from __future__ import annotations

import os
import platform
import json
from pathlib import Path


def _saved_directory(name: str) -> Path | None:
    """Read a directory selected in the application configuration without writing it."""
    try:
        data = json.loads((config_root() / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        return None
    value = data.get(name) if isinstance(data, dict) else None
    return Path(value).expanduser() if isinstance(value, str) and value else None


def cache_root() -> Path:
    """Return the Agent cache root, with explicit settings taking precedence."""
    if os.environ.get("REXGRAPH_CACHE_DIR"):
        return Path(os.environ["REXGRAPH_CACHE_DIR"]).expanduser()
    platform_override = (os.environ.get("LOCALAPPDATA") if platform.system() == "Windows"
                         else os.environ.get("XDG_CACHE_HOME"))
    if not platform_override:
        saved = _saved_directory("cache_dir")
        if saved is not None:
            return saved
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
    if setting:
        return Path(setting).expanduser()
    platform_override = (os.environ.get("LOCALAPPDATA") if platform.system() == "Windows"
                         else os.environ.get("XDG_CACHE_HOME"))
    if not (os.environ.get("REXGRAPH_CACHE_DIR") or platform_override):
        saved = _saved_directory("models_dir")
        if saved is not None:
            return saved
    return cache_root() / "models"


def sessions_root() -> Path:
    """Return explicit session storage or the configured application directory."""
    setting = os.environ.get("REXGRAPH_SESSION_DIR")
    if setting:
        return Path(setting).expanduser()
    return config_root() / "sessions"


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

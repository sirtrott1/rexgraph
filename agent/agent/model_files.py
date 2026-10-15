"""Inspect local model configuration and weight files without downloading data."""
from __future__ import annotations

import json
from pathlib import Path


def has_model_weights(directory: Path) -> bool:
    """Require a configuration and either a full weight file or all indexed shards.

    This checks file presence and nonzero size. The model loader validates their
    contents and any tokenizer or processor files required by the selected model.
    """
    directory = Path(directory)
    try:
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            return False
        for filename in ("model.safetensors", "pytorch_model.bin"):
            path = directory / filename
            if path.is_file() and path.stat().st_size > 0:
                return True
        for filename in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            index = directory / filename
            if not index.is_file():
                continue
            try:
                data = json.loads(index.read_text(encoding="utf-8"))
                weights = data.get("weight_map") if isinstance(data, dict) else None
                if not isinstance(weights, dict) or not weights:
                    continue
                paths = []
                for name in weights.values():
                    if (not isinstance(name, str) or not name or Path(name).is_absolute()
                            or ".." in Path(name).parts or "\\" in name):
                        break
                    paths.append(directory / name)
                else:
                    if all(path.is_file() and path.stat().st_size > 0 for path in paths):
                        return True
            except (OSError, ValueError, UnicodeError):
                continue
    except (OSError, ValueError, UnicodeError):
        pass
    return False


def cached_model_path(model_id: str, cache_dir: Path) -> Path | None:
    """Resolve a complete local model or Hub snapshot from the configured cache."""
    local = Path(model_id).expanduser()
    if has_model_weights(local):
        return local
    root = Path(cache_dir) / ("models--" + model_id.replace("/", "--"))
    snapshots = root / "snapshots"
    try:
        ref = root / "refs" / "main"
        if ref.is_file():
            revision = ref.read_text(encoding="utf-8").strip()
            if not revision or Path(revision).name != revision or revision in {".", ".."}:
                return None
            path = snapshots / revision
            return path if has_model_weights(path) else None
        paths = sorted(snapshots.iterdir(), key=lambda path: path.stat().st_mtime_ns, reverse=True)
        return next((path for path in paths if has_model_weights(path)), None)
    except (OSError, ValueError, UnicodeError):
        return None

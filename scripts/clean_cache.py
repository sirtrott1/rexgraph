#!/usr/bin/env python3
"""Remove checkout bytecode and pytest caches while preserving installed environments."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parent.parent


def clean(root: Path, environment: Path | None = None) -> int:
    """Remove generated caches, excluding build trees, environments and symbolic links."""
    excluded = environment.resolve() if environment is not None else None
    removed = 0
    for current, directories, files in os.walk(root, followlinks=False):
        parent = Path(current)
        for name in directories[:]:
            path = parent / name
            if (path.is_symlink() or name in {".git", ".venv", "build"}
                    or path.resolve() == excluded or (path / "pyvenv.cfg").is_file()
                    or (path / "conda-meta").is_dir()):
                directories.remove(name)
            elif name in {"__pycache__", ".pytest_cache"}:
                shutil.rmtree(path)
                directories.remove(name)
                removed += 1
        for name in files:
            path = parent / name
            if path.suffix == ".pyc" and not path.is_symlink():
                path.unlink()
                removed += 1
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path)
    args = parser.parse_args()
    print(f"Removed {clean(ROOT, args.environment)} generated cache entries")


if __name__ == "__main__":
    main()

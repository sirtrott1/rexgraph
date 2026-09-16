#!/usr/bin/env python3
"""Run checkout tests against installed wheels, without source import fallbacks.

Use a dedicated venv and ``python -I scripts/test_installed.py --package rexgraph
--package rcql -- /absolute/test/paths -q`` from a neutral working directory.
No packages are installed or environments modified by this runner.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", action="append", required=True)
    parser.add_argument("tests", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not sys.flags.isolated:
        parser.error("use python -I to exclude PYTHONPATH and the current directory")
    prefix = Path(sys.prefix).resolve()
    for name in args.package:
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(prefix):
            parser.error(f"{name} is not installed inside this environment: {path}")
        print(f"installed package: {name} = {path}", flush=True)
    # Do not let source only conftests silently add sibling distributions.
    os.environ["REXGRAPH_TEST_INSTALLED"] = "1"
    import pytest
    tests = args.tests[1:] if args.tests[:1] == ["--"] else args.tests
    if not tests:
        parser.error("supply explicit test paths after --")
    status = int(pytest.main(["--import-mode=importlib", *tests]))
    escaped = []
    for name, module in tuple(sys.modules.items()):
        if name.split(".")[0] not in args.package or ".tests" in name or name.endswith(".conftest"):
            continue
        location = getattr(module, "__file__", None)
        if location and not Path(location).resolve().is_relative_to(prefix):
            escaped.append(f"{name}: {location}")
    if escaped:
        print("runtime imports escaped the installed environment:\n" + "\n".join(escaped), file=sys.stderr)
        return 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())

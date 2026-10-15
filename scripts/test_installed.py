#!/usr/bin/env python3
"""Run checkout tests against installed wheels, without source import fallbacks.

Use a dedicated venv and ``python -I scripts/test_installed.py --package rexgraph
--package rcql -- /absolute/test/paths -q``. The runner supplies a neutral working
directory for tests and their subprocesses, resolving supplied paths first.
``--allow-editable`` also accepts source paths named by installed editable metadata.
No packages are installed or environments modified by this runner.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from urllib.parse import unquote, urlparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", action="append", required=True)
    parser.add_argument("--allow-editable", action="store_true",
                        help="also accept source roots declared by installed editable metadata")
    parser.add_argument("tests", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not sys.flags.isolated:
        parser.error("use python -I to exclude PYTHONPATH and the current directory")
    # Package imports can bind paths and initialize stores before conftests run.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _rexgraph_test_state import prepare_test_state
    prepare_test_state()
    prefix = Path(sys.prefix).resolve()
    allowed = {}
    for name in args.package:
        roots = [prefix]
        if args.allow_editable:
            dist_name = "rexgraph" if name == "rexgraph" else f"rexgraph-{name}"
            try:
                direct = distribution(dist_name).read_text("direct_url.json")
            except PackageNotFoundError:
                parser.error(f"{dist_name} is not installed in {sys.executable}; run make install-dev")
            info = json.loads(direct) if direct else {}
            if info.get("dir_info", {}).get("editable"):
                url = urlparse(info["url"])
                if url.scheme != "file" or url.netloc not in ("", "localhost"):
                    parser.error(f"{name} has a nonlocal editable source")
                source_root = Path(unquote(url.path)).resolve()
                roots.append(source_root)
                # Meson editable extensions may use a source owned build link.
                build_root = source_root / "build"
                if (source_root / "meson.build").is_file() and build_root.is_dir():
                    roots.append(build_root.resolve())
        allowed[name] = roots
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if not any(path.is_relative_to(root) for root in roots):
            parser.error(f"{name} is not installed inside this environment: {path}")
        print(f"installed package: {name} = {path}", flush=True)
    # Do not let source only conftests silently add sibling distributions.
    os.environ["REXGRAPH_TEST_INSTALLED"] = "1"
    import pytest
    tests = args.tests[1:] if args.tests[:1] == ["--"] else args.tests
    if not tests:
        parser.error("supply explicit test paths after --")
    original_dir = Path.cwd()
    resolved = []
    output_options = {"--junitxml", "--junit-xml", "--log-file", "--basetemp", "--rootdir"}
    for index, argument in enumerate(tests):
        previous = tests[index - 1] if index else ""
        option, equal, value = argument.partition("=")
        if equal and option in output_options:
            argument = option + "=" + str(Path(value).resolve())
        elif previous in output_options:
            argument = str(Path(argument).resolve())
        elif not argument.startswith("-") and previous not in {"-k", "-m"}:
            path, selector, case = argument.partition("::")
            if Path(path).exists():
                argument = str(Path(path).resolve()) + (selector + case if selector else "")
        resolved.append(argument)
    # Child interpreters inherit cwd even when this interpreter uses -I.
    with tempfile.TemporaryDirectory(prefix="rexgraph-installed-tests-") as work_dir:
        try:
            os.chdir(work_dir)
            status = int(pytest.main(["--import-mode=importlib", *resolved]))
        finally:
            os.chdir(original_dir)
    escaped = []
    for name, module in tuple(sys.modules.items()):
        if name.split(".")[0] not in args.package or ".tests" in name or name.endswith(".conftest"):
            continue
        location = getattr(module, "__file__", None)
        if location and not any(Path(location).resolve().is_relative_to(root)
                                for root in allowed[name.split(".")[0]]):
            escaped.append(f"{name}: {location}")
    if escaped:
        print("runtime imports escaped the installed environment:\n" + "\n".join(escaped), file=sys.stderr)
        return 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Check installed kernels, native actions, sibling packages and UI assets."""
from __future__ import annotations

import argparse
import importlib
import runpy
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("auto", "core", "server", "standard", "ml", "all", "dev", "dev-full"),
                        default="auto")
    args = parser.parse_args()
    if not sys.flags.isolated:
        parser.error("run with python -I to exclude the source checkout from imports")
    if args.profile == "auto":
        try:
            version("rexgraph-agent")
            args.profile = "standard"
        except PackageNotFoundError:
            args.profile = "core"
    import rexgraph
    from rexgraph.core import _laplacians

    print(f"Interpreter: {sys.executable}", flush=True)
    print(f"Core: {rexgraph.__file__}", flush=True)
    print(f"Compiled module: {_laplacians.__file__}", flush=True)
    print(f"Installed version: {version('rexgraph')}", flush=True)
    runpy.run_path(str(Path(__file__).with_name("smoke_test.py")))
    if args.profile != "core":
        for name, distribution in (("rcdb", "rexgraph-rcdb"), ("rcql", "rexgraph-rcql"),
                                   ("system", "rexgraph-system"), ("agent", "rexgraph-agent")):
            module = importlib.import_module(name)
            print(f"{distribution}: {version(distribution)} from {module.__file__}", flush=True)
        importlib.import_module("agent.server.app")
        from agent.ui_assets import validate_ui_assets
        from system.server.launch import _ensure_ui_assets

        validate_ui_assets()
        _ensure_ui_assets()
        print("Agent server and packaged Agent/System UI assets verified.", flush=True)


if __name__ == "__main__":
    main()

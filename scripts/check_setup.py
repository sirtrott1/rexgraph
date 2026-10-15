#!/usr/bin/env python3
"""Report optional Python dependencies, external tools and configured lexical data."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version

FEATURES = {
    "application server": ("server", ("fastapi", "uvicorn", "python-multipart", "httpx", "bcrypt", "limits")),
    "analysis": ("standard", ("scipy", "pandas")),
    "schema and database connectors": ("standard", ("sqlglot", "sqlalchemy", "pymongo", "neo4j")),
    "object storage and signing": ("standard", ("fsspec", "cryptography")),
    "file formats": ("standard", ("pyarrow", "h5py", "zarr", "safetensors", "openpyxl")),
    "PDF and OCR bindings": ("standard", ("pymupdf", "pytesseract")),
    "model training": ("ml", ("torch", "transformers", "peft", "datasets", "accelerate", "sentencepiece")),
    "molecular inputs": ("molecules", ("rdkit",)),
    "stored signature queries": ("analytics", ("duckdb", "polars")),
    "numerical baselines": ("baselines", ("scikit-learn", "lightgbm", ("xgboost", "xgboost-cpu"))),
    "LangChain": ("langchain", ("langchain", "langchain-core")),
    "LangGraph": ("langgraph", ("langgraph",)),
    "TensorFlow": ("Core tf", (("tensorflow", "tensorflow-cpu"),)),
}


def dependency_status(packages):
    installed, missing = {}, []
    for package in packages:
        names = (package,) if isinstance(package, str) else package
        for name in names:
            try:
                installed[name] = version(name)
                break
            except PackageNotFoundError:
                continue
        else:
            missing.append(" or ".join(names))
    return {"installed": installed, "missing": missing}


def report():
    features = {name: {"profile_or_extra": extra, **dependency_status(packages)}
                for name, (extra, packages) in FEATURES.items()}
    data = {}
    ocr = None
    try:
        version("rexgraph-agent")
    except PackageNotFoundError:
        pass
    else:
        from agent.data_paths import lexical_status
        data = lexical_status()
        from agent.cli.setup import has_tesseract
        ocr = {"english_ready": has_tesseract()}
    gpu_state = Path(sys.prefix) / "share/rexgraph/gpu-install.json"
    try:
        gpu = json.loads(gpu_state.read_text())
    except (OSError, ValueError):
        gpu = {"status": "not configured by the installer"}
    from rexgraph._hip_build import compiler_info
    return {"python_dependencies": features, "gpu_setup": gpu, "hip_toolchain": compiler_info(),
            "external_tools": {name: shutil.which(name) for name in ("tesseract", "llama-server", "node", "hipcc")},
            "lexical_data": data, "local_ocr": ocr}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = report()
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print("Optional Python dependencies (installation status):")
    for name, status in result["python_dependencies"].items():
        detail = "installed" if not status["missing"] else "missing " + ", ".join(status["missing"])
        print(f"  {name}: {detail}; profile or extra: {status['profile_or_extra']}")
    print("External tools:")
    for name, path in result["external_tools"].items():
        print(f"  {name}: {path or 'not found'}")
    print("GPU setup: " + result["gpu_setup"]["status"])
    if result["hip_toolchain"]:
        print("HIP compiler: " + result["hip_toolchain"]["compiler"])
    if result["local_ocr"] is not None:
        print("Local OCR: " + ("ready" if result["local_ocr"]["english_ready"]
                               else "requires Tesseract and English language data (eng.traineddata)"))
    if result["lexical_data"]:
        print("Optional lexical data:")
        for name, status in result["lexical_data"].items():
            print(f"  {name}: {'present' if status['present'] else 'not found'} at {status['path']}")


if __name__ == "__main__":
    main()

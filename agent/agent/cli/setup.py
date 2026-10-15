"""
agent.cli.setup: platform aware dependency installation.

Three tiers:
    1. Tesseract + pymupdf (always, no prompt)
    2. PaddleOCR models (~200MB, prompted in interactive mode)
    3. GPU model + vLLM/SGLang (only if GPU detected, opt-in)
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

from .config import (
    CACHE_DIR,
    MODELS_DIR,
    PlatformInfo,
    RexGraphConfig,
    detect_platform,
    load_config,
    save_config,
)

logger = logging.getLogger(__name__)


# Tier 1: Tesseract

def has_tesseract() -> bool:
    """Check whether the local OCR engine and English language data are ready."""
    from agent.integrations.unlimited_ocr import OfflineOCRClient
    return OfflineOCRClient._check_tesseract()


def install_tesseract(platform_info: PlatformInfo, interactive: bool = True) -> bool:
    """Install tesseract via the detected package manager.

    Returns True if tesseract is available after the attempt.
    """
    if has_tesseract():
        return True

    pm = platform_info.package_manager
    commands = {
        "apt":    ["sudo", "apt-get", "install", "-y", "tesseract-ocr", "tesseract-ocr-eng"],
        "dnf":    ["sudo", "dnf", "install", "-y", "tesseract", "tesseract-langpack-eng"],
        "pacman": ["sudo", "pacman", "-S", "--noconfirm", "tesseract", "tesseract-data-eng"],
        "brew":   ["brew", "install", "tesseract"],
        "conda":  ["conda", "install", "-y", "-c", "conda-forge", "tesseract"],
        "choco":  ["choco", "install", "-y", "tesseract"],
        "winget": ["winget", "install", "--id", "UB-Mannheim.TesseractOCR", "-e"],
    }

    cmd = commands.get(pm)
    if not cmd:
        if interactive:
            print("Cannot auto-install tesseract: no supported package manager found.")
            print("Install manually:")
            print("  Linux (apt):   sudo apt install tesseract-ocr tesseract-ocr-eng")
            print("  Linux (dnf):   sudo dnf install tesseract tesseract-langpack-eng")
            print("  Linux (Arch):  sudo pacman -S tesseract tesseract-data-eng")
            print("  macOS (brew):  brew install tesseract")
            print("  Windows:       choco install tesseract")
            print("  Conda:         conda install -c conda-forge tesseract")
        return False

    # For sudo commands, check if we can run them
    if cmd[0] == "sudo" and not platform_info.has_sudo:
        if interactive:
            print(f"Tesseract not found. Install with:\n    {' '.join(cmd)}")
        return False

    if interactive:
        print(f"Installing tesseract via {pm}...")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            logger.warning("tesseract install failed: %s", result.stderr[:200])
            if interactive:
                print(f"Install failed. Run manually:\n    {' '.join(cmd)}")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning("tesseract install error: %s", e)
        return False

    return has_tesseract()


# Tier 1: pymupdf + pytesseract (pip)

def _ensure_pip_packages(packages: list, interactive: bool = True) -> bool:
    """Let pip check installed versions, extras and distribution names."""
    if not packages:
        return True

    if interactive:
        print(f"Checking dependencies: {', '.join(packages)}")

    cmd = [sys.executable, "-m", "pip", "install", "--quiet"] + packages
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


# Tier 2: PaddleOCR

def has_paddleocr() -> bool:
    """Check if PaddleOCR and its models are available."""
    try:
        import paddleocr  # noqa: F401
        return True
    except ImportError:
        return False


def has_paddleocr_models(models_dir: Path | None = None) -> bool:
    """Check if PaddleOCR models are cached."""
    # PaddleOCR caches models in ~/.paddleocr/ by default
    paddle_cache = Path.home() / ".paddleocr"
    custom_dir = models_dir or MODELS_DIR / "paddleocr"
    return paddle_cache.exists() or custom_dir.exists()


def install_paddleocr(interactive: bool = True) -> bool:
    """Install PaddleOCR and download models."""
    # Install paddlepaddle + paddleocr
    packages = ["paddlepaddle", "paddleocr>=2.7"]
    if not _ensure_pip_packages(packages, interactive):
        if interactive:
            print("Failed to install PaddleOCR. Try manually:")
            print("    pip install paddlepaddle paddleocr")
        return False

    # Trigger model download by initializing PaddleOCR
    if interactive:
        print("Downloading PaddleOCR model files...")

    try:
        from agent.integrations.unlimited_ocr import PaddleOCRClient
        return PaddleOCRClient(use_structure=False)._get_engine() is not None
    except Exception as e:
        logger.warning("PaddleOCR model download failed: %s", e)
        if interactive:
            print(f"Model download failed: {e}")
        return False


# Tier 3: GPU model

def _detect_best_gpu_model(platform_info: PlatformInfo) -> str:
    """Choose the best GPU OCR model based on available VRAM."""
    vram = platform_info.gpu_vram_mb

    if vram >= 24000:
        return "Baidu-OCR/Unlimited-OCR"
    elif vram >= 16000:
        return "deepseek-ai/DeepSeek-OCR-2"
    elif vram >= 8000:
        return "stepfun-ai/GOT-OCR2_0"
    else:
        return ""


def _install_torch_for_gpu(platform_info: PlatformInfo, interactive: bool = True) -> bool:
    """Install PyTorch with the correct GPU backend.

    ROCm and CUDA need different torch builds. Installing the wrong
    one means the GPU is invisible to the inference server.
    """
    gpu = platform_info.gpu

    if gpu == "amd":
        try:
            import torch
            if torch.version.hip:
                return True
        except ImportError:
            pass
        index = os.environ.get("TORCH_INDEX_URL")
        if not index:
            if interactive:
                print("Set TORCH_INDEX_URL to a PyTorch build compatible with your ROCm installation.")
            return False
        if interactive:
            print(f"Installing PyTorch from {index}")
        cmd = [
            sys.executable, "-m", "pip", "install", "--quiet",
            "--upgrade", "--force-reinstall",
            "torch", "torchvision",
            "--index-url", index,
        ]
    elif gpu == "nvidia":
        # vLLM bundles its own CUDA torch, so skip explicit install
        return True
    elif gpu == "apple":
        if interactive:
            print("  Apple Silicon detected, installing PyTorch with MPS...")
        cmd = [
            sys.executable, "-m", "pip", "install", "--quiet",
            "torch", "torchvision",
        ]
    else:
        return False

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def install_gpu_backend(
    platform_info: PlatformInfo,
    backend: str = "vllm",
    model: str | None = None,
    interactive: bool = True,
) -> bool:
    """Install a selected inference runtime and download the requested OCR model.

    AMD and Apple installations require a platform runtime already installed or
    an explicit vLLM wheel. CUDA installations may resolve the standard packages.
    """
    model = model or _detect_best_gpu_model(platform_info)
    if not model:
        if interactive:
            print("No suitable GPU model for your VRAM. Need >= 8GB.")
        return False

    gpu = platform_info.gpu

    # Install PyTorch with the right GPU backend first
    if backend not in {"vllm", "sglang"} or gpu not in {"amd", "nvidia", "apple"}:
        if interactive:
            print("Select a supported inference backend and GPU platform.")
        return False
    from importlib.util import find_spec
    wheel = os.environ.get("VLLM_WHEEL_URL", "") if backend == "vllm" else ""
    existing_backend = find_spec(backend) is not None
    if gpu in {"amd", "apple"} and not wheel and not existing_backend:
        if interactive:
            print(f"Install a {backend} build for this GPU platform before running setup.")
            if backend == "vllm":
                print("A selected wheel can be supplied through VLLM_WHEEL_URL.")
                print("https://docs.vllm.ai/en/latest/getting_started/installation/gpu/")
        return False

    if gpu == "amd" and not _install_torch_for_gpu(platform_info, interactive):
        if interactive:
            print("Failed to install PyTorch with ROCm.")
            print("Install manually:")
            print('  python -m pip install torch --index-url "$TORCH_INDEX_URL"')
        return False

    # Pick the inference server
    if wheel:
        packages = [wheel]
    elif gpu in {"amd", "apple"}:
        packages = []
    elif gpu == "nvidia" and backend == "vllm":
        packages = ["vllm>=0.3"]
        if interactive:
            print("Installing vLLM (CUDA)...")
    elif backend == "sglang":
        packages = ["sglang[all]"]
        if interactive:
            print("Installing SGLang...")
    else:
        if interactive:
            print(f"Installing {backend}...")
        packages = [backend]

    if not _ensure_pip_packages(packages, interactive):
        if interactive and gpu == "amd":
            print("If vLLM install fails, try the ROCm docker image:")
            print("  docker pull rocm/vllm:latest")
        return False

    # Model download
    if interactive:
        print(f"Downloading model: {model}")

    model_dir = MODELS_DIR / model.replace("/", "--")
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        _ensure_pip_packages(["huggingface_hub"], interactive=False)
        from huggingface_hub import snapshot_download
        # See model_manager.download: the Hub removed local_dir_use_symlinks and a
        # local_dir download no longer symlinks, so the argument is now a no op warning.
        snapshot_download(
            model,
            local_dir=str(model_dir),
        )
        return True
    except Exception as e:
        logger.warning("Model download failed: %s", e)
        if interactive:
            print(f"Model download failed: {e}")
            print(f"Try manually: huggingface-cli download {model}")
        return False


# Setup flow

def auto_setup(
    interactive: bool = True,
    skip_gpu: bool = False,
    skip_paddleocr: bool = False,
    gpu_backend: str = "vllm",
    gpu_model: str | None = None,
) -> RexGraphConfig:
    """Run the full setup flow.

    If interactive, prints progress and prompts for confirmations.
    If not, installs Tier 1+2 silently.
    """
    import datetime

    cfg = load_config()
    platform_info = detect_platform()

    if interactive:
        print("rexgraph setup")
        print(f"  OS:      {platform_info.os} ({platform_info.arch})")
        print(f"  Python:  {platform_info.python}")
        print(f"  Conda:   {platform_info.conda_env or 'none'}")
        print(f"  Pkg mgr: {platform_info.package_manager}")
        print(f"  GPU:     {platform_info.gpu_name or 'none'}")
        if platform_info.gpu_vram_mb:
            print(f"  VRAM:    {platform_info.gpu_vram_mb} MB")
        if platform_info.scheduler != "none":
            print(f"  Sched:   {platform_info.scheduler}")
        print()

    # Tier 1: Tesseract
    if not has_tesseract():
        if interactive:
            print("[1/3] Installing tesseract...")
        ok = install_tesseract(platform_info, interactive)
        cfg.tesseract = ok
        if ok and interactive:
            print("  ✓ tesseract installed")
    else:
        cfg.tesseract = True
        if interactive:
            print("[1/3] tesseract: already installed ✓")

    # Ensure pymupdf and pytesseract
    _ensure_pip_packages(["pymupdf", "pytesseract"], interactive=False)

    # Tier 2: PaddleOCR
    if not skip_paddleocr:
        if not has_paddleocr() or not has_paddleocr_models():
            if interactive:
                answer = input("[2/3] Download PaddleOCR models (~200MB)? [Y/n] ").strip()
                if answer.lower() == "n":
                    print("  Skipped PaddleOCR")
                else:
                    ok = install_paddleocr(interactive)
                    cfg.paddleocr = ok
                    if ok:
                        print("  ✓ PaddleOCR installed")
            else:
                # Non interactive: install silently
                cfg.paddleocr = install_paddleocr(interactive=False)
        else:
            cfg.paddleocr = True
            if interactive:
                print("[2/3] PaddleOCR: already installed ✓")
    elif interactive:
        print("[2/3] PaddleOCR: skipped")

    # Tier 3: GPU backend
    if not skip_gpu and platform_info.gpu != "none" and platform_info.gpu_vram_mb >= 8000:
        model_name = gpu_model or _detect_best_gpu_model(platform_info)
        if model_name:
            if interactive:
                print(f"\n[3/3] GPU detected: {platform_info.gpu_name}")
                print(f"      Recommended model: {model_name}")
                answer = input("      Download and install? [y/N] ").strip()
                if answer.lower() == "y":
                    ok = install_gpu_backend(
                        platform_info, backend=gpu_backend,
                        model=model_name, interactive=interactive,
                    )
                    if ok:
                        cfg.gpu_model = model_name
                        cfg.gpu_model_path = str(
                            MODELS_DIR / model_name.replace("/", "--")
                        )
                        cfg.gpu_server_backend = gpu_backend
                        print("  ✓ GPU model installed")
                else:
                    print("  Skipped GPU model")
            # Non interactive: skip GPU (too expensive for auto)
        elif interactive:
            print("[3/3] GPU: insufficient VRAM for OCR models")
    elif interactive:
        if skip_gpu:
            print("[3/3] GPU: skipped")
        elif platform_info.gpu == "none":
            print("[3/3] GPU: none detected")
        else:
            print(f"[3/3] GPU: {platform_info.gpu_name} "
                  f"({platform_info.gpu_vram_mb}MB - need ≥8000MB)")

    # Save config
    cfg.last_setup = datetime.datetime.now().isoformat()[:10]
    cfg.platform_os = platform_info.os
    cfg.platform_gpu = platform_info.gpu
    cfg.cache_dir = str(CACHE_DIR)
    cfg.models_dir = str(MODELS_DIR)
    save_config(cfg)

    if interactive:
        print(f"\nConfig saved to {cfg.cache_dir}")
        print("Run `rexgraph-ocr status` to check availability.")

    return cfg


# Smoke test

def smoke_test_rexgraph() -> bool:
    """Quick sanity check: import rexgraph and build a tiny rex."""
    try:
        from rexgraph.graph import RexGraph
        rex = RexGraph.from_graph([0, 1, 0], [1, 2, 2])
        assert rex.betti == (1, 1, 0), f"Unexpected betti: {rex.betti}"
        return True
    except Exception as e:
        logger.warning("rexgraph smoke test failed: %s", e)
        return False


def main(argv=None) -> int:
    """CLI entry: rexgraph setup [--yes] [--skip-gpu]."""
    import argparse

    p = argparse.ArgumentParser(
        prog="rexgraph-setup",
        description="Install OCR / GPU dependencies for the RexGraph agent.",
    )
    p.add_argument("--yes", "-y", action="store_true",
                   help="Non-interactive install")
    p.add_argument("--skip-gpu", action="store_true",
                   help="Skip GPU backend installation")
    p.add_argument("--skip-paddleocr", action="store_true",
                   help="Skip PaddleOCR installation")
    args = p.parse_args(argv)
    auto_setup(interactive=not args.yes, skip_gpu=args.skip_gpu,
               skip_paddleocr=args.skip_paddleocr)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

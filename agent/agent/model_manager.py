"""
agent.model_manager: Centralized model lifecycle management.

Owns all loaded models, manages VRAM, handles multi-model loading.
Models load once and persist across requests for the server lifetime.

Usage:

    from agent.model_manager import get_manager

    mgr = get_manager()
    mgr.load("stepfun-ai/GOT-OCR-2.0-hf", purpose="ocr")
    result = mgr.infer("stepfun-ai/GOT-OCR-2.0-hf", image=img)
    mgr.status()  # -> {loaded: [...], available: [...], vram: {...}}
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ModelType(str, Enum):
    TRANSFORMERS = "transformers"   # loaded via from_pretrained
    VLLM_SERVER = "vllm"           # served via vLLM subprocess
    SGLANG_SERVER = "sglang"       # served via SGLang subprocess
    API = "api"                    # remote API (no local model)
    GGUF = "gguf"                  # llama.cpp / GGUF format
    CUSTOM = "custom"              # user-defined loader


@dataclass
class ModelAdapter:
    """Interface for custom model inference.

    Users subclass this to bring their own models:

        class MyOCRAdapter(ModelAdapter):
            def load(self, path, device):
                self.model = MyModel.from_pretrained(path)
            def infer(self, **kwargs):
                return self.model.predict(kwargs["image"])
            def unload(self):
                del self.model
    """
    name: str = ""

    def load(self, path: str, device: str = "auto") -> None:
        raise NotImplementedError

    def infer(self, **kwargs) -> Any:
        raise NotImplementedError

    def unload(self) -> None:
        pass


@dataclass
class ModelEntry:
    """Registry entry for a known model."""
    model_id: str
    model_type: ModelType = ModelType.TRANSFORMERS
    purpose: str = ""              # ocr, chat, embedding, etc.
    size_gb: float = 0
    downloaded: bool = False
    local_path: str = ""
    hf_cache_path: str = ""        # ~/.cache/huggingface/hub/models--...
    requires_gpu: bool = True
    notes: str = ""


@dataclass
class LoadedModel:
    """A model currently in memory/VRAM."""
    model_id: str
    model_type: ModelType
    purpose: str
    model_obj: Any = None          # the actual model object
    processor_obj: Any = None      # tokenizer / processor
    device: str = "auto"
    vram_mb: int = 0
    loaded_at: float = 0.0
    last_used: float = 0.0
    use_count: int = 0

    # For server-backed models
    server_pid: int = 0
    server_port: int = 0
    server_url: str = ""


# Known model registry

KNOWN_MODELS = {
    "stepfun-ai/GOT-OCR-2.0-hf": {
        "type": ModelType.TRANSFORMERS,
        "purpose": "ocr",
        "size_gb": 4,
        "notes": "HuggingFace transformers, runs directly on GPU",
    },
    "deepseek-ai/DeepSeek-OCR-2": {
        "type": ModelType.VLLM_SERVER,
        "purpose": "ocr",
        "size_gb": 8,
        "notes": "Requires vLLM server (make ocr-serve)",
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "type": ModelType.VLLM_SERVER,
        "purpose": "chat",
        "size_gb": 16,
    },
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "type": ModelType.VLLM_SERVER,
        "purpose": "chat",
        "size_gb": 14,
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "type": ModelType.VLLM_SERVER,
        "purpose": "chat",
        "size_gb": 14,
    },
}


class ModelManager:
    """Singleton model lifecycle manager.

    - Scans for downloaded models (HF cache + rexgraph cache + custom paths)
    - Loads transformers models into VRAM (persist across requests)
    - Manages vLLM/SGLang server processes (multiple simultaneously)
    - Tracks VRAM usage
    - Routes inference to the right loaded model
    """

    def __init__(self):
        self._registry: Dict[str, ModelEntry] = {}
        self._loaded: Dict[str, LoadedModel] = {}
        self._custom_paths: Dict[str, str] = {}  # model_id -> path
        self._custom_loaders: Dict[str, ModelAdapter] = {}  # model_id -> adapter
        self._pipeline_config: Dict[str, str] = {}  # purpose -> model_id
        self._scan_done = False
        self._load_custom_paths()
        self._load_pipeline_config()

    def _load_custom_paths(self):
        """Load user-configured model paths from config."""
        try:
            from agent.cli.config import load_config
            cfg = load_config()
            paths = getattr(cfg, "model_paths", None)
            if isinstance(paths, dict):
                self._custom_paths = dict(paths)
        except Exception:
            pass

    def _load_pipeline_config(self):
        """Load which model to use for each pipeline stage."""
        try:
            from agent.cli.config import load_config
            cfg = load_config()
            pc = getattr(cfg, "pipeline_models", None)
            if isinstance(pc, dict):
                self._pipeline_config = dict(pc)
        except Exception:
            pass

    def _save_custom_paths(self):
        """Persist custom model paths to config."""
        try:
            from agent.cli.config import load_config, save_config
            cfg = load_config()
            cfg.model_paths = dict(self._custom_paths)
            save_config(cfg)
        except Exception:
            pass

    def _save_pipeline_config(self):
        """Persist pipeline model assignments."""
        try:
            from agent.cli.config import load_config, save_config
            cfg = load_config()
            cfg.pipeline_models = dict(self._pipeline_config)
            save_config(cfg)
        except Exception:
            pass

    # Custom model registration

    def register_adapter(self, model_id: str, adapter: ModelAdapter,
                         purpose: str = "", path: str = "") -> ModelEntry:
        """Register a custom model with a user-defined loader.

        This is the entry point for fine-tuned models, custom architectures,
        or any model not supported by the built-in transformers loader.

        Example:

            class MyEmbedder(ModelAdapter):
                def load(self, path, device):
                    from sentence_transformers import SentenceTransformer
                    self.model = SentenceTransformer(path, device=device)
                def infer(self, texts=None, **kw):
                    return self.model.encode(texts)
                def unload(self):
                    del self.model

            mgr.register_adapter(
                "my-embedder", MyEmbedder(),
                purpose="embedding",
                path="/data/models/custom-embedder"
            )
        """
        self._custom_loaders[model_id] = adapter

        entry = ModelEntry(
            model_id=model_id,
            model_type=ModelType.CUSTOM,
            purpose=purpose,
        )
        if path:
            resolved = os.path.realpath(os.path.expanduser(path))
            entry.local_path = resolved
            entry.downloaded = os.path.isdir(resolved)
            self._custom_paths[model_id] = resolved
            self._save_custom_paths()

        self._registry[model_id] = entry
        logger.info("Registered custom model adapter: %s (purpose=%s)",
                     model_id, purpose)
        return entry

    # Pipeline model assignment

    def set_pipeline_model(self, purpose: str, model_id: str):
        """Assign a model to a pipeline stage.

        Purposes: ocr, chat, embedding, reranker, judge

        Example:

            mgr.set_pipeline_model("ocr", "stepfun-ai/GOT-OCR-2.0-hf")
            mgr.set_pipeline_model("chat", "my-finetuned-llama")
            mgr.set_pipeline_model("embedding", "my-embedder")
        """
        self._pipeline_config[purpose] = model_id
        self._save_pipeline_config()
        logger.info("Pipeline %s -> %s", purpose, model_id)

    def get_pipeline_model(self, purpose: str) -> Optional[str]:
        """Get the model assigned to a pipeline stage."""
        return self._pipeline_config.get(purpose)

    def pipeline_config(self) -> Dict[str, str]:
        """Get the full pipeline model configuration."""
        return dict(self._pipeline_config)

    def set_model_path(self, model_id: str, path: str,
                       model_type: str = "transformers",
                       purpose: str = "") -> ModelEntry:
        """Register a custom local path for a model.

        This lets users point to models stored anywhere on their
        filesystem - not just the HF cache or rexgraph cache.

        Parameters
        ----------
        model_id : identifier (e.g. "my-custom-llm" or "stepfun-ai/GOT-OCR-2.0-hf")
        path : absolute path to the model directory
        model_type : "transformers", "vllm", "sglang", "gguf"
        purpose : "ocr", "chat", "embedding", etc.
        """
        resolved = os.path.realpath(os.path.expanduser(path))
        if not os.path.isdir(resolved):
            raise FileNotFoundError("Model path does not exist: %s" % resolved)

        self._custom_paths[model_id] = resolved
        self._save_custom_paths()

        # Update or create registry entry
        entry = self._registry.get(model_id, ModelEntry(model_id=model_id))
        entry.downloaded = True
        entry.local_path = resolved
        entry.purpose = purpose or entry.purpose
        try:
            entry.model_type = ModelType(model_type)
        except ValueError:
            entry.model_type = ModelType.TRANSFORMERS
        self._registry[model_id] = entry

        logger.info("Registered model path: %s -> %s", model_id, resolved)
        return entry

    def remove_model_path(self, model_id: str) -> bool:
        """Remove a custom model path registration."""
        if model_id in self._custom_paths:
            del self._custom_paths[model_id]
            self._save_custom_paths()
            return True
        return False

    def get_model_path(self, model_id: str) -> Optional[str]:
        """Resolve the local path for a model.

        Priority: custom path -> rexgraph cache -> HF cache -> model_id as-is.
        """
        # 1. Custom user path
        if model_id in self._custom_paths:
            p = self._custom_paths[model_id]
            if os.path.isdir(p):
                return p

        # 2. Rexgraph cache
        try:
            from agent.cli.config import MODELS_DIR
            dirname = model_id.replace("/", "--")
            rex_path = MODELS_DIR / dirname
            if rex_path.exists() and any(rex_path.iterdir()):
                return str(rex_path)
        except Exception:
            pass

        # 3. HF cache
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        dirname = model_id.replace("/", "--")
        hf_dir = hf_cache / ("models--" + dirname)
        if hf_dir.exists():
            snapshots = hf_dir / "snapshots"
            if snapshots.exists() and any(snapshots.iterdir()):
                return model_id  # transformers resolves from HF cache automatically

        # 4. Check if model_id is itself a path
        if os.path.isdir(model_id):
            return model_id

        return None

    # Registry

    def scan(self) -> List[ModelEntry]:
        """Scan for all known, downloaded, and custom-path models."""
        from agent.cli.config import MODELS_DIR

        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"

        for model_id, info in KNOWN_MODELS.items():
            entry = ModelEntry(
                model_id=model_id,
                model_type=info.get("type", ModelType.TRANSFORMERS),
                purpose=info.get("purpose", ""),
                size_gb=info.get("size_gb", 0),
                requires_gpu=info.get("requires_gpu", True),
                notes=info.get("notes", ""),
            )

            # Check custom path first
            if model_id in self._custom_paths:
                p = self._custom_paths[model_id]
                if os.path.isdir(p):
                    entry.downloaded = True
                    entry.local_path = p

            # Check rexgraph cache
            if not entry.downloaded:
                dirname = model_id.replace("/", "--")
                rex_path = MODELS_DIR / dirname
                if rex_path.exists() and any(rex_path.iterdir()):
                    entry.downloaded = True
                    entry.local_path = str(rex_path)

            # Check HF cache
            if not entry.downloaded:
                dirname = model_id.replace("/", "--")
                hf_dir = hf_cache / ("models--" + dirname)
                snapshots = hf_dir / "snapshots"
                if snapshots.exists() and any(snapshots.iterdir()):
                    entry.downloaded = True
                    entry.hf_cache_path = str(hf_dir)

            self._registry[model_id] = entry

        # Add custom-path models not in KNOWN_MODELS
        for model_id, path in self._custom_paths.items():
            if model_id not in self._registry and os.path.isdir(path):
                self._registry[model_id] = ModelEntry(
                    model_id=model_id,
                    downloaded=True,
                    local_path=path,
                    notes="Custom path",
                )

        self._scan_done = True
        return list(self._registry.values())

    def list_models(self) -> List[Dict]:
        """List all models with status."""
        if not self._scan_done:
            self.scan()
        result = []
        for model_id, entry in self._registry.items():
            loaded = self._loaded.get(model_id)
            d = {
                "model_id": entry.model_id,
                "type": entry.model_type.value,
                "purpose": entry.purpose,
                "size_gb": entry.size_gb,
                "downloaded": entry.downloaded,
                "loaded": loaded is not None,
                "notes": entry.notes,
            }
            if loaded:
                d["vram_mb"] = loaded.vram_mb
                d["use_count"] = loaded.use_count
                d["loaded_at"] = loaded.loaded_at
                d["last_used"] = loaded.last_used
                d["device"] = loaded.device
                if loaded.server_url:
                    d["server_url"] = loaded.server_url
                    d["server_port"] = loaded.server_port
            result.append(d)
        return result

    # Loading

    def load(self, model_id: str, device: str = "auto",
             purpose: str = "") -> LoadedModel:
        """Load a model into VRAM. Returns cached instance if already loaded."""
        if model_id in self._loaded:
            lm = self._loaded[model_id]
            lm.last_used = time.time()
            return lm

        if not self._scan_done:
            self.scan()

        entry = self._registry.get(model_id)
        if entry is None:
            # Unknown model - create a minimal entry
            entry = ModelEntry(model_id=model_id, purpose=purpose)
            self._registry[model_id] = entry

        if entry.model_type == ModelType.TRANSFORMERS:
            return self._load_transformers(model_id, device, purpose or entry.purpose)
        elif entry.model_type in (ModelType.VLLM_SERVER, ModelType.SGLANG_SERVER):
            return self._load_server(model_id, entry.model_type, purpose or entry.purpose)
        elif entry.model_type == ModelType.CUSTOM:
            return self._load_custom(model_id, device, purpose or entry.purpose)
        else:
            raise ValueError("Unsupported model type: %s" % entry.model_type)

    def _load_transformers(self, model_id: str, device: str,
                           purpose: str) -> LoadedModel:
        """Load a HuggingFace transformers model."""
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM

        # Resolve path: custom -> rexgraph cache -> HF cache -> model_id
        model_path = self.get_model_path(model_id) or model_id
        logger.info("Loading transformers model: %s (from %s)", model_id, model_path)
        t0 = time.time()

        # Detect the right model class based on purpose / model ID
        if "got-ocr" in model_id.lower() or "GOT-OCR" in model_id:
            from transformers import GotOcr2ForConditionalGeneration
            processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True, local_files_only=True)
            model = GotOcr2ForConditionalGeneration.from_pretrained(
                model_path, trust_remote_code=True, device_map=device,
                local_files_only=True)
        else:
            processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, device_map=device,
                local_files_only=True)

        model.eval()

        # Estimate VRAM usage
        vram_mb = 0
        if torch.cuda.is_available():
            vram_mb = int(torch.cuda.memory_allocated() / (1024 * 1024))

        lm = LoadedModel(
            model_id=model_id,
            model_type=ModelType.TRANSFORMERS,
            purpose=purpose,
            model_obj=model,
            processor_obj=processor,
            device=device,
            vram_mb=vram_mb,
            loaded_at=time.time(),
            last_used=time.time(),
        )
        self._loaded[model_id] = lm

        elapsed = time.time() - t0
        logger.info("Loaded %s in %.1fs (%d MB VRAM)", model_id, elapsed, vram_mb)
        return lm

    def _load_server(self, model_id: str, model_type: ModelType,
                     purpose: str) -> LoadedModel:
        """Start a vLLM/SGLang server for a model."""
        from agent.cli.serve import serve, find_running_server, server_status

        # Check if already running
        existing = find_running_server()
        if existing:
            srv = server_status()
            lm = LoadedModel(
                model_id=model_id,
                model_type=model_type,
                purpose=purpose,
                server_port=srv.get("port", 10000),
                server_url=existing,
                server_pid=srv.get("pid", 0),
                loaded_at=time.time(),
                last_used=time.time(),
            )
            self._loaded[model_id] = lm
            return lm

        # Find a free port
        port = self._next_free_port()
        backend = "vllm" if model_type == ModelType.VLLM_SERVER else "sglang"
        ok = serve(port=port, model=model_id, backend=backend)

        if ok:
            lm = LoadedModel(
                model_id=model_id,
                model_type=model_type,
                purpose=purpose,
                server_port=port,
                server_url="http://localhost:%d" % port,
                loaded_at=time.time(),
                last_used=time.time(),
            )
            self._loaded[model_id] = lm
            return lm
        else:
            raise RuntimeError("Failed to start %s server for %s" % (backend, model_id))

    def _load_custom(self, model_id: str, device: str,
                     purpose: str) -> LoadedModel:
        """Load a model using a user-registered adapter."""
        adapter = self._custom_loaders.get(model_id)
        if adapter is None:
            raise ValueError("No adapter registered for %s" % model_id)

        path = self.get_model_path(model_id) or ""
        logger.info("Loading custom model: %s (adapter=%s)", model_id,
                     type(adapter).__name__)
        t0 = time.time()

        adapter.load(path, device)

        lm = LoadedModel(
            model_id=model_id,
            model_type=ModelType.CUSTOM,
            purpose=purpose,
            model_obj=adapter,
            device=device,
            loaded_at=time.time(),
            last_used=time.time(),
        )
        self._loaded[model_id] = lm

        elapsed = time.time() - t0
        logger.info("Loaded custom model %s in %.1fs", model_id, elapsed)
        return lm

    def _next_free_port(self) -> int:
        """Find the next free port for a server model."""
        used_ports = {lm.server_port for lm in self._loaded.values()
                      if lm.server_port > 0}
        port = 10000
        while port in used_ports:
            port += 1
        return port

    # Unloading

    def unload(self, model_id: str) -> bool:
        """Unload a model and free VRAM."""
        lm = self._loaded.pop(model_id, None)
        if lm is None:
            return False

        if lm.model_type == ModelType.TRANSFORMERS:
            # Free torch memory
            del lm.model_obj
            del lm.processor_obj
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.info("Unloaded transformers model: %s", model_id)

        elif lm.model_type == ModelType.CUSTOM:
            # Call the adapter's unload method
            if isinstance(lm.model_obj, ModelAdapter):
                try:
                    lm.model_obj.unload()
                except Exception:
                    pass
            del lm.model_obj
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.info("Unloaded custom model: %s", model_id)

        elif lm.model_type in (ModelType.VLLM_SERVER, ModelType.SGLANG_SERVER):
            # Stop the server process
            if lm.server_pid:
                try:
                    import signal
                    os.kill(lm.server_pid, signal.SIGTERM)
                except (ProcessLookupError, OSError):
                    pass
            logger.info("Stopped server model: %s (port %d)", model_id, lm.server_port)

        return True

    def unload_all(self):
        """Unload all models."""
        for model_id in list(self._loaded.keys()):
            self.unload(model_id)

    # Access

    def get(self, model_id: str) -> Optional[LoadedModel]:
        """Get a loaded model. Does NOT auto-load."""
        lm = self._loaded.get(model_id)
        if lm:
            lm.last_used = time.time()
            lm.use_count += 1
        return lm

    def get_or_load(self, model_id: str, **kwargs) -> LoadedModel:
        """Get a loaded model or load it on demand."""
        lm = self.get(model_id)
        if lm:
            return lm
        return self.load(model_id, **kwargs)

    def get_by_purpose(self, purpose: str) -> Optional[LoadedModel]:
        """Get the first loaded model matching a purpose (ocr, chat, etc.)."""
        for lm in self._loaded.values():
            if lm.purpose == purpose:
                lm.last_used = time.time()
                lm.use_count += 1
                return lm
        return None

    # Status

    def status(self) -> Dict:
        """Full status of all models."""
        if not self._scan_done:
            self.scan()
        loaded = []
        available = []
        for model_id, entry in self._registry.items():
            lm = self._loaded.get(model_id)
            info = {
                "model_id": model_id,
                "type": entry.model_type.value,
                "purpose": entry.purpose,
                "size_gb": entry.size_gb,
                "downloaded": entry.downloaded,
            }
            if lm:
                info["loaded"] = True
                info["vram_mb"] = lm.vram_mb
                info["use_count"] = lm.use_count
                info["uptime_s"] = int(time.time() - lm.loaded_at) if lm.loaded_at else 0
                if lm.server_url:
                    info["server_url"] = lm.server_url
                loaded.append(info)
            elif entry.downloaded:
                info["loaded"] = False
                available.append(info)

        return {
            "loaded": loaded,
            "available": available,
            "n_loaded": len(loaded),
            "n_available": len(available),
            "vram_total_mb": sum(lm.vram_mb for lm in self._loaded.values()),
        }

    def download(self, model_id: str, callback: Optional[Callable] = None) -> str:
        """Download a model from HuggingFace. Returns local path."""
        from agent.cli.config import MODELS_DIR
        from huggingface_hub import snapshot_download

        dirname = model_id.replace("/", "--")
        local_path = MODELS_DIR / dirname
        local_path.mkdir(parents=True, exist_ok=True)

        logger.info("Downloading model: %s", model_id)
        snapshot_download(
            model_id,
            local_dir=str(local_path),
            local_dir_use_symlinks=False,
        )

        # Update registry
        entry = self._registry.get(model_id)
        if entry:
            entry.downloaded = True
            entry.local_path = str(local_path)

        logger.info("Downloaded %s to %s", model_id, local_path)
        return str(local_path)


# Singleton

_manager: Optional[ModelManager] = None


def get_manager() -> ModelManager:
    """Get the singleton ModelManager instance."""
    global _manager
    if _manager is None:
        _manager = ModelManager()
    return _manager

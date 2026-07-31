"""
agent.chat_model: unified generation layer for chat/synthesis.

A single place that resolves *which* model to talk to and provides a
best-effort ``generate()``. Everything that wants an LLM (the chat route,
the pipeline's query synthesis, the query engine) goes through here, so
"model setup" is one concept instead of four copies of the resolution
chain.

Resolution order (first hit wins):
    1. an explicit runtime override set via ``configure()``
    2. ModelManager's pipeline model for the "chat" role (Models tab)
    3. ``CHAT_MODEL_URL`` env var
    4. a locally running GPU server (``find_running_server``)
    5. ``UNLIMITED_OCR_URL`` env var

If none resolve, ``is_available()`` is False and ``generate()`` returns
None - callers fall back to the structural (LLM-free) answer path.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Optional, List, Dict

# Runtime override set through configure() / the model-config endpoint.
_override_lock = threading.Lock()
_override: Dict[str, str] = {}


@dataclass
class ModelTarget:
    """A resolved generation target."""
    url: str = ""
    model: str = ""
    source: str = ""          # where the target came from (for the UI)
    api_key: str = ""

    @property
    def available(self) -> bool:
        return bool(self.url)


def configure(url: Optional[str] = None, model: Optional[str] = None,
              api_key: Optional[str] = None) -> None:
    """Set a runtime override for the chat model (from the setup UI).

    Passing ``url=""`` clears the override and falls back to the
    resolution chain.
    """
    with _override_lock:
        if url is not None:
            if url:
                _override["url"] = url.rstrip("/")
            else:
                _override.pop("url", None)
        if model is not None:
            _override["model"] = model
        if api_key is not None:
            if api_key:
                _override["api_key"] = api_key
            else:
                _override.pop("api_key", None)


def _resolve() -> ModelTarget:
    with _override_lock:
        ov = dict(_override)
    if ov.get("url"):
        return ModelTarget(url=ov["url"], model=ov.get("model", ""),
                           api_key=ov.get("api_key", ""), source="configured")

    # ModelManager pipeline model for the chat role.
    try:
        from agent.model_manager import get_manager
        mgr = get_manager()
        chat_id = mgr.get_pipeline_model("chat")
        if chat_id:
            lm = mgr.get(chat_id)
            if lm and getattr(lm, "server_url", ""):
                return ModelTarget(url=lm.server_url.rstrip("/"), model=chat_id,
                                   source="model_manager")
    except Exception:
        pass

    url = os.environ.get("CHAT_MODEL_URL", "").rstrip("/")
    if url:
        return ModelTarget(url=url, model=os.environ.get("CHAT_MODEL_NAME", ""),
                           source="env:CHAT_MODEL_URL")

    try:
        from agent.cli.serve import find_running_server
        running = find_running_server()
        if running:
            return ModelTarget(url=running.rstrip("/"),
                               model=os.environ.get("CHAT_MODEL_NAME", ""),
                               source="gpu_server")
    except Exception:
        pass

    url = os.environ.get("UNLIMITED_OCR_URL", "").rstrip("/")
    if url:
        return ModelTarget(url=url, model=os.environ.get("CHAT_MODEL_NAME", ""),
                           source="env:UNLIMITED_OCR_URL")

    return ModelTarget()


def target() -> ModelTarget:
    """The currently resolved generation target (may be unavailable)."""
    return _resolve()


def is_available() -> bool:
    return _resolve().available


def status() -> dict:
    """Model-setup status for the UI (no secrets leaked)."""
    t = _resolve()
    return {
        "available": t.available,
        "source": t.source or "none",
        "model": t.model or None,
        # host only, never the full URL with any embedded credentials
        "endpoint": (t.url.split("://")[-1].split("/")[0] if t.url else None),
        "has_api_key": bool(t.api_key),
    }


def generate(prompt: str, system: Optional[str] = None,
             max_tokens: int = 512, temperature: float = 0.3,
             timeout: float = 120.0) -> Optional[str]:
    """Generate a completion, or return None if no model is available.

    Synchronous and dependency-light so it works from any context
    (async routes, sync tests, the pipeline). Uses the OpenAI-compatible
    ``/v1/chat/completions`` contract the GPU server speaks.
    """
    t = _resolve()
    if not t.available:
        return None
    try:
        import httpx
    except Exception:
        return None

    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload: Dict = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if t.model:
        payload["model"] = t.model
    headers = {}
    if t.api_key:
        headers["Authorization"] = f"Bearer {t.api_key}"

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(f"{t.url}/v1/chat/completions",
                               json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        text = ""
        for choice in data.get("choices", []):
            text += choice.get("message", {}).get("content", "")
        text = text.strip()
        if not text:
            return None
        try:
            from agent.server.security import sanitize_model_response
            text = sanitize_model_response(text)
        except Exception:
            pass
        return text
    except Exception:
        return None


def generate_with_metrics(prompt: str, system: Optional[str] = None,
                          max_tokens: int = 512, temperature: float = 0.3,
                          timeout: float = 120.0) -> Optional[Dict]:
    """Generate a completion AND its token-level LLM metrics (perplexity, mean
    surprisal, varentropy) from the model's logprobs - the standard LLM metrics,
    computed with the same Rényi/varentropy calculus as the structural metrics
    (``agent.metrics``). Returns ``{'text': str, 'metrics': dict}`` or None; 'metrics'
    is empty if the server did not return logprobs (older/limited backends)."""
    t = _resolve()
    if not t.available:
        return None
    try:
        import httpx
    except Exception:
        return None

    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload: Dict = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
        "logprobs": True,          # OpenAI-compatible: per-token logprobs
    }
    if t.model:
        payload["model"] = t.model
    headers = {}
    if t.api_key:
        headers["Authorization"] = f"Bearer {t.api_key}"

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(f"{t.url}/v1/chat/completions",
                               json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        text, logprobs = "", []
        for choice in data.get("choices", []):
            text += choice.get("message", {}).get("content", "")
            lp = choice.get("logprobs") or {}
            for tok in (lp.get("content") or []):
                v = tok.get("logprob")
                if v is not None:
                    logprobs.append(float(v))
        text = text.strip()
        if not text:
            return None
        try:
            from agent.server.security import sanitize_model_response
            text = sanitize_model_response(text)
        except Exception:
            pass
        from agent.metrics import token_metrics
        return {"text": text, "metrics": token_metrics(logprobs) if logprobs else {}}
    except Exception:
        return None

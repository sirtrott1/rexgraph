"""
Server-Sent Events (SSE) streaming for progressive analysis results.

The pipeline computes stages sequentially. Each stage emits an SSE event
so the frontend can update as results arrive rather than waiting for
the full computation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import concurrent.futures
from typing import AsyncGenerator

from agent.pipeline import AnalysisPipeline


_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)


class _SafeEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


async def stream_pipeline(pipeline: AnalysisPipeline, depth: str = "standard") -> AsyncGenerator[str, None]:
    """Yield SSE events as analysis stages complete.

    Each event has the format:
        event: stage
        data: {"stage": "topology", "results": {...}}

    Final event:
        event: done
        data: {}
    """
    queue: asyncio.Queue = asyncio.Queue()

    def on_stage(name: str, data: dict):
        queue.put_nowait((name, data))

    pipeline.on_stage(on_stage)

    loop = asyncio.get_event_loop()

    # Run the pipeline in a thread (Cython is CPU-bound, can't be async)
    future = loop.run_in_executor(_executor, pipeline.run, depth)

    # Yield events as they arrive
    stages_expected = len(getattr(pipeline, f"STAGES_{depth.upper()}", pipeline.STAGES_STANDARD))
    stages_received = 0

    while stages_received < stages_expected:
        try:
            name, data = await asyncio.wait_for(queue.get(), timeout=120.0)
            payload = json.dumps({"stage": name, "results": data}, cls=_SafeEncoder)
            yield f"event: stage\ndata: {payload}\n\n"
            stages_received += 1
        except asyncio.TimeoutError:
            yield f"event: error\ndata: {{\"error\": \"Stage timeout\"}}\n\n"
            break

    # Wait for the pipeline to finish fully
    try:
        await future
    except Exception as e:
        # Log the detail server-side; return a generic, properly-escaped message
        # (never interpolate str(e) into the SSE frame - it leaks internals and
        # breaks the JSON when the message contains quotes/newlines).
        logging.getLogger(__name__).exception("Pipeline stream failed")
        payload = json.dumps({"error": "Analysis failed"})
        yield f"event: error\ndata: {payload}\n\n"

    yield f"event: done\ndata: {{}}\n\n"

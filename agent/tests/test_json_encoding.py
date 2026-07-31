"""The server's JSON surfaces share the one encoder and never emit invalid JSON.

stream/explore/chat each carried their own copy that did `float(o)` for np.floating,
and none of them ran for np.float64 at all (float subclass), so a NaN metric went out
as a bare `NaN` token. Browsers reject that: JSON.parse throws, the SSE handler dies
mid-analysis, and the route 500s. integrations already returned null for non-finite
values -- that is the policy the wire surfaces now share.
"""

import json

import numpy as np
import pytest


def _strict_loads(text):
    def _reject(token):
        raise ValueError(f"not valid JSON: bare {token} token")
    return json.loads(text, parse_constant=_reject)


PAYLOAD = {"kappa": np.float64("nan"), "ratio": float("inf"),
           "small": np.float32("nan"), "arr": np.array([1.0, np.nan]),
           "n": np.int64(7), "ok": np.bool_(True)}


def test_sse_stream_encoder_emits_strict_json():
    from agent.server.stream import _encode

    back = _strict_loads(_encode(dict(PAYLOAD)))
    assert back["kappa"] is None and back["ratio"] is None
    assert back["arr"] == [1.0, None]
    assert back["n"] == 7 and back["ok"] is True


@pytest.mark.parametrize("module", ["agent.server.routes.explore",
                                    "agent.server.routes.chat",
                                    "agent.server.routes.integrations"])
def test_route_encoders_are_the_shared_one(module):
    import importlib

    from rexgraph.io._compat import json_sanitize

    mod = importlib.import_module(module)
    fn = getattr(mod, "_sanitize", None)
    assert fn is not None, f"{module} has no _sanitize"
    out = fn(dict(PAYLOAD))
    assert out["kappa"] is None and out["ratio"] is None
    assert out["arr"] == [1.0, None]
    # and it is genuinely the shared implementation, not another near-copy
    assert json_sanitize(dict(PAYLOAD), nan="null") == out


def test_dashboard_payload_survives_a_nonfinite_metric(tmp_path):
    """allow_nan=False turned a float64 NaN into a ValueError that killed the whole
    dashboard render rather than one metric."""
    from rexgraph.viz.dashboard import _inject_data

    jsx = "const DATA = /*__REX_DATA__*/null;"
    out = _inject_data(jsx, {"kappa": np.float64("nan"), "beta": [1, 2]})
    payload = out[out.index("=") + 1:].rstrip(";").strip()
    assert "NaN" not in payload and "Infinity" not in payload
    assert _strict_loads(payload)["kappa"] == 0.0

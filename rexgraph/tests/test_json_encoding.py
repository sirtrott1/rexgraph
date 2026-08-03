"""One JSON encoder, one NaN policy, applied everywhere JSON is written.

The io layer grew nine independent numpy-JSON encoders with four different NaN
policies, and none of them worked for float64. np.float64 subclasses Python float,
so json.dumps serializes it directly and never consults default() -- the encoder
that documents "NaN/Inf become 0" silently emitted a bare NaN token instead, which
is not JSON and no strict reader will parse.
"""

import json
import os

import numpy as np
import pytest

from rexgraph.graph import RexGraph


def _strict_loads(text):
    """json.loads accepts NaN/Infinity as a Python extension; real readers do not."""
    def _reject(token):
        raise ValueError(f"not valid JSON: bare {token} token")
    return json.loads(text, parse_constant=_reject)


def _rex():
    return RexGraph(sources=np.array([0, 1, 2], np.int32),
                    targets=np.array([1, 2, 0], np.int32))


NONFINITE = {
    "f64_nan": np.float64("nan"), "f64_inf": np.float64("inf"),
    "f64_ninf": np.float64("-inf"), "f32_nan": np.float32("nan"),
    "py_nan": float("nan"), "py_inf": float("inf"),
}


def test_nan_policy_applies_to_every_float_width_not_just_float32():
    """np.float64 is a float subclass, so it bypasses JSONEncoder.default entirely.
    The policy has to be enforced before dumps, not only in the fallback."""
    from rexgraph.io._compat import dumps

    text = dumps(dict(NONFINITE))
    assert "NaN" not in text and "Infinity" not in text, text
    back = _strict_loads(text)
    assert set(back) == set(NONFINITE)


def test_nan_policies_are_selectable_and_honest():
    from rexgraph.io._compat import dumps

    assert _strict_loads(dumps({"x": float("nan")}, nan="zero"))["x"] == 0.0
    assert _strict_loads(dumps({"x": float("nan")}, nan="null"))["x"] is None
    with pytest.raises(ValueError):
        dumps({"x": float("nan")}, nan="raise")


def test_nested_containers_and_arrays_are_covered():
    from rexgraph.io._compat import dumps

    payload = {"a": [float("nan"), {"b": (np.float64("inf"), 1.5)}],
               "arr": np.array([1.0, np.nan, np.inf])}
    text = dumps(payload, nan="null")
    back = _strict_loads(text)
    assert back["a"][0] is None
    assert back["a"][1]["b"] == [None, 1.5]
    assert back["arr"] == [1.0, None, None]


def test_numpy_dict_keys_survive():
    """np.int64 is not an int subclass; an unsanitized numpy key is a TypeError."""
    from rexgraph.io._compat import dumps

    assert _strict_loads(dumps({np.int64(3): 1.0})) == {"3": 1.0}


def test_a_bundle_manifest_is_always_strict_json(tmp_path):
    """A .rex bundle carrying a NaN metric wrote a MANIFEST.json that JSON.parse
    rejects -- the GUI and every external consumer choke on it."""
    from rexgraph.io.bundle import save_rex

    g = _rex()
    g._agent_meta = {"score": np.float64("nan"), "ratio": float("inf")}
    save_rex(str(tmp_path / "g.rex"), g)

    seen = 0
    for root, _, files in os.walk(tmp_path):
        for f in files:
            if f.endswith(".json"):
                seen += 1
                _strict_loads(open(os.path.join(root, f)).read())
    assert seen, "no manifest was written, the test proves nothing"


def test_every_json_writer_shares_the_one_encoder():
    """Nine copies of the same twelve lines is nine places for the policy to drift.

    Checks the property, not the plumbing: no io module may DEFINE its own encoder,
    and each must route its writes through _compat.dumps. An earlier version of this
    test asserted a back-compat alias instead, and so failed when the (by then dead)
    alias was cleaned up -- while the property it cared about still held.
    """
    import inspect

    import rexgraph.io.arrow_bridge as arrow
    import rexgraph.io.bundle as bundle
    import rexgraph.io.parquet_bridge as parquet
    import rexgraph.io.sql_bridge as sql
    from rexgraph.io._compat import dumps

    for mod in (bundle, parquet, arrow, sql):
        own = getattr(mod, "_json_default", None)
        assert not inspect.isfunction(own) or own.__module__ == "rexgraph.io._compat", (
            f"{mod.__name__} defines its own encoder again")
        assert getattr(mod, "_dumps", None) is dumps, (
            f"{mod.__name__} does not write through the shared dumps")


def test_no_module_reintroduces_a_raw_numpy_dumps():
    """The nine copies came back one call site at a time. `default=` cannot enforce a
    NaN policy on a float subclass, so any new use of it is the same bug returning."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    banned = ("default=json_default", "default=_json_default", "cls=NumpyJSONEncoder")
    offenders, scanned = [], 0
    for py in list(root.glob("rexgraph/**/*.py")) + list(root.glob("agent/agent/**/*.py")):
        if "tests" in py.parts or py.name == "_compat.py":
            continue
        scanned += 1
        text = py.read_text(encoding="utf-8", errors="ignore")
        for line_no, line in enumerate(text.splitlines(), 1):
            if any(b in line for b in banned):
                offenders.append(f"{py.relative_to(root)}:{line_no}")
    assert scanned > 100, f"only scanned {scanned} files, the glob is wrong"
    assert not offenders, (
        "use rexgraph.io._compat.dumps instead:\n  " + "\n  ".join(offenders))

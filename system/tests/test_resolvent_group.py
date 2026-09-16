"""Group previews are bounded and do not solve or traverse source state."""
import json

from rexgraph import RexGraph
from rexgraph.channel_operator import channel_operator
from rexgraph.rational_operator import ResolventGroup
from system.serialize import json_value


def test_bounded_group_preview(monkeypatch):
    r = RexGraph.from_cells([3, [[0, 1], [1, 2]]])
    group = ResolventGroup([channel_operator(r, "T")], [1], [1, 1, 1])
    monkeypatch.setattr(ResolventGroup, "apply", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("solve ran")))
    out = json.loads(json.dumps(json_value(group, max_values=1)))
    assert out["kind"] == "ResolventGroup" and out["word"] == [1]
    assert out["word_length"] == 3 and out["truncated"] and "source" not in out

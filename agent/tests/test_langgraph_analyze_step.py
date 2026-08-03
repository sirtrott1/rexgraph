"""langgraph_analyze reports its diagnostics on a successful analysis.

should_continue, cycle detection and path decomposition read the complex rather than
the analysis result, so they belong outside the analysis fallback. Indented into it,
they ran only when rsg.analyze() raised, which meant a healthy graph returned none of
them and the unreachable second handler never recorded an error either.
"""

from __future__ import annotations

import pytest
from agent.builder import _step_langgraph_analyze, _step_langgraph_init


@pytest.fixture
def state():
    st: dict = {}
    _step_langgraph_init([], st, {})
    rsg = st.get("state_graph")
    if rsg is None:
        pytest.skip("langgraph_init did not produce a state graph")
    for a, b in (("start", "load"), ("load", "work"), ("work", "check"),
                 ("check", "work"), ("check", "done")):
        for fn in ("add_transition", "transition", "record_transition"):
            f = getattr(rsg, fn, None)
            if callable(f):
                try:
                    f(a, b)
                    break
                except Exception:
                    continue
    return st


def test_the_step_reports_analysis(state):
    out = _step_langgraph_analyze([], state, {})
    assert "skipped" not in out
    assert "analysis" in out


def test_should_continue_is_reported_on_the_success_path(state):
    """The regression: present only when analyze() raised."""
    out = _step_langgraph_analyze([], state, {})
    assert "should_continue" in out or "error" in out


def test_cycles_are_reported_on_the_success_path(state):
    out = _step_langgraph_analyze([], state, {})
    assert "cycles" in out or "error" in out


def test_the_diagnostics_travel_together(state):
    """Either both structural reads happened or the failure was recorded. Silently
    returning an analysis with neither is the shape the indentation produced."""
    out = _step_langgraph_analyze([], state, {})
    assert ("should_continue" in out and "cycles" in out) or "error" in out


def test_result_lands_on_the_state(state):
    _step_langgraph_analyze([], state, {})
    assert "graph_analysis" in state


def test_no_state_graph_is_skipped_not_an_error():
    assert "skipped" in _step_langgraph_analyze([], {}, {})

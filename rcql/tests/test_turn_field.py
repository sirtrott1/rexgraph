"""Conversation source contracts and numerical preview provenance."""
import pytest
from rcql import BoundSource, Executor, SourcePolicy, parse
from rexgraph.flow.turn_field import TurnField


@pytest.fixture
def field():
    field = TurnField()
    field.observe("alpha beta gamma")
    field.observe("gamma delta")
    return field


def test_snapshot_and_preview_are_read_only(field):
    result = Executor(sources={"chat": field}).execute(parse(
        'FROM $chat LET p=PATH_CHANGE("alpha delta") '
        'RETURN TURN_FIELD(),p,p.event,p.status,p.n_turns,p.baseline_turns'))
    assert result.values[0].T == 2
    assert result.values[1] == field.preview("alpha delta")
    assert result.values[-2:] == (3, 2) and field.n_turns == 2
    assert field.observe("alpha delta") == result.values[1]


def test_rebind_capture_for_native_temporal_reads(field):
    history = Executor(sources={"chat": field}).execute(parse('FROM $chat RETURN TURN_FIELD(1,2)')).values[0]
    result = Executor(sources={"history": history}).execute(parse('FROM AT($history,0) RETURN COUNT(CELLS(1))'))
    assert result.values == (2,)


def test_explain_does_not_capture_or_preview(field, monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("runtime read during explain")
    monkeypatch.setattr(field, "snapshot", fail)
    monkeypatch.setattr(field, "preview", fail)
    assert not Executor(sources={"chat": field}).execute(parse(
        'EXPLAIN FROM $chat RETURN TURN_FIELD(),PATH_CHANGE("candidate")')).execution


@pytest.mark.parametrize("expression", ['TURN_FIELD(-1)', 'TURN_FIELD(0,3)',
    'TURN_FIELD(true)', 'TURN_FIELD(1,0)', 'PATH_CHANGE(3)', 'PATH_CHANGE("x").private'])
def test_bad_arguments_are_refused(field, expression, monkeypatch):
    monkeypatch.setattr(field, "preview", lambda *args: pytest.fail("preview called"))
    with pytest.raises((TypeError, ValueError)):
        Executor(sources={"chat": field}).execute(parse(f'FROM $chat RETURN {expression}'))


@pytest.mark.parametrize("missing", ["read", "identity", "history", "agent_read"])
@pytest.mark.parametrize("expression", ['TURN_FIELD()', 'PATH_CHANGE("x")'])
def test_capabilities(field, missing, expression):
    allowed = {"read", "identity", "history", "agent_read"} - {missing}
    source = BoundSource(field, SourcePolicy.allow(*allowed))
    with pytest.raises(PermissionError):
        Executor(sources={"chat": source}).execute(parse(f'FROM $chat RETURN {expression}'))

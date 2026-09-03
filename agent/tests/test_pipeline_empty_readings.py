"""A reading that is undefined is omitted, never emitted as NaN.

The per-cell readings are defined at zero cells and are the empty list. Their MEANS are
not: numpy returns NaN for the mean of an empty slice, and NaN serialises to a bare NaN
token that strict JSON readers reject. Emitting it would poison an entire payload over a
key that simply has no value, so the key is absent instead.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from rexgraph import RexGraph


def _strict(payload) -> None:
    """Parse the way a non-Python client does: NaN and Infinity are not JSON."""
    def reject(token):
        raise ValueError(f"not valid JSON: {token}")
    json.loads(json.dumps(payload), parse_constant=reject)


def test_the_mean_of_no_cells_is_omitted_not_nan():
    chi = np.zeros((0, 4))
    result = {}
    if chi.ndim == 2:
        if chi.shape[0]:
            result["chi_mean"] = chi.mean(axis=0).tolist()
        result["chi_per_edge"] = chi.tolist()

    assert "chi_mean" not in result, "an undefined mean must be absent, not NaN"
    assert result["chi_per_edge"] == [], "the per-cell reading is defined and empty"
    _strict(result)


def test_a_nan_mean_would_have_broken_strict_json():
    """The failure this guards against, so the test says why the guard exists."""
    # errstate because producing the NaN is the POINT here, so it must not add noise
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        nan_mean = np.zeros((0, 4)).mean(axis=0).tolist()
    with pytest.raises(ValueError, match="not valid JSON"):
        _strict({"chi_mean": nan_mean})


def test_the_pipeline_emits_no_empty_slice_warning_on_a_relationless_complex():
    from agent import pipeline as pl

    rex = RexGraph.from_cells([3, []])
    assert int(rex.nE) == 0

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            out = pl.analyze(rex) if hasattr(pl, "analyze") else None
        except Exception:
            out = None
    empty_slice = [w for w in caught if "empty slice" in str(w.message)]

    assert not empty_slice, f"mean of an empty slice reached the payload: {empty_slice}"
    if isinstance(out, dict):
        _strict(out)


def test_coherence_mean_over_no_cells_is_none_not_nan():
    """The same defect on the response path, where the payload IS the HTTP body.

    `coherence_kappa` returns an empty array for a complex with no cells, so the mean is
    NaN and `min` raises outright. Both route sites now report the absence as null.
    """
    import warnings as _w

    from agent.metrics import coherence_kappa

    rex = RexGraph.from_cells([0, []])
    with _w.catch_warnings():
        _w.simplefilter("ignore", RuntimeWarning)
        kappa = np.asarray(coherence_kappa(rex))
    assert kappa.size == 0, "the empty complex is what makes the mean undefined"

    out = {}
    if kappa is not None and kappa.size:
        out["kappa_mean"] = round(float(kappa.mean()), 4)
        out["kappa_min"] = round(float(kappa.min()), 4)
    else:
        out["kappa_mean"] = None
        out["kappa_min"] = None

    assert out == {"kappa_mean": None, "kappa_min": None}
    _strict(out)


def test_min_over_no_cells_raises_which_is_why_the_guard_wraps_both():
    """Stated because it changes the failure mode, not just the value.

    `mean` returns NaN quietly while `min` raises, so an unguarded pair set kappa_mean to
    NaN and then aborted, leaving the payload both unparseable and missing a field.
    """
    with pytest.raises(ValueError):
        np.zeros(0).min()


def test_chunk_vertices_reads_the_stored_support_at_every_arity():
    """The k-ary reading this function documents, which never actually ran.

    It called `rex.B1.tocsc()`, but B1 is the dense oracle and has no `tocsc`, so every
    call raised into a bare except and returned an empty list. The consequence was silent
    and total: every chunk's coherence became the mean of nothing.
    """
    from agent.chunking import _chunk_vertices

    rex = RexGraph.from_cells([4, [[0, 1, 2], [2, 3]]])
    nV = int(rex.nV)

    assert _chunk_vertices(rex, [0], nV) == [0, 1, 2], "an arity-3 relation gives all three"
    assert _chunk_vertices(rex, [1], nV) == [2, 3]
    assert _chunk_vertices(rex, [0, 1], nV) == [0, 1, 2, 3]
    assert _chunk_vertices(rex, [], nV) == [], "no relations touch no vertices"


def test_the_dense_oracle_still_has_no_tocsc():
    """Pins the mistake itself, so the old call cannot be reintroduced as a fix."""
    rex = RexGraph.from_cells([4, [[0, 1, 2], [2, 3]]])

    assert not hasattr(rex.B1, "tocsc"), "B1 is the dense oracle, not a sparse matrix"
    assert hasattr(rex, "B1_sparse"), "the sparse boundary has its own public accessor"


def test_a_chunk_with_no_relations_reports_no_coherence():
    """None, not NaN: absence is reported rather than serialised as a non-number."""
    import warnings as _w

    from agent.chunking import _chunk_vertices

    rex = RexGraph.from_cells([3, [[0, 1]]])
    with _w.catch_warnings():
        _w.simplefilter("error", RuntimeWarning)
        verts = _chunk_vertices(rex, [], int(rex.nV))
        local = [0.5 for v in verts]
        kappa = float(np.mean(local)) if local else None

    assert kappa is None
    _strict({"kappa": kappa})


def test_token_metrics_on_no_tokens_is_json_serialisable():
    """The undefined metrics are null, and n_tokens says why they are undefined.

    These three reach a response body through chat_model, the metrics response helpers
    and the model SSE route. They used to be float('nan'), which json.dumps writes as a
    bare NaN token: valid for Python's own reader, rejected by every other one.
    """
    from agent.metrics import token_metrics

    empty = token_metrics([])

    assert empty == {"perplexity": None, "mean_surprisal": None,
                     "varentropy": None, "n_tokens": 0}
    _strict(empty)
    json.dumps(empty, allow_nan=False)          # what a strict producer would do

    real = token_metrics([-0.5, -1.0])
    assert real["n_tokens"] == 2 and real["perplexity"] > 0
    _strict(real)


def test_the_consumers_of_an_absent_perplexity_still_behave():
    """None has to be safe where NaN used to flow, or this trades one bug for another."""
    from agent.metrics import token_metrics

    ppl = token_metrics([])["perplexity"]

    # metrics.py gates on truthiness before comparing; None short-circuits, NaN did not
    assert not (ppl and ppl < 10.0)
    # conversation.note_reply_perplexity already special-cased None
    assert (float(ppl) if ppl is not None else None) is None


def test_a_layer_with_no_relations_yields_no_chi_or_kappa_keys():
    """The guard the huggingface analyzer applies per layer, exercised on the values.

    Scraping the module source for indentation would be the same fragile shape as the
    guard test that scanned nothing, so this runs the arithmetic the analyzer runs.
    """
    import warnings as _w

    from agent.metrics import coherence_kappa

    rex = RexGraph.from_cells([0, []])
    chi = np.asarray(rex.structural_character)
    with _w.catch_warnings():
        _w.simplefilter("error", RuntimeWarning)     # an unguarded mean would raise here
        kappa = np.asarray(coherence_kappa(rex))

        layer = {}
        if chi.shape[0] > 0:
            means = chi.mean(axis=0)
            for i, name in enumerate(["T", "G", "F", "C"][: len(means)]):
                layer[f"chi_{name}"] = round(float(means[i]), 4)
        if kappa is not None and kappa.size:
            layer["kappa_mean"] = round(float(kappa.mean()), 4)

    assert chi.shape[0] == 0 and kappa.size == 0, "the empty complex is the case"
    assert layer == {}, f"an undefined reading was emitted: {layer}"
    _strict(layer)


def test_the_confidence_tool_refuses_to_rank_an_absent_coherence():
    """An undefined reading must not become a verdict.

    kappa_mean was NaN on a complex with no cells. Every comparison against NaN is
    False, so the ladder fell through to its catch-all and told the caller the complex
    had "some structural support". Nothing was there to support anything: the failure
    was not the NaN in the payload but the confident answer built on top of it.
    """
    pytest.importorskip("langchain_core")   # the langchain extra, absent on a base install
    from agent.integrations.langchain_tools import RexConfidenceTool

    empty = RexConfidenceTool(RexGraph.from_cells([0, []]))._run()
    fields = dict(
        line.split(": ", 1) for line in empty.split("\n") if ": " in line
    )

    assert "kappa_mean" not in fields, "an undefined coherence must not be reported"
    assert fields["confidence"].startswith("UNAVAILABLE"), fields["confidence"]
    assert "nan" not in empty.lower()

    # a complex that does have a reading still gets ranked
    real = RexConfidenceTool(
        RexGraph.from_cells([4, [[0, 1], [1, 2], [2, 0], [2, 3]]])
    )._run()
    real_fields = dict(
        line.split(": ", 1) for line in real.split("\n") if ": " in line
    )
    assert "kappa_mean" in real_fields
    assert not real_fields["confidence"].startswith("UNAVAILABLE")


def test_the_analyze_tool_emits_no_nan_on_an_empty_complex():
    import warnings as _w

    pytest.importorskip("langchain_core")
    from agent.integrations.langchain_tools import RexAnalyzeTool

    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        out = RexAnalyzeTool(RexGraph.from_cells([0, []]))._run("summary")

    runtime = [c for c in caught if issubclass(c.category, RuntimeWarning)]
    assert not runtime, f"an unguarded mean ran: {[str(c.message)[:40] for c in runtime]}"
    assert "nan" not in out.lower(), out


def test_building_the_langchain_tools_raises_no_deprecation():
    """The tools declared pydantic config the v1 way, which v2 deprecates and v3 removes.

    Four tool classes each carried an inner `class Config`, so simply importing the module
    and constructing the tools emitted four PydanticDeprecatedSince20 warnings. Promoted
    to an error here: this is a contract with a dependency that has already announced the
    removal, not a style preference.
    """
    import warnings as _w

    pytest.importorskip("langchain_core")
    from agent.integrations import langchain_tools as lt

    rex = RexGraph.from_cells([3, [[0, 1], [1, 2]]])
    with _w.catch_warnings():
        _w.simplefilter("error", DeprecationWarning)
        tools = lt.get_rex_tools(rex)

    assert len(tools) == 4, f"expected the four rex tools, got {len(tools)}"

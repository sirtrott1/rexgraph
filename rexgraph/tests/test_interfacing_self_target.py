"""Self-interfacing in one solve.

Reading a query's induced flow through the typed channels means scoring psi against
itself, but `target_signal` had to be supplied by the caller and psi is computed
inside the bundle. Callers therefore ran the bundle twice -- once with a throwaway
zero target purely to obtain psi, then again with it -- paying two L0^+ solves for
one reading. That is the dominant per-candidate cost in store-backed retrieval.

`target_signal=None` now means "score psi against itself", which the bundle can do
with the psi it already has.
"""

import numpy as np

from rexgraph.graph import RexGraph


def _graph(nV=40, extra=30, seed=0):
    rng = np.random.default_rng(seed)
    src = list(range(nV - 1))
    tgt = list(range(1, nV))
    for _ in range(extra):
        a, b = int(rng.integers(nV)), int(rng.integers(nV))
        if a != b:
            src.append(a)
            tgt.append(b)
    return RexGraph(sources=np.array(src, np.int32), targets=np.array(tgt, np.int32))


def _targets(rex, k=5):
    return np.arange(k, dtype=np.int32), np.ones(k, dtype=np.float64)


def test_none_target_equals_passing_psi_back_in():
    """The contract: the one-solve path must give exactly the answer the two-call
    dance gave, not merely a similar one."""
    rex = _graph()
    ti, tw = _targets(rex)

    psi = np.asarray(rex.interfacing_vector(
        ti, tw, np.zeros(int(rex.nE), dtype=np.float64))["psi"], dtype=np.float64)
    two_call = rex.interfacing_vector(ti, tw, psi)
    one_call = rex.interfacing_vector(ti, tw, None)

    assert np.allclose(one_call["iv"], two_call["iv"], rtol=1e-9, atol=1e-12)
    assert np.allclose(one_call["sphere_pos"], two_call["sphere_pos"],
                       rtol=1e-9, atol=1e-12)
    assert np.allclose(one_call["psi"], two_call["psi"], rtol=1e-9, atol=1e-12)
    assert np.isclose(one_call["coverage"], two_call["coverage"], rtol=1e-9)


def test_none_target_still_returns_the_whole_bundle():
    rex = _graph()
    ti, tw = _targets(rex)
    iv = rex.interfacing_vector(ti, tw, None)
    for key in ("rho", "psi", "scores", "schrodinger", "iv", "sphere_pos",
                "signal_magnitude", "coverage", "efficiency", "confidence"):
        assert key in iv, f"{key} missing"


def test_an_explicit_target_is_unaffected():
    """A caller with a real phenotype vector must get exactly what it always got."""
    rex = _graph()
    ti, tw = _targets(rex)
    rng = np.random.default_rng(3)
    target = rng.standard_normal(int(rex.nE))
    a = rex.interfacing_vector(ti, tw, target)
    b = rex.interfacing_vector(ti, tw, target)
    assert np.allclose(a["iv"], b["iv"])


def test_the_scorer_no_longer_builds_a_whole_bundle_per_document():
    """The scorer used to call interfacing_vector per candidate, paying a whole
    interfacing bundle -- and passing target=None, which scores psi against itself
    rather than interfacing with anything. It reads coherence_response now, which is
    demand-driven at the seed, so interfacing_vector is not on that path at all."""
    import inspect

    from agent import scoring

    src = inspect.getsource(scoring)
    assert "interfacing_vector" not in src.split('"""')[2], \
        "the scorer is back on the whole-bundle path"
    assert "coherence_response" in src


def test_the_scorer_reads_only_the_seed():
    """coherence_response is O(|seed|), so scoring must not touch the full field.
    Reading rex.coherence would compute every vertex to answer about a handful."""
    import numpy as np
    from agent.scoring import interfacing_score

    rex = _graph(nV=60, extra=50)
    labels = [f"w{i}" for i in range(rex.nV)]
    r = interfacing_score(rex, labels, ["W0", "W1", "W2", "W3"], reading=False)
    assert r["n_shared"] == 4
    assert len(r["kappa"]) == 4, "kappa was returned for more than the seed"
    assert np.allclose(r["kappa"],
                       np.asarray(rex.coherence_response(np.array([0, 1, 2, 3], np.int32))))

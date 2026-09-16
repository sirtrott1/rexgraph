"""Cold native dependency gate. Run from a neutral directory against installed wheels.

``python -I /path/to/scripts/smoke_no_scipy.py [--platform]``
The optional platform gate adds RCQL execution and an RCDB native record roundtrip.
This deliberately does NOT certify every legacy analysis, spectral oracle, index,
or export API: those remaining SciPy routes are documented separately.
"""
from __future__ import annotations

import argparse
import importlib.abc
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import subprocess
from fractions import Fraction as Q


class NoScipy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "scipy" or fullname.startswith("scipy."):
            raise ModuleNotFoundError("SciPy is disabled by the native dependency gate", name=fullname)
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", action="store_true")
    parser.add_argument("--require-absent", action="store_true",
                        help="also require SciPy to be absent from installed distributions")
    parser.add_argument("--operators", action="store_true",
                        help="also execute the RCQL native operator inventory cases")
    args = parser.parse_args()
    try:
        scipy_version = importlib.metadata.version("scipy")
    except importlib.metadata.PackageNotFoundError:
        scipy_version = None
    if args.require_absent:
        assert scipy_version is None, "this gate requires an installation without SciPy"
    # Check unblocked imports too: a caught optional import would evade the
    # blocked gate while still eagerly loading SciPy when it is installed.
    subprocess.run([sys.executable, "-I", "-c",
        "import sys, rexgraph; import rexgraph.io._compat; "
        "assert not rexgraph.core._failed; "
        "assert not any(n == 'scipy' or n.startswith('scipy.') for n in sys.modules)"],
        check=True)
    assert not any(k == "scipy" or k.startswith("scipy.") for k in sys.modules), "must start cold"
    sys.meta_path.insert(0, NoScipy())

    import numpy as np

    import rexgraph
    from rexgraph import RexGraph, core
    from rexgraph.adjugate_operator import AdjugateOperator
    from rexgraph.chain_map import ChainHomotopy, CoordinateComplex, GradedMap
    from rexgraph.channel_operator import channel_operator
    from rexgraph.cochain import Chain
    from rexgraph.cells import Cell
    from rexgraph.cell_neighborhood import hyperslice
    from rexgraph.column_expansion import ColumnExpansion, primary_lift
    from rexgraph.flow import FieldNavigator, flow_step
    from rexgraph.graph import TemporalRex
    from rexgraph.green import GreenOperator
    from rexgraph.linear_operator import boundary_operator, coboundary_operator
    from rexgraph.native_rank import boundary_rank
    from rexgraph.native_homology import homology_split
    from rexgraph.native_sparse import empty_native
    from rexgraph.sheaf import ExactSheaf, Sheaf
    from rexgraph.sigma_operator import critical_commutator, critical_rate, sigma_operator
    from rexgraph.weighted_dirac import GradedChain, weighted_dirac
    from rexgraph.weighted_hodge import weighted_hodge

    assert not core._failed, core._failed
    # Ordinary construction must not require a compatibility matrix package.
    built = RexGraph.from_cells([4, [[0, 1, 2, 3], [0, 1, 2, 3], [0, 0], [3]],
                               [[(0, 1), (1, -1)]], []])
    assert built.betti_tower == (2, 1, 0, 0)
    split = homology_split(built, 1)
    assert (split.simple, split.multiplicity, split.chain_multiplicity) == (1, 0, 1)
    face_column = [(0, 1), (1, 1), (2, -1)]
    difference = [(0, 1), (1, -1)]
    high_cells = RexGraph.from_cells([3, [[0, 1], [1, 2], [0, 2]],
                                     [face_column, face_column], [difference, difference], [difference]])
    assert high_cells.betti_tower == (1, 0, 0, 0, 0)
    from rexgraph.partition import document_field, section_response
    from rexgraph.sectioning import add_sectioning, sectionings_of
    from rexgraph.tower import semantic_closure
    add_sectioning(high_cells, "all", {"all": [0, 1, 2]})
    np.testing.assert_array_equal(document_field(high_cells, [0]).numpy(), [Q(1, 2), Q(0), Q(1, 2)])
    assert section_response(high_cells, sectionings_of(high_cells)["all"], [0], exact=True)[0].tolist() == [Q(1)]
    assert semantic_closure(high_cells, 0)["steps"][0]["betti"] == [1, 0, 0, 0, 0]
    from rexgraph.tensor_diff import difference_tensor
    from rexgraph.field_delta import field_delta_moment
    before = RexGraph.from_cells([1, []])
    after = RexGraph.from_cells([1, [[0]]])
    delta, reading = difference_tensor(before, after)
    assert delta.entries == ((0, 0, Q(1)),) and reading["max_column_sum"] == 1
    np.testing.assert_array_equal(delta.apply(np.array([3])), [3])
    c = CoordinateComplex.from_rex(high_cells)
    correspondence = GradedMap(c, c, tuple(tuple((i, i, 1) for i in range(n)) for n in c.sizes))
    assert field_delta_moment(Chain(4, np.ones(1, dtype=int), source=high_cells), correspondence) == 0
    filled = high_cells.fill_cycle([1, 1, -1])
    assert filled.nF == 3 and high_cells.nF == 2
    assert filled.harmonic_shadow == {"shadow_dim": 1, "beta_1_at_d1": 1, "beta_1_at_d2": 0}
    assert filled.betti_tower == (1, 0, 1, 0, 0)
    assert homology_split(high_cells, 3).chain_multiplicity == 1
    rex = RexGraph(boundary_ptr=[0, 4, 6], boundary_idx=[0, 1, 2, 3, 1, 0],
                   w_E=[Q(2, 3), Q(3, 2)])
    x = Chain(1, np.array([3, 1]), source=rex)
    b = boundary_operator(rex, 1)
    np.testing.assert_array_equal(b.apply(x.values, exact=True), [-2, 0, 1, 1])
    np.testing.assert_allclose(b.apply(x.values), [-2, 0, 1, 1])
    assert coboundary_operator(rex, 1).shape == (0, 2)
    assert boundary_rank(rex, 1) == b.exact_rank_factory()[0] == 2
    assert rex.betti_tower == (2, 0) and rex.betti == (2, 0, 0)
    lone_branch = RexGraph.from_hypergraph([0, 4], [0, 1, 2, 3])
    assert lone_branch.betti_tower == (3, 0)  # not its one support component
    high = RexGraph.from_graph([0], [1])
    assert core._boundary.compute_rank(high._B1_dual) == 1
    high._graded_duals = [empty_native((0, 3)).dual,
        core._sparse.dual_from_coo([0, 1], [0, 0], [-1., 1.], 3, 1)]
    assert high.betti_tower == (1, 0, 0, 2, 0)
    for value, grade in ((rex, 1), (high, 4)):
        expansion = ColumnExpansion(boundary_operator(value, grade))
        lifted = primary_lift(expansion.legs, expansion.lift)
        unit = np.ones(lifted.shape[1], dtype=int)
        np.testing.assert_array_equal(lifted.apply(unit, exact=True), expansion.boundary.apply(unit, exact=True))
        np.testing.assert_allclose(lifted.apply(unit), expansion.boundary.apply(unit))
    assert hyperslice(Cell(high, 3, 0)).above.indices == (0,)
    assert hyperslice(Cell(high, 4, 0)).above.indices == ()
    assert hyperslice(Cell(rex, 1, 0)).below.indices == (0, 1, 2, 3)
    d = weighted_dirac(rex)
    assert d.apply(GradedChain(rex, [x]), exact=True).component(0).values.tolist() == [-2, 0, 1, 1]
    for name in "TGFC":
        action = channel_operator(rex, name)
        np.testing.assert_allclose(action.apply(x.values), np.asarray(action.apply(x.values, exact=True), float))
        np.testing.assert_allclose(action.diagonal(), np.asarray(action.diagonal(exact=True), float))
    h = weighted_hodge(rex, 1)
    green = GreenOperator.resolvent(h, .5)
    y, info = green.solve_with_info(x.values)
    np.testing.assert_allclose(y + .5*h.apply(y), x.values, atol=1e-10)
    assert info["kernel"] == "native-metric-block-cg"
    assert CoordinateComplex.from_rex(rex).sizes == (4, 2)
    sigma_args = ([1, 2, 3, 4], [-1, -2, -3, -4], list("TGFC"), "share")
    sigma = sigma_operator(rex, 0.5, *sigma_args)
    assert np.isfinite(sigma.apply(x.values)).all()
    assert np.isfinite(sigma.derivative.apply(x.values)).all()
    assert not np.any(critical_commutator(rex, 0.5, *sigma_args).apply(x.values))
    np.testing.assert_allclose(critical_rate(rex, *sigma_args, "slope").apply(x.values),
                               -2 * critical_rate(rex, *sigma_args).apply(x.values))
    channel = channel_operator(rex, "T")
    np.testing.assert_array_equal(AdjugateOperator(channel).apply(x.values, exact=True),
        sum(channel.diagonal(exact=True)) * x.values - channel.apply(x.values, exact=True))
    complex_ = CoordinateComplex.from_rex(rex)
    identity = GradedMap(complex_, complex_, tuple(tuple((i, i, 1) for i in range(n)) for n in complex_.sizes))
    assert ChainHomotopy(identity, identity, ((), ())).residuals == (0, 0)
    section = ExactSheaf(rex)
    section.assign(0, [Q(1, 3)])
    section.assign(1, [Q(1, 3)])
    assert section.check_section().compatible and section.glue().ratio == 1
    face = RexGraph.from_simplicial([0, 1, 0], [1, 2, 2], [[0, 1, 2]])
    assert face.nF_hodge == 1 and ExactSheaf(face, grade=2).check_section().compatible
    numerical = Sheaf(face)
    angles = numerical.gradient_angles(np.array([1., 2., 3.]))
    np.testing.assert_array_equal(numerical.holonomy(angles), [0])

    cycle = RexGraph.from_graph([0, 1, 2], [1, 2, 0])
    flow = flow_step(cycle, np.arange(cycle.nE))
    np.testing.assert_allclose(flow["draining"], 0, atol=1e-12)
    np.testing.assert_allclose(flow["circulating"], 1, atol=1e-12)
    branching_flow = flow_step(rex, np.arange(rex.nE))
    np.testing.assert_allclose(branching_flow["draining"] + branching_flow["circulating"], 1)
    snapshots = [
        ([0, 0, 1], [1, 2, 3]),
        ([0, 0, 1, 2], [1, 2, 3, 4]),
        ([0, 0, 1, 2, 3], [1, 2, 3, 4, 5]),
        ([0, 0, 1, 2, 3, 4], [1, 2, 3, 4, 5, 6]),
        ([0, 0, 1, 2, 3, 4, 4], [1, 2, 3, 4, 5, 6, 0]),
    ]
    history = TemporalRex([(np.asarray(s, np.int32), np.asarray(t, np.int32))
                           for s, t in snapshots])
    navigator = FieldNavigator()
    events = navigator.run(history)
    assert navigator.flow_calls == sum(event["event"] for event in events) > 0

    if args.platform:
        from rcdb import MemoryStore
        from rcdb import index as accession
        from rcdb.core import ComplexRecord
        from rcql import Executor, parse
        from rexgraph.flow.turn_field import TurnField
        chat = TurnField()
        chat.observe("alpha beta")
        chat.observe("beta gamma delta")
        chat.observe("alpha beta")
        conversation = Executor(sources={"chat": chat}).execute(parse(
            'FROM $chat RETURN TURN_FIELD(),PATH_CHANGE("delta epsilon")'))
        assert conversation.values[0].at(2).relation_ids.tolist() == [0, 1, 2]
        assert conversation.values[0].at(2)._agent_meta["vertex_labels"] == ["alpha", "beta", "gamma", "delta"]
        assert conversation.values[1] == chat.preview("delta epsilon") and chat.n_turns == 3
        corpus = accession.build([
            ("a", ComplexRecord(id="a", created=0.0, signature={"tags": ["x", "y"]})),
            ("b", ComplexRecord(id="b", created=0.0, signature={"tags": ["x"]})),
        ])
        assert accession.record_response_exact(corpus, ["x"]) == {0: Q(1, 4), 1: Q(1, 2)}
        np.testing.assert_array_equal(
            accession.record_response(corpus, ["x"])[0], [0.25, 0.5])
        assert accession.record_response_exact(corpus, ["x"], reading="existence") == {
            0: Q(1, 2), 1: Q(1, 2)}
        corpus_store = MemoryStore()
        try:
            corpus_store.put("r", rex, tags=["x"], analytics=False)
            corpus_result = Executor(sources={"db": corpus_store}).execute(parse(
                'FROM $db LET f=CORPUS_FIELD(["x"]) RETURN f.ids,f.versions,f.scores'))
            assert corpus_result.values == (("r",), (1,), (Q(1),))
        finally:
            corpus_store.close()
        result = Executor(sources={"r": rex}, params={"x": x, "section": section}).execute(parse(
            'FROM $r LET h=HODGE_SUM(1) LET g=RESOLVENT(h,0.5) '
            'RETURN GRADE(), BOUNDARY(1,$x), APPLY(h,$x,true), '
            'GREEN_SOLVE(g,$x), SECTION_CHECK($section).compatible, GLUE($section).ratio'))
        assert result.values[0] == 1 and result.values[-2:] == (True, Q(1))
        np.testing.assert_allclose(result.values[3].values, y)
        ranks = Executor(sources={"r": high}).execute(parse(
            'FROM $r RETURN BETTI(3), RANK(4), NULLITY(3), RANK(BOUNDARY(4)), '
            'NULLITY(COBOUNDARY(4))'))
        assert ranks.values == (2, 1, 3, 1, 1)
        dimensions = Executor(sources={"r": built}).execute(parse(
            'FROM $r RETURN SIMPLE_HOMOLOGY(grade=1), MULTIPLICITY_HOMOLOGY(grade=1)'))
        assert dimensions.values == (1, 0)
        selected = Executor(sources={"r": high_cells}).execute(parse(
            'FROM $r LET p=PARTITION(CELL(4,0)) RETURN p, RESTRICT(CELL(0,0)), COUNT(FACES(CELLS(1)))'))
        assert selected.values[0].cell_maps == ((0, 1, 2), (0, 1, 2), (0, 1), (0, 1), (0,))
        assert selected.values[1].nV == 1 and selected.values[2] == 2
        store = MemoryStore()
        try:
            store.put("r", rex, analytics=False)
            stored = Executor(sources={"db": store}).execute(parse(
                'FROM RCDB_GET($db,"r") RETURN GRADE(), STATE_HASH(), '
                'BOUNDARY(1,ZERO(1,"chain")), RANK(1), NULLITY(1), BETTI(0)'))
            assert stored.values[0] == 1
            assert stored.values[1] == store.read_record("r").state_digest
            assert stored.values[2].values.tolist() == [Q(0)]*4
            assert stored.values[3:] == (2, 0, 2)
            family = '[1,2,3,4], [-1,-2,-3,-4], ["T","G","F","C"], "share"'
            certified = Executor(sources={"db": store}).execute(parse(
                'FROM RCDB_GET($db,"r") LET x=INDICATOR(CELL(1,0)) '
                f'RETURN APPLY(SIGMA_OPERATOR(0.5,{family}),x), '
                f'APPLY(CRITICAL_COMMUTATOR(0.5,{family}),x), '
                f'APPLY(CRITICAL_RATE({family}),x), APPLY(ADJUGATE(CHANNEL("T")),x,true)'))
            assert not np.any(certified.values[1].values)
            np.testing.assert_allclose(certified.values[0].values, sigma.apply(np.array([1, 0])))
            assert all(isinstance(v, Q) for v in certified.values[3].values)
            coordinates = Executor(sources={"db": store}).execute(parse(
                'FROM RCDB_GET($db,"r") LET e=COLUMN_EXPANSION(BOUNDARY(1)) '
                'LET b=PRIMARY_LIFT(e.legs,e.lift) '
                'RETURN APPLY(b,ZERO(1,"chain"),true), HYPERSLICE(CELL(1,0)).below, RANK(b)'))
            assert coordinates.values[0].values.tolist() == [Q(0)] * 4
            assert coordinates.values[1].indices == (0, 1, 2, 3)
            assert coordinates.values[2] == 2
            store.put("high", high, analytics=False)
            higher = Executor(sources={"db": store}).execute(parse(
                'FROM RCDB_GET($db,"high") RETURN BETTI(3), RANK(4), NULLITY(3)'))
            assert higher.values == (2, 1, 3)
            store.configure_security(require_commits=True)
            edited = Executor(sources={"db": store}, params={"r": rex}).execute(parse(
                'FROM $db MUTATE "edited" SET state=$r, expected_version=0 '
                'REMOVE [1] ADD [[0,1,2,3,4]] COMMIT'))
            assert edited.values[0].version == 1 and store.verify_commits("edited")
            matched = Executor(sources={"db": store}).execute(parse(
                'FROM RCDB_GET($db,"edited") MATCH e IN CELLS(1) '
                'WHERE ARITY(e) >= 4 RETURN e.index, ARITY(e) ORDER BY e.index'))
            assert matched.values == (((0, 4), (1, 5)),)
            from rexgraph.io.temporal_state import to_temporal_state, from_temporal_state
            from rexgraph.io.catalog import object_digest
            history = TemporalRex([], general=True)
            history.append_snapshot(rex, at=5)
            history.append_snapshot(store.read_record("edited").value, at=9)
            recovered = from_temporal_state(to_temporal_state(history))
            assert object_digest(recovered) == object_digest(history)
        finally:
            store.close()

    # Negative control: the explicit compatibility export really is unavailable.
    try:
        b.as_scipy()
    except ModuleNotFoundError as exc:
        assert exc.name == "scipy"
    else:
        raise AssertionError("SciPy export unexpectedly succeeded")
    assert not any(k == "scipy" or k.startswith("scipy.") for k in sys.modules)
    if args.operators:
        import pytest
        os.environ["REXGRAPH_TEST_INSTALLED"] = "1"
        tests = Path(__file__).resolve().parents[1] / "rcql/tests/test_operator_inventory.py"
        status = pytest.main([str(tests), "--import-mode=importlib", "-q", "--tb=short", "-k",
                              "test_native_direct_adapter_and_typed_contract_agree or test_temporal_direct_adapter_and_typed_contract_agree"])
        if status:
            raise SystemExit(status)
        assert not any(k == "scipy" or k.startswith("scipy.") for k in sys.modules)
    print(json.dumps({"status": "passed", "platform": args.platform,
        "operator_inventory": args.operators, "scipy_distribution": scipy_version,
        "interpreter": sys.executable, "rexgraph": rexgraph.__file__,
        "sparse_kernel": core._sparse.__file__, "compiled_modules": len(core._loaded),
        "metadata": importlib.metadata.version("rexgraph"), "scipy_modules_loaded": 0}))


if __name__ == "__main__":
    main()

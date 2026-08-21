"""The index and the log are binary, and a record has to survive both exactly.

Both encoders drop anything they cannot name, and a dropped field is silent: the store
reopens, the record is there, and one key is missing. So these check identity on the
awkward shapes rather than on a well formed record, since the well formed one passes
under any encoding.
"""
from __future__ import annotations

import numpy as np
import pytest

from agent import rcdb_index as ix
from agent.rcdb import ComplexRecord


def _rec(rid, sig_extra=None, meta=None):
    sig = {"object_type": "RexGraph", "source": "obo", "nV": 10, "nE": 12, "nF": 0,
           "betti": [1, 3, 0], "betti1": 3, "chain_valid": True, "kappa_mean": 0.5,
           "coherence_method": "local", "n_voids": 0, "n_labels": 4,
           "tags": ["a", "b"], "labels_sample": ["x", "y"]}
    sig.update(sig_extra or {})
    return ComplexRecord(id=rid, signature=sig, created=1.0, meta=meta or {},
                         version=1, tx_from=1.0, tx_to=None, valid_from=None,
                         valid_to=None)


#: the shapes an encoder gets wrong: nesting, ragged lists, empty containers, a value
#: that is None rather than absent, and keys chosen to break a joined path.
AWKWARD = {
    "nested": {"a": {"b": [1.5, 2.5]}, "c": True},
    "ragged": [1, "two", {"three": 3}],
    "a.b": "a dotted key",
    "0": "a key that reads as a list index",
    "#0": "a key that reads as an index marker",
    "empty_dict": {},
    "empty_list": [],
    "none": None,
    "unicode": "β₁ = 3",
    "float": 3.141592653589793,
    "big_int": 2 ** 53 + 1,
    "bools": [True, False, True],
}

CASES = [
    ("plain", _rec("plain")),
    ("temporal", _rec("temporal", {"object_type": "TemporalRex", "T": 7,
                                   "checkpoint_times": [0, 3, 6],
                                   "t_first": 0.5, "t_last": None})),
    ("awkward", _rec("awkward", meta={"vertex_labels": ["v1", "v2"], **AWKWARD})),
    ("clash", _rec("clash", meta={"nested": "the same path, now a string"})),
]


def _same(a: ComplexRecord, b: ComplexRecord):
    assert a.id == b.id and a.version == b.version
    assert (a.created, a.tx_from, a.tx_to) == (b.created, b.tx_from, b.tx_to)
    assert (a.valid_from, a.valid_to) == (b.valid_from, b.valid_to)
    assert a.meta == b.meta
    for k, v in (b.signature or {}).items():
        assert a.signature.get(k) == v, k


@pytest.mark.parametrize("name", [c[0] for c in CASES])
def test_the_index_round_trips_a_record(tmp_path, name):
    orig = dict(CASES)[name]
    p = tmp_path / "index.rexidx"
    ix.write(p, ix.build([(orig.id, orig)]))
    _same(ix.record_at(ix.read(p), 0), orig)


def test_the_index_round_trips_every_case_together(tmp_path):
    """One store, so the shared tables and the path columns carry all of them at once."""
    p = tmp_path / "index.rexidx"
    ix.write(p, ix.build([(r.id, r) for _n, r in CASES]))
    back = ix.read(p)
    assert back["n"] == len(CASES)
    for row, (_n, orig) in enumerate(CASES):
        _same(ix.record_at(back, row), orig)


@pytest.mark.parametrize("name", [c[0] for c in CASES])
def test_the_log_round_trips_a_record(tmp_path, name):
    orig = dict(CASES)[name]
    p = tmp_path / "index.rexlog"
    ix.log_append(p, "put", orig.id, orig)
    (op, rid, back, extra), = ix.log_read(p)
    assert extra is None, "a frame written without one carries no extra row"
    assert (op, rid) == ("put", orig.id)
    _same(back, orig)


def test_the_log_keeps_its_order_and_its_deletes(tmp_path):
    p = tmp_path / "index.rexlog"
    for _n, r in CASES:
        ix.log_append(p, "put", r.id, r)
    ix.log_append(p, "delete", "plain", None)
    got = list(ix.log_read(p))
    assert [g[0] for g in got] == ["put"] * len(CASES) + ["delete"]
    assert [g[1] for g in got] == [r.id for _n, r in CASES] + ["plain"]
    assert got[-1][2] is None


def test_a_torn_log_tail_stops_the_read(tmp_path):
    """A crash mid append truncates the last frame. The frames before it still read."""
    p = tmp_path / "index.rexlog"
    for _n, r in CASES:
        ix.log_append(p, "put", r.id, r)
    whole = p.read_bytes()
    for cut in (len(whole) - 1, len(whole) - 31, len(whole) // 2, 8, 3):
        torn = tmp_path / f"torn{cut}.rexlog"
        torn.write_bytes(whole[:cut])
        got = list(ix.log_read(torn))          # must not raise
        assert len(got) < len(CASES)
        for _op, _rid, rec, _extra in got:
            assert rec is None or rec.signature["nV"] == 10


def test_a_numpy_array_in_meta_reads_back_as_a_list(tmp_path):
    """meta is caller supplied, so an array lands there. It is data, not a tensor to
    preserve: it comes back as the list it encodes."""
    rec = _rec("arr", meta={"a": np.arange(3), "b": np.array([1.5, 2.5])})
    p = tmp_path / "index.rexidx"
    ix.write(p, ix.build([("arr", rec)]))
    back = ix.record_at(ix.read(p), 0)
    assert back.meta == {"a": [0, 1, 2], "b": [1.5, 2.5]}


def test_a_shared_term_is_stored_once(tmp_path):
    """The index grows with the VOCABULARY, not with the corpus.

    Stated as the comparison rather than a byte budget: a hundred records sharing one
    set of twelve labels and a hundred records with twelve distinct labels each hold the
    same number of incidence entries, so any size difference between them is vocabulary
    alone. A fixed byte threshold would assert the encoding's current constants instead.
    """
    vocab = [f"GO:{i:07d}" for i in range(12)]

    def _size(n, distinct):
        idx = ix.build([
            (f"r{i}", _rec(f"r{i}", {"labels_sample":
                                     [f"GO:{i:04d}{j:03d}" for j in range(12)]
                                     if distinct else vocab}))
            for i in range(n)])
        p = tmp_path / f"{'d' if distinct else 's'}{n}.rexidx"
        ix.write(p, idx)
        return idx, p.stat().st_size

    (a100, sa100), (a200, sa200) = _size(100, False), _size(200, False)
    (b100, sb100), (b200, sb200) = _size(100, True), _size(200, True)
    # identical corpus structure either way: the incidence is the same size
    assert a100["rel_idx"].size == b100["rel_idx"].size
    # a shared vocabulary stops growing; an unshared one grows with the corpus
    assert a200["n_terms"] == a100["n_terms"]
    assert b200["n_terms"] > b100["n_terms"]
    # so the marginal cost of a record is strictly lower when its terms are shared
    assert (sa200 - sa100) < (sb200 - sb100)


def test_the_columns_answer_a_predicate_without_a_payload(tmp_path):
    recs = [(f"r{i}", _rec(f"r{i}", {"nV": i * 10, "source": "obo" if i % 2 else "owl",
                                     "betti1": i}))
            for i in range(50)]
    p = tmp_path / "index.rexidx"
    ix.write(p, ix.build(recs))
    back = ix.read(p)
    rows = ix.rows_for(back, min_nV=200, max_nV=300, source="obo")
    got = set(ix.ids_of(back, rows))
    want = {rid for rid, r in recs
            if 200 <= r.signature["nV"] <= 300 and r.signature["source"] == "obo"}
    assert got == want and got


def test_an_unreadable_format_version_is_refused(tmp_path):
    """A future index must not be read as if it were this one."""
    from safetensors.numpy import save_file
    p = tmp_path / "future.rexidx"
    save_file({"col/nV": np.zeros(1, np.int64)}, str(p),
              metadata={"format": str(ix.FORMAT_VERSION + 1), "n": "1"})
    with pytest.raises(ValueError, match="index format"):
        ix.read(p)


#### the boundary operator is the arrays the index already holds ################

def test_boundary_operator_is_built_from_the_stored_arrays_not_rebuilt():
    """`rel_ptr`/`rel_idx` ARE the column structure (relation `e` occupies
    `rel_idx[rel_ptr[e]:rel_ptr[e+1]]` with the record first) so B1 is those arrays plus
    the values arity determines. Reconstructing a RexGraph to reach the same matrix
    measured 40 s on the 61,353-record store against 1.3 s here, byte-identical, and it
    was paid by the first query of every process."""
    import numpy as np
    from rexgraph.core._sparse import to_scipy_csr

    from agent import rcdb_index as ix

    recs = [_rec(f"r{i}", meta={"vertex_labels": [f"t{i}", f"t{i+1}", "shared"]})
            for i in range(6)]
    index = ix.build([(r.id, r) for r in recs])
    B = ix.boundary_operator(index)
    ref = to_scipy_csr(ix.complex_of(index)._B1_dual).tocsc()
    assert B.shape == ref.shape
    d = (B - ref)
    assert d.nnz == 0 or float(abs(d).max()) < 1e-12, "must be the same operator"

    # the columns are boundary columns: one -1 at the record, the rest sharing
    ptr = np.asarray(index["rel_ptr"], dtype=np.int64)
    Bc = B.tocsc()
    for e in range(min(8, B.shape[1])):
        col = Bc.data[Bc.indptr[e]:Bc.indptr[e + 1]]
        k = int(ptr[e + 1] - ptr[e])
        assert (col < 0).sum() == 1, "exactly one record heads its own accession"
        if k > 1:
            assert abs(col.sum()) < 1e-12, "and the column is zero-sum"


def test_the_operator_is_cached_so_a_query_does_not_rebuild_it():
    from agent import rcdb_index as ix
    recs = [_rec(f"r{i}", meta={"vertex_labels": [f"t{i}", "shared"]}) for i in range(4)]
    index = ix.build([(r.id, r) for r in recs])
    a = ix.boundary_operator(index)
    b = ix.boundary_operator(index)
    assert a is b


#### the accession reading has axes, and one of them is structurally zero ########

def test_the_record_scalar_is_its_channel_profile_summed():
    """`record_response` returns one float per record, and that float is the profile
    summed over (topology, geometry, frustration, coparticipation). Asking for the
    profile adds nothing to the computation; it stops discarding the axes."""
    import numpy as np

    from agent import rcdb_index as ix

    recs = [_rec(f"r{i}", meta={"vertex_labels": [f"t{i}", f"t{i+1}", "shared"]})
            for i in range(5)]
    index = ix.build([(r.id, r) for r in recs])
    sc, ids = ix.record_response(index, ["t0", "t1"])
    prof, ids2, names = ix.record_response(index, ["t0", "t1"], channels=True)
    assert ids == ids2 and prof.shape == (len(ids), 4)
    assert names[:2] == ["topology", "geometry"]
    assert np.allclose(sc, prof.sum(axis=1)), "the scalar IS the summed profile"


def test_frustration_is_structurally_zero_at_the_accession_grade():
    """Not a property of this data. F measures where the signed and unsigned readings
    disagree, which needs a vertex that HEADS one relation and is an ARGUMENT of another.
    Records occupy [0, n) and terms [n, n + n_terms), disjoint, and a record heads every
    relation it is in while a term is always an argument, so no vertex is ever both and
    the mismatch cannot arise. What is left is topology against co-participation."""
    import numpy as np

    from agent import rcdb_index as ix

    recs = [_rec(f"r{i}", meta={"vertex_labels": [f"t{i}", "shared"]}) for i in range(4)]
    index = ix.build([(r.id, r) for r in recs])
    B = ix.boundary_operator(index).toarray()
    T = B.T @ B
    G = np.abs(B).T @ np.abs(B)
    off = np.abs(T - G)
    np.fill_diagonal(off, 0.0)
    assert off.max() < 1e-12, "T and G agree off-diagonal, so F is zero"

    d, _n = ix.channel_diagonals(index)
    assert np.allclose(d[:, 0], d[:, 1]), "T and G share their diagonal"
    assert np.allclose(d[:, 2], 0.0), "F is identically zero here"
    # C is two matvecs standing in for an off-diagonal row sum: check it against the
    # assembled operator, which is only affordable at this size
    ref = G.sum(axis=1) - np.diag(G)
    assert np.allclose(d[:, 3], ref), "C must equal the assembled row sum"


def test_the_seed_degree_is_per_vertex_and_not_per_relation():
    """`boundary_operator` returns CSC, so `np.diff(B.indptr)` counts per COLUMN (per
    relation) and the seeds are VERTEX indices. Using the wrong one mis-weights every
    seed silently and then raises IndexError once a vertex id exceeds the relation count.

    That is exactly what happened: it passed on small fixtures because there every vertex
    id is below nE, and on the 61,353-record store it raised inside a bare `except` that
    turned an 88 s scan fallback into the observed behaviour. This asserts the shape
    relationship that makes the two distinguishable at fixture size."""
    import numpy as np

    from agent import rcdb_index as ix

    # many terms per record, so nV is comfortably larger than the relation count
    recs = [_rec(f"r{i}", meta={"vertex_labels": [f"t{i}_{j}" for j in range(9)]})
            for i in range(6)]
    index = ix.build([(r.id, r) for r in recs])
    B = ix.boundary_operator(index)
    nV, nE = B.shape
    assert nV > nE, "the fixture must have more vertices than relations to bite"
    assert len(B.indptr) - 1 == nE, "CSC indptr is over columns"

    deg = np.bincount(np.asarray(index["rel_idx"], dtype=np.int64), minlength=nV)
    assert deg.size == nV
    # a seed at the highest vertex must work rather than raise
    top_term = index["vocab"][-1]
    sc, ids = ix.record_response(index, [top_term])
    assert sc.shape == (index["n"],)
    assert sc.sum() > 0, "the last term in the vocabulary must still reach its record"


def test_vertex_degree_is_cached_and_counts_per_vertex():
    """`deg` was rebuilt per call from `rel_idx` with an int64 cast, allocating a fresh
    copy of every nonzero each time. It is cached beside the operator now, and counted
    off `B.indices`, which is the same array already widened."""
    from agent import rcdb_index as ix

    index = ix.build([("r0", _rec("r0", {"tags": ["alpha", "beta"]})),
                      ("r1", _rec("r1", {"tags": ["beta", "gamma"]}))])
    B = ix.boundary_operator(index)
    deg = ix._vertex_degree(index, B)

    assert deg.shape[0] == B.shape[0]                 # per VERTEX, not per relation
    assert index["_deg"] is deg                       # cached
    assert ix._vertex_degree(index, B) is deg         # and returned from the cache
    assert float(deg.sum()) == float(B.nnz)           # every incidence counted once
    # `beta` is in both records and `alpha` in one, which is what 1/deg reads
    codes = ix._term_codes(index)
    assert deg[codes["beta"]] == 2 and deg[codes["alpha"]] == 1


def test_the_two_towers_read_the_same_boundary_and_differ_by_the_share():
    """`share` divides each term's contribution by the record's width, so it reads a
    DENSITY and ranks short records first. `existence` reads the {0,1} incidence, so it
    reads MASS. Both are exact; neither is a normalisation of the other."""
    from agent import rcdb_index as ix

    # one short record and one long one, both holding the query term exactly once
    short = _rec("short", {"tags": ["alpha", "beta"]})
    long_ = _rec("long", {"tags": ["alpha"] + [f"f{i}" for i in range(40)]})
    index = ix.build([("short", short), ("long", long_)])

    share, ids = ix.record_response(index, {"alpha"}, reading="share")
    mass, _ = ix.record_response(index, {"alpha"}, reading="existence")
    i_s, i_l = ids.index("short"), ids.index("long")

    # the share reading prefers the short record purely on width
    assert share[i_s] > share[i_l]
    # the mass reading does not: both hold the term once, so neither is ahead on it
    assert mass[i_s] == pytest.approx(mass[i_l])


def test_an_unknown_reading_is_refused_rather_than_silently_defaulted():
    from agent import rcdb_index as ix
    index = ix.build([("r", _rec("r", {"tags": ["a", "b"]}))])
    with pytest.raises(ValueError, match="share.*existence"):
        ix.record_response(index, {"a"}, reading="bogus")


def test_a_frame_carries_a_backend_s_own_row(tmp_path):
    """`extra` is how a store keeps an aligned int64 row of its own in the shared log,
    signalled by the presence byte reading 2. RexStore puts its blob address there."""
    p = tmp_path / "index.rexlog"
    r = dict(CASES)["plain"]
    ix.log_append(p, "put", r.id, r, extra=[4096, 271])
    ix.log_append(p, "put", "second", r)
    got = list(ix.log_read(p))
    assert [int(x) for x in got[0][3]] == [4096, 271]
    assert got[1][3] is None, "a frame without one is unchanged"
    _same(got[0][2], r)


def test_the_log_resumes_at_a_byte_offset(tmp_path):
    """A store replays only the frames its snapshot does not hold, so the reader takes
    the offset it recorded rather than re-reading from the start."""
    p = tmp_path / "index.rexlog"
    ix.log_append(p, "put", "a", dict(CASES)["plain"])
    at = p.stat().st_size
    ix.log_append(p, "put", "b", dict(CASES)["plain"])
    assert [g[1] for g in ix.log_read(p)] == ["a", "b"]
    assert [g[1] for g in ix.log_read(p, at)] == ["b"]


def test_the_compiled_scan_and_the_python_codec_agree(tmp_path, monkeypatch):
    """`log_read` runs the compiled scan where the core is built and the python codec
    where it is not. Two readers of one format have to return the same records."""
    p = tmp_path / "index.rexlog"
    for _n, r in CASES:
        ix.log_append(p, "put", r.id, r, extra=[11, 22])
    ix.log_append(p, "delete", "plain", None)

    compiled = list(ix.log_read(p))
    monkeypatch.setattr(ix, "_read_frames", None)
    pure = list(ix.log_read(p))

    assert ix._read_frames is None
    assert [(g[0], g[1]) for g in compiled] == [(g[0], g[1]) for g in pure]
    assert len(compiled) == len(CASES) + 1
    for a, b in zip(compiled, pure, strict=True):
        if a[2] is None:
            assert b[2] is None
            continue
        _same(a[2], b[2])
        assert [int(x) for x in a[3]] == [int(x) for x in b[3]]


def test_both_readers_stop_at_the_same_torn_tail(tmp_path, monkeypatch):
    """A short read must end the scan identically in either codec, since a partial
    frame that one of them interprets is a record the other does not have."""
    p = tmp_path / "index.rexlog"
    for _n, r in CASES:
        ix.log_append(p, "put", r.id, r)
    whole = p.read_bytes()
    for cut in (len(whole) - 1, len(whole) - 31, len(whole) // 2, 12, 8, 3):
        torn = tmp_path / f"t{cut}.rexlog"
        torn.write_bytes(whole[:cut])
        compiled = [(g[0], g[1]) for g in ix.log_read(torn)]
        monkeypatch.setattr(ix, "_read_frames", None)
        pure = [(g[0], g[1]) for g in ix.log_read(torn)]
        monkeypatch.undo()
        assert compiled == pure, f"cut {cut}"

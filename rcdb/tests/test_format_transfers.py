"""Format conversion and checked store copies preserve complete selected versions."""
from contextlib import closing
from fractions import Fraction

import pytest

from rcdb import LocalStore, MemoryStore, NativeObjectStore, SQLStore, copy_record
from rexgraph import RexGraph
from rexgraph.object_identity import object_digest


def open_target(kind, path):
    if kind == "local":
        return LocalStore(path)
    if kind == "sqlite":
        pytest.importorskip("sqlalchemy")
        return SQLStore(f"sqlite:///{path}.sqlite")
    pytest.importorskip("fsspec")
    return NativeObjectStore("file://" + str(path))


@pytest.mark.parametrize("kind", ["local", "sqlite", "object-local"])
@pytest.mark.parametrize("format", ["rcbd", "safetensors", "h5", "zarr", "json", "arrow"])
def test_conversion_copy_and_cold_time_selection_retain_the_same_state(kind, format, tmp_path):
    dependency = {"safetensors": "safetensors", "h5": "h5py", "zarr": "zarr", "arrow": "pyarrow"}.get(format)
    if dependency:
        pytest.importorskip(dependency)
    graph = RexGraph.from_hypergraph([0, 3, 5], [0, 1, 2, 2, 2])
    graph.attach_metadata(1, 0, "field", {"exact": Fraction(3, 7), "large": 2**512})
    if format == "arrow":
        from rexgraph.io.arrow_bridge import rex_to_arrow, arrow_to_rex
        converted = arrow_to_rex(rex_to_arrow(graph))
    else:
        from rexgraph.io import save, load
        path = tmp_path / ("input." + format)
        save(path, graph)
        converted = load(path)
    expected = object_digest(graph)
    assert object_digest(converted) == expected
    root = tmp_path / "store"
    with closing(MemoryStore()) as source, closing(open_target(kind, root)) as target:
        source.put("snapshot", converted, analytics=False, _tx_time=10.0)
        source.put("snapshot", RexGraph.from_graph([0], [1]), analytics=False, _tx_time=20.0)
        receipt = copy_record(source, target, source.get_record("snapshot", as_of=15.0),
                              return_receipt=True, tx_time=10.0)
        assert receipt.source_digest == receipt.destination_digest
        target.put("snapshot", RexGraph.from_graph([0], [1]), analytics=False, _tx_time=20.0)
    with closing(open_target(kind, root)) as restored:
        old = restored.get("snapshot", as_of=15.0)
        assert object_digest(old) == expected
        assert old.get_metadata(1, 0, "field")["exact"] == Fraction(3, 7)
        assert restored.get("snapshot", as_of=25.0).nE == 1

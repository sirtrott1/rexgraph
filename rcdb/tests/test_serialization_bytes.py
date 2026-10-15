"""Record serialization uses the common byte codec and reads prior file payloads."""
import tempfile

import numpy as np
import pytest

from rexgraph import RexGraph, TemporalRex
from rexgraph.object_identity import object_digest
from rcdb import serialize_complex, deserialize_complex


@pytest.mark.parametrize("temporal", [False, True])
def test_serialization_needs_no_temporary_file(temporal, monkeypatch):
    graph = (TemporalRex([(np.array([0]), np.array([1]))]) if temporal
             else RexGraph.from_graph([0], [1]))
    def refuse(*args, **kwargs):
        pytest.fail("record serialization created a temporary file")
    monkeypatch.setattr(tempfile, "mkstemp", refuse)
    assert object_digest(deserialize_complex(serialize_complex(graph))) == object_digest(graph)


@pytest.mark.parametrize("temporal", [False, True])
def test_file_payload_and_byte_payload_restore_the_same_state(temporal, tmp_path):
    from rexgraph.io import save_safetensors
    from rcdb.core import compress_blob
    graph = (TemporalRex([(np.array([0]), np.array([1]))]) if temporal
             else RexGraph.from_graph([0], [1]))
    path = tmp_path / "old-writer.safetensors"
    save_safetensors(path, graph)
    old = deserialize_complex(compress_blob(path.read_bytes()))
    new = deserialize_complex(serialize_complex(graph))
    assert object_digest(old) == object_digest(new) == object_digest(graph)

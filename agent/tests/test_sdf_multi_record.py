"""Multi-record SDF reading.

`$$$$` separates records in an SDF, which is what distinguishes it from a single-record
MOL file. Every record is read into one complex; records do not share atom numbering, so
each lands on its own vertex block and forms its own component.

Correctness is checked against chemistry: benzene has one ring, so a file of three
benzenes has beta_0 = 3 and beta_1 = 3.
"""

import numpy as np
import pytest
from agent.adapters import formats as F

from rexgraph.graph import RexGraph

BENZENE = [(1, 2, 2), (2, 3, 1), (3, 4, 2), (4, 5, 1), (5, 6, 2), (6, 1, 1)]
ETHANE = [(1, 2, 1)]


def _mol(name, n_atoms, bonds):
    head = f"{name}\n  RexGraph\n\n{n_atoms:3d}{len(bonds):3d}  0  0  0  0  0  0  0  0999 V2000\n"
    atoms = "".join("    0.0000    0.0000    0.0000 C   0  0\n" for _ in range(n_atoms))
    bond_block = "".join(f"{a:3d}{b:3d}{o:3d}  0\n" for a, b, o in bonds)
    return head + atoms + bond_block + "M  END\n"


def _write(tmp_path, name, *records):
    p = tmp_path / name
    p.write_text("$$$$\n".join(records) + ("$$$$\n" if records else ""))
    return str(p)


def _complex(ec):
    return RexGraph(sources=np.asarray(ec.sources, np.int32),
                    targets=np.asarray(ec.targets, np.int32))


def _betti(ec):
    return [int(b) for b in _complex(ec).betti]


def test_every_record_is_read(tmp_path):
    """Benzene and ethane together are 6+2 atoms and 6+1 bonds."""
    ec = F.read(_write(tmp_path, "two.sdf", _mol("benzene", 6, BENZENE),
                       _mol("ethane", 2, ETHANE)))
    assert len(ec.vertex_labels) == 8
    assert len(ec.sources) == 7


def test_each_molecule_is_its_own_component(tmp_path):
    """Atom indices restart at 1 in each record, so nothing is merged across them.
    beta_0 counts the molecules and beta_1 counts the rings among them."""
    ec = F.read(_write(tmp_path, "two.sdf", _mol("benzene", 6, BENZENE),
                       _mol("ethane", 2, ETHANE)))
    b = _betti(ec)
    assert b[0] == 2
    assert b[1] == 1


def test_ring_count_scales_with_the_records(tmp_path):
    """Three benzenes are three components and three rings."""
    ec = F.read(_write(tmp_path, "three.sdf", *[_mol(f"b{i}", 6, BENZENE) for i in range(3)]))
    assert len(ec.vertex_labels) == 18
    assert _betti(ec)[:2] == [3, 3]


def test_a_single_record_mol_keeps_plain_labels(tmp_path):
    """A MOL file has no delimiter, so its atoms carry no record prefix."""
    p = tmp_path / "one.mol"
    p.write_text(_mol("benzene", 6, BENZENE))
    ec = F.read(str(p))
    assert len(ec.vertex_labels) == 6
    assert ec.vertex_labels[0] == "C1"
    assert _betti(ec)[1] == 1


def test_later_records_are_labelled_by_record(tmp_path):
    """Atom 1 exists in every record, so labels carry the record number to stay unique."""
    ec = F.read(_write(tmp_path, "two.sdf", _mol("a", 6, BENZENE), _mol("b", 2, ETHANE)))
    assert ec.vertex_labels[6] == "m1:C1"
    assert len(set(ec.vertex_labels)) == len(ec.vertex_labels)


def test_bond_orders_are_typed_across_all_records(tmp_path):
    """Edge typing is global: a single bond in record 2 shares the type of a single bond
    in record 1."""
    ec = F.read(_write(tmp_path, "two.sdf", _mol("a", 6, BENZENE), _mol("b", 2, ETHANE)))
    assert sorted(ec.type_names) == ["bond_order_1", "bond_order_2"]
    assert ec.n_types == 2


def test_trailing_delimiter_yields_no_extra_record(tmp_path):
    """The chunk after the final `$$$$` is empty and is skipped."""
    p = tmp_path / "trail.sdf"
    p.write_text(_mol("benzene", 6, BENZENE) + "$$$$\n\n\n")
    ec = F.read(str(p))
    assert len(ec.vertex_labels) == 6
    assert _betti(ec)[0] == 1


def test_a_file_with_no_readable_record_raises(tmp_path):
    p = tmp_path / "junk.sdf"
    p.write_text("not\nan\nsdf\n")
    with pytest.raises(ValueError, match="no readable SDF/MOL record"):
        F.read(str(p))

"""What each reader parses, it should say.

Every loader read structured fields and then flattened them into a label string. A PDB
line carries a chain and a residue number, both parsed for the backbone pass and then
discarded; a GFF line carries a whole `key=value` column already parsed into a dict, of
which two keys were used; an SDF atom carries an element and a formal charge. The only
way back from a label is to parse the name, and a name is not a schema: "CL1" is chlorine
atom 1 or carbon atom L1 depending on who wrote it.

`EdgeConstruction.attributes` has the same shape as `RexGraph._cell_metadata`, so
`build_rex_from_edges` hands it straight to `attach_metadata` and it serialises columnar
through `rex_state`, sparse and typed, indexed by cell index into the boundary tensors.
"""
from __future__ import annotations

import pytest

from agent.adapters.formats import load_bed, load_gff, load_pdb, load_sdf, load_vcf
from agent.auto import build_rex_from_edges

_BENZENE = """benzene
  test

  6  6  0  0  0  0  0  0  0  0999 V2000
    0.0000    1.4000    0.0000 C   0  0
    1.2124    0.7000    0.0000 C   0  0
    1.2124   -0.7000    0.0000 C   0  0
    0.0000   -1.4000    0.0000 C   0  0
   -1.2124   -0.7000    0.0000 C   0  0
   -1.2124    0.7000    0.0000 N   0  0
  1  2  4  0
  2  3  4  0
  3  4  4  0
  4  5  4  0
  5  6  4  0
  6  1  4  0
M  END
$$$$
"""

_PDB = """ATOM      1  N   ALA A   1      11.104   6.134  -6.504  1.00 20.00           N
ATOM      2  CA  ALA A   1      11.639   6.071  -5.147  1.00 21.50           C
ATOM      3  N   GLY B   2      11.549   3.895  -4.117  1.00 18.70           N
HETATM    4 ZN    ZN A 100      15.000   1.000  -1.000  0.50 30.00          ZN
END
"""

_GFF = """##gff-version 3
chr1\thavana\tgene\t100\t500\t12.5\t+\t.\tID=g1;Name=ALPHA;biotype=protein_coding
chr1\thavana\tgene\t400\t900\t3.0\t-\t.\tID=g2;Name=BETA;biotype=lncRNA
"""

_BED = "chr1\t100\t500\tfeatA\t900\t+\nchr1\t400\t900\tfeatB\t12\t-\n"

_VCF = """##fileformat=VCFv4.2
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2
chr1\t100\trs1\tA\tG\t50.0\tPASS\t.\tGT\t0/1\t1/1
chr1\t200\t.\tAT\tA\t20.0\tLowQual\t.\tGT\t0/0\t0/1
"""


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def _built(construction):
    return build_rex_from_edges(construction, face_selection="none")


#### the attributes reach the complex


def test_the_construction_shape_matches_the_store(tmp_path):
    """Same `{grade: {index: {key: value}}}`, so it is a hand-off not a translation."""
    attributes = load_sdf(_write(tmp_path, "b.sdf", _BENZENE)).attributes
    assert set(attributes) <= {0, 1, 2}
    assert all(isinstance(k, int) for cells in attributes.values() for k in cells)


def test_they_survive_onto_the_cells(tmp_path):
    rex = _built(load_sdf(_write(tmp_path, "b.sdf", _BENZENE)))
    assert rex.get_metadata(0, 0, "element") == "C"
    assert rex.get_metadata(0, 5, "element") == "N"


def test_they_round_trip_through_the_state(tmp_path):
    """Columnar and typed through rex_state, so an attribute travels with the tensors."""
    from rexgraph.io.rex_state import from_state, to_state

    rex = _built(load_sdf(_write(tmp_path, "b.sdf", _BENZENE)))
    back = from_state(to_state(rex))
    assert back.get_metadata(0, 5, "element") == "N"
    assert back.get_metadata(1, 0, "bond_order") == 4


#### what each reader now says


def test_sdf_carries_the_element_and_the_bond_order(tmp_path):
    rex = _built(load_sdf(_write(tmp_path, "b.sdf", _BENZENE)))
    assert [rex.get_metadata(0, v, "element") for v in range(6)] == ["C"] * 5 + ["N"]
    assert rex.get_metadata(1, 0, "bond_order") == 4


def test_a_delocalised_relation_says_so_rather_than_borrowing_an_order(tmp_path):
    """It is not a bond, so it has no order. Putting a sentinel under `bond_order` also
    made that column mixed-type, and `_pack_cell_metadata` packs a mixed column as
    strings, so every bond order came back as "4" after a round trip."""
    rex = _built(load_sdf(_write(tmp_path, "b.sdf", _BENZENE)))
    wide = rex.nE - 1
    assert rex.get_metadata(1, wide, "relation_kind") == "delocalised"
    assert rex.get_metadata(1, wide, "bond_order") is None
    assert rex.get_metadata(1, 0, "relation_kind") == "bond"


def test_a_mixed_type_column_is_packed_as_strings(tmp_path):
    """Worth pinning, because it is silent: one string among the numbers under a key
    coerces the whole column. Keep a key to one type."""
    from rexgraph.io.rex_state import from_state, to_state

    rex = _built(load_sdf(_write(tmp_path, "b.sdf", _BENZENE)))
    rex.attach_metadata(1, 0, "mixed", 4)
    rex.attach_metadata(1, 1, "mixed", "four")
    assert from_state(to_state(rex)).get_metadata(1, 0, "mixed") == "4"


def test_pdb_keeps_the_chain_it_used_to_discard(tmp_path):
    """Parsed for the backbone pass, then dropped, so a file could not afterwards be
    filtered to one chain."""
    rex = _built(load_pdb(_write(tmp_path, "t.pdb", _PDB)))
    assert {rex.get_metadata(0, v, "chain") for v in range(rex.nV)} == {"A", "B"}
    assert [v for v in range(rex.nV) if rex.get_metadata(0, v, "chain") == "B"] == [2]


def test_pdb_reads_the_columns_it_never_opened(tmp_path):
    rex = _built(load_pdb(_write(tmp_path, "t.pdb", _PDB)))
    assert rex.get_metadata(0, 3, "element") == "ZN"
    assert rex.get_metadata(0, 3, "record") == "HETATM"
    assert rex.get_metadata(0, 3, "occupancy") == pytest.approx(0.5)
    assert rex.get_metadata(0, 1, "b_factor") == pytest.approx(21.5)


def test_pdb_coordinates_are_exact(tmp_path):
    """Fixed columns, three decimal places, so each is a Fraction over 10^3."""
    from fractions import Fraction

    construction = load_pdb(_write(tmp_path, "t.pdb", _PDB))
    assert construction.embedding[0][0] == Fraction("11.104")


def test_gff_keeps_the_whole_attribute_column(tmp_path):
    """Already parsed into a dict, of which two keys were read."""
    rex = _built(load_gff(_write(tmp_path, "t.gff", _GFF)))
    assert rex.get_metadata(0, 0, "biotype") == "protein_coding"
    assert rex.get_metadata(0, 1, "biotype") == "lncRNA"


def test_gff_keeps_the_standard_columns(tmp_path):
    rex = _built(load_gff(_write(tmp_path, "t.gff", _GFF)))
    assert rex.get_metadata(0, 0, "strand") == "+"
    assert rex.get_metadata(0, 1, "strand") == "-"
    assert rex.get_metadata(0, 0, "score") == pytest.approx(12.5)
    assert rex.get_metadata(0, 0, "source") == "havana"


def test_bed_keeps_score_and_strand(tmp_path):
    rex = _built(load_bed(_write(tmp_path, "t.bed", _BED)))
    assert rex.get_metadata(0, 0, "score") == pytest.approx(900.0)
    assert rex.get_metadata(0, 1, "strand") == "-"


def test_vcf_separates_the_variant_from_the_sample(tmp_path):
    """A bipartite complex, and the two sides carry different attributes."""
    rex = _built(load_vcf(_write(tmp_path, "t.vcf", _VCF)))
    assert rex.get_metadata(0, 0, "role") == "sample"
    assert rex.get_metadata(0, 2, "chrom") == "chr1"
    assert rex.get_metadata(0, 2, "filter") == "PASS"


def test_vcf_classifies_the_variant(tmp_path):
    rex = _built(load_vcf(_write(tmp_path, "t.vcf", _VCF)))
    assert rex.get_metadata(0, 2, "variant_type") == "snv"
    assert rex.get_metadata(0, 3, "variant_type") == "indel"


#### and the point of it: they are selectable


def test_a_selection_needs_no_label_parsing(tmp_path):
    """Which is the whole reason: "C12" is not a schema."""
    rex = _built(load_sdf(_write(tmp_path, "b.sdf", _BENZENE)))
    carbons = [v for v in range(rex.nV) if rex.get_metadata(0, v, "element") == "C"]
    assert carbons == [0, 1, 2, 3, 4]


def test_the_existing_criteria_filter_reaches_them(tmp_path):
    """subcomplex_by_criteria reads relation metadata, which now exists."""
    rex = _built(load_sdf(_write(tmp_path, "b.sdf", _BENZENE)))
    filtered = rex.subcomplex_by_criteria({"bond_order": 4})
    assert filtered.nE == 6


def test_a_reader_with_nothing_to_add_stays_empty(tmp_path):
    from agent.adapters.formats import load_fasta

    path = _write(tmp_path, "t.fa", ">a\nACGTACGTAC\n>b\nACGTACGTAC\n")
    assert load_fasta(path).attributes == {}

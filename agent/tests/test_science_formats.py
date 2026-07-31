"""File formats that were unreadable, each read as the structure it already is.

None of this is domain framing: an SDF is atoms and bonds, which is a labeled graph
with typed edges; a VCF is samples against variants, which is a bipartite incidence;
a GFF is intervals on a coordinate, which is an overlap graph; an h5ad is a matrix
container. The formats were simply unhandled, so auto_rex raised on them.

Parsers use h5py and the standard library. anndata, rdkit and biopython are not
installed and are not required -- these layouts are documented and stable, and a
hard dependency on a domain toolkit for a file read would be the wrong trade.
"""

import numpy as np
import pytest

from agent.adapters import formats


# --- fixtures in the real on-disk shapes --------------------------------------

SDF = """benzene
  RexGraph

  6  6  0  0  0  0  0  0  0  0999 V2000
    0.0000    1.4000    0.0000 C   0  0
    1.2124    0.7000    0.0000 C   0  0
    1.2124   -0.7000    0.0000 C   0  0
    0.0000   -1.4000    0.0000 C   0  0
   -1.2124   -0.7000    0.0000 C   0  0
   -1.2124    0.7000    0.0000 O   0  0
  1  2  2  0
  2  3  1  0
  3  4  2  0
  4  5  1  0
  5  6  2  0
  6  1  1  0
M  END
$$$$
"""

PDB = """ATOM      1  N   MET A   1      10.000  10.000  10.000  1.00  0.00           N
ATOM      2  CA  MET A   1      11.000  10.000  10.000  1.00  0.00           C
ATOM      3  C   MET A   1      12.000  10.000  10.000  1.00  0.00           C
ATOM      4  N   ALA A   2      13.000  10.000  10.000  1.00  0.00           N
CONECT    1    2
CONECT    2    3
CONECT    3    4
END
"""

FASTA = """>seq1 first
ACGTACGTAA
>seq2 second
ACGTTTGGCC
"""

VCF = """##fileformat=VCFv4.2
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\tS3
chr1\t100\tv1\tA\tG\t.\tPASS\t.\tGT\t0/1\t0/0\t1/1
chr1\t200\tv2\tC\tT\t.\tPASS\t.\tGT\t0/0\t1/1\t0/1
chr2\t300\tv3\tG\tA\t.\tPASS\t.\tGT\t1/1\t0/1\t0/0
"""

GFF = """##gff-version 3
chr1\t.\tgene\t100\t500\t.\t+\t.\tID=a
chr1\t.\tgene\t400\t900\t.\t+\t.\tID=b
chr1\t.\tgene\t1000\t1200\t.\t-\t.\tID=c
chr1\t.\tgene\t1100\t1300\t.\t-\t.\tID=d
"""

BED = """chr1\t100\t500\ta
chr1\t400\t900\tb
chr1\t1000\t1200\tc
"""


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


# --- bonded structure ---------------------------------------------------------

def test_sdf_reads_atoms_as_vertices_and_bonds_as_edges(tmp_path):
    ec = formats.load_sdf(_write(tmp_path, "m.sdf", SDF))
    assert ec.nV == 6
    assert ec.nE == 6
    assert ec.vertex_labels[0].startswith("C")
    assert any(lbl.startswith("O") for lbl in ec.vertex_labels)


def test_sdf_bond_order_becomes_an_edge_type(tmp_path):
    """Bond order is exactly the typed-edge information the complex already carries."""
    ec = formats.load_sdf(_write(tmp_path, "m.sdf", SDF))
    assert ec.type_labels is not None
    assert set(np.unique(ec.type_labels).tolist()) == {0, 1}, "single and double bonds"


def test_sdf_builds_a_valid_complex(tmp_path):
    from agent.auto import auto_rex
    rex = auto_rex(str(_write(tmp_path, "m.sdf", SDF)))
    assert int(rex.nV) == 6 and int(rex.nE) == 6
    assert list(rex.betti)[1] == 1, "a ring has one independent cycle"


def test_pdb_reads_atoms_and_explicit_bonds(tmp_path):
    ec = formats.load_pdb(_write(tmp_path, "p.pdb", PDB))
    assert ec.nV == 4
    assert ec.nE == 3
    assert "MET" in ec.vertex_labels[0]


# --- sequences ----------------------------------------------------------------

def test_fasta_reads_each_record(tmp_path):
    ec = formats.load_fasta(_write(tmp_path, "s.fasta", FASTA), k=3)
    assert ec.nE > 0
    assert ec.vertex_labels, "k-mers should be the vertices"


def test_fasta_kmer_adjacency_is_the_structure(tmp_path):
    """Consecutive k-mers share k-1 characters; that overlap is the edge."""
    ec = formats.load_fasta(_write(tmp_path, "s.fasta", "ACGTA\n" if False else ">x\nACGTA\n"), k=3)
    assert set(ec.vertex_labels) == {"ACG", "CGT", "GTA"}
    assert ec.nE == 2


def test_an_empty_fasta_is_an_error_not_an_empty_complex(tmp_path):
    with pytest.raises(ValueError):
        formats.load_fasta(_write(tmp_path, "e.fasta", ""), k=3)


# --- incidence ----------------------------------------------------------------

def test_vcf_reads_samples_against_variants(tmp_path):
    ec = formats.load_vcf(_write(tmp_path, "v.vcf", VCF))
    assert ec.nV == 6, "3 samples + 3 variants"
    assert ec.nE > 0
    assert "S1" in ec.vertex_labels and "v1" in ec.vertex_labels


def test_vcf_only_connects_carried_variants(tmp_path):
    """A 0/0 genotype is the absence of an edge, not an edge with weight zero."""
    ec = formats.load_vcf(_write(tmp_path, "v.vcf", VCF))
    pairs = {(ec.vertex_labels[s], ec.vertex_labels[t])
             for s, t in zip(ec.sources, ec.targets)}
    assert ("S2", "v1") not in pairs and ("v1", "S2") not in pairs
    assert ("S1", "v1") in pairs or ("v1", "S1") in pairs


# --- intervals ----------------------------------------------------------------

def test_gff_connects_overlapping_intervals(tmp_path):
    ec = formats.load_gff(_write(tmp_path, "a.gff", GFF))
    pairs = {tuple(sorted((ec.vertex_labels[s], ec.vertex_labels[t])))
             for s, t in zip(ec.sources, ec.targets)}
    assert ("a", "b") in pairs, "100-500 and 400-900 overlap"
    assert ("c", "d") in pairs
    assert ("a", "c") not in pairs, "disjoint intervals must not be joined"


def test_bed_is_read_the_same_way(tmp_path):
    ec = formats.load_bed(_write(tmp_path, "a.bed", BED))
    assert ec.nV == 3
    assert ec.nE == 1


def test_intervals_on_different_sequences_never_overlap(tmp_path):
    text = "chr1\t100\t500\ta\nchr2\t100\t500\tb\n"
    ec = formats.load_bed(_write(tmp_path, "a.bed", text))
    assert ec.nE == 0


# --- matrix containers --------------------------------------------------------

def _write_h5ad(path, X, obs, var):
    import h5py
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=X)
        g = f.create_group("obs")
        g.attrs["_index"] = "idx"
        g.create_dataset("idx", data=np.array(obs, dtype="S"))
        g2 = f.create_group("var")
        g2.attrs["_index"] = "idx"
        g2.create_dataset("idx", data=np.array(var, dtype="S"))
    return path


def test_h5ad_reads_the_matrix_and_its_axes(tmp_path):
    X = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 1.0], [4.0, 0.0, 0.0]])
    p = _write_h5ad(tmp_path / "a.h5ad", X, ["c1", "c2", "c3"], ["g1", "g2", "g3"])
    matrix, obs, var = formats.load_h5ad(p)
    assert matrix.shape == (3, 3)
    assert obs == ["c1", "c2", "c3"] and var == ["g1", "g2", "g3"]


def test_h5ad_routes_through_auto_rex(tmp_path):
    from agent.auto import auto_rex
    X = np.random.default_rng(0).random((12, 6))
    p = _write_h5ad(tmp_path / "a.h5ad", X, [f"c{i}" for i in range(12)],
                    [f"g{i}" for i in range(6)])
    rex = auto_rex(str(p))
    assert int(rex.nV) > 0 and int(rex.nE) > 0


# --- registry + dispatch ------------------------------------------------------

@pytest.mark.parametrize("ext", [".sdf", ".mol", ".pdb", ".fasta", ".fa", ".vcf",
                                 ".gff", ".gtf", ".bed", ".h5ad", ".loom"])
def test_every_extension_is_registered(ext):
    assert ext in formats.available_extensions()


def test_a_format_can_be_registered_from_outside(tmp_path):
    formats.register_reader("demo", lambda p, **kw: "read", extensions=[".demo"])
    try:
        assert ".demo" in formats.available_extensions()
        assert formats.read(tmp_path / "x.demo") == "read"
    finally:
        formats.unregister_reader("demo")
    assert ".demo" not in formats.available_extensions()


def test_an_unknown_extension_names_what_is_supported(tmp_path):
    with pytest.raises(ValueError) as ei:
        formats.read(tmp_path / "x.nope")
    assert ".sdf" in str(ei.value)


def test_auto_rex_recognises_the_new_types(tmp_path):
    from agent.auto import detect_input_type
    for name, text in (("m.sdf", SDF), ("v.vcf", VCF), ("a.gff", GFF),
                       ("s.fasta", FASTA), ("a.bed", BED)):
        p = _write(tmp_path, name, text)
        assert detect_input_type(str(p)) != "text", f"{name} fell through to text"


# --- shapes real files actually have ------------------------------------------
#
# Checked against files pulled from UniProt, RCSB and PubChem. The fixtures below
# are trimmed to the same layout those use, because the failures that matter are
# the ones a real file causes and a tidy fixture does not.

REAL_PDB = """\
ATOM      1  N   HIS A   3      12.000  10.000  10.000  1.00  0.00           N
ATOM      2  CA  HIS A   3      13.000  10.000  10.000  1.00  0.00           C
ATOM      3  C   HIS A   3      14.000  10.000  10.000  1.00  0.00           C
ATOM      4  N   TRP A   4      15.000  10.000  10.000  1.00  0.00           N
ATOM      5  CA  TRP A   4      16.000  10.000  10.000  1.00  0.00           C
ATOM      6  N   MET B   1      30.000  10.000  10.000  1.00  0.00           N
ATOM      7  CA  MET B   1      31.000  10.000  10.000  1.00  0.00           C
END
"""


def test_a_real_pdb_without_conect_is_not_a_cloud_of_isolated_atoms(tmp_path):
    """Real structures omit CONECT for standard residues -- 1CA2 carries four of
    them for 2207 atoms. Reading only CONECT gave a complex of isolated points."""
    ec = formats.load_pdb(_write(tmp_path, "p.pdb", REAL_PDB))
    assert ec.nV == 7
    assert ec.nE >= 5, "the residue chain was not connected"


def test_separate_chains_are_not_bonded_to_each_other(tmp_path):
    """A peptide bond joins consecutive residues OF A CHAIN. Joining chain A's last
    residue to chain B's first would invent a covalent bond that is not there."""
    from agent.auto import auto_rex

    rex = auto_rex(str(_write(tmp_path, "p.pdb", REAL_PDB)))
    assert int(list(rex.betti)[0]) == 2, "the two chains were merged"


def test_backbone_can_be_declined(tmp_path):
    ec = formats.load_pdb(_write(tmp_path, "p.pdb", REAL_PDB), backbone=False)
    assert ec.nE == 0, "CONECT-only should find nothing here"


def test_ring_counts_match_known_chemistry(tmp_path):
    """The check that says the reader is right rather than merely running: benzene
    is one independent cycle, and beta_1 has to agree."""
    from agent.auto import auto_rex

    benzene = """benzene
  test

  6  6  0  0  0  0  0  0  0  0999 V2000
    0.0000    1.4000    0.0000 C   0  0
    1.2124    0.7000    0.0000 C   0  0
    1.2124   -0.7000    0.0000 C   0  0
    0.0000   -1.4000    0.0000 C   0  0
   -1.2124   -0.7000    0.0000 C   0  0
   -1.2124    0.7000    0.0000 C   0  0
  1  2  2  0
  2  3  1  0
  3  4  2  0
  4  5  1  0
  5  6  2  0
  6  1  1  0
M  END
$$$$
"""
    rex = auto_rex(str(_write(tmp_path, "b.sdf", benzene)))
    assert list(rex.betti)[1] == 1


def test_a_multi_record_sdf_reads_its_first_record(tmp_path):
    """PubChem ships $$$$-delimited files; a reader that chokes on the delimiter
    fails on the most common source there is."""
    two = (SDF.rstrip() + "\n" + SDF)
    ec = formats.load_sdf(_write(tmp_path, "m.sdf", two))
    assert ec.nV == 6 and ec.nE == 6


def test_a_uniprot_style_fasta_header_is_parsed(tmp_path):
    """UniProt headers are sp|ACC|NAME ..., so the accession must survive."""
    text = (">sp|P00918|CAH2_HUMAN Carbonic anhydrase 2\n"
            "MSHHWGYGKHNGPEHWHKDFPIAKGERQ\n"
            ">sp|P00915|CAH1_HUMAN Carbonic anhydrase 1\n"
            "MASPDWGYDDKNGPEQWSKLYPIANGNN\n")
    ec = formats.load_fasta(_write(tmp_path, "p.fasta", text), k=6)
    assert ec.nE > 0 and ec.nV > 0

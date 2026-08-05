"""Every input type in, a complex out, through every serializer, and back.

Real files, not fixtures-of-fixtures. This is the walkthrough a user performs when
they point the software at their data: whatever they have loads, and whatever loads
survives being stored and read again.
"""
from __future__ import annotations

import numpy as np
import pytest

#### one real sample per supported input type
TEXT_FILES = {
    "prose.txt": ("Alpha connects beta. Beta connects gamma. Gamma connects alpha "
                  "and delta. Delta connects epsilon. Epsilon connects alpha."),
    "edges.csv": "source,target\na,b\nb,c\nc,a\nc,d\nd,e\ne,a\n",
    "edges.tsv": "source\ttarget\na\tb\nb\tc\nc\ta\nc\td\n",
    "edges.json": ('{"edges":[{"source":"a","target":"b"},{"source":"b","target":"c"},'
                   '{"source":"c","target":"a"},{"source":"c","target":"d"}]}'),
    "nx.json": ('{"nodes":[{"id":"a"},{"id":"b"},{"id":"c"}],'
                '"links":[{"source":"a","target":"b"},{"source":"b","target":"c"}]}'),
    "adj.json": "[[0,1,0,1],[1,0,1,0],[0,1,0,1],[1,0,1,0]]",
    "pairs.json": '{"edges":[["a","b"],["b","c"],["c","a"]]}',
    "s.fasta": (">seq1 first\nMKTAYIAKQRQISFVK\n>seq2 second\nMKTAYIAKQRQISFVL\n"
                ">seq3 third\nMKTAYIAKQRQISFVM\n"),
    "r.bed": ("chr1\t100\t200\tfeatA\t0\t+\nchr1\t150\t250\tfeatB\t0\t-\n"
              "chr1\t300\t400\tfeatC\t0\t+\nchr2\t100\t200\tfeatD\t0\t+\n"),
    "a.gff": ("##gff-version 3\n"
              "chr1\t.\tgene\t100\t200\t.\t+\t.\tID=g1;Name=alpha\n"
              "chr1\t.\tmRNA\t100\t200\t.\t+\t.\tID=m1;Parent=g1\n"
              "chr1\t.\texon\t100\t150\t.\t+\t.\tID=e1;Parent=m1\n"
              "chr1\t.\texon\t160\t200\t.\t+\t.\tID=e2;Parent=m1\n"),
    "v.vcf": ("##fileformat=VCFv4.2\n"
              "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\n"
              "chr1\t100\trs1\tA\tG\t50\tPASS\t.\tGT\t0/1\t1/1\n"
              "chr1\t200\trs2\tC\tT\t60\tPASS\t.\tGT\t0/0\t0/1\n"
              "chr1\t300\trs3\tG\tA\t70\tPASS\t.\tGT\t1/1\t0/1\n"),
}


def _feature_csv() -> str:
    rows = ("\n".join(",".join(f"{v:.3f}" for v in np.random.RandomState(i).rand(6))
                      for i in range(12)))
    return "f1,f2,f3,f4,f5,f6\n" + rows


def _pdb() -> str:
    return "".join(
        f"ATOM  {i:5d}  CA  ALA A{i:4d}    "
        f"{i * 3.8:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C\n"
        for i in range(1, 9)) + "END\n"


def _sdf() -> str:
    return ("mol1\n  test\n\n  4  3  0  0  0  0  0  0  0  0999 V2000\n"
            + "".join(f"    {i:.4f}    0.0000    0.0000 C   0  0\n" for i in range(4))
            + "  1  2  1  0\n  2  3  1  0\n  3  4  1  0\nM  END\n$$$$\n")


@pytest.fixture(scope="module")
def samples(tmp_path_factory) -> dict:
    d = tmp_path_factory.mktemp("formats")
    out = {}
    for name, body in TEXT_FILES.items():
        p = d / name
        p.write_text(body)
        out[name] = str(p)
    for name, body in (("feat.csv", _feature_csv()), ("p.pdb", _pdb()), ("m.sdf", _sdf())):
        p = d / name
        p.write_text(body)
        out[name] = str(p)
    return out


FILE_CASES = ["prose.txt", "edges.csv", "edges.tsv", "feat.csv", "edges.json",
              "nx.json", "adj.json", "pairs.json", "s.fasta", "r.bed", "a.gff",
              "v.vcf", "p.pdb", "m.sdf"]


@pytest.mark.parametrize("name", FILE_CASES)
def test_every_file_type_builds_a_complex(samples, name):
    from agent.auto import auto_rex
    rex = auto_rex(samples[name])
    assert rex is not None, f"{name} produced nothing"
    assert rex.nV > 0 and rex.nE > 0, f"{name} produced an empty complex"


@pytest.mark.parametrize("obj_name", ["feature_matrix", "correlation", "adjacency", "text"])
def test_every_in_memory_input_builds_a_complex(obj_name):
    from agent.auto import auto_rex
    data = {
        "feature_matrix": np.random.RandomState(0).rand(24, 6),
        "correlation": np.corrcoef(np.random.RandomState(1).rand(8, 30)),
        "adjacency": (np.random.RandomState(2).rand(9, 9) > 0.6).astype(float),
        "text": ("Sodium binds chloride. Chloride binds potassium. "
                 "Potassium binds sodium and calcium."),
    }[obj_name]
    rex = auto_rex(data)
    assert rex is not None and rex.nV > 0 and rex.nE > 0


def _shape(rex):
    return (int(rex.nV), int(rex.nE), int(getattr(rex, "nF", 0) or 0))


@pytest.mark.parametrize("name", ["prose.txt", "edges.csv", "feat.csv", "s.fasta", "v.vcf"])
def test_a_complex_survives_every_serializer(samples, tmp_path, name):
    """.rex, safetensors, zarr, hdf5 and the RCDB's own encoder all have to return
    the complex that went in. `load_safetensors` returns a dict by design, with the
    complex under "object"."""
    from agent.auto import auto_rex
    from agent.rcdb import deserialize_complex, serialize_complex
    from rexgraph.io import (load_hdf5, load_rex, load_safetensors, load_zarr,
                             save_hdf5, save_rex, save_safetensors, save_zarr)
    rex = auto_rex(samples[name])
    want = _shape(rex)
    stem = str(tmp_path / name.replace(".", "_"))

    save_rex(stem + ".rex", rex)
    assert _shape(load_rex(stem + ".rex")) == want, ".rex"

    save_safetensors(stem + ".safetensors", rex)
    assert _shape(load_safetensors(stem + ".safetensors")["object"]) == want, "safetensors"

    save_zarr(stem + ".zarr", rex)
    assert _shape(load_zarr(stem + ".zarr")) == want, "zarr"

    save_hdf5(stem + ".h5", rex)
    assert _shape(load_hdf5(stem + ".h5")) == want, "hdf5"

    assert _shape(deserialize_complex(serialize_complex(rex))) == want, "rcdb"

    # arrow round-trips the complex through rex_to_arrow / arrow_to_rex.
    # write_arrow_ipc is the lower-level array-dict transport, not this.
    from rexgraph.io import HAS_ARROW, arrow_to_rex, rex_to_arrow
    if HAS_ARROW:
        assert _shape(arrow_to_rex(rex_to_arrow(rex))) == want, "arrow"


def test_json_edge_lists_accept_pairs_and_objects(tmp_path):
    """`{"edges": [["a","b"]]}` is a common shape and used to raise AttributeError
    from inside the key finder. Numeric lists of lists stay adjacency matrices,
    because [[0,1],[1,0]] is a matrix and reading it as edges changes the file."""
    from rexgraph.io import load_json
    cases = {
        "wrapped_pairs.json": ('{"edges":[["a","b"],["b","c"],["c","a"]]}', 3, 3),
        "bare_pairs.json": ('[["a","b"],["b","c"],["c","a"]]', 3, 3),
        "objects.json": ('{"edges":[{"source":"a","target":"b"}]}', 2, 1),
        "matrix4.json": ("[[0,1,0,1],[1,0,1,0],[0,1,0,1],[1,0,1,0]]", 4, 4),
        "matrix2.json": ("[[0,1],[1,0]]", 2, 1),
    }
    for fname, (body, nv, ne) in cases.items():
        p = tmp_path / fname
        p.write_text(body)
        rex = load_json(str(p))
        assert (rex.nV, rex.nE) == (nv, ne), f"{fname}: got {(rex.nV, rex.nE)}"


def test_a_malformed_edge_list_says_what_is_wrong(tmp_path):
    """The failure has to name the offending entry, not surface as an AttributeError
    from three frames down."""
    from rexgraph.io import load_json
    p = tmp_path / "ragged.json"
    p.write_text('{"edges":[["a","b"],["c"]]}')
    with pytest.raises(ValueError, match="pair"):
        load_json(str(p))
    p2 = tmp_path / "scalars.json"
    p2.write_text('{"edges":["a","b"]}')
    with pytest.raises(ValueError, match="source/target"):
        load_json(str(p2))


#### defects found by walking every reader with real files


def test_an_isolated_record_survives_wherever_it_sits(tmp_path):
    """nV came from max(edge index)+1, so a record with no relation was kept when
    another followed it and dropped when it was last. The same five intervals in a
    different order gave a different complex and a different beta_0."""
    from agent.auto import auto_rex
    rows = ["chr1\t100\t200\tA", "chr1\t150\t250\tB",
            "chr1\t300\t400\tC", "chr1\t350\t450\tD"]
    lonely = "chr1\t900\t950\tLONELY"
    shapes = []
    for pos in (2, len(rows)):
        body = "\n".join(rows[:pos] + [lonely] + rows[pos:]) + "\n"
        p = tmp_path / f"bed{pos}.bed"
        p.write_text(body)
        rex = auto_rex(str(p))
        labels = (rex._agent_meta or {}).get("vertex_labels") or []
        assert rex.nV == len(labels), f"nV {rex.nV} != {len(labels)} labels"
        shapes.append((rex.nV, rex.nE, int(rex.betti[0])))
    assert shapes[0] == shapes[1], f"order changed the complex: {shapes}"
    assert shapes[0][0] == 5, shapes


def test_an_unbonded_atom_is_its_own_component(tmp_path):
    """A counter-ion bonds to nothing and is still an atom."""
    from agent.auto import auto_rex
    p = tmp_path / "iso.sdf"
    p.write_text("m\n t\n\n  3  1  0  0  0  0  0  0  0  0999 V2000\n"
                 + "".join(f"    {i:.4f}    0.0000    0.0000 C   0  0\n" for i in range(2))
                 + "    9.0000    0.0000    0.0000 Na  0  0\n  1  2  1  0\nM  END\n$$$$\n")
    rex = auto_rex(str(p))
    assert rex.nV == 3, rex.nV
    assert int(rex.betti[0]) == 2, f"beta_0 {rex.betti[0]}, the ion was absorbed"


def _pdb_line(rec, serial, name, res, chain, resseq, x):
    return (f"{rec:<6}{serial:5d}  {name:<3} {res} {chain}{resseq:4d}    "
            f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C\n")


def test_waters_are_not_bonded_to_the_chain(tmp_path):
    """Consecutive residues were joined by a peptide bond with no test that either
    was a polymer residue, so the waters of any ordinary structure got chained."""
    from agent.auto import auto_rex
    body = "".join(_pdb_line("ATOM", i, "CA", r, "A", i, i * 3.8)
                   for i, r in enumerate(["ALA", "GLY", "SER"], start=1))
    body += "".join(_pdb_line("HETATM", j, "O", "HOH", "A", 100 + j, j * 9.0)
                    for j in range(4, 7))
    p = tmp_path / "prot.pdb"
    p.write_text(body + "END\n")
    rex = auto_rex(str(p))
    assert rex.nV == 6, rex.nV
    assert rex.nE == 2, f"{rex.nE} bonds; the waters were bonded to something"
    assert int(rex.betti[0]) == 4, f"beta_0 {rex.betti[0]}: chain + 3 free waters"


def test_a_multi_model_pdb_loads(tmp_path):
    """Every NMR structure repeats its serials per MODEL. Round-tripping index to
    serial and back made that a KeyError."""
    from agent.auto import auto_rex
    body = "".join(
        f"MODEL     {m}\n"
        + "".join(_pdb_line("ATOM", i, "CA", "ALA", "A", i, i * 3.8) for i in range(1, 5))
        + "ENDMDL\n" for m in (1, 2))
    p = tmp_path / "nmr.pdb"
    p.write_text(body + "END\n")
    rex = auto_rex(str(p))
    assert rex.nV == 8, f"{rex.nV}: a model was lost"
    assert rex.nE > 0


def test_a_plain_chain_still_bonds(tmp_path):
    """The guard must not cost the peptide bonds it exists to qualify."""
    from agent.auto import auto_rex
    p = tmp_path / "chain.pdb"
    p.write_text("".join(_pdb_line("ATOM", i, "CA", "ALA", "A", i, i * 3.8)
                         for i in range(1, 6)) + "END\n")
    rex = auto_rex(str(p))
    assert (rex.nV, rex.nE, int(rex.betti[0])) == (5, 4, 1)


def test_vcf_finds_gt_where_format_says_it_is(tmp_path):
    """GT was read as sub-field 0 unconditionally, so a DP-only record's read depth
    became "carries" and a 0/0 sample behind a DP field got an edge."""
    from agent.auto import auto_rex
    head = ("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO"
            "\tFORMAT\tS1\tS2\n")

    p = tmp_path / "nogt.vcf"
    p.write_text(head + "chr1\t100\trsA\tA\tG\t50\tPASS\t.\tDP\t30\t0\n")
    try:
        assert auto_rex(str(p)).nE == 0, "read depth was read as a genotype"
    except ValueError:
        pass                                   # no edges at all is also correct

    p = tmp_path / "gtlast.vcf"
    p.write_text(head + "chr1\t100\trsA\tA\tG\t50\tPASS\t.\tDP:GT\t30:0/1\t20:0/0\n")
    assert auto_rex(str(p)).nE == 1, "the 0/0 sample was counted as a carrier"

    p = tmp_path / "plain.vcf"
    p.write_text(head + "chr1\t100\trsA\tA\tG\t50\tPASS\t.\tGT\t0/1\t1/1\n"
                        "chr1\t200\trsB\tC\tT\t60\tPASS\t.\tGT\t0/0\t0/1\n")
    assert auto_rex(str(p)).nE == 3, "the ordinary case regressed"


def test_gff_coordinates_are_inclusive(tmp_path):
    """GFF is 1-based inclusive; the overlap sweep is half-open like BED. Passing
    them through unconverted made features sharing one base read as disjoint."""
    from agent.auto import auto_rex
    head = "##gff-version 3\n"
    p = tmp_path / "adj.gff"
    p.write_text(head + "chr1\t.\tgene\t100\t200\t.\t+\t.\tID=A\n"
                        "chr1\t.\tgene\t200\t300\t.\t+\t.\tID=B\n")
    assert auto_rex(str(p)).nE == 1, "features sharing base 200 did not overlap"

    p = tmp_path / "dis.gff"
    p.write_text(head + "chr1\t.\tgene\t100\t200\t.\t+\t.\tID=A\n"
                        "chr1\t.\tgene\t201\t300\t.\t+\t.\tID=B\n")
    assert auto_rex(str(p)).nE == 0, "adjacent-but-disjoint features were joined"


def test_bed_track_and_browser_are_directives_not_prefixes(tmp_path):
    """`startswith("track")` also matched a contig named track1."""
    from agent.auto import auto_rex
    p = tmp_path / "tc.bed"
    p.write_text("track name=x\ntrack1\t100\t200\tA\ntrack1\t150\t250\tB\n"
                 "browser9\t100\t200\tC\nbrowser9\t150\t250\tD\nchr1\t10\t20\tE\n")
    rex = auto_rex(str(p))
    labels = (rex._agent_meta or {}).get("vertex_labels") or []
    assert labels == ["A", "B", "C", "D", "E"], labels


def test_gtf_labels_the_feature_not_only_its_gene(tmp_path):
    """GTF has no ID=, so gene_id won for a gene, its transcripts and its exons
    alike and every row of one gene carried the same label."""
    from agent.auto import auto_rex
    p = tmp_path / "t.gtf"
    p.write_text('chr1\t.\tgene\t100\t200\t.\t+\t.\tgene_id "G1";\n'
                 'chr1\t.\ttranscript\t100\t200\t.\t+\t.\tgene_id "G1"; transcript_id "T1";\n'
                 'chr1\t.\texon\t100\t150\t.\t+\t.\tgene_id "G1"; transcript_id "T1"; '
                 'exon_number "1";\n')
    labels = (auto_rex(str(p))._agent_meta or {}).get("vertex_labels") or []
    assert len(set(labels)) == 3, f"features are not distinguishable: {labels}"


def test_a_malformed_mol_record_is_an_error_not_filler(tmp_path):
    """A counts line claiming more atoms than the record holds made the reader take
    the bond table and `M  END` as atoms, then read the real bonds past the end of
    the file and lose them: 5 labels, 3 of them invented, and no bonds at all."""
    from agent.auto import auto_rex
    p = tmp_path / "trunc.sdf"
    p.write_text("m\n t\n\n  5  4  0  0  0  0  0  0  0  0999 V2000\n"
                 "    0.0000    0.0000    0.0000 C   0  0\n"
                 "    1.0000    0.0000    0.0000 C   0  0\n"
                 "  1  2  1  0\nM  END\n$$$$\n")
    with pytest.raises(ValueError, match="counts line"):
        auto_rex(str(p))


def test_multi_record_sdf_still_reads_every_record(tmp_path):
    """The counts check must not cost the valid multi-record case."""
    from agent.auto import auto_rex

    def rec(n):
        return (f"m{n}\n t\n\n  3  2  0  0  0  0  0  0  0  0999 V2000\n"
                + "".join(f"    {i:.4f}    0.0000    0.0000 C   0  0\n" for i in range(3))
                + "  1  2  1  0\n  2  3  2  0\nM  END\n$$$$\n")
    p = tmp_path / "multi.sdf"
    p.write_text(rec(1) + rec(2))
    rex = auto_rex(str(p))
    assert (rex.nV, rex.nE) == (6, 4), (rex.nV, rex.nE)


def test_h5ad_uses_a_name_column_not_whatever_sorts_first(tmp_path):
    """With no `_index`, the reader took the first key in HDF5 order, so a boolean
    flag column became the gene names."""
    h5py = pytest.importorskip("h5py")
    import numpy as np
    from agent.adapters.formats import read
    p = tmp_path / "n.h5ad"
    with h5py.File(p, "w") as f:
        f.create_dataset("X", data=np.random.RandomState(0).rand(6, 4))
        v = f.create_group("var")
        v.create_dataset("a_flag", data=np.array([True, False, True, False]))
        v.create_dataset("gene_symbol",
                         data=np.array([b"GENE1", b"GENE2", b"GENE3", b"GENE4"]))
        f.create_group("obs").create_dataset(
            "_index", data=np.array([f"c{i}".encode() for i in range(6)]))
    assert read(str(p))[2] == ["GENE1", "GENE2", "GENE3", "GENE4"]


def test_loom_finds_gene_names_under_the_names_loompy_writes(tmp_path):
    """Only `Gene`/`gene` were looked for, so a file using `Name` left every gene
    unlabelled and the vertices fell back to f0, f1, f2."""
    h5py = pytest.importorskip("h5py")
    import numpy as np
    from agent.adapters.formats import read
    p = tmp_path / "n.loom"
    with h5py.File(p, "w") as f:
        f.create_dataset("matrix", data=np.random.RandomState(0).rand(6, 4).T)
        f.create_group("row_attrs").create_dataset(
            "Name", data=np.array([b"GENE1", b"GENE2", b"GENE3", b"GENE4"]))
        f.create_group("col_attrs").create_dataset(
            "CellID", data=np.array([f"c{i}".encode() for i in range(6)]))
    assert read(str(p))[2] == ["GENE1", "GENE2", "GENE3", "GENE4"]


def test_bcf_says_it_is_binary(tmp_path):
    """`.bcf` was registered to the text VCF parser, which failed with "no #CHROM
    header": true, and a description of the wrong problem."""
    import gzip
    from agent.auto import auto_rex
    p = tmp_path / "v.bcf"
    p.write_bytes(b"BCF\x02\x02" + gzip.compress(b"body"))
    with pytest.raises(ValueError, match="binary"):
        auto_rex(str(p))


def test_reader_options_reach_the_reader(tmp_path):
    """auto_rex filtered kwargs to ("k",), so every other documented switch was
    accepted and dropped: load_pdb(backbone=False) had no effect through it."""
    from agent.adapters.formats import read
    from agent.auto import auto_rex
    p = tmp_path / "p.pdb"
    p.write_text("".join(_pdb_line("ATOM", i, "CA", "ALA", "A", i, i * 3.8)
                         for i in range(1, 6)) + "END\n")
    assert auto_rex(str(p)).nE == 4, "the default lost its backbone"
    assert auto_rex(str(p), backbone=False).nE == read(str(p), backbone=False).nE == 0

    f = tmp_path / "s.fasta"
    f.write_text(">a\nACGTACGTAA\n>b\nACGTACGTAC\n")
    assert auto_rex(str(f), k=3).nV > 0 and auto_rex(str(f), k=5).nV > 0


def test_a_face_is_a_filled_cycle_of_any_gon(tmp_path):
    """Faces come from `rexgraph.faces`, which solves B1 c = 0 exactly and reads the
    gon off the cycle basis. The agent path used to run a triangle-only, type-gated
    rule instead, so a 4-gon could never close and a ring with a double bond in it
    was rejected for having edges that disagreed."""
    import numpy as np
    from rexgraph.graph import RexGraph
    from rexgraph.faces import autoface

    square = RexGraph(sources=np.array([0, 1, 2, 3], np.int32),
                      targets=np.array([1, 2, 3, 0], np.int32))
    assert autoface(square, k=3) == 0, "a square is not a triangle"
    assert autoface(square, k=4) == 1
    assert int(square.betti[1]) == 0

    mixed = RexGraph(sources=np.array([0, 1, 2, 3, 4, 5, 6], np.int32),
                     targets=np.array([1, 2, 0, 4, 5, 6, 3], np.int32))
    assert autoface(mixed, k=[3, 4]) == 2, "k has to accept several gons"
    assert int(mixed.betti[1]) == 0


def test_bond_order_weights_the_complex_and_does_not_gate_faces(tmp_path):
    """Bond order is a magnitude, so it is w_E. A ring is a ring whichever bonds
    close it: the ring below has a double bond and still closes."""
    from agent.adapters.formats import read
    from agent.auto import auto_rex
    p = tmp_path / "ring.sdf"
    p.write_text("cyclopropene\n t\n\n  3  3  0  0  0  0  0  0  0  0999 V2000\n"
                 + "".join(f"    {i:.4f}    0.0000    0.0000 C   0  0\n" for i in range(3))
                 + "  1  2  1  0\n  2  3  2  0\n  1  3  1  0\nM  END\n$$$$\n")
    assert list(read(str(p)).weights) == [1.0, 2.0, 1.0], "bond order is not the weight"

    rex = auto_rex(str(p), face_selection="auto")
    assert rex.nF == 1 and int(rex.betti[1]) == 0, "the ring did not close"
    assert list(rex.w_E) == [1.0, 2.0, 1.0], "w_E did not reach the complex"


def test_faces_are_not_assumed(tmp_path):
    """Asserting a face asserts something is enclosed. That is the caller's claim
    about their data, so nothing fills unless asked."""
    from agent.auto import auto_rex
    p = tmp_path / "ring.sdf"
    p.write_text("c\n t\n\n  3  3  0  0  0  0  0  0  0  0999 V2000\n"
                 + "".join(f"    {i:.4f}    0.0000    0.0000 C   0  0\n" for i in range(3))
                 + "  1  2  1  0\n  2  3  1  0\n  1  3  1  0\nM  END\n$$$$\n")
    bare = auto_rex(str(p))
    assert bare.nF == 0 and int(bare.betti[1]) == 1
    asked = auto_rex(str(p), face_selection="auto")
    assert asked.nF == 1 and int(asked.betti[1]) == 0


def test_a_bipartite_complex_closes_at_its_own_gon(tmp_path):
    """A VCF is samples against variants, so its cycles are 4-gons. Under the old
    triangle rule it could never close one."""
    from agent.auto import auto_rex
    p = tmp_path / "t.vcf"
    p.write_text("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\t"
                 "INFO\tFORMAT\tS1\tS2\tS3\n"
                 + "".join(f"chr1\t{100 + i}\trs{i}\tA\tG\t50\tPASS\t.\tGT\t0/1\t1/1\t0/1\n"
                           for i in range(3)))
    assert auto_rex(str(p), face_selection=3).nF == 0, "there are no triangles here"
    closed = auto_rex(str(p), face_selection="auto")
    assert closed.nF == 4 and int(closed.betti[1]) == 0


def test_an_isolated_vertex_survives_a_save(tmp_path):
    """beta_0 must not move across a save.

    The boundary arrays only witness vertices that carry a relation, so a 0-cell
    incident to nothing is invisible to them. The state header records nV and the
    reader dropped it, so an isolated vertex lived in memory and vanished on reload.
    """
    import numpy as np
    from agent.rcdb import deserialize_complex, serialize_complex
    from rexgraph.graph import RexGraph
    from rexgraph.io import (load_hdf5, load_rex, load_safetensors, load_zarr,
                             save_hdf5, save_rex, save_safetensors, save_zarr)

    rex = RexGraph(sources=np.array([0, 1], np.int32), targets=np.array([1, 2], np.int32))
    rex._nV = 6                                   # three isolated 0-cells
    assert (rex.nV, int(rex.betti[0])) == (6, 4)

    stem = str(tmp_path / "iso")
    save_rex(stem + ".rex", rex)
    save_zarr(stem + ".zarr", rex)
    save_hdf5(stem + ".h5", rex)
    save_safetensors(stem + ".st", rex)
    got = {
        ".rex": load_rex(stem + ".rex").nV,
        ".zarr": load_zarr(stem + ".zarr").nV,
        ".h5": load_hdf5(stem + ".h5").nV,
        ".safetensors": load_safetensors(stem + ".st")["object"].nV,
        "rcdb": deserialize_complex(serialize_complex(rex)).nV,
    }
    assert all(v == 6 for v in got.values()), got


def test_a_bed_file_with_isolated_intervals_round_trips(tmp_path):
    """The case the roundtrip matrix caught: four intervals, two of which overlap
    nothing, stored and read back."""
    from agent.auto import auto_rex
    from rexgraph.io import load_rex, save_rex
    p = tmp_path / "r.bed"
    p.write_text("chr1\t100\t200\tA\nchr1\t150\t250\tB\n"
                 "chr1\t900\t950\tC\nchr2\t100\t200\tD\n")
    rex = auto_rex(str(p))
    assert rex.nV == 4, rex.nV
    out = str(tmp_path / "r.rex")
    save_rex(out, rex)
    assert load_rex(out).nV == 4


def test_face_selection_rejects_a_value_it_cannot_honour():
    """An unrecognised rule used to reach `autoface` and fail there with a TypeError
    about comparing str and int."""
    import pytest as _pytest
    from agent.auto import auto_rex
    txt = "Alpha connects beta. Beta connects gamma. Gamma connects alpha."
    with _pytest.raises(ValueError, match="not a face rule"):
        auto_rex(txt, face_selection="nonsense")
    # every word the codebase actually uses resolves
    for word in ("none", "auto", "all", "promote", "hyper", "typed"):
        auto_rex(txt, face_selection=word)

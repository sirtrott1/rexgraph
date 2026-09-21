from fractions import Fraction as Q
import gzip
import numpy as np
import pytest
from agent.adapters.smiles import read_smiles, load_reaction_smiles, match_smarts
from agent.adapters.formats import reader_for, read
from agent.auto import build_rex_from_edges
from rexgraph.molecular_field import MolecularView, molecular_changes
from rexgraph.coordinate_map import CoordinateMap
from rexgraph.chain_map import CoordinateComplex
from rexgraph.native_field import NativeFieldCalculus
from rexgraph.tensor_field import apply_tensor
from rexgraph.io.rex_state import to_state, from_state
from .helpers import make, declaration


def test_registered_reader(tmp_path):
    p,b,g=make(tmp_path)
    assert reader_for(p)=='smiles'
    assert read(p,document_id='fixture').nE==3
    v=MolecularView.from_source(g,'chain')
    assert tuple(v.field().values)==(Q(1),)*3
    assert v.balance()==((1,0,10),(6,0,4))


@pytest.mark.parametrize('text', ['CCO', 'OCC', 'COC', 'C(Cl)(Br)F', '[13CH3][C@H](O)C', 'F/C=C/F', 'F/C=C\\F', 'C%12CCCCC%12', 'C1CCCCC1C1CC1', '[Na+].[Cl-]', '[H][C]([H])([H])[H]', 'c1ccc2ccccc2c1'])
def test_syntax_and_source_spans(tmp_path,text):
    p,b,g=make(tmp_path,text+' mol\n',conformers={})
    v=MolecularView.from_source(g,'mol')
    raw=p.read_bytes()
    for t in v.record['tokens']:assert raw[t['start']:t['stop']].decode()==t['text']
    for bond in v.record['bonds']:
        attachment=g._cell_metadata[1][bond['cell']]['molecular_source']
        assert all(raw[int(a):int(z)] for _,a,z in attachment.text.components)
    assert len(v.record['atoms'])==len([t for t in v.record['tokens'] if t['kind']=='atom'])


def test_aromatic_primary_group_does_not_replace_bond_view(tmp_path):
    _,_,g=make(tmp_path,'c1ccccc1 ring\n',conformers={})
    v=MolecularView.from_source(g,'ring')
    assert g.nE==7 and len(v.record['bonds'])==6
    c=v.bond_complex()
    b=CoordinateMap(c.spaces[1],c.spaces[0],c.boundaries[0]).as_sparse()
    assert 6-len(b.rref()[1])==1
    assert tuple(v.field().values)==(Q(3,2),)*6
    assert set(g._agent_meta['type_names'])=={'AROMATIC','aromatic_system'}


def test_no_automatic_faces_and_isolated_witness(tmp_path):
    _,_,g=make(tmp_path,'[Na+].[Cl-] salts\n',conformers={})
    v=MolecularView.from_source(g,'salts')
    assert g.nE==2 and g.nF==0
    assert v.bond_complex().sizes==(2,0)
    assert tuple(v.field('charge').values)==(1,-1)


def test_conformer_and_atoms_roundtrip(tmp_path):
    _,_,g=make(tmp_path)
    before=MolecularView.from_source(g,'chain').conformation('plus')
    restored=from_state(to_state(g))
    after=MolecularView.from_source(restored,'chain').conformation('plus')
    assert np.array_equal(before.values,after.values)
    assert before.axes==after.axes
    assert before.source.state_digest==after.source.state_digest
    assert to_state(g).header['format_version']>=5 if 'format_version' in to_state(g).header else True


def test_large_exact_coordinates(tmp_path):
    vals={'map/1':[2**90+1,0,0], 'map/2':[0,0,0], 'map/3':[1,0,0], 'map/4':[1,Q(1,3),Q(2,7)]}
    _,_,g=make(tmp_path,conformers={'chain':{'large':declaration(values=vals)}})
    restored=from_state(to_state(g));v=MolecularView.from_source(restored,'chain').conformation('large')
    assert v.values[0,0]==2**90+1 and v.values[3,1]==Q(1,3)


def test_same_mapped_molecule_different_traversal(tmp_path):
    text='[CH3:1][CH2:2][OH:3] a\n[OH:3][CH2:2][CH3:1] b\n'
    _,_,g=make(tmp_path,text,conformers={})
    a,b=(MolecularView.from_source(g,k) for k in ('a','b'))
    diff=molecular_changes(a,b)
    assert all(not any(diff.field(k).values) for k in diff.names if k.endswith('/delta'))
    la,lb=a.bond_lift(),b.bond_lift()
    assert [v for _,_,v in la.entries]==[1,1]
    assert [v for _,_,v in lb.entries]==[-1,-1]


def test_unmapped_comparison_requires_explicit_alignment(tmp_path):
    _,_,g=make(tmp_path,'CCO a\nOCC b\n',conformers={})
    a,b=(MolecularView.from_source(g,k) for k in ('a','b'))
    with pytest.raises(ValueError,match='alignment'):molecular_changes(a,b)
    mapping=(('occurrence/0','occurrence/2','a'),('occurrence/1','occurrence/1','b'),('occurrence/2','occurrence/0','c'))
    assert not any(molecular_changes(a,b,mapping).field('bond_order/delta').values)


def test_isomers_differ_with_same_token_multiset(tmp_path):
    _,_,g=make(tmp_path,'[CH3:1][CH2:2][OH:3] ethanol\n[CH3:1][O:3][CH3:2] ether\n',conformers={})
    delta=molecular_changes(MolecularView.from_source(g,'ethanol'),MolecularView.from_source(g,'ether'))
    assert sum(abs(x) for x in delta.field('bond_presence/delta').values)==2
    assert not any(delta.field('inventory/delta').values)


def test_mapped_ring_opening(tmp_path):
    pytest.importorskip("rdkit")
    p=tmp_path/'reaction.rsmi'
    p.write_text('[CH2:1]1[CH2:2][O:3]1.[OH2:4]>>[CH2:1]([OH:4])[CH2:2][OH:3] opening\n')
    g=build_rex_from_edges(load_reaction_smiles(p,document_id='opening'))
    a,b=(MolecularView.from_source(g,'opening/'+k) for k in ('reactants','products'))
    result=molecular_changes(a,b)
    entries=dict(zip(result.field('bond_presence/delta').space.keys,result.field('bond_presence/delta').values,strict=False))
    assert entries['["map/1","map/3"]']==-1
    assert entries['["map/1","map/4"]']==1
    assert sum(abs(x) for x in entries.values())==2
    assert not any(result.field('inventory/delta').values)
    assert a.balance()==((1,0,6),(6,0,2),(8,0,2))


def test_duplicate_atom_maps_do_not_infer_lineage(tmp_path):
    _,_,g=make(tmp_path,'[CH3:1][OH:1] m\n',conformers={})
    v=MolecularView.from_source(g,'m')
    assert not v.record['complete_mapping']
    with pytest.raises(ValueError):molecular_changes(v,v)


def test_bond_order_change_without_connectivity_change(tmp_path):
    _,_,g=make(tmp_path,'[CH3:1][CH3:2] a\n[CH2:1]=[CH2:2] b\n',conformers={})
    delta=molecular_changes(MolecularView.from_source(g,'a'),MolecularView.from_source(g,'b'))
    assert tuple(delta.field('bond_presence/delta').values)==(0,)
    assert tuple(delta.field('bond_order/delta').values)==(1,)
    assert tuple(delta.field('hydrogens/delta').values)==(-1,-1)
    assert any(delta.field('inventory/delta').values)


def test_selected_smarts_has_real_source_spans(tmp_path):
    p,_,g=make(tmp_path,'CC(=O)O acid\n',conformers={})
    matches=match_smarts(g,'acid','C(=O)O')
    assert len(matches)==1 and len(matches[0]['atom_keys'])==3 and len(matches[0]['bond_cells'])==2
    spans=matches[0]['byte_spans']
    assert (1,2) in spans and (4,5) in spans and (6,7) in spans
    with pytest.raises(ValueError):match_smarts(g,'acid','C(',max_matches=5)
    with pytest.raises(ValueError,match='exceeds'):match_smarts(g,'acid','O',max_matches=1)


def test_green_native_bonds(tmp_path):
    _,_,g=make(tmp_path,'[CH3:1][CH2:2][OH:3] ethanol\n',conformers={})
    view=MolecularView.from_source(g,'ethanol')
    f=view.field('bond_presence',native=True).with_values([0,1])
    response=apply_tensor(NativeFieldCalculus.from_rex(g).green(1),f)
    assert tuple(response.values)==(Q(1,8),Q(3,8))


@pytest.mark.parametrize('text', ['C1CC', 'C(', 'CC|foo|', 'C->N', 'C C\nC C\n', 'C%123', 'C[bad]', 'C>O'])
def test_invalid_notation(tmp_path,text):
    pytest.importorskip("rdkit")
    p=tmp_path/'invalid.smi';p.write_text(text+'\n')
    with pytest.raises((ValueError,UnicodeError)):read_smiles(p,document_id='invalid')


def test_utf8_crlf_gzip_source_positions(tmp_path):
    pytest.importorskip("rdkit")
    p=tmp_path/'m.smi.gz'; raw=b'  CCO ethanol\r\nC1CC1 '+ '环'.encode()+b'\r\n'
    p.write_bytes(gzip.compress(raw))
    g=build_rex_from_edges(read_smiles(p,document_id='utf8').construction())
    v=MolecularView.from_source(g,'环')
    for t in v.record['tokens']:assert raw[t['start']:t['stop']].decode()==t['text']


def test_float_coordinates_are_refused(tmp_path):
    vals=declaration();vals['coordinates']['map/1'][0]=0.1
    with pytest.raises(TypeError,match='exact'):make(tmp_path,conformers={'chain':{'bad':vals}})


def test_unknown_or_missing_coordinate_atom_refused(tmp_path):
    vals=declaration();vals['coordinates'].pop('map/1')
    with pytest.raises(ValueError,match='every atom'):make(tmp_path,conformers={'chain':{'bad':vals}})


def test_double_bond_stereo_delta(tmp_path):
    text='[F:1]/[CH:2]=[CH:3]/[F:4] a\n[F:1]/[CH:2]=[CH:3]\\[F:4] b\n'
    _,_,g=make(tmp_path,text,conformers={})
    delta=molecular_changes(MolecularView.from_source(g,'a'),MolecularView.from_source(g,'b'))
    assert not any(delta.field('bond_presence/delta').values)
    assert not any(delta.field('bond_order/delta').values)
    assert sum(abs(x) for x in delta.field('bond_stereo/delta').values)==2


def test_primary_connection_ids_survive_mapped_source_reorder(tmp_path):
    a_dir=tmp_path/'a';b_dir=tmp_path/'b';a_dir.mkdir();b_dir.mkdir()
    _,_,a=make(a_dir,'[CH3:1][CH2:2][OH:3] molecule\n',conformers={})
    _,_,b=make(b_dir,'[OH:3][CH2:2][CH3:1] molecule\n',conformers={})
    assert set(a.relation_ids)==set(b.relation_ids)
    assert tuple(a.relation_ids)==tuple(reversed(b.relation_ids))


def test_bond_and_atom_lifts_commute(tmp_path):
    _,_,g=make(tmp_path,'c1ccccc1 ring\n',conformers={})
    view=MolecularView.from_source(g,'ring')
    c=view.bond_complex();native=CoordinateComplex.from_rex(g)
    b=CoordinateMap(c.spaces[1],c.spaces[0],c.boundaries[0])
    bn=CoordinateMap(native.spaces[1],native.spaces[0],native.boundaries[0])
    left=view.bond_lift().compose(bn);right=b.compose(view.atom_lift())
    assert left.entries==right.entries


def test_view_defends_against_mutated_private_selection(tmp_path):
    _,_,g=make(tmp_path)
    view=MolecularView.from_source(g,'chain')
    view._record['atoms'][0]['charge']=12
    with pytest.raises(ValueError,match='changed'):view.field('charge')


def test_bundle_identity_is_the_native_declaration_identity(tmp_path):
    _, bundle, source = make(tmp_path)
    assert bundle.digest == bundle.construction().source_manifest['bundle_digest']
    assert bundle.digest == source._agent_meta['source_manifest']['bundle_digest']


def test_lazy_native_exports():
    import rexgraph
    from rexgraph.molecular_field import MolecularView
    from rexgraph.process_field import response_direction
    assert rexgraph.MolecularView is MolecularView
    assert rexgraph.response_direction is response_direction

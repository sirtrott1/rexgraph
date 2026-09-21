from fractions import Fraction as Q
import pytest
from agent.adapters.smiles import read_smiles
from agent.auto import build_rex_from_edges

SMILES = '[CH3:1][CH2:2][CH2:3][CH3:4]'


def coordinates(t=Q(0)):
    t = Q(t)
    c, s = (1-t*t)/(1+t*t), 2*t/(1+t*t)
    return {'map/1': [0, 1, 0], 'map/2': [0, 0, 0], 'map/3': [1, 0, 0], 'map/4': [1, c, s]}


def declaration(t=Q(0), values=None):
    return {'coordinates': coordinates(t) if values is None else values,
            'unit': 'angstrom', 'frame': 'laboratory',
            'origin': 'synthetic rational fixture', 'source': 'explicit supplied coordinates'}


def make(tmp_path, text=SMILES+' chain\n', *, conformers=None, **kwargs):
    # SMILES is read through RDKit, the optional agent[molecules] parser.
    pytest.importorskip("rdkit")
    path = tmp_path/'source.smi'
    path.write_text(text,encoding='utf-8',newline='')
    if conformers is None and 'chain' in text:
        conformers = {'chain': {'zero': declaration(0), 'plus': declaration(1), 'minus': declaration(-1)}}
    bundle=read_smiles(path,document_id='fixture',map_namespace='mapped_atoms',conformers=conformers,**kwargs)
    source=build_rex_from_edges(bundle.construction(),face_selection='none')
    return path,bundle,source

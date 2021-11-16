import os
import subprocess
import json
import prody
import contextlib
import tempfile
import shutil
import numpy as np
from io import StringIO
from path import Path
from copy import deepcopy
from collections import OrderedDict

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

import Bio
from Bio.SubsMat import MatrixInfo as matlist
from Bio.pairwise2 import format_alignment

from .loggers import logger


# TODO: This produces annoying error log, when it hits atom number which is not in the table.
#       Need to fix it somehow.
def get_rdkit_elements():
    pt = Chem.GetPeriodicTable()
    elements = []
    for i in range(1000):
        try:
            elements.append(pt.GetElementSymbol(i))
        except:
            break
    return elements


@contextlib.contextmanager
def isolated_filesystem(dir=None, remove=True):
    """A context manager that creates a temporary folder and changes
    the current working directory to it for isolated filesystem tests.
    """
    cwd = os.getcwd()
    if dir is None:
        t = tempfile.mkdtemp(prefix='pocketdock-')
    else:
        t = dir
    os.chdir(t)
    try:
        yield t
    except Exception as e:
        logger.error(f'Error occured, temporary files are in {t}')
        raise
    else:
        os.chdir(cwd)
        if remove:
            try:
                shutil.rmtree(t)
            except (OSError, IOError):
                pass
    finally:
        os.chdir(cwd)


@contextlib.contextmanager
def cwd(dir):
    pwd = os.getcwd()
    try:
        os.chdir(dir)
        yield
    finally:
        os.chdir(pwd)


def tmp_file(**kwargs):
    handle, fname = tempfile.mkstemp(**kwargs)
    os.close(handle)
    return Path(fname)


def run_check_output(call, **kwargs):
    try:
        logger.debug('Running command:\n' + ' '.join(call))
        output = subprocess.check_output(call, **kwargs)
        logger.debug('Command output:\n' + output.decode('utf-8'))
    except subprocess.CalledProcessError as e:
        logger.debug('Command output:\n' + e.output.decode('utf-8'))
        raise
    return output.decode('utf-8')


def write_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def safe_read_ag(ag) -> prody.Atomic:
    if isinstance(ag, prody.AtomGroup):
        return ag
    elif isinstance(ag, str):
        return prody.parsePDB(ag)
    else:
        raise RuntimeError(f"Can't read atom group, 'ag' has wrong type {type(ag)}")


# from https://www.rdkit.org/docs/Cookbook.html
def read_mol_block(mol_block, no_valence_check=False, **kwargs):
    if mol_block is None:
        raise RuntimeError(f'Mol block is empty')
    sanitize = kwargs.get('sanitize', True)
    if no_valence_check:
        kwargs['sanitize'] = False
    mol = Chem.MolFromMolBlock(mol_block, **kwargs)
    if no_valence_check and sanitize:
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(mol,
                         Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
                         Chem.SanitizeFlags.SANITIZE_KEKULIZE |
                         Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                         Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
                         Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION |
                         Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                         catchErrors=True)
    return mol


def mol_to_ag(mol):
    return prody.parsePDBStream(StringIO(Chem.MolToPDBBlock(mol)))


def ag_to_mol_assign_bonds(ag, mol_template):
    output = StringIO()
    prody.writePDBStream(output, ag)
    ag_mol = AllChem.MolFromPDBBlock(output.getvalue())
    ag_mol = AllChem.AssignBondOrdersFromTemplate(mol_template, ag_mol)
    return ag_mol


def apply_prody_transform(coords, tr):
    return np.dot(coords, tr.getRotation().T) + tr.getTranslation()


def minimize_rmsd(mob_ag, ref_ag, mob_serials=None, ref_serials=None, mob_cset=None, ref_cset=None):
    if mob_serials is not None and ref_serials is not None:
        mob_sel = mob_ag.select('serial ' + ' '.join(map(str, mob_serials)))
        ref_sel = ref_ag.select('serial ' + ' '.join(map(str, ref_serials)))
        mob_s2i = dict(zip(mob_sel.getSerials(), mob_sel.getIndices()))
        ref_s2i = dict(zip(ref_sel.getSerials(), ref_sel.getIndices()))
        mob_ids = [mob_s2i[s] for s in mob_serials]
        ref_ids = [ref_s2i[s] for s in ref_serials]
    else:
        mob_ids = mob_ag.all.getIndices()
        ref_ids = ref_ag.all.getIndices()

    if mob_cset is not None:
        mob_crd = mob_ag.getCoordsets(mob_cset)[mob_ids]
    else:
        mob_crd = mob_ag.getCoords()[mob_ids]

    if ref_cset is not None:
        ref_crd = ref_ag.getCoordsets(ref_cset)[ref_ids]
    else:
        ref_crd = ref_ag.getCoords()[ref_ids]

    tr = prody.calcTransformation(mob_crd, ref_crd)
    rmsd_minimized = prody.calcRMSD(apply_prody_transform(mob_crd, tr), ref_crd)
    transformation = tr.getMatrix().flatten().tolist()
    return rmsd_minimized, transformation


def change_mol_coords(mol, new_coords, conf_ids=None):
    if len(new_coords.shape) == 2:
        new_coords = [new_coords]

    conf_ids = range(mol.GetNumConformers()) if conf_ids is None else conf_ids

    if len(conf_ids) != len(new_coords):
        raise RuntimeError('Number of coordinate sets is different from the number of conformers')

    for coords_id, conf_id in enumerate(conf_ids):
        conformer = mol.GetConformer(conf_id)
        new_coordset = new_coords[coords_id]

        if mol.GetNumAtoms() != new_coordset.shape[0]:
            raise ValueError(f'Number of atoms is different from the number of coordinates \
            ({mol.GetNumAtoms()} != {new_coordset.shape[0]})')

        for i in range(mol.GetNumAtoms()):
            x, y, z = new_coordset[i]
            conformer.SetAtomPosition(i, Point3D(x, y, z))


def apply_prody_transform_to_rdkit_mol(mol, tr):
    mol = deepcopy(mol)
    new_coords = apply_prody_transform(mol.GetConformer().GetPositions(), tr)
    change_mol_coords(mol, new_coords)
    return mol


def global_align(s1, s2):
    aln = Bio.pairwise2.align.globalds(s1, s2, matlist.blosum62, -14.0, -4.0)
    return aln


def mol_with_atom_index(mol):
    atoms = mol.GetNumAtoms()
    for idx in range(atoms):
        mol.GetAtomWithIdx(idx).SetProp('molAtomMapNumber', str(mol.GetAtomWithIdx(idx).GetIdx()))
    return mol


def make_list(obj):
    try:
        obj = [x for x in obj]
    except TypeError:
        obj = [obj]
    return obj


def build_d2mat(crd1, crd2):
    return np.sum((crd1[:, None, :] - crd2[None, :, :])**2, axis=2)


def parse_hhr_file(hhr_file, num_top=None):
    with open(hhr_file, 'r') as f:
        lines = f.readlines()

    assert int(lines[9].split()[0]) == 1
    q_chain = Path(hhr_file).basename().stripext()

    break_line = None
    for i, line in enumerate(lines[9:], 9):
        if line.strip() == '':
            break_line = i
            break
        int(line.split()[0])

    out = OrderedDict()
    for line in lines[9:break_line][:num_top]:
        pdb_chain = line.split()[1]
        if pdb_chain in out:
            continue
        #split = line.split()
        q_range = line[75:84].split('-')
        t_range = line[85:94].split('-')
        out[pdb_chain] = OrderedDict(
            hh_pdb=pdb_chain.upper(),
            hh_prob=float(line[35:40]),
            hh_identity=None,
            hh_evalue=None,
            query_aln='',
            target_aln='',
            query_range=[int(q_range[0])-1, int(q_range[1])],
            target_range=[int(t_range[0])-1, int(t_range[1])]
        )

    counter = 0
    t_chain = None
    skip = False

    for idx in range(break_line + 1, len(lines)):
        line = lines[idx]
        if line.startswith('>'):
            # sanity check for previous target's alignment
            if t_chain is not None:
                last_chain = out[t_chain]
                #assert len(last_chain['query_aln']) == len(last_chain['target_aln'])

            t_chain = line.split()[0][1:]

            # hhr can have multiple alignments for the same chain
            # we keep only the first one
            skip = out[t_chain]['hh_identity'] is not None
            if skip:
                continue

            # extract alignment scores
            scores = lines[idx+1].split()
            identity = scores[4]
            assert identity.startswith('Iden')
            identity = float(identity.split('=')[1].strip('%'))

            evalue = scores[1]
            assert evalue.startswith('E-value')
            evalue = float(evalue.split('=')[1].strip('%'))

            out[t_chain]['hh_identity'] = identity
            out[t_chain]['hh_evalue'] = evalue
            counter += 1

        if skip:
            continue

        # record Q alignment
        if line.startswith(f'Q {q_chain}'):
            s = line.split()
            out[t_chain]['query_aln'] += s[3]

            #begin, end = int(s[2]), int(s[4])
            #cur_range = out[t_chain]['query_range']
            # sanity check that alignment chunks in hhr file are contiguous
            #if len(cur_range) > 0:
            #    assert begin-1 == cur_range[-1]
            #out[t_chain]['query_range'] += [begin-1, end]

        # record target alignment
        if t_chain is not None and line.startswith(f'T {t_chain}'):
            s = line.split()
            out[t_chain]['target_aln'] += s[3]

            #begin, end = int(s[2]), int(s[4])
            #cur_range = out[t_chain]['target_range']
            # sanity check that alignment chunks in hhr file are contiguous
            #if len(cur_range) > 0:
            #    assert begin-1 == cur_range[-1]
            #out[t_chain]['target_range'] += [begin-1, end]

        if num_top is not None and counter >= num_top:
            break

    #for item in out.values():
    #    item['query_range'] = [min(item['query_range']), max(item['query_range'])]
    #    item['target_range'] = [min(item['target_range']), max(item['target_range'])]
    #    #print(item)
    #    #assert (item['target_range'][1] - item['target_range'][0]) == (item['query_range'][1] - item['query_range'][0])

    return out

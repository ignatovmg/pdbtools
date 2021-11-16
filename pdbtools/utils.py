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

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

import Bio
from Bio.SubsMat import MatrixInfo as matlist
from Bio.pairwise2 import format_alignment

from .loggers import logger


def _get_rdkit_elements():
    pt = Chem.GetPeriodicTable()
    elements = []
    for i in range(1000):
        try:
            elements.append(pt.GetElementSymbol(i))
        except:
            break
    return elements


# TODO: This produces annoying error log, when it hits atom number which is not in the table.
#       Need to fix it somehow.
RDKIT_ELEMENTS = _get_rdkit_elements()


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


def read_mol_block(mol_block, **kwargs):
    if mol_block is None:
        raise RuntimeError(f'Mol block is empty')
    mol = Chem.MolFromMolBlock(mol_block, **kwargs)
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
import numpy as np
import gzip
import json
import prody
import multiprocessing as mp
from io import StringIO
from path import Path
from collections import OrderedDict
from itertools import chain
from tqdm import tqdm
import pybel
import traceback

from rdkit import DataStructs
from rdkit.Chem import rdFMCS
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit import Chem
from rdkit.Chem import AllChem

from .ligtree import LigandTree
from .paths import DATA_DIR
from .loggers import logger
from .database import Database
from . import utils
from .pdb import get_atom_group


def _parse_sdf(path):
    sdf_dict = {}
    with open(path, 'r') as f:
        chemid = None
        for l in f:
            if chemid is None:
                chemid = l.strip()
                sdf_dict[chemid] = ''
            if not l.startswith('$$$$'):
                sdf_dict[chemid] += l
            else:
                chemid = None
    return sdf_dict


def _parse_all_sdf_ids(names):
    d = {}
    for name in names:
        s = name.split('_')
        pdb = s[0]
        chemid = s[1]
        key = pdb + '_' + chemid
        val = d.get(key, [])
        val.append(name)
        if len(val) == 1:
            d[key] = val
    return d


def _read_all_sdf(db_file, index_file):
    if not Path(index_file).exists():
        logger.info(f'Index file {index_file} does not exist, creating..')
        sep = ['(^[0-9A-Za-z]{4}_[0-9A-Za-z]{1,3}_[0-9]+_[0-9A-Za-z]+_.*$)\n', None]
        db = Database(db_file, sep=sep)
        db.write_index_file(index_file)
    else:
        db = Database(db_file, index_file)
    return db


class LigandExpo(object):
    # https://bitbucket.org/abc-group/ligtbm/raw/1455c1c10eec304a7bc18a90bd05a37915bf43b8/backend/ligtbm/data/Ligand_Expo/chemid_to_liginfo.json
    _CHEMID_TO_LIGINFO = utils.read_json(DATA_DIR / 'ligand_expo' / 'chemid_to_liginfo.json')

    # https://bitbucket.org/abc-group/ligtbm/raw/1455c1c10eec304a7bc18a90bd05a37915bf43b8/backend/ligtbm/data/Ligand_Expo/chemid_to_pdbids.json
    _CC_TO_PDB = utils.read_json(DATA_DIR / 'ligand_expo' / 'chemid_to_pdbids.json')

    # https://bitbucket.org/abc-group/ligtbm/raw/1455c1c10eec304a7bc18a90bd05a37915bf43b8/backend/ligtbm/data/Ligand_Expo/pdbid_to_chemids.json
    _PDB_TO_CC = utils.read_json(DATA_DIR / 'ligand_expo' / 'pdbid_to_chemids.json')

    # Ideal geometries
    # http://ligand-expo.rcsb.org/dictionaries/Components-pub.sdf.gz
    _CHEMID_TO_IDEAL_SDF = _parse_sdf(DATA_DIR / 'ligand_expo' / 'Components-pub.sdf')

    # All ligands in PDB: http://ligand-expo.rcsb.org/dictionaries/all-sdf.sdf.gz
    # Index file was made using:
    # python ../../src/database.py all-sdf.sdf --sep_start '(^.{4}_.{3}_[0-9]+_.+_[0-9]+_.+)\n' -w all-sdf.index
    _CHEMID_TO_ALL_SDF = _read_all_sdf(DATA_DIR.joinpath('ligand_expo', 'all-sdf.sdf'),
                                       DATA_DIR.joinpath('ligand_expo', 'all-sdf.index'))
    _PDB_CHEMID_TO_NAME = _parse_all_sdf_ids(_CHEMID_TO_ALL_SDF.keys())

    # Amino acids geometries (can be created using ROOT/scripts/download_amino_acids.py)
    _AA_TO_IDEAL_SDF = _parse_sdf(DATA_DIR.joinpath('ligand_expo', 'aa.sdf'))
    _AA_BONDS = utils.read_json(DATA_DIR.joinpath('ligand_expo', 'aa.json'))

    # Fragments
    #if DATA_DIR.joinpath('ligand_expo', 'chemid_to_frags.json').exists():
    #    _CHEMID_TO_FRAGS = utils.read_json(DATA_DIR.joinpath('ligand_expo', 'chemid_to_frags.json'))
    #    _FRAG_TO_CHEMIDS = utils.read_json(DATA_DIR.joinpath('ligand_expo', 'frag_to_chemids.json'))
    #else:
    #    _CHEMID_TO_FRAGS = None
    #    _FRAG_TO_CHEMIDS = None
    #    logger.warning(f"Cant find {DATA_DIR.joinpath('ligand_expo', 'chemids_to_frags.json')}, setting fragments to None")

    def __init__(self):
        pass

    @classmethod
    def get_smiles(cls, chemid):
        liginfo = cls._CHEMID_TO_LIGINFO.get(chemid, None)
        if liginfo is not None:
            return liginfo.get('SMILES_OpenEye', None)
        return None

    @classmethod
    def get_pdb_list(cls, chemid):
        return cls._CC_TO_PDB.get(chemid, [])

    @classmethod
    def get_chemids_list(cls, pdb):
        return cls._PDB_TO_CC.get(pdb.upper(), [])

    @classmethod
    def get_ideal_mol(cls, chemid, **kwargs):
        if chemid == 'UNX':
            logger.error(f'UNX stands for an unknown molecule, returning None')
            return None
        mol_block = cls._CHEMID_TO_IDEAL_SDF.get(chemid, None)
        try:
            mol = utils.read_mol_block(mol_block, **kwargs)
        except KeyError as e:
            logger.error(f'Key Error for ideal geometry {chemid}, returning None')
            mol = None
        except Exception as e:
            logger.error(f'RDkit reading failed for ideal geometry {chemid}, returning None ({e})')
            mol = None
        return mol

    @classmethod
    def get_all_sdf_mol(cls, all_sdf_id, **kwargs):
        """
        all_sdf_id uses ligand expo convention, ex.: 1pk8_ATP_1_B_801__R_D_
        """
        cls._CHEMID_TO_ALL_SDF.open()
        try:
            mol_block = cls._CHEMID_TO_ALL_SDF[all_sdf_id]
            mol_block = mol_block.decode('utf-8')
            mol = utils.read_mol_block(mol_block, **kwargs)
            if mol.GetNumAtoms() == 0:
                raise RuntimeError(f'Molecule does not have atoms')
            if mol.GetConformer(0).GetPositions().shape[0] == 0:
                raise RuntimeError(f'Molecule does not have coordinates')
        except KeyError as e:
            logger.error(f'Key Error for {all_sdf_id}, returning None')
            mol = None
        except Exception as e:
            logger.exception(e)
            logger.error(f'RDkit reading failed for {all_sdf_id}, returning None')
            mol = None
        finally:
            cls._CHEMID_TO_ALL_SDF.close()
        return mol

    @classmethod
    def get_all_sdf_ids(cls, pdb, chemid, first_model_only=True):
        """
        Get all ligand expo full names with the same pdb and chemid
        """
        sdf_ids = cls._PDB_CHEMID_TO_NAME.get(f'{pdb.lower()}_{chemid}', [])
        if first_model_only:
            sdf_ids = [x for x in sdf_ids if int(x.split('_')[2]) == 1]
        return sdf_ids

    @classmethod
    def get_aa_mol(cls, aa, **kwargs):
        try:
            mol = Chem.MolFromMolBlock(cls._AA_TO_IDEAL_SDF[aa], **kwargs)
        except Exception as e:
            logger.error(f'RDkit reading error - {str(e)}')
            mol = None
        return mol

    @classmethod
    def get_aa_bonds(cls):
        return cls._AA_BONDS

    #@classmethod
    #def get_chemid_to_frags(cls):
    #    if cls._CHEMID_TO_FRAGS is None:
    #        raise RuntimeError('_CHEMID_TO_FRAGS is not set')
    #    return cls._CHEMID_TO_FRAGS

    #@classmethod
    #def get_frag_to_chemids(cls):
    #    if cls._FRAG_TO_CHEMIDS is None:
    #        raise RuntimeError('_FRAG_TO_CHEMIDS is not set')
    #    return cls._FRAG_TO_CHEMIDS

    #@staticmethod
    #def remove_hs(mol):
    #    logger.info([x.GetSymbol() for x in mol.GetAtoms()])
    #    hs = [x.GetIdx() for x in mol.GetAtoms() if x.GetSymbol().strip() == 'H']
    #    logger.info(hs)
    #    rw = Chem.RWMol(mol)
    #    for a in hs:
    #        rw.RemoveAtom(a)
    #    mol = rw.GetMol()
    #    return mol

    @staticmethod
    def check_elements(mol, ag):
        mol_elements = [x.GetSymbol() for x in mol.GetAtoms()]
        pdb_elements = ag.getElements()
        if len(pdb_elements) != len(mol_elements) or not all(
                pe.upper() == me.upper() for pe, me in zip(pdb_elements, mol_elements)):
            pdb_elist = ' '.join(['%-2s' % e for e in pdb_elements])
            mol_elist = ' '.join(['%-2s' % e for e in mol_elements])
            raise RuntimeError(f'\nElements are different in pdb and sdf\n{" ".join(pdb_elist)}\n{" ".join(mol_elist)}')

    @classmethod
    def get_bonds_from_mol_and_ag(cls, mol, ag):
        mol = Chem.RemoveHs(mol)
        cls.check_elements(mol, ag)

        mol_connected_pairs = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), int(b.GetBondType())) for b in mol.GetBonds()]
        pdb_names = ag.getNames()
        bonds = OrderedDict([(name, {}) for name in pdb_names])
        for don_i, acc_i, btype in mol_connected_pairs:
            don, acc = pdb_names[don_i], pdb_names[acc_i]
            bonds[don][acc] = btype
            bonds[acc][don] = btype

        return bonds

    @classmethod
    def _get_bonds_dict_for_sdf(cls, all_sdf_id):
        mol = cls.get_all_sdf_mol(all_sdf_id)
        if mol is None:
            raise RuntimeError(f'Ligand {all_sdf_id} is missing')
        if mol.GetConformer(0).GetPositions().shape[0] == 0:
            raise RuntimeError(f'Ligand {all_sdf_id} is empty')

        pdb, chemid, model, chain, resnum = all_sdf_id.split('_')[:5]
        ag = get_atom_group(pdb, csets=int(model))
        if ag is None:
            raise RuntimeError(f'PDB {pdb} is missing')
        ag = ag.select(f'heavy and (chain {chain} resname {chemid} resnum {resnum})').copy()
        if ag is None:
            raise RuntimeError(f'Selection for {all_sdf_id} is empty')

        bonds = cls.get_bonds_from_mol_and_ag(mol, ag)
        return bonds, ag.getNames(), mol

    @classmethod
    def get_bonds_for_ag(cls, ag: prody.Atomic, max_trials=100):
        chemids = set(ag.getResnames())
        ag_bonds = {}

        for chemid in chemids:
            chemid_bonds = {}
            chemid_atom_names = set(ag.select(f'resname {chemid} and heavy').getNames())
            chemid_natoms = len(chemid_atom_names)

            # if compound is a single atom its trivial
            if len(chemid_atom_names) == 1:
                chemid_bonds[chemid_atom_names.pop()] = {}

            # if the compound is an amino acid or a nucleic acid, use the precomputed dictionary
            elif chemid in cls._AA_BONDS:
                chemid_bonds = cls._AA_BONDS[chemid]
                chemid_atom_names -= set(chemid_bonds.keys())

            # else - look the compound up
            else:
                counter = 0
                pdb_list = cls.get_pdb_list(chemid)

                for pdb in pdb_list:
                    all_sdf_ids = cls.get_all_sdf_ids(pdb, chemid)

                    for sdf_id in all_sdf_ids:
                        if counter >= max_trials:
                            logger.warning(f'{chemid}: Max number of attempts exceeded ({max_trials})')
                            break
                        try:
                            logger.debug(f'{chemid}: Trying to find bonds using {sdf_id}')
                            sdf_bonds, sdf_names = cls._get_bonds_dict_for_sdf(sdf_id)[:2]
                            for don, acc_dict in sdf_bonds.items():
                                # TODO: add warning if the same pair of atoms has a different type of bond
                                if don in chemid_bonds:
                                    chemid_bonds[don].update(acc_dict)
                                else:
                                    chemid_bonds[don] = acc_dict
                            chemid_atom_names -= set(sdf_names)
                        except Exception as e:
                            logger.warning(f'{chemid}: {e}')
                        counter += 1

                        if len(chemid_atom_names) == 0:
                            break

                    if len(chemid_atom_names) == 0 or counter >= max_trials:
                        break

            if len(chemid_atom_names) != 0:
                if len(chemid_atom_names) == chemid_natoms:
                    logger.error(f"{chemid}: Couldn't find bonds for any atoms")
                else:
                    logger.warning(
                        f"{chemid}: Couldn't find bonds for some atoms ({100 * len(chemid_atom_names) / chemid_natoms:.2f}%) ({' '.join(chemid_atom_names)}")
            else:
                logger.debug(f"{chemid}: Found bonds for all atoms")
            chemid_bonds['exc'] = list(chemid_atom_names)
            ag_bonds[chemid] = chemid_bonds

        return ag_bonds

    @staticmethod
    def calc_mcs(mol1, mol2, mcs_flags=[], timeout=60):
        if 'aa' in mcs_flags:
            atomcompare = rdFMCS.AtomCompare.CompareAny
        elif 'ai' in mcs_flags:
            # CompareIsotopes matches based on the isotope label
            # isotope labels can be used to implement user-defined atom types
            atomcompare = rdFMCS.AtomCompare.CompareIsotopes
        else:
            atomcompare = rdFMCS.AtomCompare.CompareElements

        if 'ba' in mcs_flags:
            bondcompare = rdFMCS.BondCompare.CompareAny
        elif 'be' in mcs_flags:
            bondcompare = rdFMCS.BondCompare.CompareOrderExact
        else:
            bondcompare = rdFMCS.BondCompare.CompareOrder

        if 'v' in mcs_flags:
            matchvalences = True
        else:
            matchvalences = False

        if 'chiral' in mcs_flags:
            matchchiraltag = True
        else:
            matchchiraltag = False

        if 'r' in mcs_flags:
            ringmatchesringonly = True
        else:
            ringmatchesringonly = False

        if 'cr' in mcs_flags:
            completeringsonly = True
        else:
            completeringsonly = False

        maximizebonds = True

        mols = [mol1, mol2]
        try:
            mcs_result = rdFMCS.FindMCS(mols,
                                        timeout=timeout,
                                        atomCompare=atomcompare,
                                        bondCompare=bondcompare,
                                        matchValences=matchvalences,
                                        ringMatchesRingOnly=ringmatchesringonly,
                                        completeRingsOnly=completeringsonly,
                                        matchChiralTag=matchchiraltag,
                                        maximizeBonds=maximizebonds)
        except:
            # sometimes Boost (RDKit uses it) errors occur
            raise RuntimeError('MCS calculation failed')
        if mcs_result.canceled:
            raise RuntimeError('MCS calculation ran out of time')

        return mcs_result.smartsString, mcs_result.numAtoms, mcs_result.numBonds

    @staticmethod
    def calc_fp_tanimoto(mol1, mol2):
        fp1 = FingerprintMols.FingerprintMol(mol1)
        fp2 = FingerprintMols.FingerprintMol(mol2)
        fp_tanimoto = DataStructs.FingerprintSimilarity(fp1, fp2)
        return fp_tanimoto

    @staticmethod
    def _mol_from_smiles(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise RuntimeError(f'SMILES {smiles} is not RDKit-correct')
        return mol

    @staticmethod
    def _calc_mcs_mp(trg_mol, ref_chemid, ref_smiles,
                     mcs_size_min=0, mcs_cov_min=0,
                     fp_tanimoto_min=0, mcs_flags=['v'],
                     max_mol_size=None):
        mcs_size_min = min(mcs_size_min, trg_mol.GetNumAtoms())

        output = {'ref_chemid': ref_chemid,
                  'ref_smiles': ref_smiles,
                  'mcs_flags': mcs_flags}

        try:
            ref_mol = LigandExpo._mol_from_smiles(ref_smiles)
            mcs_smarts, mcs_natoms, mcs_nbonds = LigandExpo.calc_mcs(trg_mol, ref_mol, mcs_flags=mcs_flags)
            mcs_strict_smarts, mcs_strict_natoms, mcs_strict_nbonds = LigandExpo.calc_mcs(trg_mol, ref_mol, mcs_flags=['v', 'r'])
        except Exception as e:
            output.update({'exc': str(e)})
            return output

        mcs_coverage = float(mcs_natoms) / trg_mol.GetNumAtoms()
        mcs_strict_coverage = float(mcs_strict_natoms) / trg_mol.GetNumAtoms()
        mcs_tanimoto = float(mcs_natoms) / (trg_mol.GetNumAtoms() + ref_mol.GetNumAtoms() - mcs_natoms)
        fp_tanimoto = LigandExpo.calc_fp_tanimoto(trg_mol, ref_mol)

        if (mcs_coverage >= mcs_cov_min) and (fp_tanimoto >= fp_tanimoto_min) and (mcs_natoms >= mcs_size_min) \
                and (max_mol_size is None or ref_mol.GetNumHeavyAtoms() <= max_mol_size):
            output.update({'fp_tanimoto': fp_tanimoto,
                           'mcs_coverage': mcs_coverage,
                           'mcs_strict_coverage': mcs_strict_coverage,
                           'mcs_tanimoto': mcs_tanimoto,
                           'mcs_smarts': mcs_smarts,
                           'mcs_num_atoms': mcs_natoms,
                           'mcs_num_bonds': mcs_nbonds})
            return output
        return None

    @classmethod
    def compute_mcs_for_smiles(cls,
                               smiles,
                               pdb_ids=None,
                               mcs_size_min=0,
                               mcs_cov_min=0.0,
                               fp_tanimoto_min=0.0,
                               max_mol_size=None,
                               mcs_flags=[],
                               nprocs=1,
                               max_results=None):

        # Get chemid list
        if pdb_ids is None:
            chemid_list = list(cls._CHEMID_TO_LIGINFO.keys())
        else:
            chemid_list = set()
            for pdb_id in pdb_ids:
                chemid_list.update(cls._PDB_TO_CC.get(pdb_id.upper(), []))  # Ligand Expo might be out-of-date
            chemid_list = list(chemid_list)

        # Get ligand list
        chemid_smiles_list = [(chemid, cls.get_smiles(chemid)) for chemid in chemid_list]
        absent_chemids = [x[0] for x in chemid_smiles_list if x[1] is None]
        if len(absent_chemids) != 0:
            logger.warning(
                'The following chemids are missing from Ligand Expo Database - {}'.format(' '.join(absent_chemids)))
        chemid_smiles_list = [x for x in chemid_smiles_list if x[1] is not None]

        # Prepare args
        trg_mol = LigandExpo._mol_from_smiles(smiles)
        mp_inputs = [(trg_mol, chemid, smiles, mcs_size_min, mcs_cov_min, fp_tanimoto_min, mcs_flags, max_mol_size)
                     for chemid, smiles in chemid_smiles_list if smiles]

        # Run
        with mp.Pool(nprocs) as pool:
            logger.info(f'Started MCS with PDB ligands ({nprocs} workers for {len(mp_inputs)} ligands)')
            mp_outputs = pool.starmap(cls._calc_mcs_mp, mp_inputs)
            logger.info('Finished MCS calculation')

        # Make dict chemid -> mcs output
        chemid_to_mcs = {o['ref_chemid']: o for o in mp_outputs if o and 'exc' not in o}
        for output in mp_outputs:
            if output and 'exc' in output:
                logger.warning(f"MCS for {output['ref_chemid']} failed")
        logger.info(f'Found {len(chemid_to_mcs)} chemids')

        # Sort by strict coverage and fingerprint
        if max_results is not None and len(chemid_to_mcs) > max_results:
            logger.info(f'Number of chemids is larger than permitted ({max_results}), truncating')
            chemid_scores = [(mcs['mcs_strict_coverage'], mcs['fp_tanimoto'], chemid)
                             for chemid, mcs in chemid_to_mcs.items()]
            chemid_scores = sorted(chemid_scores, reverse=True)
            best_chemids = list(zip(*chemid_scores))[2][:max_results]

            chemid_to_mcs = {chemid: mcs for chemid, mcs in chemid_to_mcs.items() if chemid in best_chemids}
            logger.info(f'Keeping {len(chemid_to_mcs)} chemids with best MCS coverage and fingerprint score')

        return chemid_to_mcs

    @classmethod
    def get_chemid_to_frags(cls, chemids=None, min_frag_size=0):
        if chemids is None:
            chemids = list(cls._CHEMID_TO_LIGINFO.keys())

        chemid_to_frags = {}
        missing_chemids = []
        failed_chemids = []

        for chemid in tqdm(chemids):
            chemid_smiles = cls.get_smiles(chemid)
            if chemid_smiles is None:
                missing_chemids.append(chemid)
                continue

            mol = Chem.MolFromSmiles(chemid_smiles)
            #AllChem.EmbedMolecule(mol)
            AllChem.Compute2DCoords(mol)

            try:
                tree = LigandTree(mol)
                frags = []

                for n in tree.tree.all_nodes():
                    frag_sel = tree.node_to_selection(n).heavy
                    if frag_sel is None:
                        continue

                    frag_size = len(frag_sel)
                    if frag_size < min_frag_size:
                        continue

                    frag = {'atoms': tree.node_to_mol_ids(n),
                            'smiles': tree.node_to_smiles(n),
                            'smarts': tree.node_to_smarts(n),
                            'size': frag_size}
                    frags.append(frag)

                chemid_to_frags[chemid] = frags
            except Exception as e:
                logger.warning(f'Failed making fragments for {chemid}')
                logger.exception(e)
                failed_chemids.append(chemid)

        if len(missing_chemids) > 0:
            logger.warning('Failed to read the following compounds: ' + ' '.join(missing_chemids))

        if len(failed_chemids) > 0:
            logger.warning('Failed breaking down the following compounds: ' + ' '.join(failed_chemids))

        return chemid_to_frags

    @staticmethod
    def __mol_from_smiles_(smiles):
        # kekulization fails sometimes, which makes us skip such fragments as Nc1ncnc2ncnc12
        m = Chem.MolFromSmiles(smiles, sanitize=False)
        Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL - Chem.SanitizeFlags.SANITIZE_KEKULIZE, catchErrors=True)
        return m

    @classmethod
    def get_nonred_smiles_set(cls, smiles_list, use_mcs=False, mcs_flags=[]):
        smiles_to_ids = {}

        mol_list = [(i, cls.__mol_from_smiles_(s)) for i, s in enumerate(smiles_list)]
        mol_list = [x for x in mol_list if x[1] is not None]
        mol_dict = OrderedDict(mol_list)
        failed_smiles = [(x[0], smiles_list[x[0]]) for x in mol_list if x[1] is None]
        if len(failed_smiles) > 0:
            logger.warning('The following smiles failed: ' + str(failed_smiles))

        for i1, m1 in tqdm(mol_list):
            identical_smiles = None
            s1 = smiles_list[i1]

            for s2, ids in smiles_to_ids.items():
                if use_mcs:
                    m2 = mol_dict[ids[0]]
                    num1, num2 = m1.GetNumAtoms(), m2.GetNumAtoms()
                    if num1 != num2:
                        continue
                    if num1 == 1:
                        identical_smiles = 'single_atom'
                        continue
                    try:
                        smarts, natoms, nbonds = cls.calc_mcs(m1, m2, mcs_flags=mcs_flags, timeout=60)
                        if natoms == num1:
                            identical_smiles = s2
                    except Exception as e:
                        logger.error(f'Failure for fragments {s1} and {s2} ({e})')
                else:
                    if s1 == s2:
                        identical_smiles = s2

            # if None then the molecule doesnt have twins in the dict
            if identical_smiles is None:
                identical_smiles = s1

            # add smiles index to the dict
            ids = smiles_to_ids.get(identical_smiles, [])
            ids.append(i1)
            if len(ids) == 1:
                smiles_to_ids[identical_smiles] = ids

        if 'single_atom' in smiles_to_ids:
            single_atoms = smiles_to_ids['single_atom']
            del smiles_to_ids['single_atom']
        else:
            single_atoms = []

        return smiles_to_ids, single_atoms

    @classmethod
    def get_frag_to_chemids(cls, chemid_to_frags, use_mcs=False, mcs_flags=[]):
        chemid_list = []
        smiles_list = []
        for chemid, frags in chemid_to_frags.items():
            for x in frags:
                smiles_list.append(x['smiles'])
                chemid_list.append(chemid)

        smiles_to_ids, single_atoms = cls.get_nonred_smiles_set(smiles_list, use_mcs, mcs_flags=mcs_flags)
        smiles_to_chemids = {s: list(sorted(list(set([chemid_list[i] for i in ids])))) for s, ids in smiles_to_ids.items()}
        return smiles_to_chemids


def add_hs(out_mol, in_mol):
    # add hydrogens
    mol = next(pybel.readfile('mol', in_mol))
    mol.addh()
    mol.localopt('mmff94', steps=500)
    mol.write('mol', out_mol, overwrite=True)

    # fix back the coordinates
    mol = Chem.MolFromMolFile(out_mol, removeHs=False)
    ag = utils.mol_to_ag(mol)
    ref_ag = utils.mol_to_ag(Chem.MolFromMolFile(in_mol, removeHs=True))
    tr = prody.calcTransformation(ag.heavy, ref_ag.heavy)
    ag = tr.apply(ag)
    prody.writePDB(Path(out_mol).stripext() + '.pdb', ag)

    utils.change_mol_coords(mol, ag.getCoords())
    AllChem.ComputeGasteigerCharges(mol, throwOnParamFailure=True)

    Chem.MolToMolFile(mol, out_mol)


def get_fragments(outdir, mol, prefix='rigid_'):
    frag_list = []
    try:
        mol_tree = LigandTree(mol)
        mol_ag = utils.mol_to_ag(mol)
        frag_id = 0
        # iterate rigid fragments
        for node in mol_tree.tree.all_nodes():
            node_ids = mol_tree.node_to_mol_ids(node)
            if len(mol_ag[node_ids].heavy) < 3:
                continue

            # match fragment back to mol
            node_smiles = mol_tree.node_to_smiles(node)
            frag_mol = Chem.MolFromSmiles(node_smiles)
            match = None
            for m in mol.GetSubstructMatches(frag_mol):
                if set(m) == set(node_ids):
                    match = m
                    break
            if match is None:
                print("Couldn't match fragment to it's ligand", node_smiles, outdir)
                continue

            # add coordinates to fragment
            AllChem.Compute2DCoords(frag_mol)
            new_coords = mol_ag.getCoords()[list(match)]
            assert new_coords.shape[0] == frag_mol.GetNumAtoms()
            utils.change_mol_coords(frag_mol, new_coords)

            out_name = f'{prefix}{frag_id:03d}.mol'
            Chem.MolToMolFile(frag_mol, outdir / out_name)

            add_hs(outdir / f'{prefix}{frag_id:03d}_ah.mol', outdir / out_name)

            frag_list.append(
                OrderedDict(
                    mol_path=f'{prefix}{frag_id:03d}_ah.mol',
                    atom_ids=node_ids,
                    smiles=node_smiles
                )
            )
            frag_id += 1

        if len(frag_list) == 0:
            raise RuntimeError("Couldn't find any fragments")
    except:
        print(traceback.format_exc())
        return []

    return frag_list
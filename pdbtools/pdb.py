import gzip
import prody
from io import StringIO
from path import Path
from pymol import cmd

from .paths import DATA_DIR
from .loggers import logger
from . import utils


# PDB database in .ent format
_PDB_DIR = DATA_DIR / 'pdb'
if not _PDB_DIR.exists():
    raise RuntimeError(f"Path {_PDB_DIR} doesn't exist")


SEQ_CLUSTERS = {}
CHAIN_TO_CLUSTER = {}


def _fill_clusters():
    for path in (DATA_DIR / 'seq_clusters').glob(f'bc-*.out'):
        perc = int(path.basename().stripext().split('-')[-1])
        with open(path, 'r') as f:
            SEQ_CLUSTERS[perc] = [x.split() for x in f]

        CHAIN_TO_CLUSTER[perc] = {}
        for _cluster_id, _cluster in enumerate(SEQ_CLUSTERS[perc]):
            for _chain in _cluster:
                CHAIN_TO_CLUSTER[perc][_chain] = _cluster_id


_fill_clusters()


def pdb_file_path(pdb):
    return _PDB_DIR.joinpath(pdb[1:3].lower(), f'pdb{pdb.lower()}.ent.gz')


def _get_pdb_content(pdb):
    path = pdb_file_path(pdb)
    if not path.exists():
        return ''

    with gzip.open(path, 'rb') as f:
        content = f.read()
    return content.decode('utf-8')


def get_atom_group(pdb, csets=1, header=False, **kwargs):  # csets is 1-based
    content = _get_pdb_content(pdb)
    if content == '':
        return None

    if csets is None:
        csets = [1]
    if isinstance(csets, int):
        csets = [csets]
    if len(csets) == 1:
        ag = prody.parsePDBStream(StringIO(content), model=csets[0], header=header, **kwargs)
        if header:
            header = ag[1]
            ag = ag[0]
    else:
        ag = prody.parsePDBStream(StringIO(content), header=header, **kwargs)
        if header:
            header = ag[1]
            ag = ag[0]
        ag.setCoords(ag.getCoordsets(indices=csets))

    if header:
        return ag, header
    else:
        return ag


def fetch_atom_group(pdb):
    rec_path = Path(prody.fetchPDBviaFTP(pdb.lower()))
    ag = prody.parsePDB(rec_path)
    rec_path.remove_p()
    return ag


def cealign(mob_ag, ref_ag, mob_chain=None, ref_chain=None):
    with utils.isolated_filesystem():
        prody.writePDB('mob.pdb', mob_ag)
        prody.writePDB('ref.pdb', ref_ag)
        cmd.set('pdb_retain_ids', 1)
        cmd.set('retain_order', 1)
        cmd.delete('all')
        cmd.load('mob.pdb', 'mob')
        cmd.load('ref.pdb', 'ref')
        ref_sel = 'ref' if ref_chain is None else 'ref and chain ' + ref_chain
        mob_sel = 'mob' if mob_chain is None else 'mob and chain ' + mob_chain
        rmsd = cmd.cealign(ref_sel, mob_sel)['RMSD']
        cmd.save('tmp.pdb', 'mob')
        aligned = prody.parsePDB('tmp.pdb')
        cmd.delete('all')
    return aligned, rmsd


def align(mob_ag, ref_ag, mob_chain=None, ref_chain=None):
    """
    From pymol wiki:

    align() returns a list with 7 items:
        RMSD after refinement
        Number of aligned atoms after refinement
        Number of refinement cycles
        RMSD before refinement
        Number of aligned atoms before refinement
        Raw alignment score
        Number of residues aligned
    """
    with utils.isolated_filesystem():
        prody.writePDB('mob.pdb', mob_ag)
        prody.writePDB('ref.pdb', ref_ag)
        cmd.set('pdb_retain_ids', 1)
        cmd.set('retain_order', 1)
        cmd.delete('all')
        cmd.load('mob.pdb', 'mob')
        cmd.load('ref.pdb', 'ref')
        ref_sel = 'ref' if ref_chain is None else 'ref and chain ' + ref_chain
        mob_sel = 'mob' if mob_chain is None else 'mob and chain ' + mob_chain
        result = cmd.align(mob_sel, ref_sel, object='alnobj')
        cmd.save('alignment.aln', 'alnobj')
        cmd.save('tmp.pdb', 'mob')
        aligned = prody.parsePDB('tmp.pdb')

        with open('alignment.aln', 'r') as f:
            lines = f.readlines()
        mob_aln = ''.join([x.split()[1].strip() for x in lines if x.startswith('mob')])
        ref_aln = ''.join([x.split()[1].strip() for x in lines if x.startswith('ref')])

        cmd.delete('all')
    return aligned, result, (mob_aln, ref_aln)


def get_symmetry_mates(pdb, cutoff=6):
    pdb_file = pdb_file_path(pdb)
    cmd.reinitialize()
    cmd.load(pdb_file, 'orig')
    cmd.symexp('symm', 'orig', 'orig', cutoff)
    tmp_file = utils.tmp_file(prefix='pdbtools-', suffix='.pdb')
    cmd.save(tmp_file, 'symm*')
    symm_ag = prody.parsePDB(tmp_file)
    Path(tmp_file).remove_p()
    cmd.delete('all')
    return symm_ag


def get_sequence(pdb, chain):
    ag = get_atom_group(pdb, chain=chain)
    if ag is None:
        return None
    ag = ag.select('protein and name CA')
    if ag is None:
        return None
    return ag.getSequence()


def get_sequence_alignment(pdb1, chain1, pdb2, chain2):
    s1 = get_sequence(pdb1, chain1)
    s2 = get_sequence(pdb2, chain2)
    if s1 is None or s2 is None:
        return None
    aln = utils.global_align(s1, s2)[0]
    nident = sum([x == y and x != '-' for x, y in zip(aln[0], aln[1])])
    return aln, nident / len(s1) * 100, nident / len(s2) * 100


def nonred_chain_ids(chain_list, identity=90):
    nonred_chain_ids = []
    present_clusters = []
    for chain_id, chain in enumerate(chain_list):
        cluster_id = CHAIN_TO_CLUSTER[identity].get(chain, None)
        if cluster_id not in present_clusters:
            nonred_chain_ids.append(chain_id)
            present_clusters.append(cluster_id)
        if cluster_id is None:
            nonred_chain_ids.append(chain_id)
            logger.warning(f'Chain {chain} was not found in sequence clusters')
    return nonred_chain_ids


def find_similar_to_pdb(pdb, chain=None, identity=90):
    similar_chains = set()
    c2c = CHAIN_TO_CLUSTER[identity]
    s2c = SEQ_CLUSTERS[identity]
    if chain is None:
        for key, cluster_id in c2c.items():
            if key.split('_')[0] == pdb.upper():
                similar_chains |= set(s2c[cluster_id])
    else:
        similar_chains = set(s2c[c2c[pdb.upper() + '_' + chain]])

    return list(similar_chains)

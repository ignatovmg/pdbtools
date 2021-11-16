import subprocess
import treelib
import prody
from rdkit import Chem
from io import StringIO
from path import Path

from .utils import run_check_output, isolated_filesystem
from .paths import VENV_DIR


_MGLTOOLS_BIN = VENV_DIR / 'mgltools_env' / 'bin'


class LigandTree(object):
    def __init__(self, mol, backend='mgltools'):
        """
        :param backend: "mgltools" or "babel"
        """
        self.mol = mol
        if backend == 'mgltools':
            self.pdbqt_text = self.mol_to_pdbqdt_block_mgltools(mol)
        elif backend == 'babel':
            self.pdbqt_text = self.mol_to_pdbqdt_block_babel(mol)
        else:
            raise ValueError('Backend must be "mgltools" or "babel"')
        self.ag = self.prody_pdbqt(self.pdbqt_text)
        self.a2m, self.m2a = self._get_mol2ag(self.mol, self.ag)
        self.tree = self._read_pdbqt(self.pdbqt_text)
        self._fill_subtrees_atoms()

    @staticmethod
    def prody_pdbqt(pdbqt_text):
        pdbqt_text = '\n'.join(filter(lambda x: x.startswith('ATOM'), pdbqt_text.split('\n')))
        s = StringIO(pdbqt_text)
        ag = prody.parsePDBStream(s)
        return ag

    @staticmethod
    def mol_to_pdbqdt_block_babel(mol):
        call = ['obabel', '-imol', '-opdbqt']
        p = subprocess.Popen(call, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        output, output_err = p.communicate(Chem.MolToMolBlock(mol).encode('utf-8'))
        if p.returncode != 0:
            raise RuntimeError('Conversion to pdbqt failed')
        return output.decode('utf-8')

    @staticmethod
    def mol_to_pdbqdt_block_mgltools(mol):
        with isolated_filesystem():
            Chem.MolToMolFile(mol, 'lig.mol')
            run_check_output(['obabel', '-imol', 'lig.mol', '-omol2', '-O', 'lig.mol2'])

            call = [_MGLTOOLS_BIN / 'python2', _MGLTOOLS_BIN / 'prepare_ligand4.py', '-l', 'lig.mol2']
            run_check_output(call)
            if not Path('lig.pdbqt').exists():
                raise RuntimeError('Conversion to pdbqt failed')
            with open('lig.pdbqt', 'r') as f:
                output = f.read()
        return output

    @staticmethod
    def _get_mol2ag(mol, ag):
        mol_crd = mol.GetConformer(0).GetPositions()
        ag_crd = ag.getCoords()
        dmat = prody.buildDistMatrix(ag_crd, mol_crd)
        ids = list(zip(*map(list, (dmat < 0.01).nonzero())))
        a2m = {a.item(): m.item() for a, m in ids}
        m2a = {m.item(): a.item() for a, m in ids}
        return a2m, m2a

    def _get_selection(self, serials):
        return self.ag.select('serial ' + ' '.join(serials))

    def _read_pdbqt(self, pdbqt_text):
        tree = treelib.Tree()

        pointer = None

        for line in pdbqt_text.split('\n'):
            if line.startswith('ATOM'):
                serial = line[6:11].strip()
                tree.get_node(pointer).data['atoms'].append(serial)

            if line.startswith('ROOT'):
                tree.create_node('ROOT', 'root')
                tree.get_node('root').data = {'atoms': []}
                pointer = 'root'

            if line.startswith('BRANCH'):
                nid = '_'.join(line.strip().split()[1:])
                tree.create_node(nid, nid, parent=pointer, data={'atoms': []})
                pointer = nid

            if line.startswith('ENDROOT') or line.startswith('ENDBRANCH'):
                node = tree.get_node(pointer)
                node.data['sel'] = self._get_selection(node.data['atoms'])
                if pointer != 'root':
                    pointer = node.bpointer

        return tree

    def _fill_subtrees_atoms(self):
        tree = self.tree
        for n in tree.all_nodes():
            n.data['all_atoms'] = [x for x in n.data['atoms']]

        for leaf in tree.leaves(tree.root):
            nid = leaf.identifier
            while nid != tree.root:
                parent = tree.parent(nid)
                parent.data['all_atoms'] += tree.get_node(nid).data['all_atoms']
                nid = parent.identifier

    def atoms_to_smiles(self, ag_atoms):
        mol_ids = [self.a2m[x] for x in ag_atoms]
        smiles = Chem.MolFragmentToSmiles(self.mol, mol_ids)
        if smiles is None:
            raise RuntimeError('Failed to compute fragment SMILES')
        return smiles

    def atoms_to_smarts(self, ag_atoms):
        mol_ids = [self.a2m[x] for x in ag_atoms]
        smarts = Chem.MolFragmentToSmarts(self.mol, mol_ids)
        if smarts is None:
            raise RuntimeError('Failed to compute fragment SMILES')
        return smarts

    def node_to_smiles(self, node):
        return self.atoms_to_smiles(node.data['sel'].getIndices())

    def node_to_smarts(self, node):
        return self.atoms_to_smarts(node.data['sel'].getIndices())

    def node_to_selection(self, node):
        return node.data['sel']

    def node_to_ag_ids(self, node):
        return utils.numpy_to_list(node.data['sel'].getIndices())

    def node_to_mol_ids(self, node):
        return [self.a2m[x] for x in self.node_to_ag_ids(node)]


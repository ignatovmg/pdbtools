import urllib.request
from collections import OrderedDict
import json
import prody
from rdkit import Chem
from io import StringIO

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))


if __name__ == '__main__':
    d_1aa = {'SER': 'S', 'LYS': 'K', 'PRO': 'P', 'ALA': 'A', 'ASP': 'D', 'ARG': 'R', 'VAL': 'V', 'CYS': 'C', 'HIS': 'H',
             'PHE': 'F', 'MET': 'M', 'LEU': 'L', 'ASN': 'N', 'TYR': 'Y', 'ILE': 'I', 'GLN': 'Q', 'THR': 'T', 'GLY': 'G',
             'TRP': 'W', 'GLU': 'E'}

    aas = list(sorted(d_1aa.keys()))
    mols = {}
    pdbs = {}
    bonds = {}

    aas += ['DA', 'DG', 'DC', 'DT', 'A', 'G', 'C', 'T', 'U', 'HEM', 'OUR', 'MSE']

    for aa in aas:
        mols[aa] = urllib.request.urlopen(f'http://ligand-expo.rcsb.org/reports/{aa[0]}/{aa}/{aa}_ideal.sdf').read().decode('utf-8')
        pdbs[aa] = urllib.request.urlopen(f'http://ligand-expo.rcsb.org/reports/{aa[0]}/{aa}/{aa}_ideal.pdb').read().decode('utf-8')

    with open('aa.sdf', 'w') as f:
        f.write(''.join(mols.values()))

    #for aa in aas:
    #    with open(f'{aa}.pdb', 'w') as f:
    #        f.write(pdbs[aa])

    for aa in aas:
        print(f'Detecting bonds for {aa}')
        mol = Chem.MolFromMolBlock(mols[aa])
        if mol is None:
            print(f'RDkit could not read {aa}.mol')
            mol = Chem.MolFromPDBBlock(pdbs[aa])
            if mol is None:
                print(f'RDkit could not read {aa}.pdb, skipping..')
                continue
        mol_elements = [x.GetSymbol() for x in mol.GetAtoms()]
        mol_connected_pairs = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), int(b.GetBondType())) for b in mol.GetBonds()]

        ag = prody.parsePDBStream(StringIO(pdbs[aa])).heavy
        pdb_names = ag.getNames()
        pdb_elements = ag.getElements()

        if len(pdb_elements) != len(mol_elements) or not all(
                pe.upper() == me.upper() for pe, me in zip(pdb_elements, mol_elements)):
            pdb_elist = ' '.join(['%-2s' % e for e in pdb_elements])
            mol_elist = ' '.join(['%-2s' % e for e in mol_elements])
            raise RuntimeError(f'\nElements are different in pdb and sdf for {aa}-\n{" ".join(pdb_elist)}\n{" ".join(mol_elist)}')

        bond_dict = OrderedDict([(name, {}) for name in pdb_names])
        for don_i, acc_i, btype in mol_connected_pairs:
            don, acc = pdb_names[don_i], pdb_names[acc_i]
            bond_dict[don][acc] = btype
            bond_dict[acc][don] = btype

        bonds[aa] = bond_dict

    with open('aa.json', 'w') as f:
        json.dump(bonds, f, indent=4)

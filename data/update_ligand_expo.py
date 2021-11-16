import os
import re
import json
import shutil
import urllib.request
from rdkit import Chem

chemid_pdbids_file = 'cc-to-pdb.tdd'
plan = {'InChI': 'Components-inchi.ich',
        'SMILES_OpenEye': 'Components-smiles-stereo-oe.smi',
        'SMILES_CACTVS': 'Components-smiles-stereo-cactvs.smi'}

url_template = 'http://ligand-expo.rcsb.org/dictionaries/{}'
files = list(plan.values()) + [chemid_pdbids_file]
for file in files:
    url = url_template.format(file)
    with urllib.request.urlopen(url) as f, open(file, 'wb') as g:
        shutil.copyfileobj(f, g)

chemid_to_pdbids = {}
pdbid_to_chemids = {}
with open(chemid_pdbids_file) as f:
    for line in f:
        chemid, pdbids = line.strip().split('\t')
        pdbids = [pdbid.upper() for pdbid in pdbids.split()]
        chemid_to_pdbids[chemid] = pdbids
        for pdbid in pdbids:
            if pdbid not in pdbid_to_chemids:
                pdbid_to_chemids[pdbid] = []
            pdbid_to_chemids[pdbid].append(chemid)

with open('chemid_to_pdbids.json', 'w') as f:
    json.dump(chemid_to_pdbids, f, indent=4)
with open('pdbid_to_chemids.json', 'w') as f:
    json.dump(pdbid_to_chemids, f, indent=4)

chemid_to_liginfo = {}
pattern = '^([^\n\t ]*)\t+([^\n\t ]+)\t+[^\t]*$'
for key, file in plan.items():
    print(f'----------{key}----------\n')
    with open(file) as f:
        content = f.read()
    check = re.sub(pattern, '', content, flags=re.MULTILINE).replace('\n', '')
    if check:
        print(f'Warning: unparsed text!\n{repr(check)}\n')
        continue

    for m in re.finditer(pattern, content, flags=re.MULTILINE):
        x = m.group(1)
        chemid = m.group(2)
        if x:
            if key == 'InChI':
                mol = Chem.MolFromInchi(x)
            else:
                mol = Chem.MolFromSmiles(x)

            if mol:
                if chemid not in chemid_to_liginfo:
                    chemid_to_liginfo[chemid] = {}
                chemid_to_liginfo[chemid][key] = x
            else:
                print(f'Warning: {chemid} {x} is RDKit-incorrect!\n')
        else:
            print(f'Warning: {chemid} is empty!\n')

with open('chemid_to_liginfo.json', 'w') as f:
    json.dump(chemid_to_liginfo, f, indent=4)

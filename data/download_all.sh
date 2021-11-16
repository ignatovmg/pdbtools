#!/bin/bash -e

oldpwd=$(pwd)

mkdir -p data/ligand_expo && cd data/ligand_expo
wget http://ligand-expo.rcsb.org/dictionaries/{Components-pub.sdf.gz,all-sdf.sdf.gz}
gzip -d all-sdf.sdf.gz Components-pub.sdf.gz
python $oldpwd/../pdbtools/database.py all-sdf.sdf --sep_start '(^.{4}_.{3}_[0-9]+_.+_[0-9]+_.+)\n' -w all-sdf.index
python $oldpwd/update_ligand_expo.py
python $oldpwd/update_amino_acids.py
cd $oldpwd

mkdir -p data/seq_clusters && cd data/seq_clusters
$oldpwd/update_seqclus.sh
cd $oldpwd
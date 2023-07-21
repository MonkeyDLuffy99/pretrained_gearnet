import pandas as pd
import numpy as np

f = open("dc50_labels.txt", "r")
lines = f.readlines()
refined_lines = []
for l in lines:
    l = l.replace("(n/a)", "")
    l = l.replace("\n", "")
    l = l.replace("nM", "")
    l = l.replace(">", "")
    l = l.replace("<", "")
    l = l.replace("=", "")
    l = l.replace("~", "")
    refined_lines.append(l)

final_labels = []
for l in refined_lines:
    arr = l.split('/')
    if len(arr) > 1:
        smallest_so_far = 999999
        for el in arr:
            if el == "":
                continue
            if el.find("~") > 0 or el.find("-") > 0:
                continue
            el = float(el)
            if el < smallest_so_far:
                smallest_so_far = el
        final_labels.append(smallest_so_far)
    else:
        final_labels.append(arr[0])

labeled_indices = []
for i, val in enumerate(final_labels):
    val = str (val)
    if val == "":
        continue
    else:
        labeled_indices.append(i)

df = pd.read_csv("filtered_protacs.csv")
target_proteins = np.array(df["Target Protein"])
e3_targets = np.array(df["E3 Target"])
uniprots_poi = np.array(df["Uniprot POI"])
uniprots_e3 = np.array(df["Uniprot E3"])

target_proteins = target_proteins[labeled_indices]
e3_targets = e3_targets[labeled_indices]
uniprots_poi = uniprots_poi[labeled_indices]
uniprots_e3 = uniprots_e3[labeled_indices]

del final_labels
del lines
del refined_lines

print(len(e3_targets))
print(len(target_proteins))
print(len(uniprots_poi))
print(len(uniprots_e3))

from torchdrug import data, utils
from tqdm import tqdm as tqdm

uniprot_df = pd.read_csv("uniprot_to_pdb_id.csv")
uniprots = list(uniprot_df["From"])
pdb = list(uniprot_df["To"])

uniprot_to_pdb_map = {}
for u, p in zip(uniprots, pdb):
    uniprot_to_pdb_map[u] = p


correct_indices = []
my_target_proteins = []
my_e3_ligases = []
for i, (poi, e3) in tqdm(enumerate(zip(uniprots_poi, uniprots_e3))):
    try:
        poi_pdb_id = uniprot_to_pdb_map[poi]
    except:
        print("Key error encountered, no pdb file exists for", poi)
        continue
    try:
        e3_pdb_id = uniprot_to_pdb_map[e3]
    except:
        print("Key error encountered, no pdb file exists for", e3)
        continue
    url_poi = "https://files.rcsb.org/download/"
    url_e3 = "https://files.rcsb.org/download/"
    url_poi += poi_pdb_id + ".pdb"
    url_e3 += e3_pdb_id + ".pdb"
    poi_pdb_file = utils.download(url_poi, "pdb_files")
    e3_pdb_file = utils.download(url_e3, "pdb_files")
    my_poi = data.Protein.from_pdb(poi_pdb_file, atom_feature="position", bond_feature="length", residue_feature="symbol")
    my_e3 = data.Protein.from_pdb(e3_pdb_file, atom_feature="position", bond_feature="length", residue_feature="symbol")
    my_target_proteins.append(my_poi)
    my_e3_ligases.append(my_e3)
    correct_indices.append(i)
print(len(correct_indices))

my_target_proteins = data.Protein.pack(my_target_proteins)

del uniprots_poi
del uniprots_e3

print(len(my_target_proteins))
print(len(my_e3_ligases))

import pickle

f = open("my_target_proteins", "wb")
pickle.dump(my_target_proteins, f)

f = open("my_e3_ligases", "wb")
pickle.dump(my_e3_ligases, f)

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
warheads = np.array(df["Warhead"])
linkers = np.array(df["Linker"])
e3_ligands = np.array(df["E3 Ligand"])
uniprots_poi = np.array(df["Uniprot POI"])
uniprots_e3 = np.array(df["Uniprot E3"])

target_proteins = target_proteins[labeled_indices]
e3_targets = e3_targets[labeled_indices]
warheads = warheads[labeled_indices]
linkers = linkers[labeled_indices]
e3_ligands = e3_ligands[labeled_indices]
uniprots_poi = uniprots_poi[labeled_indices]
uniprots_e3 = uniprots_e3[labeled_indices]

final_labels = np.array(final_labels)
my_labels = final_labels[labeled_indices]

print(len(e3_targets))
print(len(target_proteins))
print(len(warheads))
print(len(linkers))
print(len(e3_ligands))
print(len(uniprots_poi))
print(len(uniprots_e3))
print(len(my_labels))

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
    print(poi_pdb_id, e3_pdb_id)
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
my_e3_ligases = data.Protein.pack(my_e3_ligases)

warheads = warheads[correct_indices]
linkers = linkers[correct_indices]
e3_ligands = e3_ligands[correct_indices]

my_labels = my_labels[correct_indices]

print(len(my_target_proteins))
print(len(my_e3_ligases))
print(len(warheads))
print(len(linkers))
print(len(e3_ligands))
print(len(my_labels))

from torchdrug import layers, models
from torchdrug.layers import geometry
import torch
import math
import pickle

device = torch.device("cpu")

graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()],
                                                    edge_layers=[geometry.SequentialEdge(max_distance=2),
                                                                 geometry.SpatialEdge(radius=10.0, min_distance=5),
                                                                 geometry.KNNEdge(k=10, min_distance=5)],
                                                    edge_feature="gearnet")

model = models.GearNet(input_dim=21,
                         hidden_dims=[512, 512, 512, 512, 512, 512],
                         num_relation=7, edge_input_dim=59,
                         batch_norm=True, activation="relu",
                         concat_hidden=True, short_cut=True,
                         readout="sum", num_angle_bin=8).to(device)

model_dict = torch.load("angle_gearnet_edge.pth", map_location=torch.device("cpu"))
model.load_state_dict(model_dict)
print(my_target_proteins)

# distance gearnet does not work
def create_embeddings(proteins):
    idx_to_embedding = {}
    for i, t in tqdm(enumerate(proteins)):
        if i == 19 or i == 63:
            continue
        try:
            t = data.Protein.pack(t).to(device)
            t = graph_construction_model(t).to(device)
            res = model.forward(t, t.residue_feature.float())
            idx_to_embedding[i] = res
        except:
            print("Encountered issue at index:", i)
    return idx_to_embedding

distance_embeddings_tp = create_embeddings(my_target_proteins)
distance_embeddings_e3 = create_embeddings(my_e3_ligases)

f = open("angle_gearnet_tp", "wb")
pickle.dump(distance_embeddings_tp, f)

f = open("angle_gearnet_e3", "wb")
pickle.dump(distance_embeddings_e3, f)

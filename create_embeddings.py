import pickle

with open('my_target_proteins.pkl', 'rb') as f:
        my_target_proteins = pickle.load(f)

with open('my_e3_ligases.pkl', 'rb') as f:
        my_e3_ligases = pickle.load(f)

with open('correct_indices.pkl', 'rb') as f:
    correct_indices =pickle.load(f)

from torchdrug import layers, models, data
from torchdrug.layers import geometry
from tqdm import tqdm as tqdm
import torch
import numpy as np
import pandas as pd

df = pd.read_csv("filtered_protacs.csv")
warheads = np.array(df["Warhead"])
linkers = np.array(df["Linker"])
e3_ligands = np.array(df["E3 Ligand"])

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


warheads = warheads[labeled_indices]
linkers = linkers[labeled_indices]
e3_ligands = e3_ligands[labeled_indices]
final_labels = np.array(final_labels)
my_labels = final_labels[labeled_indices]

warheads = warheads[correct_indices]
linkers = linkers[correct_indices]
e3_ligands = e3_ligands[correct_indices]
my_labels = my_labels[correct_indices]


device = torch.device("cpu")

graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()],
                                                    edge_layers=[geometry.SequentialEdge(max_distance=2),
                                                                 geometry.SpatialEdge(radius=10.0, min_distance=5),
                                                                 geometry.KNNEdge(k=10, min_distance=5)],
                                                    edge_feature="gearnet")

gearnet = models.GearNet(input_dim=21,
                         hidden_dims=[512, 512, 512, 512, 512, 512],
                         num_relation=7, edge_input_dim=59,
                         batch_norm=True, activation="relu",
                         concat_hidden=True, short_cut=True,
                         readout="sum", num_angle_bin=8)

model_dict = torch.load("distance_gearnet_edge.pth", map_location=torch.device("cpu"))
gearnet.load_state_dict(model_dict)

restricted_residues = [1101, 1444, 1738, 1153, 2291, 390, 563, 504] # residues that cause segmentation fault
useful_indexes = []
discarded_indexes = []

def get_indexes(proteins):
    for i, t in tqdm(enumerate(proteins)):
        if t.num_residue in restricted_residues:
            discarded_indexes.append(i)
            continue

        useful_indexes.append(i)
    return useful_indexes

# some proteins cause a segmentation fault
useful_indexes = get_indexes(my_target_proteins)

# get finderprints from PROTAC molecules
from molfeat.calc import FPCalculator

def calc_fingerprint(calc, arr):
    to_return = []
    for el in arr:
        to_return.append(calc(el))
    return np.array(to_return)

calc = FPCalculator("secfp")

warhead_features = calc_fingerprint(calc, warheads)
linker_features = calc_fingerprint(calc, linkers)
e3_ligand_features = calc_fingerprint(calc, e3_ligands)


# get the labels and construct protac dataset

labels = []
for m in my_labels:
    if m.find("~") > 0:
        m.split("~")
        m = float (m[0])
    elif m.find("-") > 0:
        m.split("-")
        m = float (m[0])
    else:
        m = float (m)
    labels.append(m)

protac_set = np.column_stack((warhead_features, linker_features, e3_ligand_features))

labels = np.array(labels)
labels = np.log(labels)

from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA

# standardise data
scaler = StandardScaler(with_mean=True, with_std=True)
standardised_dataset = scaler.fit_transform(protac_set)

# PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(standardised_dataset)

from sklearn.cluster import KMeans

# MIX DATA FROM EACH CLUSTER TO EACH FOLD
kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(X_reduced)

cluster_0_indices = []
cluster_1_indices = []
cluster_2_indices = []

for i, k in enumerate(kmeans.labels_):
    if k == 0:
        cluster_0_indices.append(i)
    elif k == 1:
        cluster_1_indices.append(i)
    elif k == 2:
        cluster_2_indices.append(i)

fold_0_indices = []
fold_1_indices = []
fold_2_indices = []

i = 0
while i < len(cluster_0_indices):
    fold_0_indices.append(cluster_0_indices[i])
    i += 1
    if i >= len(cluster_0_indices):
        break
    fold_1_indices.append(cluster_0_indices[i])
    i += 1
    if i >= len(cluster_0_indices):
        break
    fold_2_indices.append(cluster_0_indices[i])
    i += 1

i = 0
while i < len(cluster_1_indices):
    fold_0_indices.append(cluster_1_indices[i])
    i += 1
    if i >= len(cluster_1_indices):
        break
    fold_1_indices.append(cluster_1_indices[i])
    i += 1
    if i >= len(cluster_1_indices):
        break
    fold_2_indices.append(cluster_1_indices[i])
    i += 1

i = 0
while i < len(cluster_2_indices):
    fold_0_indices.append(cluster_2_indices[i])
    i += 1
    if i >= len(cluster_2_indices):
        break
    fold_1_indices.append(cluster_2_indices[i])
    i += 1
    if i >= len(cluster_2_indices):
        break
    fold_2_indices.append(cluster_2_indices[i])
    i += 1

def gearnet_forward(proteins):
    my_list = [proteins]
    t = data.Protein.pack(my_list)
    t = graph_construction_model(t)
    res = gearnet.forward(t, t.residue_feature.float())
    return res

import torch.nn as nn
class MyModel(nn.Module):
    def __init__(self,
                 hidden_neurons,
                 output_size):
        super().__init__()
        self.fc1 = nn.Linear(2048*3+3072*2, hidden_neurons)
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.fc2 = nn.Linear(hidden_neurons, hidden_neurons)
        self.fc3 = nn.Linear(hidden_neurons, output_size)

    def forward(self,
                protac_components,
                poi_graph,
                e3_graph):
        v_0 = gearnet_forward(poi_graph)
        v_0 = v_0["graph_feature"]
        v_1 = gearnet_forward(e3_graph)
        v_1 = v_1["graph_feature"]
        protac_components = protac_components.reshape(1, 6144)
        v_f = torch.cat((v_0, v_1, protac_components), 1)
        v_f = self.relu(self.fc1(v_f))
        v_f = self.relu(self.fc2(v_f))
        v_f = self.fc3(v_f)
        return v_f

def slice_array(the_array, the_indices):
    arr = []
    for idx in the_indices:
        arr.append(the_array[idx])

    return arr

model = MyModel(1024,1)

def train(learning_rate, model, num_epoch=10):
    r2 = []
    spearman = []
    pearson = []
    rmse = []
    criterion = nn.MSELoss()
    for k in range(3):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        if k == 0:
            train_indices = fold_0_indices + fold_1_indices
            test_indices = fold_2_indices
        elif k == 1:
            train_indices = fold_1_indices + fold_2_indices
            test_idnices = fold_0_indices
        else:
            train_indices = fold_0_indices + fold_2_indices
            test_indices = fold_1_indices

        model.train()

        train_protacs = protac_set[train_indices]
        train_poi = slice_array(my_target_proteins, train_indices)
        train_e3 = slice_array(my_e3_ligases, train_indices)

        train_labels = labels[train_indices]

        epoch_no = []
        all_losses = []
        for j, epoch in enumerate(range(num_epoch)):
            losses = []
            for protac, poi, e3, label in zip(train_protacs, train_poi, train_e3, train_labels):
                optimizer.zero_grad()
                protac = torch.tensor(protac)
                output = model.forward(protac, poi, e3).reshape(1,)
                label = np.array(label).reshape(1,)
                label = torch.from_numpy(label)
                loss = criterion(output, label)
                my_loss = float (loss)
                losses.append(my_loss)
                optimizer.step()
            losses = np.array(losses)
            epoch_no.append(j+1)
            all_losses.append(loses.sum())
            print("Epoch no.", epoch, "loss:", losses.sum())

        rmse_, coeffr_, coeffs_, r2_ = val(model, test_idnices)
        rmse.append(rmse_)
        pearson.append(coeffr_)
        spearman.append(coeffs_)
        r2.append(r2_)
    print("rmse:", np.array(rmse).mean())
    print("pearson coefficient:", np.array(pearson).mean())
    print("spearman rank:", np.array(spearman).mean())
    print("r2:", np.array(r2).mean())

def val(model, test_idnices):
    test_protacs = protac_set[test_idnices]
    test_poi = slice_array(my_target_proteins, test_idnices)
    test_e3 = slice_array(my_e3_ligases, test_idnices)

    test_labels = labels[test_idnices]

    outputs = []
    answers = []
    for protac, poi, e3, label in zip(test_protacs, test_poi, test_e3, test_labels):
        with torch.no_grad():
            output = model.forward(protac, poi, e3)
            output = float (output)
            outputs.append(output)
            label = float (labels)
            answers.append(label)
    outputs = np.array(outputs)
    answers = np.array(answers)
    coeffr, _ = pearsonr(answers, outputs)
    coeffs, _ = spearmanr(answers, outputs)
    rmse = math.sqrt(mean_squared_error(answers, outputs))
    r2 = r2_score(answers, outputs)
    print("rmse:", rmse)
    print("pearson correlation coefficient:", coeffr)
    print("spearman correlation coefficient:", coeffs)
    print("r2 score:", r2)
    return rmse, coeffr, coeffs, r2

train(0.001, model)




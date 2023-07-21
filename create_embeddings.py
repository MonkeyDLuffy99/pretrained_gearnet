import pickle

with open('target_protein_graphs', 'a') as f:
        tp_graphs = pickle.load(f)

with open('e3_target_graphs', 'a') as f:
        e3_graphs = pickle.load(f)

from torchdrug import models
from tqdm import tqdm as tqdm
import torch

device = torch.device("cpu")

model = models.GearNet(input_dim=21,
                         hidden_dims=[512, 512, 512, 512, 512, 512],
                         num_relation=7, edge_input_dim=59,
                         batch_norm=True, activation="relu",
                         concat_hidden=True, short_cut=True,
                         readout="sum", num_angle_bin=8)

model_dict = torch.load("distance_gearnet_edge.pth", map_location=torch.device("cpu"))
model.load_state_dict(model_dict)

def get_embeddings(proteins):
    protein_embeddings = []
    for t in tqdm(proteins):
        res = model.forward(t, t.residue_feature.float())
        protein_embeddings.append(res)
    return protein_embeddings

distance_tb_embeddings = get_embeddings(tp_graphs)
distance_e3_embeddings = get_embeddings(e3_graphs)

f = open("distance_tb_embeddings", "wb")
pickle.dump(distance_tb_embeddings, f)

f = open("distance_e3_embeddings", "wb")
pickle.dump(distance_e3_embeddings, f)

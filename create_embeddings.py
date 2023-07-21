import pickle

with open('my_target_proteins', 'a') as f:
        my_target_proteins = pickle.load(f)

with open('my_e3_ligases', 'a') as f:
        my_e3_ligases = pickle.load(f)

from torchdrug import layers, models, data
from torchdrug.layers import geometry
from tqdm import tqdm as tqdm
import torch

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
                         readout="sum", num_angle_bin=8)

model_dict = torch.load("distance_gearnet_edge.pth", map_location=torch.device("cpu"))
model.load_state_dict(model_dict)

def get_embeddings(proteins):
    protein_embeddings = []
    for t in tqdm(proteins):
        mask = torch.zeros(t.num_residue, dtype=torch.bool, device="cpu")
        mask[0:500] = True
        t = t.subresidue(mask)
        t = data.Protein.pack(t)
        t = graph_construction_model(t)
        res = model.forward(t, t.residue_feature.float())
        protein_embeddings.append(res)
    return protein_embeddings

distance_tp_embeddings = get_embeddings(my_target_proteins)
distance_e3_embeddings = get_embeddings(my_e3_ligases)

f = open("distance_tp_embeddings", "wb")
pickle.dump(distance_tp_embeddings, f)

f = open("distance_e3_embeddings", "wb")
pickle.dump(distance_e3_embeddings, f)

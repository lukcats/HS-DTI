# coding=utf-8

import networkx as nx
import numpy as np
import scipy as sc
import os
import re
import random
import logging
import torch
from torch.utils.data import Dataset, DataLoader
import hparams_lib
from models.global_variables import *

def read_graphfile(xd, smile_graph, max_nodes=100):

    smiles = xd  # SMILES
    c_size, features, edge_index = smile_graph[smiles]  # smile_graph dict

    G = nx.from_edgelist(edge_index)  #edge_index:[(), (), ().....]， smile_to_graph()

    for u in G.nodes():
        G.nodes[u]['feat'] = features[u]

    mapping = {}
    it = 0
    if float(nx.__version__) < 2.0:
        for n in G.nodes():
            mapping[n] = it
            it += 1
    else:
        for n in G.nodes:
            mapping[n] = it
            it += 1
    graph = nx.relabel_nodes(G, mapping)
    graph_tmp_dict = {}
    adj = np.array(nx.to_numpy_matrix(graph))
    node_tmp_feature = np.zeros((max_nodes, 78))

    for index, feature in enumerate(graph.nodes()):
        if index < max_nodes:
            node_tmp_feature[index, :] = graph.nodes[index]['feat']

    num_nodes = adj.shape[0]

    graph_tmp_dict[g_key.node_num] = torch.tensor(num_nodes, dtype=torch.int16)

    graph_tmp_dict[g_key.x] = torch.tensor(node_tmp_feature, dtype=torch.float32)

    graph_tmp_dict[g_key.adj_mat] = torch.zeros(max_nodes, max_nodes)
    if index < max_nodes:
        graph_tmp_dict[g_key.adj_mat][:num_nodes, :num_nodes] = torch.tensor(adj, dtype=torch.float32)

    return graph_tmp_dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from models.gcn_hpool_submodel import GcnHpoolSubmodel
import argparse
import models.hparam as hparam
from models.protein import protein_order
# GINConv model
class GINConvNet(torch.nn.Module):
    def __init__(self, n_output=1,num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):
        super(GINConvNet, self).__init__()
        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        hparams = hparam.HParams()
        hparams.from_yaml('./hpool/hparams_testdb.yml')
        self.gcn_hpool_layer = GcnHpoolSubmodel(  # channel_list[78, 30, 30, 30, 30]  node_list[78, 10, 10]
            78, hparams.channel_list[3], hparams.channel_list[4],
            hparams.node_list[0], hparams.node_list[1], hparams.node_list[2], hparams)  # 128 30 30  78 10 10
        self.embedding_xt = nn.Embedding(num_features_xt + 1,125)  # 25  125
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)  # 1000 32 8
        self.fc1_xt = nn.Linear(32*121, output_dim)  # 128
        self.fc1 = nn.Linear(218, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)        # n_output = 1 for regression task

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target
        gra = data.hp
        adj = torch.reshape(gra['adj_mat'],(-1,100,100))
        feature = torch.reshape(gra['x'],(-1,100,78))
        num = gra['node_num']
        max_num_nodes = adj.size()[1]
        embedding_mask = self.construct_mask(max_num_nodes, num)
        drug, _, _, _ = self.gcn_hpool_layer(feature, feature, adj, embedding_mask)
        x = drug
        embedded_xt = self.embedding_xt(target)
        s=protein_order(target)
        s=s.to("cuda:0")
        embedded_xt = torch.cat((embedded_xt, s), 2)
        conv_xt = self.conv_xt_1(embedded_xt)
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc1_xt(xt)

        # concat
        xc = torch.cat((x, xt), 1)
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out, x

    def construct_mask(self, max_nodes, batch_num_nodes):
        ''' For each num_nodes in batch_num_nodes, the first num_nodes entries of the
        corresponding column are 1's, and the rest are 0's (to be masked out).
        Dimension of mask: [batch_size x max_nodes x 1]
        '''

        batch_num_nodes[batch_num_nodes > max_nodes] = max_nodes
        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)  # 20
        out_tensor = torch.zeros(batch_size, max_nodes)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, :batch_num_nodes[i]] = mask
        return out_tensor.unsqueeze(2).to('cuda:0')

import torch
from torch.nn.modules.module import Module
import torch.nn.functional as F
from global_variables import g_key
import hparams_lib
from gcn_hpool_submodel import GcnHpoolSubmodel
import gcn_layer
import torch.nn as nn

class GcnHpoolEncoder(Module):

    def __init__(self, hparams):
        super(GcnHpoolEncoder, self).__init__()
        self._hparams = hparams_lib.copy_hparams(hparams)
        self.build_graph()
        self.reset_parameters()
        self._device = torch.device(self._hparams.device)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, gcn_layer.GraphConvolution):
                m.weight.data = torch.nn.init.xavier_uniform_(m.weight.data, gain=torch.nn.init.calculate_gain('relu'))

                if m.bias is not None:
                    m.bias.data = torch.nn.init.constant_(m.bias.data, 0.0)

    def build_graph(self):
        self.gcn_hpool_layer = GcnHpoolSubmodel(
            128, self._hparams.channel_list[3], self._hparams.channel_list[4],
            self._hparams.node_list[0], self._hparams.node_list[1], self._hparams.node_list[2],
            self._hparams
        )


    def forward(self, graph_input):
        node_feature = graph_input[g_key.x]
        adjacency_mat = graph_input[g_key.adj_mat]
        batch_num_nodes = graph_input[g_key.node_num]
        target = graph_input[g_key.t]

        max_num_nodes = adjacency_mat.size()[1]
        embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)

        drug, _, _, _ = self.gcn_hpool_layer(
            node_feature, node_feature, adjacency_mat, embedding_mask)

        return drug


    def apply_bn(self, x):
        ''' Batch normalization of 3D tensor x
        '''

        bn_module = torch.nn.BatchNorm1d(x.size()[1]).to(self._device)
        return bn_module(x)

    def construct_mask(self, max_nodes, batch_num_nodes):

        ''' For each num_nodes in batch_num_nodes, the first num_nodes entries of the
        corresponding column are 1's, and the rest are 0's (to be masked out).
        Dimension of mask: [batch_size x max_nodes x 1]
        '''
        # masks
        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)  # 20
        out_tensor = torch.zeros(batch_size, max_nodes)

        for i, mask in enumerate(packed_masks):
            out_tensor[i, :batch_num_nodes[i]] = mask
        return out_tensor.unsqueeze(2).to(self._device)

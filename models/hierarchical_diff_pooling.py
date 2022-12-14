# coding=utf-8

import torch
from models.global_variables import *


def dense_diff_pool(x, adj, s, mask=None):  # l层：特征 邻接矩阵 转换矩阵
  r"""Differentiable pooling operator from the `"Hierarchical Graph
  Representation Learning with Differentiable Pooling"
  <https://arxiv.org/abs/1806.08804>`_ paper

  Directly use the implementation of torch_geometric
  https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#module-torch_geometric.nn.dense.diff_pool
  .. math::
      \mathbf{X}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
      \mathbf{X}

      \mathbf{A}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
      \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})

  based on dense learned assignments :math:`\mathbf{S} \in \mathbb{R}^{B
  \times N \times C}`.
  Returns pooled node feature matrix, coarsened adjacency matrix and the
  auxiliary link prediction objective :math:`\| \mathbf{A} -
  \mathrm{softmax}(\mathbf{S}) \cdot {\mathrm{softmax}(\mathbf{S})}^{\top}
  \|_F`.

  Args:
      x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
          \times N \times F}` with batch-size :math:`B`, (maximum)
          number of nodes :math:`N` for each graph, and feature dimension
          :math:`F`.
      adj (Tensor): Adjacency tensor :math:`\mathbf{A} \in \mathbb{R}^{B
          \times N \times N}`.
      s (Tensor): Assignment tensor :math:`\mathbf{S} \in \mathbb{R}^{B
          \times N \times C}` with number of clusters :math:`C`. The softmax
          does not have to be applied beforehand, since it is executed
          within this method.
      mask (ByteTensor, optional): Mask matrix
          :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
          the valid nodes for each graph. (default: :obj:`None`)

  :rtype: (:class:`Tensor`, :class:`Tensor`, :class:`Tensor`,
      :class:`Tensor`)
  """
  #print('former x.size:{} adj.size:{}'.format(x.size(),adj.size()))
  x = x.unsqueeze(0) if x.dim() == 2 else x
  adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
  s = s.unsqueeze(0) if s.dim() == 2 else s

  batch_size, num_nodes, _ = x.size()

  if mask is not None:
    mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
    x, s = x * mask, s * mask

  out = torch.matmul(s.transpose(1, 2), x)  # SX
  out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)  # S(T)AS


  link_loss = adj - torch.matmul(s, s.transpose(1, 2)) + EPS
  link_loss = torch.norm(link_loss, p=2)
  link_loss = link_loss / adj.numel()

  ent_loss = (-s * torch.log(s + EPS)).sum(dim=-1).mean()

  return out, out_adj, link_loss, ent_loss

# coding=utf-8

class GKey(object):

  def __init__(self):
    self.adj_mat = 'adj_mat'
    self.x = 'x'  # feature of atom
    self.y = 'y'  # label
    self.t = 't'  # target
    self.node_num = 'node_num'


g_key = GKey()
EPS = 1e-30
writer_batch_idx = [0, 3, 6, 9]

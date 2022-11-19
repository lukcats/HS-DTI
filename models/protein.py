import torch
import torch.nn as nn
import numpy as np
from utils import *
import math

"""
# ****************** standard value****************** 
def nor_anci():  # 标准转换亲水性，疏水性，侧链质量
    hs = [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8, 1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3]
    hq = [-0.5, -1.0, 2.5, 2.5, -2.5, 0.0, -0.5, -1.8, 3, -1.8, -1.3, 0.2, -1.4, 0.2, 3.0, 0.3, -0.4, -1.5, -3.4, -2.3]
    m = [89.1, 121.2, 133.1, 147.1, 165.2, 75.1, 155.2, 131.2, 146.2, 131.2, 149.2, 132.1, 115.1, 146.2, 174.2, 105.1, 119.1, 117.1, 204.2, 181.2]

    hs = np.array(hs) # convert
    low = np.sqrt(sum([pow((i - sum(hs / 20)), 2) for i in hs]) / 20)
    high = [(i - sum(hs / 20)) for i in hs]
    return high / low
"""

def protein_order(target):
    """
        max_seq_len = 6
        def seq_cat(prot):  #
            x = np.zeros(max_seq_len)  # 1000
            for i, ch in enumerate(prot[:max_seq_len]):  #  1:A 2:B 3:C
                x[i] = seq_dict[ch]  # A:1 B:2 C:3
            return x
        seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
        seq_dict = {v: (i + 1.0) for i, v in enumerate(seq_voc)}  # A:1 B:2 C:3)
        test_prots = [['Y', 'A', 'G', 'C', 'E', 'F'], ['A', 'C', 'D', 'A', 'G', 'C'], ['E', 'F', 'S', 'A', 'G', 'Y']]
        XT = [seq_cat(t) for t in test_prots]  # list--> np.array--->torch.LongTensor
    """
    target=target.cpu()
    x_arry = np.asarray(target)
    XT = x_arry.tolist()
    for i in range(len(x_arry)):
        for j in range(0, 26):
            temp = x_arry[i][x_arry[i] == j].size / float(1000)
            x_arry[i][x_arry[i] == j] = temp
    x_arry = torch.tensor(x_arry)
    x_arry = torch.unsqueeze(x_arry, 2)
    hs = {23: 0.0, 0: 0.0, 1: 0.7865794, 3: 1.02701852, 4: -1.03388821, 5: -1.03388821, 6: 1.13006386, 7: 0.0309136,
          8: -0.93084287, 9: 1.71398743, 10: -1.17128199, 11: 1.47354831, 12: 0.82092785, 13: -1.03388821,
          15: -0.38126775, 16: -1.03388821, 17: -1.37737267, 18: -0.10648018, 19: -0.07213174, 21: 1.6109421,
          22: -0.14082863, 24: -0.27822241}
    hq = {23: 0.0, 0: 0.0, 1: -0.09112998, 3: -0.36728142, 4: 1.56577867, 5: 1.56577867, 6: -1.19573574, 7: 0.18502147,
          8: -0.09112998, 9: -0.80912372, 10: 1.84193011, 11: -0.80912372, 12: -0.53297228, 13: 0.29548204,
          15: -0.58820257, 16: 0.29548204, 17: 1.84193011, 18: 0.35071233, 19: -0.03589969, 21: -0.64343286,
          22: -1.69280833, 24: -1.08527516}
    m = {23: 0.0, 0: 0.0, 1: -1.58928737, 3: -0.52211606, 4: -0.12649803, 5: 0.33893494, 6: 0.94067328, 7: -2.05472034,
         8: 0.60822116, 9: -0.18966394, 10: 0.30901425, 11: -0.18966394, 12: 0.40874988, 13: -0.15974324,
         15: -0.72491185, 16: 0.30901425, 17: 1.23988019, 18: -1.05736398, 19: -0.591931, 21: -0.65842143,
         22: 2.23723656, 24: 1.47259668}

    first_order = []
    second_order = []
    for protein in XT:
        first_temp = []
        second_temp = []
        for num in range(len(protein) - 2):
            try:
                 first_temp.append(pow(hs[protein[num]] - hs[protein[num + 1]], 2) / 3 + pow(hq[protein[num]] - hq[protein[num + 1]],2) / 3 + pow(m[protein[num]] - m[protein[num + 1]], 2) / 3)
                 second_temp.append(pow(hs[protein[num]] - hs[protein[num + 2]], 2) / 3 + pow(hq[protein[num]] - hq[protein[num + 2]],2) / 3 + pow(m[protein[num]] - m[protein[num + 2]], 2) / 3)
                 if num == len(protein) - 3:  # 最后两个节点的一阶 二阶信息特殊处理
                     first_temp.append(pow(hs[protein[num + 1]] - hs[protein[num + 2]], 2) / 3 + pow(hq[protein[num + 1]] - hq[protein[num + 2]], 2) / 3 + pow(m[protein[num + 1]] - m[protein[num + 2]],2) / 3)
                     first_temp.extend(first_temp[-1:])
                     second_temp.extend(second_temp[-2:])
            except:
                print(protein[num])

        first_order.append(first_temp)
        second_order.append(second_temp)
    first_order = torch.unsqueeze(torch.tensor(first_order), 2)
    second_order = torch.unsqueeze(torch.tensor(second_order), 2)

    order = torch.cat((first_order, second_order), 2)
    return torch.cat((order, x_arry), 2)

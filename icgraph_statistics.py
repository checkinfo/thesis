import os
import pickle
import torch
import numpy as np
from scipy.stats import pearsonr
from torch_geometric.utils import to_dense_adj

def load_graphs(graph_path):
    adj_list = []
    st = 0
    total_days = 2431
    for i in range(st, st+total_days):
        adj = np.load(os.path.join(graph_path, f"date_{st}.npz"))['adj']
        adj_list.append(torch.from_numpy(adj).long())  # edge index should be longtensor
    assert len(adj_list) == total_days
    print(f"load {len(adj_list)} graphs successful!")
    return adj_list


# 20100104在第974行
stock_num = 1931

mask = torch.from_numpy(np.load("../data/mask_3404_1931.npy"))


# 一年大概250个交易日， 一个月大概20个交易日
min_ic_list = [0.6, 0.8]
window_size_list = [5, 20]  # 5(week), 20(month), 60(quater), 250(1year), 750(3years), 1250(5years)

for min_ic in min_ic_list:
    for window_size in window_size_list:
        adj_path = f"../data/icgraph_window_{window_size}_{min_ic}/"
        adj_list = load_graphs(adj_path)

        # 2431 days
        stock_cnt = [0 for i in range(2431)]
        edges_cnt = [0 for i in range(2431)]


        for i in range(2431):
            cur_mask = mask[973+i, :]  # mask not shifted
            cur_adj = to_dense_adj(adj_list[i], max_num_nodes=1931).squeeze(0)
            cur_adj = cur_adj.fill_diagonal_(0)  # remove self loop
            # print(type(cur_adj), type(cur_adj.numpy()), type(cur_mask))
            
            stock_cnt[i] = torch.sum(cur_mask).item()
            edges_cnt[i] = torch.sum(torch.multiply(cur_adj, cur_mask.reshape(-1, 1))).item()

        with open(f"../data/stock_cnt_icgraph_{window_size}_{min_ic}.pickle", "wb") as fp:   #Pickling
            pickle.dump(stock_cnt, fp)
            # unpickle: 'rb', res = pickle.load(fp)
        with open(f"../data/edges_cnt_without_selfloop_icgraph_{window_size}_{min_ic}.pickle", "wb") as fp:   #Pickling
            pickle.dump(edges_cnt, fp)
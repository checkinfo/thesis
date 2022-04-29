import os
import numpy as np
from scipy.stats import pearsonr

# 20100104在第974行
stock_num = 1931
# 一年大概250个交易日， 一个月大概20个交易日

prices_ffill = np.load("../data/close_price_ffil_3404_1931.npy")  # labels not shifted
mask = np.load("../data/mask_3404_1931.npy")

min_ic_list = [0.6, 0.8]
window_size_list = [5, 20]  # 5(week), 20(month), 60(quater), 250(1year), 750(3years), 1250(5years)

for min_ic in min_ic_list:
    for window_size in window_size_list:
        for i in range(973, 3404):
            st, ed = i-window_size, i  # [i-window, i-1]
            cur_corr = np.corrcoef(prices_ffill[st:ed, :], rowvar=False)  # each column is a variable
            new_corr = np.nan_to_num(cur_corr) # fillna with zero
            new_corr[new_corr < min_ic] = 0
            # edge_index = np.array(new_corr.nonzero())  # 1.2s, 1.9s
            edge_index = np.stack(new_corr.nonzero(), axis=0)  # 1.6s, 1.3s

            if not os.path.exists(f"../data/icgraph_window_{window_size}_{min_ic}/"):
                os.makedirs(f"../data/icgraph_window_{window_size}_{min_ic}/")
            np.savez_compressed(os.path.join(f"../data/icgraph_window_{window_size}_{min_ic}/", f"date_{i-973}"), adj=edge_index)
            
    
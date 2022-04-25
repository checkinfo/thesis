from email.mime import base
import os
import copy
from typing import Callable, Optional, Tuple
import torch
import random
import datetime
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import Dataset as PygDataset
from torch_geometric.utils import to_dense_adj

from utils import normalize, sparse_mx_to_torch_sparse_tensor, fill_window
from sklearn.preprocessing import MinMaxScaler, StandardScaler



class TimeDataset(Dataset):
	def __init__(self, data_path, mask_path, dataset_type, args) -> None:
		super(TimeDataset, self).__init__()
		self.days = args.num_days
		self.dataset_type = dataset_type
		self.args = args
		self.data = torch.from_numpy(np.load(data_path).astype('float32'))
		self.mask = torch.from_numpy(np.load(mask_path))

	def __len__(self) -> int:
		return len(self.data) - self.days + 1  
		# no need to remove the last day
	
	def __getitem__(self, idx) -> Tuple[torch.Tensor]:  # return (x, y, mask, ann placeholder)
		end_idx = idx + self.days  # mask not shifted
		return (self.data[idx:end_idx, :, self.args.label_cnt:], 
				self.data[end_idx-1, :, :self.args.label_cnt], 
				self.mask[min(end_idx, len(self.data)-1), :], 
				torch.Tensor([idx]))



class NASDAQTimeDataset(Dataset):
	def __init__(self, data_path, mask_path, dataset_type, args) -> None:
		super(NASDAQTimeDataset, self).__init__()
		self.seq_len = args.num_days
		self.dataset_type = dataset_type
		self.steps = 1

		data_path = args.rsr_data_path  #"../Temporal_Relational_Stock_Ranking/data/2013-01-01"  # p
		market = args.market  # m
		tickers_fname = market + "_tickers_qualify_dr-0.98_min-5_smooth.csv"  # t

		tickers = np.genfromtxt(os.path.join(data_path, '..', tickers_fname), \
			dtype=str, delimiter='\t', skip_header=False)
		print('#tickers selected:', len(tickers))  # 1026

		self.eod_data, self.mask_data, self.gt_data, self.price_data = \
					self.load_EOD_data(data_path, market, tickers, steps=1)
		# [stocks, days, input_dim]
		print(self.eod_data.shape, self.mask_data.shape, self.gt_data.shape, self.price_data.shape)
	
	def load_EOD_data(self, data_path, market_name, tickers, steps=1):
		# tickers: stocks
		eod_data = []
		masks = []
		ground_truth = []
		base_price = []
		for index, ticker in enumerate(tickers):
			single_EOD = np.genfromtxt(
				os.path.join(data_path, market_name + '_' + ticker + '_1.csv'),
				dtype=np.float32, delimiter=',', skip_header=False
			)
			if market_name == 'NASDAQ':
				# remove the last day since lots of missing data
				single_EOD = single_EOD[:-1, :]
			if index == 0:
				print('single EOD data shape:', single_EOD.shape)
				eod_data = np.zeros([len(tickers), single_EOD.shape[0],
									single_EOD.shape[1] - 1], dtype=np.float32)
				masks = np.ones([len(tickers), single_EOD.shape[0]],
								dtype=np.float32)
				ground_truth = np.zeros([len(tickers), single_EOD.shape[0]],
										dtype=np.float32)
				base_price = np.zeros([len(tickers), single_EOD.shape[0]],
									dtype=np.float32)
			for row in range(single_EOD.shape[0]):
				if abs(single_EOD[row][-1] + 1234) < 1e-8:
					masks[index][row] = 0.0
				elif row > steps - 1 and abs(single_EOD[row - steps][-1] + 1234) \
						> 1e-8:
					ground_truth[index][row] = \
						(single_EOD[row][-1] - single_EOD[row - steps][-1]) / \
						single_EOD[row - steps][-1]
				for col in range(single_EOD.shape[1]):
					if abs(single_EOD[row][col] + 1234) < 1e-8:
						single_EOD[row][col] = 1.1
			eod_data[index, :, :] = single_EOD[:, 1:]
			base_price[index, :] = single_EOD[:, -1]
		return eod_data, masks, ground_truth, base_price

	def get_batch(self, offset=None):
		if offset is None:
			offset = random.randrange(0, self.valid_index)
		
		mask_batch = self.mask_data[:, offset: offset + self.seq_len + self.steps]
		mask_batch = np.min(mask_batch, axis=1)
		return self.eod_data[:, offset:offset + self.seq_len, :], \
               	mask_batch, \
            	np.expand_dims(self.price_data[:, offset + self.seq_len - 1], axis=1),\
                np.expand_dims(self.gt_data[:, offset + self.seq_len + self.steps - 1], axis=1)

	def __len__(self) -> int:
		return 756 if self.dataset_type=='train' else 237
		# no need to remove the last day
	
	def __getitem__(self, idx) -> Tuple[torch.Tensor]:  # return (x, y, mask, ann placeholder)
		if self.dataset_type != 'train':
			idx += 1008  # 756+252+237
		eod, mask, base_price, ground_truth = self.get_batch(idx)
		eod = np.transpose(eod, (1,0,2))  # [num_days, num_stocks, input_dim]

		return (torch.from_numpy(eod),\
				torch.from_numpy(ground_truth),\
				torch.from_numpy(mask), \
				torch.Tensor([idx]))


class AnnTimeDataset(TimeDataset):
	def __init__(self, data_path, mask_path, dataset_type, args) -> None:
		super(AnnTimeDataset, self).__init__(data_path, mask_path, dataset_type, args)
		self.ann = torch.from_numpy(np.load(args.ann_path)['data']).long()

	def __len__(self) -> int:
		return super().__len__()
	
	def __getitem__(self, idx) -> Tuple[torch.Tensor]:  # return (x, y, mask, ann)
		end_idx = idx + self.days  # mask not shifted
		return (self.data[idx:end_idx, :, self.args.label_cnt:], 
				self.data[end_idx-1, :, :self.args.label_cnt], 
				self.mask[min(end_idx, len(self.data)-1), :], 
				self.ann[idx:end_idx, :, :])


class MaskedTimeDataset(TimeDataset):
	def __init__(self, data_path, mask_path, dataset_type, args) -> None:
		super(MaskedTimeDataset, self).__init__(data_path, mask_path, dataset_type, args)

		index = np.argwhere(self.mask>0)
		self.idx2pair = {i:index[i] for i in range(index.shape[0]) \
            if index[i][0] < self.data.shape[0]-self.days+1}  # no need to remove the last day
		self.idx_arr = list(self.idx2pair.keys())
		print(len(self.idx2pair))

	def __len__(self) -> int:
		return len(self.idx2pair)

	def __getitem__(self, idx) -> Tuple[torch.Tensor]:  # return (x, y, mask, day_idx)
		pair = self.idx2pair[self.idx_arr[idx]]  # [day_idx, stock_idx]
		start_idx, end_idx = pair[0], pair[0] + self.days

		x = self.data[start_idx: end_idx, pair[1], self.args.label_cnt:]
		y = self.data[end_idx-1, pair[1], :self.args.label_cnt]
		return (x, y, torch.Tensor([1]), torch.Tensor([start_idx]))


class AdjTimeDataset(TimeDataset):
	def __init__(self, data_path, mask_path, dataset_type, args) -> None:
		super().__init__(data_path, mask_path, dataset_type, args)
		self.adj = torch.from_numpy(np.load(args.adj_path).astype('float32'))

	def __len__(self) -> int:
		return super().__len__()
	
	def __getitem__(self, idx) -> Tuple[torch.Tensor]:
		# returns (x, y, mask, adj, end_idx, ann placeholder)
		end_idx = idx + self.days
		cur_mask = self.mask[end_idx-1, :]  # mask not shifted
		if self.args.mask_adj:
			cur_adj = torch.mul(self.adj, cur_mask.reshape(-1, 1))  # broadcast: [n*n] * [n*1] -> [n*n]
		else:
			cur_adj = self.adj

		if self.args.use_adj:
			if self.args.normalize_adj:
				cur_adj = normalize(cur_adj)
		else:
			cur_adj = cur_adj.nonzero().t()

		return (self.data[idx:end_idx, :, self.args.label_cnt:],
				self.data[end_idx-1, :, :self.args.label_cnt],
				self.mask[min(end_idx, len(self.data)-1), :],
				cur_adj.float() if self.args.use_adj else cur_adj.long(),
				torch.LongTensor([cur_adj.size(1)]),  # meaningless
				torch.LongTensor([0]))  # meaningless


class AdjAnnTimeDataset(AdjTimeDataset):
	def __init__(self, data_path, mask_path, dataset_type, args) -> None:
		super().__init__(data_path, mask_path, dataset_type, args)
		self.ann = torch.from_numpy(np.load(args.ann_path)['data']).long()

	def __len__(self) -> int:
		return super().__len__()
	
	def __getitem__(self, idx) -> Tuple[torch.Tensor]:
		# returns (x, y, mask, adj, end_idx, ann)
		end_idx = idx + self.days
		cur_mask = self.mask[end_idx-1, :]  # mask not shifted
		if self.args.mask_adj:
			cur_adj = torch.mul(self.adj, cur_mask.reshape(-1, 1))  # broadcast: [n*n] * [n*1] -> [n*n]
		else:
			cur_adj = self.adj
		
		if self.args.use_adj:
			if self.args.normalize_adj:
				cur_adj = normalize(cur_adj)
		else:
			cur_adj = cur_adj.nonzero().t()
		
		return (self.data[idx:end_idx, :, self.args.label_cnt:],
				self.data[end_idx-1, :, :self.args.label_cnt],
				self.mask[min(end_idx, len(self.data)-1), :],
				cur_adj.float() if self.args.use_adj else cur_adj.long(),
				torch.LongTensor([cur_adj.size(1)]), # meaningless
				self.ann[idx:end_idx, :, :])  


class AdjSeqTimeDataset(AdjTimeDataset):
	def __init__(self, data_path, mask_path, dataset_type, args) -> None:
		super().__init__(data_path, mask_path, dataset_type, args)

	def __len__(self) -> int:
		return super().__len__()
	
	def __getitem__(self, idx) -> Tuple[torch.Tensor]:
		# returns (x, y, mask, adjs, end_idxs)
		end_idx = idx + self.days
		adjs, edgenum = [], []
		for i in range(idx, end_idx):  # [idx, end_idx-1]
			cur_mask = self.mask[i, :]  # mask not shifted
			cur_adj = torch.mul(self.adj, cur_mask.reshape(-1, 1)) \
				if self.args.mask_adj else self.adj # broadcast: [n*n] * [n*1] -> [n*n]
			
			if self.args.use_adj:
				if self.args.normalize_adj:
					cur_adj = normalize(cur_adj)
			else:
				cur_adj = cur_adj.nonzero().t()
				edgenum.append(cur_adj.size(1))

			adjs.append(cur_adj)

		if self.args.use_adj:  # stacked adjs
			return (self.data[idx:end_idx, :, self.args.label_cnt:], \
					self.data[end_idx-1, :, :self.args.label_cnt], \
					self.mask[min(end_idx, len(self.data)-1), :], \
					torch.stack(adjs, dim=0).float(), \
					torch.LongTensor([1931]),  # meaningless
					torch.LongTensor([0]))  # meaningless
		else:  # concated edge index, also return end idx for every edge index
			return (self.data[idx:end_idx, :, self.args.label_cnt:], \
					self.data[end_idx-1, :, :self.args.label_cnt], \
					self.mask[min(end_idx, len(self.data)-1), :], \
					torch.cat(adjs, dim=1).long(), \
					torch.LongTensor(edgenum),  # default 
					torch.LongTensor([0])) 


class AdjAnnSeqTimeDataset(AdjAnnTimeDataset):
	def __init__(self, data_path, mask_path, dataset_type, args) -> None:
		super().__init__(data_path, mask_path, dataset_type, args)

	def __len__(self) -> int:
		return super().__len__()
	
	def __getitem__(self, idx) -> Tuple[torch.Tensor]:
		# returns (x, y, mask, adjs, end_idxs)
		end_idx = idx + self.days
		adjs, edgenum = [], []
		for i in range(idx, end_idx):  # [idx, end_idx-1]
			cur_mask = self.mask[i, :]  # mask not shifted
			cur_adj = torch.mul(self.adj, cur_mask.reshape(-1, 1)) \
				if self.args.mask_adj else self.adj # broadcast: [n*n] * [n*1] -> [n*n]
			
			if self.args.use_adj:
				if self.args.normalize_adj:
					cur_adj = normalize(cur_adj)
			else:
				cur_adj = cur_adj.nonzero().t()
				edgenum.append(cur_adj.size(1))
			adjs.append(cur_adj)

		if self.args.use_adj:  # stacked adjs
			return (self.data[idx:end_idx, :, self.args.label_cnt:], \
					self.data[end_idx-1, :, :self.args.label_cnt], \
					self.mask[min(end_idx, len(self.data)-1), :], \
					torch.stack(adjs, dim=0).float(), \
					torch.LongTensor([1931]),  # meaningless
					self.ann[idx:end_idx, :, :])  
		else:  # concated edge index, also return end idx for every edge index
			return (self.data[idx:end_idx, :, self.args.label_cnt:], \
					self.data[end_idx-1, :, :self.args.label_cnt], \
					self.mask[min(end_idx, len(self.data)-1), :], \
					torch.cat(adjs, dim=1).long(), \
					torch.LongTensor(edgenum), # default 
					self.ann[idx:end_idx, :, :]) 


class SparseAdjTimeDataset(TimeDataset):
	def __init__(self, data_path, mask_path, dataset_type, args) -> None:
		super().__init__(data_path, mask_path, dataset_type, args)
		self.adj_list = self.load_graphs(args.sparse_adj_path)  # list of edge indexs
	
	def load_graphs(self, graph_path):
		adj_list = []
		st = 0 if self.dataset_type == "train" else 2305
		total_days = 2305 if self.dataset_type=='train' else 126
		for i in range(st, st+total_days):
			adj = np.load(os.path.join(graph_path, f"date_{st}.npz"))['adj']
			adj_list.append(torch.from_numpy(adj).long())  # edge index should be longtensor
		assert len(adj_list) == total_days
		print(f"load {len(adj_list)} {self.dataset_type} graphs successful!")
		return adj_list

	def __len__(self) -> int:
		return super().__len__()
	
	def __getitem__(self, idx) -> Tuple[torch.Tensor]:
		# returns (x, y, mask, adj, end_idx, ann placeholder)
		end_idx = idx + self.days
		cur_mask = self.mask[end_idx-1, :]  # mask not shifted
		cur_adj = to_dense_adj(self.adj_list[end_idx-1], max_num_nodes=self.args.stock_num).squeeze(0),
		if self.args.mask_adj:
			cur_adj = torch.mul(cur_adj, cur_mask.reshape(-1, 1))  # broadcast: [n*n] * [n*1] -> [n*n]

		if self.args.use_adj:
			if self.args.normalize_adj:
				cur_adj = normalize(cur_adj)
		else:
			cur_adj = cur_adj.nonzero().t()

		return (self.data[idx:end_idx, :, self.args.label_cnt:],
				self.data[end_idx-1, :, :self.args.label_cnt],
				self.mask[min(end_idx, len(self.data)-1), :],
				cur_adj.float() if self.args.use_adj else cur_adj.long(),
				torch.LongTensor([cur_adj.size(1)]),  # meaningless
				torch.LongTensor([0]))  # meaningless


class SparseAdjSeqTimeDataset(SparseAdjTimeDataset):
	def __init__(self, data_path, mask_path, dataset_type, args) -> None:
		super().__init__(data_path, mask_path, dataset_type, args)

	def __len__(self) -> int:
		return super().__len__()
	
	def __getitem__(self, idx) -> Tuple[torch.Tensor]:
		# returns (x, y, mask, adjs, end_idxs, ann)
		end_idx = idx + self.days
		adjs, edgenum = [], []
		for i in range(idx, end_idx):  # [idx, end_idx-1]
			cur_mask = self.mask[i, :]  # mask not shifted
			cur_adj = torch.mul(to_dense_adj(self.adj_list[i], max_num_nodes=self.args.stock_num).squeeze(0), cur_mask.reshape(-1, 1)) \
				if self.args.mask_adj else self.adj # broadcast: [n*n] * [n*1] -> [n*n]
			
			if self.args.use_adj:
				if self.args.normalize_adj:
					cur_adj = normalize(cur_adj)
			else:
				cur_adj = cur_adj.nonzero().t()
				edgenum.append(cur_adj.size(1))
			adjs.append(cur_adj)
			
		if self.args.use_adj:  # stacked adjs
			return (self.data[idx:end_idx, :, self.args.label_cnt:], \
					self.data[end_idx-1, :, :self.args.label_cnt], \
					self.mask[min(end_idx, len(self.data)-1), :], \
					torch.stack(adjs, dim=0).float(), \
					torch.LongTensor([1931]),  # meaningless
					torch.LongTensor([0]))  # meaningless
		else:  # concated edge index, also return end idx for every edge index
			return (self.data[idx:end_idx, :, self.args.label_cnt:], \
					self.data[end_idx-1, :, :self.args.label_cnt], \
					self.mask[min(end_idx, len(self.data)-1), :], \
					torch.cat(adjs, dim=1).long(), \
					torch.LongTensor(edgenum),
					torch.LongTensor([0]))


class SparseAdjAnnSeqTimeDataset(SparseAdjTimeDataset):
	def __init__(self, data_path, mask_path, dataset_type, args) -> None:
		super().__init__(data_path, mask_path, dataset_type, args)
		self.ann = torch.from_numpy(np.load(args.ann_path)['data']).long()

	def __len__(self) -> int:
		return super().__len__()
	
	def __getitem__(self, idx) -> Tuple[torch.Tensor]:
		# returns (x, y, mask, adjs, end_idxs, ann)
		end_idx = idx + self.days
		adjs, edgenum = [], []
		for i in range(idx, end_idx):  # [idx, end_idx-1]
			cur_mask = self.mask[i, :]  # mask not shifted
			cur_adj = torch.mul(to_dense_adj(self.adj_list[i], max_num_nodes=self.args.stock_num).squeeze(0), cur_mask.reshape(-1, 1)) \
				if self.args.mask_adj else self.adj # broadcast: [n*n] * [n*1] -> [n*n]
			
			if self.args.use_adj:
				if self.args.normalize_adj:
					cur_adj = normalize(cur_adj)
			else:
				cur_adj = cur_adj.nonzero().t()
				edgenum.append(cur_adj.size(1))
			adjs.append(cur_adj)
			
		if self.args.use_adj:  # stacked adjs
			return (self.data[idx:end_idx, :, self.args.label_cnt:], \
					self.data[end_idx-1, :, :self.args.label_cnt], \
					self.mask[min(end_idx, len(self.data)-1), :], \
					torch.stack(adjs, dim=0).float(), \
					torch.LongTensor([1931]),  # meaningless
					self.ann[idx:end_idx, :, :])  
		else:  # concated edge index, also return end idx for every edge index
			return (self.data[idx:end_idx, :, self.args.label_cnt:], \
					self.data[end_idx-1, :, :self.args.label_cnt], \
					self.mask[min(end_idx, len(self.data)-1), :], \
					torch.cat(adjs, dim=1).long(), \
					torch.LongTensor(edgenum),
					self.ann[idx:end_idx, :, :])


class RGCNTimeDataset(SparseAdjSeqTimeDataset):
	def __init__(self, data_path, mask_path, dataset_type, args) -> None:
		super().__init__(data_path, mask_path, dataset_type, args)

	def __len__(self) -> int:
		return super().__len__()
	
	def __getitem__(self, idx) -> Tuple[torch.Tensor]:
		# returns (x, y, mask, adjs, end_idx)
		# adjs shape: [rel_nums, nodes, nodes]
		end_idx = idx + self.days
		cur_mask = self.mask[end_idx-1, :]  # mask not shifted
		cur_adj1 = self.adj
		cur_adj2 = to_dense_adj(self.adj_list[end_idx-1], max_num_nodes=self.args.stock_num).squeeze(0),
		if self.args.mask_adj:
			cur_adj1, cur_adj2 = torch.mul(cur_adj1, cur_mask.reshape(-1, 1)), torch.mul(cur_adj2, cur_mask.reshape(-1, 1))  # broadcast: [n*n] * [n*1] -> [n*n]

		if self.args.use_adj and self.args.normalize_adj:
			cur_adj1, cur_adj2 = normalize(cur_adj1), normalize(cur_adj2)
		else:
			raise NotImplementedError("need to set --use_adj  and --normalize_adj in argfile")

		cur_adj = torch.stack((cur_adj1, cur_adj2), dim=0)

		return (self.data[idx:end_idx, :, self.args.label_cnt:],
				self.data[end_idx-1, :, :self.args.label_cnt],
				self.mask[min(end_idx, len(self.data)-1), :],
				cur_adj.float(),
				torch.LongTensor([cur_adj.size(1)]),  # meaningless
				torch.LongTensor([0]))  # meaningless


class RGCNSeqTimeDataset(SparseAdjSeqTimeDataset):
	def __init__(self, data_path, mask_path, dataset_type, args) -> None:
		super().__init__(data_path, mask_path, dataset_type, args)
		self.adj = torch.from_numpy(np.load(args.adj_path).astype('float32'))

	def __len__(self) -> int:
		return super().__len__()

	def __getitem__(self, idx ) -> Tuple[torch.Tensor]:
		# returns (x, y, mask, adjs, end_idxs)
		# adjs shape: [num_days, rel_nums, nodes, nodes]
		end_idx = idx + self.days
		adjs, edgenum = [], []
		for i in range(idx, end_idx):  # [idx, end_idx-1]
			cur_mask = self.mask[i, :]  # mask not shifted
			cur_adj1 = torch.mul(self.adj, cur_mask.reshape(-1, 1)) \
				if self.args.mask_adj else self.adj # broadcast: [n*n] * [n*1] -> [n*n]
			cur_adj2 = torch.mul(to_dense_adj(self.adj_list[i], max_num_nodes=self.args.stock_num).squeeze(0), cur_mask.reshape(-1, 1)) \
				if self.args.mask_adj else self.adj # broadcast: [n*n] * [n*1] -> [n*n]
			
			if self.args.use_adj and self.args.normalize_adj:
				cur_adj1, cur_adj2 = normalize(cur_adj1), normalize(cur_adj2)
			else:
				raise NotImplementedError("need to set --use_adj  and --normalize_adj in argfile")
			adjs.append(torch.stack((cur_adj1, cur_adj2), dim=0))
			
		# return stacked adjs
		return (self.data[idx:end_idx, :, self.args.label_cnt:], \
				self.data[end_idx-1, :, :self.args.label_cnt], \
				self.mask[min(end_idx, len(self.data)-1), :], \
				torch.stack(adjs, dim=0).float(), \
				torch.LongTensor([1931]),  # meaningless
				torch.LongTensor([0]))  # meaningless


class GraphDataset(PygDataset):
	def __init__(self, root: Optional[str] = None, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None, pre_filter: Optional[Callable] = None):
		super().__init__(root, transform, pre_transform, pre_filter)
	
	def __len__(self) -> int:
		return super().__len__()

	def get(self, idx: int) -> Data:
		# data.x: [num_nodes, num_node_features]
		# data.edge_index: [2, num_edges]
		# data.edge_attr: [num_edges, num_edge_features]
		return super().get(idx)
	
	def process(self):
		return super().process()
	
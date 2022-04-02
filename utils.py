import math
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import scipy.sparse as sp

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
#from dgl import backend as F
#from dgl.batch import batch
#from dgl.heterograph import DGLHeteroGraph as DGLGraph


def set_seed(seed=10086): # 1029
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	'''
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.enabled = False
	'''

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def lr_lambda(current_step):
	num_warmup_steps = 40*3  # 3 epochs
	num_training_steps = 40*30
	num_cycles = 0.5
	if current_step < num_warmup_steps:
		return float(current_step) / float(max(1,num_warmup_steps))
	progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps))
	return max(0.0, 0,5*(1.0*math.cos(math.pi*float(num_cycles)*2.0*progress)))

'''
def graph_collate_fn(items):
	elem = items[0]
	elem_type = type(elem)
	if isinstance(elem, DGLGraph):
		batched_graphs = batch(items)
		return batched_graphs
	elif F.is_tensor(elem):
		return F.cat(items, 0)
		# return F.stack(items, 0)
	elif isinstance(elem, Sequence):
		item_iter = iter(items)
		elem_size = len(next(item_ter))
		if not all(len(elem)==elem for elem in item_iter):
			raise RuntimeError('hy: each element in list of batch should be of equal size')
		transposed = zip(*items)
		return [graph_collate_fn(samples) for samples in transposed]
'''

def optimizer_to(optim, device):
	for param in optim.state.values():
		if isinstance(param, torch.Tensor):
			param.data = param.data.to(device)
		elif isinstance(param, dict):
			for subparam in param.values():
				if isinstance(subparam, torch.Tensor):
					subparam.data = subparam.data.to(device)
					if subparam._grad is not None:
						subparam._grad.data = subparam._grad.data.to(device)


def print_grad_norm(model):
	total_norm = 0.0
	parameters = [(name, p) for (name, p) in model.named_parameters() if p.grad is not None and p.requires_grad]
	'''
	for name, param in model.named_parameters():
		print(name, param.size(), param.grad, param.requires_grad)
	assert len(parameters) > 0
	'''
	for name, p in parameters:
		param_norm = p.grad.detach().data.norm(2)
		if not (param_norm < 20 and param_norm > -20):
			print("norm clip needed: ", p.grad.detach().data, param_norm, name, p.size())
		total_norm += param_norm.item()**2
	return total_norm


def is_loss(x1, x2):
	if len(x1.size())==1 and len(x2.size())==1:
		x1 = x1.unsqueeze(0)
		x2 = x2.unsqueeze(0)
	cos = nn.CosineSimilarity(dim=1, eps=1e-6)
	person = cos(x1-x1.mean(dim=1, keepdm=True), x2-x2.mean(dim=1, keepdim=True))
	return person


def fill_window(data, window=8):
	n1, n2, n3 = data.shape
	data = data.tolist()
	for i in range(n1-1, window-1, -1):
		for j in range(n2):
			for k in range(n3):
				if data[i][j][k] == 0 and sum([ data[x][j][k] for x in range(i-window,i) ]) > 0:
					data[i][j][k] = max([ data[x][j][k] for x in range(i-window, i) ])
	return np.array(data)


def normalize(mx):
	# row nornalize sparse matrix
	rowsum = np.array(mx.sum(1), dtype=np.float32)  # m*n -> m*1
	r_inv = np.power(rowsum, -1).flatten()
	r_inv[np.isinf(r_inv)] = 0
	r_mat_inv = sp.diags(r_inv)
	mx = r_mat_inv.dot(mx)
	return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
	# scipy sparse matrix to torch sparse tensor
	sparse_mx = sparse_mx.tocoo()
	indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
	values = torch.Tensor(sparse_mx.data)
	shape = torch.Size(sparse_mx.shape)
	return torch.sparse.FloatTensor(indices, values, shape)


def weighted_mse_loss(inputs, target, weight):
	return torch.mean(weight * (inputs - target)**2)


def increment_date(date_str, days=1):
    date = datetime(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
    date += timedelta(days=days)
    return date.strftime('%Y%m%d')

def get_stock_list(stock_list_path):
    stock_list = []
    with open(stock_list_path, "r") as f:
        for line in f:
            stock_list.append(line.strip())
    return stock_list


def get_trading_days(traiding_days_path):
    trading_days = []
    with open(traiding_days_path, "r") as f:
        for line in f:
            trading_days.append(line.strip())
    return trading_days


def get_stock_id_mapping(stock_list_path):
    # key:stock_code, value:idx
    stock_list = get_stock_list(stock_list_path)
    return dict(zip(stock_list, list(range(len(stock_list)))))


def get_days_id_mapping(traiding_days_path):
    # key:date, value:idx
    trading_days = get_trading_days(traiding_days_path)
    return dict(zip(trading_days, list(range(len(trading_days)))))


def fill_zeros_with_last(arr):
    # fill zero values of 1d numpy array with last non-zero values
    prev = np.arange(len(arr))
    prev[arr == 0] = 0
    prev = np.maximum.accumulate(prev)
    return arr[prev]

def fill_na_with_last(tmp):
    tmp = tmp.reshape(tmp.shape[0], -1)
    # tmp[tmp==0] = np.nan
    m = np.isnan(tmp)
    print("mask", m.shape)
    pos = np.where(~m, np.expand_dims(np.arange(tmp.shape[0]), axis=1), 0)
    print("pos", pos.shape)
    np.maximum_accumulate(pos, axis=0, out=pos)
    tmp[m] = tmp[pos[m], np.nonzero(m)[-1]]
    print("tmp:", tmp.shape)
    tmp = tmp.fillna(0).reshape((2305, 4096, -1))
    
def fill_ma_nan(df):
    # 'ma5','ma10','ma20','ma30'
    prev_row = df.iloc[-1]
    for idx,row in df[::-1].iterrows():
        # print(idx)
        if row.isnull().sum().sum() == 0:
            # print(idx)
            break
        # print(row['ma30'], row['ma30'] == np.nan, pd.isna(row['ma30']))
        if pd.isna(row['ma30']):
            if not pd.isna(prev_row['ma20']):
                df.loc[idx, 'ma30'] = prev_row['ma20']
            elif not pd.isna(prev_row['ma10']):
                df.loc[idx,'ma30'] = prev_row['ma10']
            elif not pd.isna(prev_row['ma5']):
                df.loc[idx,'ma30'] = prev_row['ma5']
            else:
                df.loc[idx,'ma30'] = row['close']
        
        if pd.isna(row['ma20']):
            if not pd.isna(prev_row['ma10']):
                df.loc[idx,'ma20'] = prev_row['ma10']
            elif not pd.isna(prev_row['ma5']):
                df.loc[idx,'ma20'] = prev_row['ma5']
            else:
                df.loc[idx,'ma20'] = row['close']
                
        if pd.isna(row['ma10']):
            if not pd.isna(prev_row['ma5']):
                df.loc[idx,'ma10'] = prev_row['ma5']
            else:
                df.loc[idx,'ma10'] = row['close']
        
        if pd.isna(row['ma5']):
            df.loc[idx,'ma5'] = row['close']
                
        prev_row = row

"""Math utils functions."""

import torch


def cosh(x, clamp=15):
    return x.clamp(-clamp, clamp).cosh()


def sinh(x, clamp=15):
    return x.clamp(-clamp, clamp).sinh()


def tanh(x, clamp=15):
    return x.clamp(-clamp, clamp).tanh()


def arcosh(x):
    return Arcosh.apply(x)


def arsinh(x):
    return Arsinh.apply(x)


def artanh(x):
    return Artanh.apply(x)


class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-7, 1 - 1e-7)  # 1e-15 -> 1e-7
        ctx.save_for_backward(x)
        z = x.double()

        assert torch.eq(1-z, 0).any() == False
        assert torch.isinf(torch.log(1-z)).any() == False
        
        return (torch.log_(1 + z).sub_(torch.log_(1 - z))).mul_(0.5).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


class Arsinh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        z = x.double()
        return (z + torch.sqrt_(1 + z.pow(2))).clamp_min_(1e-15).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 + input ** 2) ** 0.5


class Arcosh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(min=1.0 + 1e-15)
        ctx.save_for_backward(x)
        z = x.double()
        return (z + torch.sqrt_(z.pow(2) - 1)).clamp_min_(1e-15).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (input ** 2 - 1) ** 0.5
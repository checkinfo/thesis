import os
from turtle import forward
import torch
import datetime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from collections import namedtuple
from torch_geometric.nn import GINConv, SAGEConv, GraphConv, GATConv

from gclstm import GLSTMCell, RGCN, GraphAttentionLayer, GAT


class Temporal_Attention_layer(nn.Module):
    def __init__(self, in_channels, num_of_vertices, num_of_timesteps):
        super(Temporal_Attention_layer, self).__init__()
        self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices))
        self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices))
        self.U3 = nn.Parameter(torch.FloatTensor(in_channels))
        self.be = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps))
        self.Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps))

    def forward(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B, T, T)
        '''
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        # print(self.U1)

        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
        # x:(B, N, F_in, T) -> (B, T, F_in, N)
        # (B, T, F_in, N)(N) -> (B,T,F_in)
        # (B,T,F_in)(F_in,N)->(B,T,N)
        # print('lhs',lhs)
        rhs = torch.matmul(self.U3, x)  # (F)(B,N,F,T)->(B, N, T)
        # print('rhs', rhs)
        product = torch.matmul(lhs, rhs)  # (B,T,N)(B,N,T)->(B,T,T)
        # print('product', product)
        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)
        # print('E', E)
        E_normalized = F.softmax(E, dim=1)
        # print('E_norm', E_normalized)
        return E_normalized


class MlpModel(nn.Module):
	def __init__(self, args):
		super(MlpModel, self).__init__()
		self.args = args
		self.fc1 = nn.Linear(args.input_dim, args.hidden_dim)
		self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
		self.fc3 = nn.Linear(args.hidden_dim, args.hidden_dim)
		self.fc4 = nn.Linear(args.hidden_dim, 1)

		self.dropout = nn.Dropout(args.dout)
		self.seq_len = args.num_days
		self.relu = nn.ReLU()

		if self.args.side_info:
			self.type_embed = nn.Embedding(args.ann_embed_num, args.ann_embed_dim)  # [89,128]
			self.fuse_type = nn.Linear(args.hidden_dim*2, args.hidden_dim)

	def forward(self, x, side_info=None):
		assert torch.isnan(x).any() == False
		batch_size, seq_len, num_stocks, num_features = x.size()
		
		if self.args.side_info:
			side_info, _ = torch.max(self.type_embed(side_info), dim=3)  # values, indices
			# [batch, num_days, num_stocks, 25, embed_dim] -> [batch, num_days, num_stocks, embed_dim] 

		output = self.relu(self.fc1(x))
		output = self.relu(self.fc2(output))
		output = self.relu(self.fc3(output))

		if self.args.side_info:
			output = self.fuse_type(torch.cat([output, side_info], dim=-1))

		output = self.fc4(output)[:, -1, :, :]
		return output.reshape((batch_size, num_stocks, -1))


class BaseLSTM(nn.Module):
	def __init__(self, args):
		super(BaseLSTM, self).__init__()
		self.args = args
		self.input_size, self.hidden_size = args.input_dim, args.hidden_dim
		self.rnn1 = nn.LSTM(args.input_dim, args.hidden_dim, args.lstm_layers, dropout=args.dout, bidirectional=True, batch_first=True)

		self.fc0 = nn.Linear(2*args.hidden_dim, args.hidden_dim)
		self.predict = nn.Linear(args.hidden_dim, 1)

		self.relu = nn.PReLU()
		self.dropout = nn.Dropout(args.dout)
		self.seq_len = args.num_days

	def forward(self, x, side_info=None):
		batch_size, seq_len, num_stocks, num_features = x.size()
		new_x = x.permute((0,2,1,3)).reshape((batch_size*num_stocks, seq_len, num_features))
		# [batch_size, seq_len, num_stocks, num_features] -> [batch_size*num_stocks, seq_len, num_features]

		out_rnn, state = self.rnn1(new_x)
		out_rnn = out_rnn[:, -1, :]
		out_rnn = self.relu(self.dropout(self.fc0(out_rnn)))
		output = self.predict(out_rnn)

		return output.reshape((batch_size, num_stocks, -1))


class TransformerModel(nn.Module):
	def __init__(self):
		super().__init__()


class GNNModel(nn.Module):
	def __init__(self, args):
		super(GNNModel, self).__init__()
		self.args = args
		self.input_size, self.hidden_size = args.input_dim, args.hidden_dim
		self.fc1 = nn.Linear(args.input_dim, args.hidden_dim)
		self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
		self.fc3 = nn.Linear(args.hidden_dim*args.num_heads, args.hidden_dim)
		self.fc4 = nn.Linear(args.hidden_dim, 1)

		self.gnns = nn.ModuleList([GraphConv(in_channels=args.hidden_dim, \
			out_channels=args.hidden_dim, aggr='mean')] * args.gnn_layers)

		self.dropout = nn.Dropout(args.dout)
		self.seq_len = args.num_days
		self.relu = nn.LeakyReLU()

		if self.args.side_info:
			self.type_embed = nn.Embedding(args.ann_embed_num, args.ann_embed_dim)  # [89,128]
			self.fuse_type = nn.Linear(args.hidden_dim*2, args.hidden_dim)

	def forward(self, x, edge_indexs, edgenum, side_info=None):
		batch_size, seq_len, num_stocks, num_features = x.size()
		assert batch_size == 1

		if self.args.side_info:
			side_info, _ = torch.max(self.type_embed(side_info), dim=3)  # values, indices
			# [batch, num_days, num_stocks, 25, embed_dim] -> [batch, num_days, num_stocks, embed_dim]
		
		edge_indexs = edge_indexs.squeeze(0)

		graphs = []
		if edge_indexs.size(0) == 2:  # concated edge index [2, n]
			graphs = torch.split(edge_indexs, edgenum.flatten().tolist(), dim=1)
		else:  # stacked dense adjs [num_days, stocknum, stocknum]
			for i in range(edge_indexs.size(0)):
				graphs.append(edge_indexs[i].nonzero().t())
				
		output = self.relu(self.fc1(x))  # x: [num_ndoes, input_size]
		output = self.relu(self.fc2(output))

		if self.args.side_info:
			output = self.fuse_type(torch.cat([output, side_info], dim=-1))

		output = torch.reshape(output, (seq_len, num_stocks, self.hidden_size))

		new_out = []
		for i in range(seq_len):
			for j in range(len(self.gnns)):
				out = self.relu(self.gnns[j](output[i], graphs[i]))  # input: x, edge_index
				out = torch.reshape(out, (num_stocks, -1))
			new_out.append(out)

		new_out = torch.stack(new_out, dim=0)
		new_out = torch.reshape(new_out, (batch_size, seq_len, num_stocks, -1))
		new_out = self.relu(self.fc3(new_out))
		new_out = self.fc4(new_out)[:, -1, :, :]
		
		return new_out.reshape((batch_size, num_stocks, -1))


class RGCNModel(nn.Module):
	def __init__(self, args):
		super().__init__()
		self.args = args
		self.input_size, self.hidden_size = args.input_dim, args.hidden_dim
		self.fc1 = nn.Linear(args.input_dim, args.hidden_dim)
		self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
		self.fc3 = nn.Linear(args.hidden_dim*(args.num_heads**args.gnn_layers), args.hidden_dim)  # TODO
		self.fc4 = nn.Linear(args.hidden_dim, 1)

		self.gnns = nn.ModuleList([RGCN(args.hidden_dim if i==0 else args.hidden_dim*args.num_heads, args.hidden_dim, args.dout, args.relation_num, args.graph_attn, args.num_heads) for i in range(args.gnn_layers)])

		self.dropout = nn.Dropout(args.dout)
		self.seq_len = args.num_days
		self.relu = nn.LeakyReLU()

		if self.args.side_info:
			self.type_embed = nn.Embedding(args.ann_embed_num, args.ann_embed_dim)  # [89,128]
			self.fuse_type = nn.Linear(args.hidden_dim*2, args.hidden_dim)

	def forward(self, x, adjs, edgenum, side_info=None):
		batch_size, seq_len, num_stocks, num_features = x.size()
		assert batch_size == 1
		adjs = adjs.squeeze(0)

		if self.args.side_info:
			side_info, _ = torch.max(self.type_embed(side_info), dim=3)  # values, indices
			# [batch, num_days, num_stocks, 25, embed_dim] -> [batch, num_days, num_stocks, embed_dim]
		
		output = self.relu(self.fc1(x))  # x: [num_ndoes, input_size]
		output = self.relu(self.fc2(output))

		if self.args.side_info:
			output = self.fuse_type(torch.cat([output, side_info], dim=-1))

		output = torch.reshape(output, (seq_len, num_stocks, self.hidden_size))

		new_out = []
		for i in range(seq_len):
			out = output[i]
			for j in range(len(self.gnns)):
				out = self.relu(self.gnns[j](out, adjs[i]))  # input: x, edge_index
				out = torch.reshape(out, (num_stocks, -1))
			new_out.append(out)

		new_out = torch.stack(new_out, dim=0)
		new_out = torch.reshape(new_out, (batch_size, seq_len, num_stocks, -1))
		new_out = self.relu(self.fc3(new_out))
		new_out = self.fc4(new_out)[:, -1, :, :]
		
		return new_out.reshape((batch_size, num_stocks, -1))


class GLSTM(nn.Module):
	def __init__(self, args):
		super(GLSTM, self).__init__()
		self.args = args
		self.hidden_dim = args.hidden_dim
		self.input_to_hidden = nn.Linear(args.input_dim, args.hidden_dim)
		self.glstm_cell = GLSTMCell(args.hidden_dim, args.hidden_dim, args.relation_num, args.dout, args)
		self.w_out = nn.Linear(args.hidden_dim, 1)
		self.num_layers = args.gnn_layers
		self.dropout = torch.nn.Dropout(args.dout)

	def forward(self, x, edge_indexs, edgenum, side_info=None):
		'''
        :param x:   (batch, time, node, feature_size)
        :param adjs: (batch, time, node, node)
        :return:    (batch_size, node, label)
        '''
		batch_size, seq_len, num_stocks, num_features = x.size()
		assert batch_size == 1
		edge_indexs = edge_indexs.squeeze(0)

		graphs = []
		if edge_indexs.size(0) == 2:  # concated edge index [2, n]
			graphs = torch.split(edge_indexs, edgenum.flatten().tolist(), dim=1)
		else:  # stacked dense adjs [num_days, stocknum, stocknum]
			for i in range(edge_indexs.size(0)):
				graphs.append(edge_indexs[i].nonzero().t())


		x = self.input_to_hidden(x)
		last_h_time_price = torch.squeeze(x[0], 0)  # node feature 
		last_c_time_price = torch.squeeze(x[0], 0)  # node feature 
       
		for t in range(seq_len):
			last_h_layer_price = last_h_time_price
			last_c_layer_price = last_c_time_price
            
			for l in range(self.num_layers):
                # x, h, c, h_t, adj
				last_h_layer_price, last_c_layer_price = self.glstm_cell(torch.squeeze(x[t], 0), \
					last_h_layer_price, last_c_layer_price, last_h_time_price, graphs[t])
			last_h_time_price, last_c_time_price = last_h_layer_price, last_c_layer_price

		return self.w_out(self.dropout(last_h_time_price)).reshape(batch_size, num_stocks, 1)


class BiGLSTM(nn.Module):
	def __init__(self, args):
		super(BiGLSTM, self).__init__()
		self.args = args
		self.hidden_dim = args.hidden_dim
		self.input_to_hidden = nn.Linear(args.input_dim, args.hidden_dim)
		self.forward_cells = nn.ModuleList([GLSTMCell(args.hidden_dim, args.hidden_dim, args.relation_num, args.dout, args)] * args.glstm_layers)
		self.backward_cells = nn.ModuleList([GLSTMCell(args.hidden_dim, args.hidden_dim, args.relation_num, args.dout, args)] * args.glstm_layers)
		self.fc0 = nn.ModuleList([nn.Linear(2*args.hidden_dim, args.hidden_dim)] * args.glstm_layers)
		self.w_out = nn.Linear(args.hidden_dim, 1)
		self.num_layers = args.gnn_layers
		self.dropout = torch.nn.Dropout(args.dout)

	def forward(self, x, adjs, edgenum, side_info=None):
		'''
        :param x:   (batch, time, node, feature_size)
        :param adjs: (batch, time, node, node)
        :return:    (batch_size, node, label)
        '''
		batch_size, seq_len, num_stocks, num_features = x.size()
		assert batch_size == 1
		x, adjs = x.squeeze(0), adjs.squeeze(0)
		if len(adjs.size())==2:
			adjs = [adjs] * seq_len

		x = self.input_to_hidden(x)
		# [seq_len, num_stocks, hidden_size]
		for i in range(self.args.glstm_layers):
			# forward pass
			last_h_time_price = torch.squeeze(x[0], 0)  # node feature 
			last_c_time_price = torch.squeeze(x[0], 0)  # node feature
			h_forward = []
		
			for t in range(seq_len):
				last_h_layer_price = last_h_time_price
				last_c_layer_price = last_c_time_price
				
				for l in range(self.num_layers):
					# x, h, c, h_t, adj
					last_h_layer_price, last_c_layer_price = self.forward_cells[i](torch.squeeze(x[t], 0), \
						last_h_layer_price, last_c_layer_price, last_h_time_price, adjs[t])
				last_h_time_price, last_c_time_price = last_h_layer_price, last_c_layer_price
				h_forward.append(last_h_layer_price)

			# backward pass
			last_h_time_price_b = torch.squeeze(x[-1], 0)  # node feature 
			last_c_time_price_b = torch.squeeze(x[-1], 0)  # node feature 
			h_backward = []
		
			for t in range(seq_len-1, -1, -1):
				last_h_layer_price_b = last_h_time_price_b
				last_c_layer_price_b = last_c_time_price_b
				
				for l in range(self.num_layers):
					# x, h, c, h_t, adj
					last_h_layer_price_b, last_c_layer_price_b = self.backward_cells[i](torch.squeeze(x[t], 0), \
						last_h_layer_price_b, last_c_layer_price_b, last_h_time_price_b, adjs[t])
				last_h_time_price_b, last_c_time_price_b = last_h_layer_price_b, last_c_layer_price_b
				h_backward.append(last_h_layer_price_b)
			
			h_f = torch.stack(h_forward, dim=0)  # seq_len, num_stocks, hidden_size]
			h_b = torch.stack(h_backward, dim=0)  # seq_len, num_stocks, hidden_size]
			
			out = self.fc0[i](self.dropout(torch.cat([h_f, h_b], dim=-1)))
			# [seq_len, num_stocks, hidden_size]
			x = out

		return self.w_out(self.dropout(out[-1])).reshape(batch_size, num_stocks, 1)


class ReRaLSTM(nn.Module):
	def __init__(self, args) -> None:
		super().__init__()
		self.args = args
		self.input_size, self.hidden_size = args.input_dim, args.hidden_dim
		self.rnn1 = nn.LSTM(args.input_dim, args.hidden_dim, args.lstm_layers, dropout=args.dout, bidirectional=True, batch_first=True)

		self.fc0 = nn.Linear(2*args.hidden_dim, args.hidden_dim)
		self.fc1 = nn.Linear(824, 1)
		self.fc2 = nn.Linear(args.hidden_dim, 1)
		self.fc3 = nn.Linear(args.hidden_dim, 1)

		self.predict = nn.Linear(args.hidden_dim*2, 1)

		self.relu = nn.LeakyReLU()
		self.dropout = nn.Dropout(args.dout)
		self.seq_len = args.num_days

		# relation data
		self.all_one = nn.Parameter(torch.ones((args.stock_num, 1)))
		self.rel_encoding, self.rel_mask = self.load_relation_data(args.adj_path)
		self.rel_encoding, self.rel_mask = nn.Parameter(self.rel_encoding, requires_grad=False), nn.Parameter(self.rel_mask, requires_grad=False)

		print('relation encoding shape:', self.rel_encoding.shape, self.rel_encoding.dtype)
		print('relation mask shape:', self.rel_mask.shape, self.rel_mask.dtype)
	
	def load_relation_data(self, relation_file):
		relation_encoding = np.load(relation_file)
		rel_shape = [relation_encoding.shape[0], relation_encoding.shape[1]]
		mask_flags = np.equal(np.zeros(rel_shape, dtype=int),
							np.sum(relation_encoding, axis=2))
		mask = np.where(mask_flags, np.ones(rel_shape) * -1e9, np.zeros(rel_shape))
		return torch.from_numpy(relation_encoding), torch.from_numpy(mask).float()

	def forward(self, x, side_info=None):
		batch_size, seq_len, num_stocks, num_features = x.size()

		new_x = x.permute((0,2,1,3)).reshape((batch_size*num_stocks, seq_len, num_features))
		# [batch_size, seq_len, num_stocks, num_features] -> [batch_size*num_stocks, seq_len, num_features]

		out_rnn, state = self.rnn1(new_x)
		out_rnn = out_rnn[:, -1, :]
		out_rnn = self.relu(self.dropout(self.fc0(out_rnn)))

		output = out_rnn.reshape((batch_size, num_stocks, -1))  # batch_size == 1
		# [batch_size*num_stocks, hidden_dim] -> [batch, num_stocks, hidden_dim]

		rel_weight = self.relu(self.fc1(self.rel_encoding))
		# [num_stocks, num_stocks, xxx] -> [num_stocks, num_stocks, 1]
		
		if self.args.inner_prod: # explicit
			# print('rsr: inner product weight')
			# [batch, num_stocks, hid] * [batch, hid, num_stocks] -> [batch, num_stocks, num_stocks]
			inner_weight = torch.matmul(output, output.transpose(1, 2))
			# [batch, num_stocks, num_stocks]*[num_stocks, num_stocks] -> [batch, num_stocks, num_stocks]
			weight = torch.mul(inner_weight, rel_weight[:,:, -1])
		else:  # implcit
			# print('rsr: sum weight')
			head_weight = self.relu(self.fc2(output))
			tail_weight = self.relu(self.fc3(output))
			# [batch, num_stocks, 1]*[1, num_stocks] + [num_stocks, 1]*[batch, 1, num_stocks] + [num_stocks, num_stocks]
			# print(head_weight.size(), tail_weight.size(), self.all_one.size())
			weight = torch.matmul(head_weight, self.all_one.transpose(0, 1)) + \
					torch.matmul(self.all_one, tail_weight.transpose(1, 2)) + \
					rel_weight[:, :, -1]
		
		# [batch, num_stocks, num_stocks]
		weight_masked = F.softmax(self.rel_mask + weight, dim=-1)
		# [batch, num_stocks, num_stocks] * [batch, num_stocks, hid] -> [batch, num_stocks, hid]
		outputs_proped = torch.matmul(weight_masked, output)
		outputs_concated = torch.concat([output, outputs_proped], axis=-1)
		
		prediction = self.relu(self.predict(outputs_concated))  # [num_stocks, 1]
		return prediction.reshape((batch_size, num_stocks, -1))


class HYperStockGAT(nn.Module):
	def __init__(self) -> None:
		super().__init__()

	def forward(self, ):
		pass

class HypModel(nn.Module):
	def __init__(self, input_size, hidden_size, graph_path, use_hyp=False):
		super(MlpModel, self).__init__()
		self.hidden_size = hidden_size
		self.use_hyp = use_hyp

		self.fc1 = nn.Linear(input_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.fc3 = nn.Linear(hidden_size, hidden_size//2)
		self.fc4 = nn.Linear(hidden_size//2, 1)

		self.tat = Temporal_Attention_layer(hidden_size, 4096, 8)
		self.tat2 = Temporal_Attention_layer(hidden_size, 4096, 8)

		self.time_conv = nn.Conv2d(8, 8, kernel_size=(1,3), stride=(1,1), padding=(0,1))
		self.time_conv2 = nn.Conv2d(8, 8, kernel_size=(1,3), stride=(1,1), padding=(0,1))
		self.time_conv3 = nn.Conv2d(8, 1, kernel_size=(1,3), stride=(1,1), padding=(0,1))
		
		self.num_gnn_layers = 1

		if self.use_hyp:
			self.hyp_dim = hidden_size
			self.manifold = getattr(manifolds, "PoincareBall")()
			# self.manifold = PoincareBall(k=1.0, learnable=True)
			# trainable curvature
			self.curvatures = [nn.Parameter(torch.Tensor([1.])) for _ in  range(self.num_gnn_layers+1)]

			hgc_layers = []
			# dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
			for i in range(self.num_gnn_layers):
				c_in, c_out = self.curvatures[i], self.curvatures[i+1]
				in_dim, out_dim = self.hyp_dim, self.hyp_dim
				print("hyp layer: ", in_dim, out_dim, c_in, c_out)
				act = F.relu
				hgc_layers.append(
					hyp_layers.HyperbolicGraphConvolution(
						self.manifold, in_dim, out_dim, c_in, c_out, dropout=0.3, act=act, use_bias=1, use_att=0
						)
					)
			self.layers = nn.Sequential(*hgc_layers)
		else:
			self.layers = nn.ModuleList([GINConv(nn.Linear(hidden_size, hidden_size), 'mean')]*2)

		gg, _ = dgl.load_graphs(graph_path)
		self.g = gg[0]
		adj = gg[0].adj(scipy_fmt='coo')
		# adj = np.load("graph.npy")
		# adj = sp.coo_matrix(adj, dtype=np.float32)
		# adj = normalize(adj+sp.eye(adj.shape[0]))
		adj = normalize(adj)  # row_normalize
		self.adj = sparse_mx_to_torch_tensor(adj)

		self.debug = False

	def forward(self, x, graph=None): 
		time_flg = True if len(x.size()) == 4 else False

		if time_flg:
			batch_size, num_days, num_stocks, input_size = x.size()
			x = x.reshape((-1, num_stocks, input_size))

		output = F.leaky_relu(self.fc1(x))
		output = F.leaky_relu(self.fc2(output))

		if time_flg:
			# output: [batch_size, num_days, num_stocks, num_features]
			output = output.reshape((batch_size, num_days, num_stocks, -1))

			x_TAt = output
			# assert torch.isinf(x_TAt).any() == False
			# assert torch.isnan(x_TAt).any() == False
			# if (x_TAt==0).nonzero().size(0) != 0:
			# print("x_TAt:", (x_TAt==0).nonzero().size())

			'''
			temporal_At = self.tat(output.permute(0,2,3,1))  # [batch_size, num_days, num_days]
			# [1, 4096*256, 8] * [1,8,8] -> [1, 4096*256, 8] -> (reshape) [1,4096, 256, 8]
			x_TAt = torch.matmul(output.permute(0,2,3,1).reshape(batch_size, -1, num_days), temporal_At)
			x_TAt = x_TAt.reshape(batch_size, num_stocks, -1, num_days).permute(0,3,1,2)
			# x_TAt: [batch_size*num_days, num_stocks, num_features]
			x_TATt = F.leaky_relu(x_TAt)

			x_TAt = self.time_conv(x_TAt.reshape(batch_size, num_days, num_stocks, -1))
			# x_TAt: [batch_size, num_days, num_stocks, num_features]

			outputs = []
			for day in range(num_days):
				y = x_TAt[:, day, :, :]
				y = y.reshape((num_stocks, self.hidden_size//4))

				temp = self.hgc_layers[0](self.g, y)  # returns (h, adj) in hyperbolic space
				outputs.append(temp.reshape(1, num_stocks, self.hidden_size//4))

			# spatial_At: [num_days, batch_size, num_stocks, num_features] -> [batch_size, num_days, num_stocks, num_features]
			spatial_At = F.leaky_relu(torch.stack(outputs)).permute(1,0,2,3)
			'''
			

			outputs = []
			for day in range(num_days):
				y = x_TAt[:, day, :, :]
				y = y.reshape((num_stocks, self.hidden_size//4))

				if self.use_hyp:
					x_tan = self.manifold.proj_tan0(y, self.curvatures[0])
					x_hyp = self.manifold.expmap0(x_tan, c=self.curvatures[0])
					x_hyp = self.manifold.proj(x_hyp, c=self.curvatures[0])
					temp, _ = self.layers.forward((x_hyp, self.adj))
					temp = self.manifold.logmap0(temp, self.curvatures[0])
					# temp = self.manifold.proj_tan0(temp, self.curvatures[0])
				else:
					temp = self.layers[0](self.g, y)

				outputs.append(temp.reshape(1, num_stocks, self.hidden_size//4))
				
			# spatial_At: [num_days, batch_size, num_stocks, num_features] -> [batch_size, num_days, num_stocks, num_features]
			spatial_At = F.leaky_relu(torch.stack(outputs)).permute(1,0,2,3)

			'''
			temporal_At2 = self.tat2(spatial_At.permute(0,2,3,1))  # [batch_size, num_days, num_days]
			# [1,4096*256,8] * [1,8,8] -> [1. 4096*256, 8] -> (reshape) [1,4096, 256, 8]
			x_TAt2 = torch.matmul(spatial_At.permute(0,2,3,1).reshape(batch_size, -1, num_days), temporal_At2)
			x_TAt2 = x_TAt2.reshape(batch_size, num_stocks, -1, num_days).permute(0,3,1,2)
			# x_TAt2: [batch_size*num_days, num_stocks, num_features]

			x_TAT2 = self.time_conv2(x_TAt2.reshape(batch_size, num_days, num_stocks, -1)).reshape(num_days*batch_size, num_stocks, -1)
			'''
		x_TAt2 = spatial_At

		# output = [batch_size, num_stocks, num_features]
		output = F.leaky_relu(self.time_conv3(x_TAt2.reshape(batch_size, num_days, num_stocks, -1))).squeeze(1)
		output = F.leaky_relu(self.fc3(output))
		output = self.fc4(output)

		return output

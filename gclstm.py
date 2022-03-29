import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch_geometric.nn import GINConv, SAGEConv, GraphConv, GATConv


class Attentive_Pooling(nn.Module):
    def __init__(self, hidden_size):
        super(Attentive_Pooling, self).__init__()
        self.w_1 = nn.Linear(hidden_size, hidden_size)
        self.w_2 = nn.Linear(hidden_size, hidden_size)
        self.u = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, memory, query=None, mask=None):
        '''
        :param query:   (node, hidden)
        :param memory: (node, hidden)
        :param mask:
        :return:
        '''
        if query is None:
            h = torch.tanh(self.w_1(memory))  # node, hidden
        else:
            h = torch.tanh(self.w_1(memory) + self.w_2(query))
        score = torch.squeeze(self.u(h), -1)  # node,
        if mask is not None:
            score = score.masked_fill(mask.eq(0), -1e9)
        alpha = F.softmax(score, -1)  # node,
        s = torch.sum(torch.unsqueeze(alpha, -1) * memory, -2)
        return s


class RGCN(nn.Module):
    def __init__(self, in_features, out_features, relation_num):
        super(RGCN, self).__init__()
        self.relation_num = relation_num
        self.linear_1 = nn.Linear(in_features, out_features)
        self.linear_2 = nn.Linear(in_features, out_features)
        self.linears = [self.linear_1, self.linear_2]
        self.activation = nn.Tanh()

    def gcn(self, relation, input, adj):
        support = self.linears[relation](input)
        output = torch.sparse.mm(adj, support)
        return output

    def forward(self, input, adjs):
        '''
        :param input:   (node, hidden)
        :param adjs:    (relationnum, node, node)
        :return:
        '''
        if len(adjs.size()) == 2:
            adjs = torch.unsqueeze(adjs, 0)
        transform = []
        for r in range(self.relation_num):
            transform.append(self.gcn(r, input, adjs[r]))
        # (node, relation, hidden) -> (node, hidden)
        return self.activation(torch.sum(torch.stack(transform, 1), 1))


class GLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, relation_num, dropout, args):
        super(GLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.args = args
        self.dropout = torch.nn.Dropout(dropout)
        self.Wh = nn.Linear(hidden_size, hidden_size * 5, bias=False)
        self.Wn = nn.Linear(hidden_size, hidden_size * 5, bias=False)
        self.Wt = nn.Linear(hidden_size, hidden_size * 5, bias=False)
        self.U = nn.Linear(input_size, hidden_size * 5, bias=False)
        self.V = nn.Linear(hidden_size, hidden_size * 5)
        self.relu = nn.LeakyReLU()
        if args.relation_num > 1:
            self.gnn = RGCN(hidden_size, hidden_size, relation_num)
        else:
            self.gnn = nn.ModuleList([GraphConv(in_channels=args.hidden_dim, \
                out_channels=args.hidden_dim, aggr='mean')] * args.gnn_layers)

    def forward(self, x, h, c, h_t, adjs):
        '''
        :param x:   (node, emb)
            embedding of the node, news and initial node embedding
        :param h:   (node, hidden)
            hidden state from last layer
        :param c:   candidate from last layer
        :param h_t:   (node, hidden)
            hidden state from last time
        :param adj:   (node, node)
            if use RGCN, there should be multiple gcns, each one for a relation
        '''
        for i in range(len(self.gnn)):
            hn = self.relu(self.gnn[i](h, adjs))  # input: x, edge_index
        # hn = self.gnn[0](h, adjs)
        # h: 同一个时间片上一层的，h_t：上一个时间片最后一层
        gates = self.Wh(self.dropout(h)) + self.U(self.dropout(x)) + self.Wn(self.dropout(hn)) + self.Wt(self.dropout(h_t))
        i, f, o, u, t = torch.split(gates, self.hidden_size, dim=-1)
        new_c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(u) + torch.sigmoid(t) * h_t
        new_h = torch.sigmoid(o) * torch.tanh(new_c)
        return new_h, new_c

'''
i = 0
for t, g, ent_embed in zip(time_batched_list_t, train_graphs, per_graph_ent_embeds):
    triplets, neg_tail_samples, neg_head_samples, labels = self.corrupter.single_graph_negative_sampling(t, g, self.num_ents)
    time_diff_tensor_forward = self.train_seq_len - 1 - start_time_tensor_forward[i]
    time_diff_tensor_backward = self.train_seq_len - 1 - start_time_tensor_backward[i]
    #all_embeds_g = self.get_all_embeds_Gt(ent_embed, g, t, hist_embeddings_forward[i][0], hist_embeddings_forward[i][1], time_diff_tensor_forward,
                                                            hist_embeddings_backward[i][0], hist_embeddings_backward[i][1], time_diff_tensor_backward)
    
    if self.learnable_lambda:
        adjusted_prev_graph_embeds_forward = self.decay_hidden(prev_graph_embeds_forward, time_diff_tensor_forward)
        adjusted_prev_graph_embeds_backward = self.decay_hidden(prev_graph_embeds_backward, time_diff_tensor_backward)
    else:
        adjusted_prev_graph_embeds_forward = prev_graph_embeds_forward * torch.exp(-time_diff_tensor_forward * self.inv_temperature)
        adjusted_prev_graph_embeds_backward = prev_graph_embeds_backward * torch.exp(-time_diff_tensor_backward * self.inv_temperature)

    _, hidden_forward = self.forward_rnn(node_repr.unsqueeze(0), adjusted_prev_graph_embeds_forward.expand(self.num_layers, *prev_graph_embeds_forward.shape))

    _, hidden_backward = self.backward_rnn(node_repr.unsqueeze(0), adjusted_prev_graph_embeds_backward.expand(self.num_layers, *prev_graph_embeds_backward.shape))

    if self.post_aggregation or self.post_ensemble or self.impute:
        return node_repr, hidden_forward[-1] + hidden_backward[-1], time_embedding
    else:
        return hidden_forward[-1] + hidden_backward[-1], 
    
    loss_tail = self.train_link_prediction(ent_embed, triplets, neg_tail_samples, labels, all_embeds_g, corrupt_tail=True)
    loss_head = self.train_link_prediction(ent_embed, triplets, neg_head_samples, labels, all_embeds_g, corrupt_tail=False)
    reconstruct_loss += loss_tail + loss_head
    i += 1

'''

class SARGCNLayer(nn.Module):
    # self attention + rgcn
    def __init__(self, args, in_feat, out_feat, num_rels, num_bases, total_times, bias=True, activation=None,
                 self_loop=True, dropout=0.0):
        super(SARGCNLayer, self).__init__(args, in_feat, out_feat, num_rels, num_bases, total_times, bias, activation,
                                          self_loop, dropout)
        self.num_layers = args.num_layers
        self.q_linear = nn.Linear(in_feat, in_feat, bias=False)
        self.v_linear = nn.Linear(in_feat, in_feat, bias=False)
        self.k_linear = nn.Linear(in_feat, in_feat, bias=False)
        self.in_feat = in_feat
        self.h = 8
        self.d_k = in_feat // self.h
        self.post_aggregation = args.post_aggregation
        self.post_ensemble = args.post_ensemble

        self.learnable_lambda = args.learnable_lambda
        if self.learnable_lambda:
            self.exponential_decay = nn.Linear(1, 1)

    def calc_result(self, cur_embeddings, prev_embeddings, time_diff, local_attn_mask):
        if self.learnable_lambda:
            decay_weight = -torch.clamp(self.exponential_decay(time_diff.unsqueeze(1)), min=0).squeeze()
        else:
            decay_weight = 0
        all_time_embeds = torch.cat([prev_embeddings, cur_embeddings.unsqueeze(1)], dim=1)
        bs = all_time_embeds.shape[0]
        q = self.q_linear(cur_embeddings).unsqueeze(1).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        k = self.k_linear(all_time_embeds).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(all_time_embeds).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        scores = self.attention(q, k, v, local_attn_mask, decay_weight)
        return scores.transpose(1, 2).contiguous().view(bs, self.in_feat)

    def forward_final(self, g, prev_embeddings, time_diff, local_attn_mask, time_batched_list_t, node_sizes):
        # pdb.set_trace()
        current_graph, time_embedding = self.forward(g, time_batched_list_t, node_sizes)
        cur_embeddings = current_graph.ndata['h'] + time_embedding
        concat = self.calc_result(cur_embeddings, prev_embeddings, time_diff, local_attn_mask)

        if self.post_aggregation:
            return cur_embeddings, concat
        else:
            return current_graph, concat

    def attention(self, q, k, v, local_attn_mask, decay_weight):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        normalised = F.softmax(scores.squeeze() + local_attn_mask.unsqueeze(1) + decay_weight, dim=-1)
        output = torch.matmul(normalised.unsqueeze(2), v).squeeze()
        return output

    def forward_isolated(self, node_repr, prev_embeddings, time_diff, local_attn_mask, time):
        # cur_embeddings, time_embedding = super().forward_isolated(node_repr, time)
        cur_time_embeddings = cur_embeddings + time_embedding
        concat = self.calc_result(cur_time_embeddings, prev_embeddings, time_diff, local_attn_mask)
        if self.post_aggregation:
            return cur_time_embeddings, concat
        else:
            return concat

    def forward_ema(self, g, prev_embeddings, time_batched_list_t, node_sizes, alpha, train_seq_len):
        current_graph, time_embedding = self.forward(g, time_batched_list_t, node_sizes)
        cur_embeddings = current_graph.ndata['h'] + time_embedding
        # pdb.set_trace()
        all_time_embeds = torch.cat([prev_embeddings, cur_embeddings.unsqueeze(1)], dim=1)
        ema_vec = torch.pow(1 - alpha, cuda(torch.arange(train_seq_len)))
        ema_vec[:, :-1] *= alpha
        ema_vec = ema_vec.flip(-1).unsqueeze(0)
        averaged = torch.sum(all_time_embeds.transpose(1, 2) * ema_vec, -1)
        return averaged

    def forward_ema_isolated(self, node_repr, prev_embeddings, time, alpha, train_seq_len):
        cur_embeddings, time_embedding = super().forward_isolated(node_repr, time)
        # pdb.set_trace()
        all_time_embeds = torch.cat([prev_embeddings, (cur_embeddings + time_embedding).unsqueeze(1)], dim=1)
        ema_vec = torch.pow(1 - alpha, cuda(torch.arange(train_seq_len)))
        ema_vec[:, :-1] *= alpha
        ema_vec = ema_vec.flip(-1).unsqueeze(0)
        averaged = torch.sum(all_time_embeds.transpose(1, 2) * ema_vec, -1)
        return averaged

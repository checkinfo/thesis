import os
import torch
import datetime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, TensorDataset
from collections import namedtuple
from torch_geometric.nn import GINConv as PygGINConv


class MlpModel(nn.Module):
	def __init__(self, input_size, hidden_size, seq_len=-1):
		super(MlpModel, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.fc3 = nn.Linear(hidden_size, hidden_size)
		self.fc4 = nn.Linear(hidden_size, 1)
		self.seq_len = seq_len

	def forward(self, x):
		assert torch.isnan(x).any() == False
		num_days, seq_len, num_stocks, num_features = x.size()
		output = F.relu(self.fc1(x))
		output = F.relu(self.fc2(output))
		output = F.relu(self.fc3(output))
		output = self.fc4(output)[:, -1, :, :]
		return output.reshape((num_days, num_stocks, -1))


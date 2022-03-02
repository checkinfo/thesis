import os
import sys
import torch
import datetime
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import MlpModel
from dataset import *
from utils import *
from train import train, evaluate
from argparser import get_parser

if __name__ == "__main__":
	parser = get_parser()
	args = parser.parse_args()  # sys.argv[1:]
	print(args)
	set_seed(10086)
	
	train_dataset = TimeDataset(args.train_path, args.train_mask_path, args, days=1)
	test_dataset = TimeDataset(args.test_path, args.test_mask_path, args, days=1)
	train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=False)
	test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=False)

	epochs = 20
	model = MlpModel(9, 256)
	criterion = nn.SmoothL1Loss(reduction='none')
	optimzer = torch.optim.Adam(model.parameters(), lr=args.lr)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	model.to(device)
	print(os.getpid())
	print(model)

	max_ic = 0.08
	for epoch in range(epochs):
		train_loss = train(epoch, model, train_dataloader, criterion, optimzer, device, print_inteval=500, mode='mlp')
		print('Epoch {}: train loss: {}'.format(epoch, train_loss))
		valid_loss, ic = evaluate(model, test_dataloader, criterion, device, print_inteval=500, mode='mlp')
		print('Eval: total loss: {}, ic :{} '.format(valid_loss, ic))

		if ic > max_ic:
			max_ic = max(ic, max_ic)
			# save_model(epoch, model, optimzer, ic, prefix)
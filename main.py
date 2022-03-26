import os
import sys
import torch
import datetime
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import model
import dataset
from utils import *
from train import train, evaluate, save_model
from argparser import get_parser

if __name__ == "__main__":
	parser = get_parser()
	args = parser.parse_args()  # sys.argv[1:]
	print(args)
	set_seed(args.seed)
	
	dataset_ = getattr(dataset, args.dataset_type)
	train_dataset = dataset_(args.train_path, args.train_mask_path, args)
	test_dataset = dataset_(args.test_path, args.test_mask_path, args)

	train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=16, pin_memory=False)
	test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=False)


	model_ = getattr(model, args.model_type)
	cur_model = model_(args)

	criterion = nn.SmoothL1Loss(reduction='none')
	optimzer = torch.optim.Adam(cur_model.parameters(), lr=args.lr)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	cur_model.to(device)
	print(os.getpid())
	print(cur_model)

	max_ic = 0.08
	for epoch in range(args.epochs):
		train_loss = train(epoch, cur_model, train_dataloader, criterion, optimzer, \
			device, args.print_inteval, args.input_graph, args.mask_type)
		print('Epoch {}: train loss: {}'.format(epoch, train_loss))
		valid_loss, ic = evaluate(cur_model, test_dataloader, criterion, device, \
			args.print_inteval, args.input_graph, args.mask_type)
		print('Eval: total loss: {}, ic :{} '.format(valid_loss, ic))

		if ic > max_ic:
			max_ic = max(ic, max_ic)
			save_model(epoch, cur_model, optimzer, ic, args)
	
	# python  main.py @argfile.txt > ./logs/$(date +"%Y_%m_%d_%I_%M_%S").log
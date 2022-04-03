import os
import sys
import torch
import datetime
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import model
import dataset
from utils import set_seed, seed_worker
from train import train, evaluate, save_model
from argparser import get_parser

if __name__ == "__main__":
	parser = get_parser()
	args = parser.parse_args()  # sys.argv[1:]
	print(args)
	set_seed(args.seed)
	
	dataset_ = getattr(dataset, args.dataset_type)
	train_dataset = dataset_(args.train_path, args.train_mask_path, 'train', args)
	test_dataset = dataset_(args.test_path, args.test_mask_path, 'test', args)

	g = torch.Generator()  # for reproducibility
	g.manual_seed(0)  # multiprocessing: each worker has a pytorch seed = base_seed(generator) + worker_id
	train_dataloader = DataLoader(train_dataset, \
			batch_size=args.batch_size, shuffle=args.shuffle, num_workers=16, \
			pin_memory=True, worker_init_fn=seed_worker, generator=g)
	test_dataloader = DataLoader(test_dataset, \
			batch_size=args.batch_size, shuffle=False, num_workers=16, \
			pin_memory=True, worker_init_fn=seed_worker, generator=g)

	model_ = getattr(model, args.model_type)
	cur_model = model_(args)

	criterion = nn.SmoothL1Loss(reduction='none')
	optimzer = torch.optim.Adam(cur_model.parameters(), lr=args.lr)
	device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
	
	cur_model.to(device)
	print(os.getpid())
	print(cur_model)

	max_ic = 0.08
	for epoch in range(args.epochs):
		train_loss = train(epoch, cur_model, train_dataloader, criterion, optimzer, \
			device, args.print_inteval, args.input_graph, args.mask_type)
		print('Epoch {}: {}: train loss: {}'.format(epoch, datetime.datetime.now(), train_loss))
		valid_loss, mse, ic, sharpe5, irr5, ndcg5, pnl5 = evaluate(cur_model, test_dataloader, criterion, device, \
			args.print_inteval, args.input_graph, args.mask_type)
		print('Eval: {}: total loss: {}, mse:{}, ic :{}, sharpe5:{}, irr5:{}, ndcg5:{}, pnl5:{} '\
			.format(datetime.datetime.now(), valid_loss, mse, ic, sharpe5, irr5, ndcg5, pnl5))

		if ic > max_ic:
			max_ic = max(ic, max_ic)
			save_model(epoch, cur_model, optimzer, ic, args)

# CUBLAS_WORKSPACE_CONFIG=:16:8 or CUBLAS_WORKSPACE_CONFIG=:4096:2
# CUDA_VISIBLE_DEVICES=0 python  main.py @argfiles/argfile.txt > ./logs/$(date +"%Y_%m_%d_%I_%M_%S").log
# scp data/train_2305_1931_12.npy root@182.92.96.52:~/yzs/thesis/data/
# scp data/train_mask_2305_1931.npy root@182.92.96.52:~/yzs/thesis/data/
# scp data/test_126_1931_12.npy root@182.92.96.52:~/yzs/thesis/data/
# scp data/test_mask_126_1931.npy root@182.92.96.52:~/yzs/thesis/data/
# scp test_mask_126_1931.npy root@172.16.7.18:~/yzs/data/
# root: Test@Aliyun
# 8 gpus: 172.16.7.18
# zip -r icgraph_window_250_0.8.zip icgraph_window250_0.8/
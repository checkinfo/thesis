import os
import torch
import random
import datetime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr

from utils import *

def train(epoch, model, train_dataloader, criterion, optimizer, device, print_inteval, mode, mask_type='soft'):
	model.train()
	total_steps, running_loss = 0, 0.0

	# optimizer.zero_grad()
	for i, batch in enumerate(train_dataloader):
		optimizer.zero_grad()
		if mode == 'mlp':
			(x_train, y_train, mask) = batch
			y_pred = model(x=x_train.to(device))

		elif mode == 'gnn':
			(g, x_train, y_train, mask) = batch
			y_pred = model(g.to(device), x_train.to(device)).squeeze()

		if mask_type == 'strict':
			loss = criterion(y_pred.squeeze(), y_train.flatten().to(device))
		else:
			mse_loss = criterion(y_pred.squeeze(), y_train[:, : ,-1].squeeze().to(device))
			'''
			# icloss = ic_loss(y_pred.squeeze(), y_train.squeeze().to(device))
			icloss = 0
			for j in range(y_pred.size(0)):
				icloss += ic_loss(torch.masked_select(y_pred[j].squeeze(), mask[j].bool().to(device)), \
									torch.masked_select(y_train[j].to(device).squeeze(), mask[j].bool().to(device)))
			'''
			mse_loss = (mse_loss*mask.float().to(device)).sum()
			mse_loss = mse_loss / mask.sum()
			loss = mse_loss  # - ic_loss

		assert torch.isnan(loss).any() == False
		assert torch.isinf(loss).any() == False

		# loss /= 0
		loss.backward()

		# if (i+1)%7 == 0:
		optimizer.step()
		# optimzer.zero_grad()

		running_loss += loss.item()
		total_steps += 1

		if i%print_inteval == 0:
			grad_norm = print_grad_norm(model)

			ics = []
			if mask_type == 'soft':
				for j in range(y_pred.size(0)):  # 对每天计算ic
					pred_return = torch.masked_select(y_pred[j].squeeze().detach().cpu(), mask[j].bool())
					true_return = torch.masked_select(y_train[j, :, -1].detach().cpu(), mask[j].bool())

					assert pred_return.size() == true_return.size()
					ic, p_value = pearsonr(pred_return, true_return)
					ics.append(ic)
			else:
				ics = [0]
			print("train {}, step: {}, loss: {}, grad_norm: {}, ic: {}".format(epoch, i, loss.item(), grad_norm, np.mean(ics)))

	return running_loss / total_steps


def evaluate(model, valid_dataloader, criterion, device, print_inteval, mode, mask_type='soft'):
	model.eval()
	total_steps, running_loss = 0, 0.0
	ics = []

	y_dict = {i:[] for i in range(361)}
	pred_dict = {i:[] for i in range(361)}

	with torch.no_grad():
		for i, batch in enumerate(valid_dataloader):
			if mode == 'mlp':
				if mask_type == 'strict':
					(x_valid, y_valid, day_idx) = batch
				else:
					(x_valid, y_valid, mask) = batch
				if isinstance(x_valid, list):
					y_pred = model(x=[x_.to(device) for x_ in x_valid])
				else:
					y_pred = model(x_valid.to(device))  # [batch_sze, num_stocks, 1]
			elif mode == 'gnn':
				if mask_type=='strict':
					(g, x_valid, y_valid, mask, node_cnt) = batch
				else:
					(g, x_valid, y_valid, mask) = batch
				y_pred = model(g.to(device), x_valid.to(device)).squeeze()

			if mask_type == 'strict':
				loss = criterion(y_pred.squeeze(), y_valid.squeeze().to(device))
			else:
				mse_loss = criterion(y_pred.squeeze(), y_valid[:, : ,-1].squeeze().to(device))
				mse_loss = (mse_loss*mask.float().to(device)).sum()
				mse_loss = mse_loss / mask.sum()
				loss = mse_loss

			if i%print_inteval == 0:
				print('Eval step {}: eval loss: {}'.format(i, loss.item()))

			running_loss += loss.item()
			total_steps += 1

			if mask_type == 'soft':
				# if len(y_pred.size()) == 1 or len(mask.size())==1:
				# when batch size ==1, reshape to [1*num_stocks]
				for j in range(y_pred.size(0)):  # 对每天计算ic
					pred_return = torch.masked_select(y_pred[j].squeeze().detach().cpu(), mask[j].bool())
					true_return = torch.masked_select(y_valid[j, :, -1].detach().cpu(), mask[j].bool())

					assert pred_return.size() == true_return.size()
					ic, p_value = pearsonr(pred_return, true_return)
					ics.append(ic)
			else:
				for j in range(y_pred.size(0)):
					y_dict[int(day_idx[j].item())].append(y_valid[j].item())
					pred_dict[int(day_idx[j].item())].append(y_pred[j].item())
		
		if mask_type == 'strict':
			for k in y_dict.keys():
				if len(pred_dict[k]) != 0:
					assert len(pred_dict[k]) == len(y_dict[k])
					ic, pvalue = pearsonr(pred_dict[k], y_dict[k])
					ics.append(ic)
				else:
					print("????", k)
	return running_loss / total_steps, np.mean(ics)

			


def save_model(epoch, model, optimizer, ic, prefix, model_type='mlp'):
	state = {
		"epoch": epoch,
		"state_dict": model.state_dict(),
		"optimizer": optimizer.state_dict()
	}

	filepath = datetime.datetime.now().strftime(f'%d-%m-%y-%H-%M_{model_type}_{epoch}_ic={ic}.pth')
	if not os.path.exists(f"./checkpoint/{prefix}/"):
		os.makedirs(f"./checkpoint/{prefix}/")
	torch.save(state, os.path.join(f"./checkpoint/{prefix}/", filepath))


def load_model(model_path, model, optimizer):
	saved_model = torch.load(model_path, map_location='cpu')
	start_epoch = saved_model['epoch']
	model.load_state_dict(saved_model['state_dict'])
	optimizer.load_state_dict(saved_model['optimizer'])
	# optimizer_to(optimizer, device)
	return start_epoch, model, optimizer


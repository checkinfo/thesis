import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import ndcg_score

from utils import *

def train(epoch, model, train_dataloader, criterion, optimizer, device, print_inteval, input_graph=True, mask_type='soft'):
	model.train()
	total_steps, running_loss = 0, 0.0

	for i, batch in enumerate(train_dataloader):
		optimizer.zero_grad()
		if input_graph:
			(x_train, y_train, mask, g) = batch
			y_pred = model(x_train.to(device), g.to(device))  # [batch_size, num_stocks, 1]
		else:
			(x_train, y_train, mask, day_idx) = batch
			y_pred = model(x=x_train.to(device))  # [batch_size, num_stocks, 1]

		if mask_type == 'strict':
			loss = criterion(y_pred.squeeze(), y_train.flatten().to(device))
		else:
			mse_loss = criterion(y_pred.squeeze(), y_train[:, : ,-1].squeeze().to(device))
			mse_loss = (mse_loss*mask.float().to(device)).sum()
			mse_loss = mse_loss / mask.sum()
			loss = mse_loss  # - ic_loss

		assert torch.isnan(loss).any() == False
		assert torch.isinf(loss).any() == False

		loss.backward()
		optimizer.step()

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


def evaluate(model, valid_dataloader, criterion, device, print_inteval, input_graph=True, mask_type='soft', eval_days=126):
	model.eval()
	total_steps, running_loss = 0, 0.0

	performances = {'mse': 0, 'ic': [], 'irr5': [], 'ndcg_top5': []}
	
	if mask_type=='strict':
		y_dict = [[] for i in range(eval_days)]
		pred_dict = [[] for i in range(eval_days)]  # {i:[] for i in range(eval_days)}

	with torch.no_grad():
		for i, batch in enumerate(valid_dataloader):
			if input_graph:
				(x_valid, y_valid, mask, g) = batch
				y_pred = model(x_valid.to(device), g.to(device))  # [batch_size, num_stocks, 1]
			else:
				(x_valid, y_valid, mask, day_idx) = batch
				y_pred = model(x_valid.to(device))  # [batch_size, num_stocks, 1]

			if mask_type == 'strict':
				loss = criterion(y_pred.squeeze(), y_valid[:, :, -1].squeeze().to(device))
			else:
				mse_loss = criterion(y_pred.squeeze(), y_valid[:, : ,-1].squeeze().to(device))
				mse_loss = (mse_loss*mask.float().to(device)).sum()
				mse_loss = mse_loss / mask.sum()
				loss = mse_loss

			if i%print_inteval == 0:
				print('Eval step {}: eval loss: {}'.format(i, loss.item()))

			running_loss += loss.item()
			total_steps += 1

			if mask_type=='soft':
				calc_metrics(performances, y_pred.squeeze(-1).detach().cpu(), y_valid[:, :, -1], mask, mask_type)
			elif mask_type=='strict':
				for j in range(y_pred.size(0)):
					y_dict[int(day_idx[j].item())].append(y_valid[j, : ,-1].item())
					pred_dict[int(day_idx[j].item())].append(y_pred[j].item())
		
		if mask_type == 'strict':
			calc_metrics(performances, pred_dict, y_dict, mask, mask_type)
		
	irr5 = sum(performances['irr5']) - 1 # 收益率，不算复利，每天都投入1……
	sharpe5 = (np.mean(performances['irr5'])/np.std(performances['irr5']))*15.87 #To annualize,	每日收益率的均值除以波动
	ic = np.mean(performances['ic'])
	ndcg5 = np.mean(performances['ndcg_top5'])
	mse = performances['mse'] / len(performances['ic'])
	# pnl = avg profit / avg loss, 收益率列表里面，正的求个平均，负的求个平均
	# appt = prob of profit * avg profit - prob of loss*avg loss
	return running_loss / total_steps, mse, ic, sharpe5, irr5, ndcg5

			
def calc_metrics(performances, prediction, ground_truth, mask, mask_type='soft'):
	# [batch_size, num_stocks]
	assert ground_truth.size() == prediction.size() == mask.size(), 'shape mis-match'

	performances['mse'] += np.linalg.norm((prediction - ground_truth) * mask)**2 \
		/ torch.sum(mask).item()

	for j in range(prediction.size(0)):  # days
		# ic
		pred_return = torch.masked_select(prediction[j], mask[j].bool()) \
			if mask_type=='soft' else prediction[j]
		true_return = torch.masked_select(ground_truth[j], mask[j].bool()) \
			if mask_type=='soft' else ground_truth[j]
		assert pred_return.size() == true_return.size()

		ic, p_value = pearsonr(pred_return, true_return)
		performances['ic'].append(ic)

		# ndcg top5: ground truth top5
		rank_gt = torch.argsort(ground_truth[j]) # sort all stocks on day j
		gt_top5 = set()  # set of top5 stocks

		for k in range(1, prediction.shape[1] + 1):  # stocks
			cur_rank = rank_gt[-1 * k]
			if mask_type=='soft' and mask[j][cur_rank] < 0.5:
				continue
			if len(gt_top5) < 5:
				gt_top5.add(cur_rank)
			else:
				break
		
		# ndcg top5: prediction top5
		rank_pre = torch.argsort(prediction[j])
		pre_top5 = set()

		for k in range(1, prediction.shape[1] + 1):
			cur_rank = rank_pre[-1 * k]
			if mask_type=='soft' and mask[j][cur_rank] < 0.5:
				continue
			if len(pre_top5) < 5:
				pre_top5.add(cur_rank)
			else:
				break

		performances['ndcg_top5'].append(ndcg_score(np.array(list(gt_top5)).reshape(1,-1), np.array(list(pre_top5)).reshape(1,-1)))

        # back testing each day on top 5
		real_ret_rat_top5 = 0
		for pre in pre_top5:
			real_ret_rat_top5 += ground_truth[j][pre]
		real_ret_rat_top5 /= 5
		performances['irr5'].append(real_ret_rat_top5)
		'''
		real_ret_rat_top5_gt = 0
		for pre in gt_top5:
			real_ret_rat_top5_gt += ground_truth[pre][i]
		real_ret_rat_top5_gt /= 5
		'''

def save_model(epoch, model, optimizer, ic, args):
	state = {
		"epoch": epoch,
		"state_dict": model.state_dict(),
		"optimizer": optimizer.state_dict()
	}

	time_str = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
	filepath = f'{args.model_type}_{args.dataset_type}_{epoch}_ic={ic}.pth'
	if not os.path.exists(f"./checkpoint/{time_str}/"):
		os.makedirs(f"./checkpoint/{time_str}/")
	torch.save(state, os.path.join(f"./checkpoint/{time_str}/", filepath))


def load_model(model_path, model, optimizer):
	saved_model = torch.load(model_path, map_location='cpu')
	start_epoch = saved_model['epoch']
	model.load_state_dict(saved_model['state_dict'])
	optimizer.load_state_dict(saved_model['optimizer'])
	# optimizer_to(optimizer, device)
	return start_epoch, model, optimizer


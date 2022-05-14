import os
import sys
import json
import copy
import torch
import argparse
import datetime
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import model
import dataset
from utils import *
from scipy.stats import pearsonr
from sklearn.metrics import ndcg_score
from utils import set_seed, seed_worker
from train import calc_metrics, load_model
from argparser import get_parser


def calc_metrics(performances, prediction, ground_truth, mask, new_mask, detail=False, stock_id_name_map=None, id_stock_map=None, top_stocks=5.0, mask_type='soft'):
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
		#print(pred_return.size())
		#print("pred: ", pred_return)
		#print("gt:   ", true_return)
		if len(pred_return) >= 5:
			ic, p_value = pearsonr(pred_return, true_return)
		else:
			return
		performances['ic'].append(ic)

		# ndcg top5: ground truth top5
		rank_gt = torch.argsort(ground_truth[j]) # sort all stocks on day j
		gt_top5 = set()  # set of top5 stocks

		for k in range(1, prediction.shape[1] + 1):  # stocks
			cur_rank = rank_gt[-1 * k]  # true idx for kth stock
			if mask_type=='soft' and new_mask[j][cur_rank] < 0.5:
				continue
			if len(gt_top5) < int(top_stocks):  # 5 by default
				gt_top5.add(cur_rank)
			else:
				break
		#print("gt top5: ", gt_top5, [ground_truth[j][x] for x in gt_top5])

		# ndcg top5: prediction top5
		rank_pre = torch.argsort(prediction[j])
		pre_top5 = set()

		for k in range(1, prediction.shape[1] + 1):
			cur_rank = rank_pre[-1 * k]
			if mask_type=='soft' and new_mask[j][cur_rank] < 0.5:
				continue
			if len(pre_top5) < int(top_stocks):   # 5 by default
				pre_top5.add(cur_rank)
			else:
				break
		#print("pred top5: ", pre_top5, [ground_truth[j][x] for x in pre_top5])
		# print("gt:", ground_truth[j])
		if top_stocks>1:
			performances['ndcg_top5'].append(ndcg_score(np.array(list(gt_top5)).reshape(1,-1), np.array(list(pre_top5)).reshape(1,-1)))
		else:
			performances['ndcg_top5'].append(1.0)

        # back testing each day on top 5
		real_ret_rat_top5 = 0
		if detail:
			print("pred top stocks:", end="   ")
		for pre in pre_top5:
			if detail:
				stock_code = id_stock_map[pre.item()]
				print(stock_code, stock_id_name_map[stock_code], "%.4f"%(ground_truth[j][pre].item()), end=" ")
			real_ret_rat_top5 += ground_truth[j][pre] # percentage change
		if detail:
			print(" ")

		real_ret_rat_top5 /= int(top_stocks)   # 5 by default
		performances['irr5'].append(real_ret_rat_top5)
		
		if detail:
			print("gt top stocks:  ", end="   ")
		real_ret_rat_top5_gt = 0
		for pre in gt_top5:
			if detail:
				stock_code = id_stock_map[pre.item()]
				print(stock_code, stock_id_name_map[stock_code], "%.4f"%ground_truth[j][pre].item(), end=" ")
			real_ret_rat_top5_gt += ground_truth[j][pre]
		if detail:
			print(" ")
		real_ret_rat_top5_gt /= int(top_stocks) # 5 by default
		performances['irr5_real'].append(real_ret_rat_top5_gt)
		

def evaluate(model, valid_dataloader, criterion, device, print_inteval, num_days=1, input_graph=True, mask_type='soft', eval_days=126, top_stocks=5):
	model.eval()
	total_steps, running_loss = 0, 0.0

	stock_path = "./data/stock_codes_1931.txt"
	stock_id_map = get_stock_id_mapping(stock_path)
	id_stock_map = {v: k for k, v in stock_id_map.items()}
	listed_path = "./data/stock_listintest_25.txt"
	listed_id_map = get_stock_id_mapping(listed_path)

	listed = []  # id_list
	for s in listed_id_map.keys():  # s = "000001.SH"
		listed.append(stock_id_map[s])
	listed = torch.LongTensor(listed)
	# (date_id, stock_id)  # date_id in [0, 118]
	# listed_map = {272:1272, 110:1343, 102:1369, 92:1416, 107:1483, 36:1506, 21:1519, 3:1577, 6:1580, 12:1590, 91:1654, 81:1669, 17:1686, 72:1724, 13:1727, 18:1751, 11:1767, 34:1801, 62:1812, 66:1829, 42:1891, 10:1916, 32:1924, 112:1926}
	# listed_map = {3:1319, 279:1272, 117:1343, 109:1369, 99:1416, 114:1483, 43:1506, 28:1519, 10:1577, 13:1580, 19:1590, 98:1654, 88:1669, 24:1686, 79:1724, 20:1727, 25:1751, 18:1767, 41:1801, 69:1812, 73:1829, 49:1891, 17:1916, 39:1924, 119:1926}
	listed_map = {4:1319, 280:1272, 118:1343, 110:1369, 100:1416, 115:1483, 44:1506, 29:1519, 11:1577, 14:1580, 20:1590, 99:1654, 89:1669, 25:1686, 80:1724, 21:1727, 26:1751, 19:1767, 42:1801, 70:1812, 74:1829, 50:1891, 18:1916, 40:1924, 120:1926}
	listed_map = {k-num_days:listed_map[k] for k in listed_map}
	# listed_map = {273:1272, 111:1343, 103:1369, 93:1416, 108:1483, 37:1506, 22:1519, 4:1577, 7:1580, 13:1590, 92:1654, 82:1669, 18:1686, 73:1724, 14:1727, 19:1751, 12:1767, 35:1801, 63:1812, 67:1829, 43:1891, 11:1916, 33:1924, 113:1926}
	# tensor([1272, 1319（不需要）, 1343, 1369, 1416, 1483, 1506, 1519, 1577, 1580, 1590, 1654, 1669, 1686, 1724, 1727, 1751, 1767, 1801, 1812, 1829, 1891, 1916, 1924, 1926])
	# listed_id_stock_map = {v: k for k, v in listed_id_map.items()}
	with open("./data/stock_id_name_map.json", "r") as f:
		stock_id_name_map = json.load(f)
	
	performances = {'mse': 0, 'ic': [], 'irr5': [], 'irr5_real':[], 'ndcg_top5': []}

	with torch.no_grad():
		for i, batch in enumerate(valid_dataloader):
			# print("i=", i)
			if input_graph:
				(x_valid, y_valid, mask, g, edgenum, side_info) = batch
				y_pred = model(x_valid.to(device), g.to(device), edgenum.to(device), side_info.to(device))  # [batch_size, num_stocks, 1]
			else:
				(x_valid, y_valid, mask, side_info) = batch  # (x, y, mask, day_idx) for mask_type==strict
				y_pred = model(x=x_valid.to(device), side_info=side_info.to(device))  # [batch_size, num_stocks, 1]
				if mask_type=='strict': day_idx=side_info
			
			# predict percentage change
			mse_loss = criterion(y_pred.squeeze(), y_valid[:, : ,-1].squeeze().to(device))
			mse_loss = (mse_loss*mask.float().to(device)).sum()
			mse_loss = mse_loss / mask.sum()
			loss = mse_loss

			if i%print_inteval == 0:
				print('Eval step {}: eval loss: {}'.format(i, loss.item()))

			running_loss += loss.item()
			total_steps += 1

			# print(y_pred.size(), y_valid.size(), mask.size())
            # torch.Size([1, 1931, 1]) torch.Size([1, 1931, 3]) torch.Size([1, 1931])
			#y_pred = y_pred[:, listed, :]  # [1,25,1]
			#y_valid = y_valid[:, listed, :]
			#mask = mask[:, listed]
			# print(y_pred.size(), y_valid.size(), mask.size())
			new_mask = copy.deepcopy(mask)
			if i in listed_map.keys():
				new_mask[:, listed_map[i]] = 0  # 上市第一日
			calc_metrics(performances, y_pred.squeeze(-1).detach().cpu(), y_valid[:, :, -1], mask, new_mask, detail=False,\
                stock_id_name_map=stock_id_name_map, id_stock_map=id_stock_map, top_stocks=top_stocks, mask_type=mask_type)
			
	#print("irr5:      ", performances['irr5'])
	#print("irr5_real: ", performances['irr5_real'])
	#print("ic:        ", performances['ic'])
	#print("mse:       ", performances['mse'])
	#print("ndcg5:     ", performances['ndcg_top5'])
	irr5 = sum(performances['irr5']) # 收益率，不算复利，每天都投入1。每天的收益之和……
	irr5_real = sum(performances['irr5_real'])
	sharpe5 = (np.mean(performances['irr5'])/np.std(performances['irr5']))*15.87  # To annualize,	每日收益率的均值除以波动
	sharpe5_real = (np.mean(performances['irr5_real'])/np.std(performances['irr5_real']))*15.87  # To annualize,	每日收益率的均值除以波动
	
	ic = np.mean(performances['ic'])
	ndcg5 = np.mean(performances['ndcg_top5'])
	mse = performances['mse'] / len(performances['ic'])
	pnl5 = np.mean([x for x in performances['irr5'] if x >= 0]) / (-np.mean([x for x in performances['irr5'] if x < 0]))
	# pnl5_real = np.mean([x for x in performances['irr5_real'] if x >= 0]) / (-np.mean([x for x in performances['irr5_real'] if x < 0]))
	
    # pnl = avg profit / avg loss, 收益率列表里面，正的求个平均，负的求个平均
	# appt = prob of profit * avg profit - prob of loss*avg loss
	return running_loss / total_steps, mse, ic, sharpe5, sharpe5_real, irr5, irr5_real, ndcg5, pnl5


if __name__ == "__main__":
    # model_path = "checkpoint/2022-05-04-09-36-08/BiGLSTM_RGCNSeqTimeDataset_33_ic=0.20431766768737816.pth"
	# @argfiles/ic_graph_2rels_gate_4e-4/biglstm_rgcn_attn_2rels_750_0.6.txt

    # model_path = "checkpoint/2022-04-27-17-15-59/BiGLSTM_AdjSeqTimeDataset_32_ic=0.20315421297249198.pth"
	# @argfiles/ablation/biglstm_rgcn_attn_1rel_adjseq.txt

    # model_path = "checkpoint/2022-05-09-14-43-49/BiGLSTM_AdjSeqTimeDataset_33_ic=0.2011257159074145.pth"
	# @argfiles/ablation/biglstm_rgcn_attn_1rel_nomask_aseq.txt

    # model_path = "checkpoint/2022-05-09-16-15-13/BiGLSTM_RGCNSeqTimeDataset_34_ic=0.20237389226334243.pth"
	# argfiles/ablation/biglstm_rgcn_attn_2rels_nomask_750_0.6.txt

    # model_path = "checkpoint/2022-05-08-20-39-01/BiGLSTM_RGCNSeqTimeDataset_17_ic=0.2005552900039013.pth"
	# argfiles/ablation/biglstm_rgcn_2rels_750_0.6.txt   (no attn)
    
	# model_path = "checkpoint/2022-04-28-09-16-27/BiGLSTM_RGCNSeqTimeDataset_33_ic=0.20363477823228673.pth"
	# argfiles/ablation/biglstm_rgcn_2rels_750_0.6.txt   (no attn no gate)
    
	# model_path = "checkpoint/2022-04-28-19-53-38/BiGLSTM_RGCNSeqTimeDataset_38_ic=0.19660703938898388.pth"
	# argfiles/ablation/biglstm_rgcn_2rels_750_0.6.txt   (no gate)
    
    model_path = "checkpoint/2022-05-13-01-19-27/MlpModel_TimeDataset_38_ic=0.1240933941487956.pth"
	# argfiles/ablation/biglstm_rgcn_attn_1rel_adjseq.txt
    
    parser = get_parser()
    # 'argfiles/ic_graph_2rels_gate_4e-4/biglstm_rgcn_attn_2rels_750_0.6.txt'
    args = parser.parse_args()  # sys.argv[1:]
    print(args)
    set_seed(args.seed)

    dataset_ = getattr(dataset, args.dataset_type)
    test_dataset = dataset_(args.test_path, args.test_mask_path, 'test', args)

    g = torch.Generator()  # for reproducibility
    g.manual_seed(0)  # multiprocessing: each worker has a pytorch seed = base_seed(generator) + worker_id
    test_dataloader = DataLoader(test_dataset, \
            batch_size=args.batch_size, shuffle=False, pin_memory=True)

    model_ = getattr(model, args.model_type)
    cur_model = model_(args)

    criterion = nn.SmoothL1Loss(reduction='none')
    optimizer = torch.optim.Adam(cur_model.parameters(), lr=args.lr)
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

    start_epoch, cur_model, optimzer = load_model(model_path,cur_model, optimizer)
    cur_model.to(device)

    print(os.getpid())
    print(cur_model)
    print("top stocks:", args.top_stocks)

    valid_loss, mse, ic, sharpe5, sharpe5_real, irr5, irr5_real, ndcg5, pnl5 = evaluate(cur_model, test_dataloader, criterion, device, \
        args.print_inteval, args.num_days, args.input_graph, args.mask_type, top_stocks=args.top_stocks)
    print('Eval: {}: total loss: {}, mse:{}, ic :{}, sharpe5:{}, sharpe5_real:{}, irr5:{}, irr5_real:{}, ndcg5:{}, pnl5:{} '\
        .format(datetime.now(), valid_loss, mse, ic, sharpe5, sharpe5_real, irr5, irr5_real, ndcg5, pnl5))

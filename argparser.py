import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='args description', fromfile_prefix_chars ='@') 
    parser.add_argument('--train_path', help='path of prices and other features, .npy file')
    parser.add_argument('--test_path', help='path of prices and other features, .npy file')
    parser.add_argument('--train_mask_path', help='path of stock mask, .npy file')
    parser.add_argument('--test_mask_path', help='path of stock mask, .npy file')
    parser.add_argument('--label_cnt', type=int, default=3, \
        help='labels start from idx zero, end at label_cnt, i.e. pre_close+change+pct_chg')
    parser.add_argument('--batch_size', type=int, help='batch size: how many days in a batch')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--adj_path', help='path of graph in adj format, .npy file')
    parser.add_argument('--model_type', help='desired class name')
    parser.add_argument('--dataset_type', help='desired class name')
    parser.add_argument('--seed', type=int, default=10086, help='random seed')
    parser.add_argument('--num_days', type=int, default=1, help='number of historical prices used')
    parser.add_argument('--epochs', type=int, default=20, help='maximum epochs')
    parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dim')
    parser.add_argument('--input_dim', type=int, default=9, help='model input dim')
    parser.add_argument('--dout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--lstm_layers', type=int, default=3, help='num of lstm_layers')
    parser.add_argument('--num_heads', type=int, default=1, help='num of attenton heads')
    parser.add_argument('--gnn_layers', type=int, default=2, help='num of gnn layers')
    parser.add_argument('--print_inteval', type=int, default=500, help='log printing frequency')
    parser.add_argument('--relation_num', type=int, default=1, help='num of edge type')
    parser.add_argument('--mask_type', default='soft', help='soft means dont \
        remove masked data in batch, strict means remove masked data before batching')
    
    parser.add_argument('--shuffle', dest='shuffle', action='store_true', default=False, help='whether to shuffle in dataloader')
    parser.add_argument('--input_graph', dest='input_graph', action='store_true', default=False, help='whether batched data contains graph')
    parser.add_argument('--use_adj', dest='use_adj', action='store_true', default=False, help='whether to load adj in dataloader')
    parser.add_argument('--mask_adj', dest='mask_adj', action='store_true', default=False, help='whether to mask each days adj in dataset')
    
    return parser
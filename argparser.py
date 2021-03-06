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
    parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--adj_path', help='path of graph in adj format, .npy file')
    parser.add_argument('--sparse_adj_path', help='path of edge indexs, folder of .npz files')
    parser.add_argument('--ann_path', help='path of announcements, .npz file')
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
    parser.add_argument('--stock_num', type=int, default=1931, help='num of stocks')
    parser.add_argument('--ann_embed_num', type=int, default=89, help='num of ann type embeddings')
    parser.add_argument('--ann_embed_dim', type=int, default=128, help='ann embedding dimension')
    parser.add_argument('--top_stocks', type=int, default=5, help='portfolio size, top x stocks to invest in each day')
    parser.add_argument('--glstm_layers', type=int, default=1, help='num of glstm layers')
    parser.add_argument('--mask_type', default='soft', help='soft means dont \
        remove masked data in batch, strict means remove masked data before batching')
    
    parser.add_argument('--shuffle', dest='shuffle', action='store_true', default=False, help='whether to shuffle in dataloader')
    parser.add_argument('--input_graph', dest='input_graph', action='store_true', default=False, help='whether batched data contains graph')
    parser.add_argument('--use_adj', dest='use_adj', action='store_true', default=False, help='whether to load adj in dataloader')
    parser.add_argument('--normalize_adj', dest='normalize_adj', action='store_true', default=False, help='whether to normalize adj first')
    parser.add_argument('--mask_adj', dest='mask_adj', action='store_true', default=False, help='whether to mask each days adj in dataset')
    parser.add_argument('--side_info', dest='side_info', action='store_true', default=False, help='whether to use side info type embed')
    
    parser.add_argument('--graph_attn', dest='graph_attn', action='store_true', default=False, help='whether to use graph attention')
    parser.add_argument('--rank_loss', dest='rank_loss', action='store_true', default=False, help='whether to add rank loss to mse loss')
   
    # rsr
    parser.add_argument('--inner_prod', dest='inner_prod', action='store_true', default=False, help='whether to implicit version of Relational Stock Ranking')
    parser.add_argument('--rsr_data_path', help='path of rsr(NASDAQ+NYSE) raw stock price')
    parser.add_argument('--market', help='market name, NASDAQ or NYSE')
    
    return parser
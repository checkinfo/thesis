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
    return parser
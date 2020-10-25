import argparse


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='data', help='root of data files')
    parser.add_argument('--train', default='train.txt')
    parser.add_argument('--test', default='test.txt')
    parser.add_argument('--rel', default='relation2id.txt')
    parser.add_argument('--vec', default='vec.txt')
    parser.add_argument('--ckpt', default='pretrain/ckpt/model.pth.tar')
    parser.add_argument('--processed_data_dir', default='_processed_data')
    parser.add_argument('--batch_size', default=160, type=int)
    parser.add_argument('--max_length', default=120, type=int)
    parser.add_argument('--max_pos_length', default=100, type=int)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--lr', default=0.5, type=float)
    parser.add_argument('--val_iter', default=1, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--early_stop', default=10, type=int)
    return parser.parse_args()
import argparse
import math
import os
def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency') 
    parser.add_argument('--save_freq', type=int, default=5, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--save_data_dir', type=str, default=None)

    opt = parser.parse_args()

    if not os.path.isdir(opt.save_data_dir):
        # os.makedirs(opt.save_data_dir) 
        pass

    return opt
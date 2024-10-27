import torch
import numpy as np
import pandas as pd
"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import torch
import sys

from termcolor import colored
from utils.config_fish import create_config
from utils.common_config_fish import get_train_transformations, get_val_transformations, \
    get_train_dataset, get_train_dataloader, \
    get_val_dataset, get_val_dataloader, \
    get_optimizer, get_model, get_criterion, \
    adjust_learning_rate
from utils.evaluate_utils_fish import get_predictions, scan_evaluate, cc_evaluate, hungarian_evaluate
from utils.train_utils_fish import scan_train
from utils.utils_fish import Logger, get_knn_indices
import numpy as np
import time
import pandas as pd
from tensorboardX import SummaryWriter


FLAGS = argparse.ArgumentParser(description='SCAN Loss')
FLAGS.add_argument('--config_env', help='Location of path config file', default='configs/env.yml')
FLAGS.add_argument('--config_exp', help='Location of experiments config file', default='configs/scan/scan_fish.yml')
FLAGS.add_argument('--version', type=str, default='scan_debug_topk', help='Record the version of this times')
FLAGS.add_argument('--gpu', type=str, default='0,1,2,3')
FLAGS.add_argument('--t', type=float, default=8.0)
FLAGS.add_argument('--u', type=float, default=1)
FLAGS.add_argument('--topk', type=int, default=15)
FLAGS.add_argument('--wo1', type=int, default=1)
FLAGS.add_argument('--wo2', type=int, default=1)
FLAGS.add_argument('--wo3', type=int, default=1)
args = FLAGS.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def main():
    p = create_config(args.config_env, args.config_exp, args.topk)
    # print(colored(p, 'red'))
    # 画loss的图
    writer = SummaryWriter('runs/{}'.format(p['dataset_name']))

    # log
    logfile_dir = os.path.join(os.getcwd(), 'logs/')
    logfile_name = logfile_dir + args.version + '.log'
    # sys.stdout = Logger(filename=logfile_name, stream=sys.stdout)
    localtime = time.asctime(time.localtime(time.time()))
    print('\n--------------------------------------------------------------\n')
    # print("The current time:", localtime)
    print('dataset_name: ', p['dataset_name'])

    # CUDNN
    torch.backends.cudnn.benchmark = True


    # Data
    # print(colored('Get dataset and dataloaders', 'blue'))
    train_transformations = get_train_transformations(p)
    val_transformations = get_val_transformations(p)
    # train_dataset = get_train_dataset(p, train_transformations, to_neighbors_dataset=True)
    train_dataset = get_train_dataset(p, val_transformations, to_neighbors_dataset=True)        #train_dataset使用原始未经过增强的数据进行kmeans
    val_dataset = get_val_dataset(p, val_transformations, to_neighbors_dataset=True)

if __name__ == '__main__':
    main()
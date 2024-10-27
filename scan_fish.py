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
    print(colored(p, 'red'))
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
    print(colored('Get dataset and dataloaders', 'blue'))
    train_transformations = get_train_transformations(p)
    val_transformations = get_val_transformations(p)
    train_dataset = get_train_dataset(p, train_transformations, to_neighbors_dataset=True)
    # train_dataset = get_train_dataset(p, val_transformations, to_neighbors_dataset=True)        #train_dataset使用原始未经过增强的数据进行kmeans
    val_dataset = get_val_dataset(p, val_transformations, to_neighbors_dataset=True)
    train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Train transforms:', train_transformations)
    print('Validation transforms:', val_transformations)
    print('Train samples %d - Val samples %d' % (len(train_dataset), len(val_dataset)))

    # get_knn_acc
    # from utils.common_config_fish import get_knn_acc
    # for i in range(1, 15):
    #     get_knn_acc(p, train_transformations, i)

    # Model
    print(colored('Get model', 'blue'))
    model = get_model(p)
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    print(model)

    # get knn indices
    # base_dataset = get_train_dataset(p, val_transformations, split='train', to_neighbors_dataset=True)
    # base_dataloader = get_val_dataloader(p, base_dataset)
    # state = torch.load(p['scan_model'], map_location='cpu')
    # print(p['scan_model'])
    # model.module.load_state_dict(state['model'], strict=True)
    # indices, acc = get_knn_indices(base_dataloader,topk=p['num_neighbors'])
    # np.save(p['after_top{}_neighbors_train_path'.format(p['num_neighbors'])], indices)
    # print('acc:{}'.format(acc))

    # Optimizer
    print(p['update_cluster_head_only'])
    print(colored('Get optimizer', 'blue'))
    optimizer = get_optimizer(p, model, p['update_cluster_head_only'])
    print(optimizer)

    # Warning
    if p['update_cluster_head_only']:
        print(colored('WARNING: SCAN will only update the cluster head', 'red'))

    # Loss function
    print(colored('Get loss', 'blue'))
    from losses.losses import SCANLoss
    criterion = SCANLoss(args.wo1, args.wo2, args.wo3, args.t, args.u, p['num_classes'], **p['criterion_kwargs'])
    criterion = criterion.cuda()
    print(criterion)

    # by reading the acc, check whether the model parameters are loaded correctly
    # from utils.utils import get_features_eval
    # features, targets = get_features_eval(train_dataloader, model, forward_pass='backbone')

    # from utils.evaluate_utils import kmeans
    # kmeans(features, targets)
    # print("-----------------targets")
    # print(targets)

    # Checkpoint
    if False and os.path.exists(p['scan_checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['scan_checkpoint']), 'blue'))
        checkpoint = torch.load(p['scan_checkpoint'], map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        best_loss_head = checkpoint['best_loss_head']

    else:
        print(colored('No checkpoint file at {}'.format(p['scan_checkpoint']), 'blue'))
        start_epoch = 0
        best_loss = 1e4
        best_loss_head = 0
        best_acc = 0
        best_ri = 0
        best_train_ri = 0
        best_epoch = -1
        best_clustering_stats = None

    # Main loop
    print(colored('Starting main loop', 'blue'))

    # 记录运行的开始时间
    torch.cuda.synchronize()
    start = time.time()

    results = [[] for i in range(7)]
    epoch_num = p['epochs']
    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d==========================================================================' % (
            epoch + 1, p['epochs']), 'yellow'))
        print(colored('-' * 15, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Train
        print('Train ...')

        final_loss = scan_train(train_dataloader, model, criterion, optimizer, epoch, epoch_num,
                                p['update_cluster_head_only'], writer)

        # Evaluate
        print('Make prediction on validation set ...')
        predictions = get_predictions(p, val_dataloader, model)
        predictions_trian = get_predictions(p, train_dataloader, model)

        print('Evaluate with hungarian matching algorithm ...')
        # 选择最小的MLP
        lowest_loss_head = 0
        clustering_stats = hungarian_evaluate(lowest_loss_head, predictions, compute_confusion_matrix=False)
        clustering_stats_train = hungarian_evaluate(lowest_loss_head, predictions_trian, compute_confusion_matrix=False)
        print(clustering_stats)

        for i in range(3):
            results[i].append(final_loss[i])
        results[3].append(round(clustering_stats['ACC'], 3))
        results[4].append(round(clustering_stats['ARI'], 3))
        results[5].append(round(clustering_stats['NMI'], 3))
        results[6].append(round(clustering_stats['RI'], 3))

        writer.add_scalar('train_RI', clustering_stats_train['RI'], global_step=epoch)
        writer.add_scalar('val_RI', clustering_stats['RI'], global_step=epoch)

        if epoch > 1:
            if clustering_stats_train['RI'] > best_train_ri:
                best_epoch_train = epoch
                best_train_ri = clustering_stats_train['RI']
                # best_clustering_stats_train = clustering_stats_train
            if clustering_stats['RI'] > best_ri:
                best_epoch = epoch
                best_ri = clustering_stats['RI']
                best_clustering_stats = clustering_stats
                if os.path.exists(p['scan_best_clustering_results']):
                    results_old = torch.load(p['scan_best_clustering_results'], map_location='cpu')
                    if best_ri > results_old['RI']:
                        print('saving the best scan model............................')
                        torch.save(best_clustering_stats, p['scan_best_clustering_results'],
                                   _use_new_zipfile_serialization=False)
                        torch.save({'model': model.module.state_dict(), 'head': best_loss_head}, p['scan_model'],
                                   _use_new_zipfile_serialization=False)
                else:
                    print('------create------scan_best_clustering_results')
                    print(p['scan_best_clustering_results'])
                    torch.save(best_clustering_stats, p['scan_best_clustering_results'],
                               _use_new_zipfile_serialization=False)
                    torch.save({'model': model.module.state_dict(), 'head': best_loss_head}, p['scan_model'],
                               _use_new_zipfile_serialization=False)

            print('=============================================================================================')
            print('best_test_ri: {}'.format(best_clustering_stats['RI']))
            print('best_epoch: {}'.format(best_epoch))
            # print('best_epoch_train:{}, best_train_ri: {}'.format(best_epoch_train, best_train_ri))
            print('=============================================================================================')

        if args.wo1 == 0:
            if clustering_stats['ACC'] > best_acc:
                best_acc = clustering_stats['ACC']
                best_clustering_stats = clustering_stats
                results_old = torch.load(p['scan_best_clustering_results_wo1'], map_location='cpu')
                if best_acc > results_old['ACC']:
                    torch.save(best_clustering_stats, p['scan_best_clustering_results_wo1'],
                               _use_new_zipfile_serialization=False)
                    torch.save({'model': model.module.state_dict(), 'head': best_loss_head}, p['scan_model_wo1'],
                               _use_new_zipfile_serialization=False)
        elif args.wo2 == 0:
            if clustering_stats['ACC'] > best_acc:
                best_acc = clustering_stats['ACC']
                best_clustering_stats = clustering_stats
                results_old = torch.load(p['scan_best_clustering_results_wo2'], map_location='cpu')
                if best_acc > results_old['ACC']:
                    torch.save(best_clustering_stats, p['scan_best_clustering_results_wo2'],
                               _use_new_zipfile_serialization=False)
                    torch.save({'model': model.module.state_dict(), 'head': best_loss_head}, p['scan_model_wo2'],
                               _use_new_zipfile_serialization=False)
        elif args.wo3 == 0:
            if clustering_stats['ACC'] > best_acc:
                best_acc = clustering_stats['ACC']
                best_clustering_stats = clustering_stats
                results_old = torch.load(p['scan_best_clustering_results_wo3'], map_location='cpu')
                if best_acc > results_old['ACC']:
                    torch.save(best_clustering_stats, p['scan_best_clustering_results_wo3'],
                               _use_new_zipfile_serialization=False)
                    torch.save({'model': model.module.state_dict(), 'head': best_loss_head}, p['scan_model_wo3'],
                               _use_new_zipfile_serialization=False)

        # Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                    'epoch': epoch + 1, 'best_loss': best_loss, 'best_loss_head': best_loss_head},
                   p['scan_checkpoint'])



    # 记录运行结束的时间
    torch.cuda.synchronize()
    end = time.time()
    t = end - start

    print('=============================================================================================')
    print('parameter:', '\tt:', args.t, '\tu:', args.u)
    print('best_test_ri: {}'.format(best_clustering_stats['RI']))
    print('best_epoch_test: {}'.format(best_epoch))
    print('best_epoch_train:{}, best_train_ri: {}'.format(best_epoch_train, best_train_ri))
    print('=============================================================================================')

    # write the results to the excel file

    para = p['setup'] + p['dataset_name'] + ' u=' + str(args.u) + ' t=' + str(args.t)  # the parameter name and value
    print(para)
    print(args.wo1, args.wo2, args.wo3)
    index = np.arange(1, epoch_num + 1)
    res = {para: index, "total_loss": results[0], "consistency_loss": results[1], "contrastive_loss": results[2],
           "RI": results[6], "ARI": results[4], "NMI": results[5]}
    from pandas.core.frame import DataFrame
    res = DataFrame(res)

    file_name = './path/to/results2/debug_results_scan{}.csv'.format(p['dataset_name'])
    res.to_csv(file_name, index=False, mode='a+', encoding='utf-8')

    from openpyxl import load_workbook
    file_name = '../results/results_para_2.xlsx'
    res_old = pd.read_excel(file_name, sheet_name='scan')
    with pd.ExcelWriter(file_name, mode='a', engine='openpyxl') as writer:
        book = load_workbook(file_name)
        writer.book = book
        writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
        row = res_old.shape[0]
        res.to_excel(writer, sheet_name='scan', startrow=row+2, index=False)


if __name__ == "__main__":
    main()
    #

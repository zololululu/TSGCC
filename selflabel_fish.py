"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import torch

from utils.config_fish import create_config
from utils.common_config_fish import get_train_dataset, get_train_transformations,\
                                get_val_dataset, get_val_transformations,\
                                get_train_dataloader, get_val_dataloader,\
                                get_optimizer, get_model, adjust_learning_rate,\
                                get_criterion
from utils.ema import EMA
from utils.evaluate_utils_fish import get_predictions, hungarian_evaluate
from utils.train_utils_fish import selflabel_train
from termcolor import colored
from utils.utils import Logger
import sys
import time
import numpy as np
import pandas as p

# Parser
parser = argparse.ArgumentParser(description='Self-labeling')
parser.add_argument('--config_env',
                    help='Config file for the environment', default='configs/env.yml')
parser.add_argument('--config_exp',
                    help='Config file for the experiment',default='configs/selflabel/selflabel_fish.yml')
parser.add_argument('--gpu', type=str, default='2,3')
parser.add_argument('--ct1', type=float, default=0.2)
parser.add_argument('--ct2', type=float, default=0.9)
parser.add_argument('--eta', type=float, default=0.6) # the proportion of sharpening
parser.add_argument('--topk', type=int, default=5)
parser.add_argument('--loss', type=str, default='our')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def main():
    print('args.ct1:{}'.format(args.ct1))
    print('args.ct2:{}'.format(args.ct2))
    print('args.eta:{}'.format(args.eta))
    print('args.topk:{}'.format(args.topk))

    start = time.time()
    # Retrieve config file
    p = create_config(args.config_env, args.config_exp, args.topk)
    print(colored(p, 'red'))
    print(p['top{}_neighbors_train_path'.format(p['num_neighbors'])])

    # Get model
    print(colored('Retrieve model', 'blue'))
    model = get_model(p, p['scan_model']) #best_scan_model找不到
    print(p['scan_model'])
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    # Get criterion
    print(colored('Get loss', 'blue'))
    if args.loss == "our":
        from losses.losses import ConfidenceBasedCE
        criterion = ConfidenceBasedCE(args.ct1, args.ct2, p['confidence_threshold'], p['criterion_kwargs']['apply_class_balancing'])
    elif args.loss == "scan":
        from losses.losses import ConfidenceBasedCE_scan
        criterion = ConfidenceBasedCE_scan(args.ct1, args.ct2, p['confidence_threshold'],
                                      p['criterion_kwargs']['apply_class_balancing'])
    # criterion = get_criterion(p)
    criterion.cuda()
    print(criterion)

    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue'))
    torch.backends.cudnn.benchmark = True

    # Optimizer
    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model)
    print(optimizer)

    # Dataset
    print(colored('Retrieve dataset', 'blue'))

    # Transforms
    strong_transforms = get_train_transformations(p)
    val_transforms = get_val_transformations(p)
    train_dataset = get_train_dataset(p, {'standard': val_transforms, 'augment': strong_transforms},
                                        split='train', to_neighbors_dataset=True)
    train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataset = get_val_dataset(p, val_transforms)
    val_dataloader = get_val_dataloader(p, val_dataset)
    print(colored('Train samples %d - Val samples %d' %(len(train_dataset), len(val_dataset)), 'yellow'))

    # Evaluate
    print('Evaluate ...')
    predictions = get_predictions(p, val_dataloader, model)
    clustering_stats = hungarian_evaluate(0, predictions,
                                          class_names=val_dataset.classes,
                                          compute_confusion_matrix=True,
                                          confusion_matrix_file=os.path.join(p['selflabel_dir'],
                                                                             'confusion_matrix.png'))
    print(clustering_stats)

    # Checkpoint
    if False and os.path.exists(p['selflabel_checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['selflabel_checkpoint']), 'blue'))
        checkpoint = torch.load(p['selflabel_checkpoint'], map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

    else:
        print(colored('No checkpoint file at {}'.format(p['selflabel_checkpoint']), 'blue'))
        start_epoch = 0

    best_acc = 0
    best_ri = 0
    best_epoch = -1
    best_clustering_stats = None

    # EMA
    if p['use_ema']:
        ema = EMA(model, alpha=p['ema_alpha'])
    else:
        ema = None


    # Main loop
    print(colored('Starting main loop', 'blue'))

    epoch_num = p['epochs']
    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' %(epoch+1, p['epochs']), 'yellow'))
        print(colored('-'*10, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Perform self-labeling
        print('Train ...')
        final_loss = selflabel_train(train_dataloader, model, criterion, optimizer, epoch, epoch_num, args.eta, ema=ema)

        # Evaluate (To monitor progress - Not for validation)
        print('Evaluate ...')
        predictions = get_predictions(p, val_dataloader, model)
        clustering_stats = hungarian_evaluate(0, predictions, compute_confusion_matrix=False)
        print(clustering_stats)

        # Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                    'epoch': epoch + 1}, p['selflabel_checkpoint'])

        if best_ri < clustering_stats['RI']:
            best_epoch = epoch
            best_ri = clustering_stats['RI']
            best_clustering_stats = clustering_stats
            if os.path.exists(p['best_selflabel_results']):
                results_old = torch.load(p['best_selflabel_results'], map_location='cpu')
                if best_ri > results_old['RI']:
                    torch.save(best_clustering_stats, p['best_selflabel_results'])
                    torch.save(model.module.state_dict(), p['selflabel_model'])
            else:
                print('======CREAT=======')
                torch.save(best_clustering_stats, p['best_selflabel_results'])
                torch.save(model.module.state_dict(), p['selflabel_model'])


    # Evaluate and save the final model
    print('=============================================================================')
    print(colored('Evaluate model at the end', 'blue'))
    predictions = get_predictions(p, val_dataloader, model)
    clustering_stats = hungarian_evaluate(0, predictions,
                                class_names=val_dataset.classes,
                                compute_confusion_matrix=True,
                                confusion_matrix_file=os.path.join(p['selflabel_dir'], 'confusion_matrix.png'))
    print(best_clustering_stats)
    end = time.time()
    t = end - start
    print('scan (train)' + p['dataset_name'] + ' The training time: {:.0f}min {:.0f}sec'.format(t // 60, t % 60))
    print('best_epoch: {}'.format(best_epoch))
    print('=============================================================================')



if __name__ == "__main__":
    main()

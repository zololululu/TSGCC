"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import yaml
from easydict import EasyDict
from utils.utils import mkdir_if_missing


def create_config(config_file_env, config_file_exp, topk):
    # Config for environment path
    with open(config_file_env, 'r') as stream:
        root_dir = yaml.safe_load(stream)['root_dir']

    with open(config_file_exp, 'r') as stream:
        config = yaml.safe_load(stream)

    cfg = EasyDict({ })

    # Copy
    for k, v in config.items():
        cfg[k] = v

    cfg['num_neighbors'] = topk

    # Set paths for pretext task (These directories are needed in every stage)
    base_dir = os.path.join(root_dir, cfg['train_db_name'])
    pretext_dir = os.path.join(base_dir, 'pretext')
    mkdir_if_missing(base_dir)
    mkdir_if_missing(pretext_dir)
    cfg['pretext_dir'] = pretext_dir
    cfg['pretext_checkpoint'] = os.path.join(pretext_dir, 'checkpoint.pth.tar')
    # cfg['pretext_model'] = os.path.join(pretext_dir, 'model.pth.tar')

    if cfg['dataset_name'] == 'cifar10':
        cfg['pretext_model'] = os.path.join(pretext_dir, 'simclr_cifar-10.pth.tar')
    elif cfg['dataset_name'] == 'stl10':
        cfg['pretext_model'] = os.path.join(pretext_dir, 'simclr_stl-10.pth.tar')
    elif cfg['dataset_name'] == 'cifar20':
        cfg['pretext_model'] = os.path.join(pretext_dir, 'simclr_cifar-20.pth.tar')

    cfg['top{}_neighbors_train_path'.format(cfg['num_neighbors'])] = os.path.join(pretext_dir,
                                                    'top{}-train-neighbors.npy'.format(cfg['num_neighbors']))  #  use the after
    cfg['after_top{}_neighbors_train_path'.format(cfg['num_neighbors'])] = os.path.join(pretext_dir,
                                                    'after_top{}-train-neighbors.npy'.format(cfg['num_neighbors']))
    cfg['topk_neighbors_val_path'] = os.path.join(pretext_dir, 'topk-val-neighbors.npy')

    # If we perform clustering or self-labeling step we need additional paths.
    # We also include a run identifier to support multiple runs w/ same hyperparams.
    if cfg['setup'] in ['scan', 'selflabel']:
        base_dir = os.path.join(root_dir, cfg['train_db_name'])
        scan_dir = os.path.join(base_dir, 'scan')
        selflabel_dir = os.path.join(base_dir, 'selflabel')
        sim_scan_dir = os.path.join(base_dir, 'sim_scan')
        mkdir_if_missing(base_dir)
        mkdir_if_missing(scan_dir)
        mkdir_if_missing(selflabel_dir)
        mkdir_if_missing(sim_scan_dir)
        cfg['scan_dir'] = scan_dir
        cfg['scan_checkpoint'] = os.path.join(scan_dir, 'checkpoint.tar')
        cfg['scan_model'] = os.path.join(scan_dir, 'model.pth.tar')    # using in scan
        cfg['best_scan_model'] = os.path.join(scan_dir, 'best_model8009.pth.tar')  # using in selflabel
        cfg['scan_best_clustering_results'] = os.path.join(scan_dir, 'best_clustering_results.pth.tar')
        cfg['selflabel_dir'] = selflabel_dir
        cfg['selflabel_checkpoint'] = os.path.join(selflabel_dir, 'checkpoint.pth.tar')
        cfg['selflabel_model'] = os.path.join(selflabel_dir, 'model.pth.tar')
        cfg['best_selflabel_results'] = os.path.join(selflabel_dir, 'best_selflabel_results.pth.tar')

        cfg['scan_best_clustering_results_wo1'] = os.path.join(scan_dir, 'best_clustering_results_wo1.pth.tar')
        cfg['scan_best_clustering_results_wo2'] = os.path.join(scan_dir, 'best_clustering_results_wo2.pth.tar')
        cfg['scan_best_clustering_results_wo3'] = os.path.join(scan_dir, 'best_clustering_results_wo3.pth.tar')

        cfg['scan_model_wo1'] = os.path.join(scan_dir, 'model_wo1.pth.tar')
        cfg['scan_model_wo2'] = os.path.join(scan_dir, 'model_wo2.pth.tar')
        cfg['scan_model_wo3'] = os.path.join(scan_dir, 'model_wo3.pth.tar')

    return cfg 

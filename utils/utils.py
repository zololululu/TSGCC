"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import torch
import numpy as np
import errno
import torch.nn as nn
import sys

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        # fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


@torch.no_grad()
def fill_memory_bank(loader, model, memory_bank):
    model.eval()
    memory_bank.reset()

    for i, batch in enumerate(loader):
        images = batch['image'].cuda(non_blocking=True)
        targets = batch['target'].cuda(non_blocking=True)
        output = model(images)
        memory_bank.update(output, targets)
        if i % 100 == 0:
            print('Fill Memory Bank [%d/%d]' %(i, len(loader)))


def confusion_matrix(predictions, gt, class_names, output_file=None):
    # Plot confusion_matrix and store result to output_file
    import sklearn.metrics
    import matplotlib.pyplot as plt
    confusion_matrix = sklearn.metrics.confusion_matrix(gt, predictions)
    confusion_matrix = confusion_matrix / np.sum(confusion_matrix, 1)
    
    fig, axes = plt.subplots(1)
    plt.imshow(confusion_matrix, cmap='Blues')
    axes.set_xticks([i for i in range(len(class_names))])
    axes.set_yticks([i for i in range(len(class_names))])
    axes.set_xticklabels(class_names, ha='right', fontsize=8, rotation=40)
    axes.set_yticklabels(class_names, ha='right', fontsize=8)
    
    for (i, j), z in np.ndenumerate(confusion_matrix):
        if i == j:
            axes.text(j, i, '%d' %(100*z), ha='center', va='center', color='white', fontsize=6)
        else:
            pass

    plt.tight_layout()
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


# based on centers
# def get_distant_neighbors(batch: torch.Tensor, anchors_features: torch.Tensor, centers, dk):
#
#     sim_cos = nn.CosineSimilarity(dim=2, eps=1e-6)  # cosine similarity
#     similarity = sim_cos(anchors_features.unsqueeze(1), centers.unsqueeze(0))
#     sorted, indices = torch.sort(similarity, dim=1, descending=True)
#     ind_max = indices[:, 0]
#     anchor_center = centers[ind_max]   # each anchor's center
#
#     similarity2 = sim_cos(anchor_center.unsqueeze(1), anchors_features.unsqueeze(0))  # the cosine similarity between each anchor_center and anchors_features
#
#     _, indices2 = torch.sort(similarity2, dim=1, descending=True)
#     # sub = 5
#     # inds = np.random.choice(-1*np.arange(1,dk+1), sub)
#     # inds = torch.from_numpy(inds)
#     # ind_min = indices2[:,inds]
#     ind_min = indices2[:, -10:]
#     return ind_min


# get the distant neighbors based on pseduo labels
def get_distant_neighbors(batch: torch.Tensor, anchors_output: torch.Tensor, epoch, labels):

    sim_cos = nn.CosineSimilarity(dim=2, eps=1e-6)  # cosine similarity
    similarity = sim_cos(anchors_output.unsqueeze(1), anchors_output.unsqueeze(0))
    sorted, indices = torch.sort(similarity, dim=1, descending=True)

    dk = min(1+int(epoch/25), 4)
    ind_labels = labels[indices]
    anchors_output = anchors_output[indices]

    tag = np.arange(0,10)
    disks_output = []
    for i in range(ind_labels.shape[0]):
        tag_ = list(set(tag) - set([ind_labels[i][0].item()]))
        mask = ind_labels[i].view(1,-1).repeat(len(tag_), 1) == torch.tensor(tag_).view(-1,1).cuda()
        disks_output_i = [anchors_output[i][mask[j]][-dk:] for j in range(len(mask))]
        disks_output_i = torch.cat(disks_output_i, dim=0)
        # when some clusters's samples are not enough
        if disks_output_i.shape[0] < len(tag_)*dk:
            num = len(tag_)*dk - disks_output_i.shape[0]
            disks_output_i = torch.cat([disks_output_i, disks_output_i[-num:]], dim=0)
        disks_output.append(disks_output_i.unsqueeze(0))
    disks_output = torch.cat(disks_output, dim=0)

    return disks_output

# get the centers of batch based on the pseduo labels
def get_batch_centers(anchors_output: torch.Tensor, pseduo_labels):
    values, _ = torch.max(anchors_output, dim=1)
    sorted, indices = torch.sort(values, dim=0, descending=True)
    labels = pseduo_labels[indices]
    anchors_output_sorted = anchors_output[indices]
    classes = torch.unique(labels)
    centers = []
    centers_ratio = 0.5
    for i in range(classes.shape[0]):
        mask = labels == classes[i]
        num = int(torch.sum(mask) * centers_ratio)
        centers.append(anchors_output_sorted[mask][0:num,:].mean(axis=0).unsqueeze(dim=0))
    centers = torch.cat(centers, dim=0)
    return centers


@torch.no_grad()
def count_negative_simples(disks_index, targets):
    if len(disks_index.shape) == 2:
        neg_num = disks_index.shape[0] * disks_index.shape[1]
    else:
        neg_num = disks_index.shape[0]

    true_neg_num = torch.tensor([0], dtype=torch.int64).cuda()
    for i in range(disks_index.shape[0]):
        true_neg_num += torch.sum(targets[i] != targets[disks_index[i]])

    return true_neg_num.item(), neg_num

@torch.no_grad()
def get_features_train(val_loader, model, forward_pass='default'):
    model.train()
    targets, features, indices = [], [], []
    for i, batch in enumerate(val_loader):
        # Forward pass
        input_ = batch['image'].cuda()
        target_ = batch['target'].cuda()
        index_ = batch['index'].cuda()

        with torch.no_grad():
            feature_ = model(input_, forward_pass = forward_pass)

        targets.append(target_)
        features.append(feature_[0].cpu())
        indices.append(index_)

    targets = torch.cat(targets).int()
    features = torch.cat(features)
    indices = torch.cat(indices)

    features_order = torch.zeros_like(features)
    features_order[indices] = features

    targets_order = torch.zeros_like(targets)
    targets_order[indices] = targets

    return features_order, targets_order

@torch.no_grad()
def get_features_eval(val_loader, model, forward_pass='default'):
    model.eval()
    targets, features, indices = [], [], []
    for i, batch in enumerate(val_loader):
        # Forward pass
        input_ = batch['anchor'].cuda()
        target_ = batch['target'].cuda()
        index_ = batch['index'].cuda()

        with torch.no_grad():
            feature_ = model(input_, forward_pass = forward_pass)

        targets.append(target_)
        features.append(feature_[0].cpu())
        indices.append(index_)

    targets = torch.cat(targets).int()
    features = torch.cat(features)
    indices = torch.cat(indices)

    features_order = torch.zeros_like(features)
    features_order[indices] = features

    targets_order = torch.zeros_like(targets)
    targets_order[indices] = targets

    return features_order, targets_order

@torch.no_grad()
def select_samples(feas_sim, scores, ratio_select, num_cluster=10, center_ratio=0.5):
    _, idx_max = torch.sort(scores, dim=0, descending=True)
    idx_max = idx_max.cpu()
    num_per_cluster = idx_max.shape[0] // num_cluster
    k = int(num_per_cluster * center_ratio)
    idx_max = idx_max[0:k, :]

    centers = []
    for c in range(num_cluster):
        centers.append(feas_sim[idx_max[:, c], :].mean(axis=0).unsqueeze(dim=0))

    centers = torch.cat(centers, dim=0)

    num_select_c = int(num_per_cluster * ratio_select)

    dis = torch.einsum('cd,nd->cn', [centers, feas_sim])
    idx_select = torch.argsort(dis, dim=1, descending=True)[:, 0:num_select_c].flatten()
    labels_select = torch.arange(0, num_cluster).unsqueeze(dim=1).repeat(1, num_select_c).flatten()

    return idx_select, labels_select


@torch.no_grad()
def select_centers(features, scores, num_cluster=10, center_ratio=0.4):
    _, idx_max = torch.sort(scores, dim=0, descending=True)
    idx_max = idx_max.cpu()
    num_per_cluster = idx_max.shape[0] // num_cluster
    k = int(num_per_cluster * center_ratio)
    idx_max = idx_max[0:k, :]

    centers = []
    for c in range(num_cluster):
        centers.append(features[idx_max[:, c], :].mean(axis=0).unsqueeze(dim=0))

    centers = torch.cat(centers, dim=0)

    return centers


@torch.no_grad()
def rand_bbox(size, lam):
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


@torch.no_grad()
def get_knn_indices(base_dataloader, model, topk=20):
    from utils.utils import get_features_eval
    features, targets = get_features_eval(base_dataloader, model, forward_pass='backbone')

    from utils.evaluate_utils_fish import kmeans
    kmeans(features, targets)

    import faiss
    features = features.cpu().numpy()
    n, dim = features.shape[0], features.shape[1]
    print('===============dim')
    index = faiss.IndexFlatIP(dim)  # index = faiss.IndexFlatIP(dim)
    index = faiss.index_cpu_to_all_gpus(index)
    index.add(features)
    distances, indices = index.search(features, topk + 1)  # Sample itself is included


    # evaluate
    targets = targets.cpu().numpy()
    neighbor_targets = np.take(targets, indices[:, 1:], axis=0)  # Exclude sample itself for eval
    anchor_targets = np.repeat(targets.reshape(-1, 1), topk, axis=1)
    accuracy = np.mean(neighbor_targets == anchor_targets)
    print(accuracy)

    return indices, accuracy



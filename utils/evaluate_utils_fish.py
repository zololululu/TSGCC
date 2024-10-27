"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import numpy as np
import torch
import torch.nn.functional as F
from utils.common_config import get_feature_dimensions_backbone
from utils.utils import AverageMeter, confusion_matrix
from data.custom_dataset import NeighborsDataset
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from losses.losses import entropy
from sklearn.cluster import KMeans


@torch.no_grad()
def contrastive_evaluate(val_loader, model, memory_bank):
    top1 = AverageMeter('Acc@1', ':6.2f')
    model.eval()

    for batch in val_loader:
        tss = batch['ts'].cuda(non_blocking=True)
        target = batch['target'].cuda(non_blocking=True)

        output = model(tss)
        output = memory_bank.weighted_knn(output)

        acc1 = 100 * torch.mean(torch.eq(output, target).float())
        top1.update(acc1.item(), tss.size(0))

    return top1.avg


@torch.no_grad()
def get_predictions(p, dataloader, model, forward_type='online_evaluation', return_features=False):
    # Make predictions on a dataset with neighbors
    model.eval()
    predictions = [[] for _ in range(p['num_heads'])]
    probs = [[] for _ in range(p['num_heads'])]
    targets = []
    if return_features:
        ft_dim = get_feature_dimensions_backbone(p)
        features = torch.zeros((len(dataloader.sampler), ft_dim)).cuda()

    if isinstance(dataloader.dataset, NeighborsDataset):  # Also return the neighbors
        key_ = 'anchor'
        include_neighbors = True
        neighbors = []

    else:
        key_ = 'ts'
        include_neighbors = False

    ptr = 0
    for batch in dataloader:
        tss = torch.FloatTensor(batch[key_]).cuda(non_blocking=True)
        bs = tss.shape[0]
        res = model(tss, forward_pass='return_all')
        output = res['output']
        if return_features:
            features[ptr: ptr + bs] = res['features']
            ptr += bs
        for i, output_i in enumerate(output):
            # print(output_i[0])
            predictions[i].append(torch.argmax(output_i, dim=1))
            # print(predictions[i][0])
            probs[i].append(F.softmax(output_i, dim=1))  # others, except sim_scan
            # print(probs[i][0])
            # probs[i].append(output_i)   # using in sim_scan
        targets.append(batch['target'])
        if include_neighbors:
            neighbors.append(batch['anchor_neighbors_indices'])
    predictions = [torch.cat(pred_, dim=0).cpu() for pred_ in predictions]
    probs = [torch.cat(prob_, dim=0).cpu() for prob_ in probs]
    targets = torch.cat(targets, dim=0)

    if include_neighbors:
        neighbors = torch.cat(neighbors, dim=0)
        out = [{'predictions': pred_, 'probabilities': prob_, 'targets': targets, 'neighbors': neighbors} for
               pred_, prob_ in zip(predictions, probs)]

    else:
        out = [{'predictions': pred_, 'probabilities': prob_, 'targets': targets} for pred_, prob_ in
               zip(predictions, probs)]

    if return_features:
        return out, features.cpu()
    else:
        return out


@torch.no_grad()
def get_predictions_scan_example(p, dataloader, model, ct, forward_type='online_evaluation', return_features=False):
    # Make predictions on a dataset with neighbors
    model.train()
    predictions = [[] for _ in range(p['num_heads'])]
    probs = [[] for _ in range(p['num_heads'])]
    targets = []
    if return_features:
        ft_dim = get_feature_dimensions_backbone(p)
        features = torch.zeros((len(dataloader.sampler), ft_dim)).cuda()

    if isinstance(dataloader.dataset, NeighborsDataset):  # Also return the neighbors
        key_ = 'anchor'
        include_neighbors = True
        neighbors = []

    else:
        key_ = 'ts'
        include_neighbors = False

    ptr = 0
    sum = 0
    for batch in dataloader:
        tss = batch[key_].cuda(non_blocking=True)
        bs = tss.shape[0]
        res = model(tss, forward_pass='return_all')
        output = res['output']
        if return_features:
            features[ptr: ptr + bs] = res['features']
            ptr += bs

        mask = None
        for i, output_i in enumerate(output):
            max_prob, target = torch.max(output_i.softmax(1), dim=1)
            mask = max_prob > ct
            predictions[i].append(torch.masked_select(torch.argmax(output_i, dim=1), mask.squeeze()))
            probs[i].append(F.softmax(output_i, dim=1)[mask])
            sum += torch.sum(mask > 0)
        targets.append(torch.masked_select(batch['target'], mask.squeeze()))
        if include_neighbors:
            neighbors.append(batch['anchor_neighbors_indices'][mask])
    predictions = [torch.cat(pred_, dim=0).cpu() for pred_ in predictions]
    probs = [torch.cat(prob_, dim=0).cpu() for prob_ in probs]
    targets = torch.cat(targets, dim=0)

    if include_neighbors:
        neighbors = torch.cat(neighbors, dim=0)
        out = [{'predictions': pred_, 'probabilities': prob_, 'targets': targets, 'neighbors': neighbors} for
               pred_, prob_ in zip(predictions, probs)]

    else:
        out = [{'predictions': pred_, 'probabilities': prob_, 'targets': targets} for pred_, prob_ in
               zip(predictions, probs)]

    if return_features:
        return out, features.cpu()
    else:
        return out, sum


@torch.no_grad()
def get_predictions_our_example(p, dataloader, model, ct, forward_type='online_evaluation', return_features=False):
    # Make predictions on a dataset with neighbors
    model.train()
    predictions = [[] for _ in range(p['num_heads'])]
    probs = [[] for _ in range(p['num_heads'])]
    targets = []
    if return_features:
        ft_dim = get_feature_dimensions_backbone(p)
        features = torch.zeros((len(dataloader.sampler), ft_dim)).cuda()

    if isinstance(dataloader.dataset, NeighborsDataset):  # Also return the neighbors
        key_ = 'anchor'
        include_neighbors = True
        neighbors = []

    else:
        key_ = 'ts'
        include_neighbors = False

    from utils.utils import get_features_train
    neighbors_outputs, _ = get_features_train(dataloader, model)

    ptr = 0
    sum = 0
    for batch in dataloader:
        tss = batch[key_].cuda(non_blocking=True)
        bs = tss.shape[0]
        res = model(tss, forward_pass='return_all')
        output = res['output']
        if return_features:
            features[ptr: ptr + bs] = res['features']
            ptr += bs

        ct1 = 0.6
        eta = 0.6
        neighbors_indices = batch['anchor_neighbors_indices']
        neighbors_output = neighbors_outputs[neighbors_indices].cuda(non_blocking=True)

        anchors_weak = output[0]
        neighbors2 = neighbors_output

        weak_anchors_prob = anchors_weak.softmax(1)
        neighbors_prob = neighbors2.softmax(2)

        # set ct1
        max_prob_0, target_0 = torch.max(weak_anchors_prob, dim=1)
        mask0 = max_prob_0 > ct1

        weak_anchors_prob = weak_anchors_prob[mask0]
        neighbors_prob = neighbors_prob[mask0]
        b, c = weak_anchors_prob.size()

        beta = torch.sum(torch.exp(-torch.norm(weak_anchors_prob.unsqueeze(1) - neighbors_prob, dim=2) ** 2).unsqueeze(
            2) * neighbors_prob, dim=1)
        beta = beta / beta.sum(1).view(-1, 1).expand_as(beta)

        # compute the tau
        q_beta_norm = torch.norm(weak_anchors_prob - beta, dim=1) ** 2
        topk = max(int(eta * b), 1)
        topk_min, _ = torch.topk(q_beta_norm, topk, largest=False)
        tau = topk_min[-1] / torch.exp(torch.tensor([-1.0]).cuda())
        alpha = -torch.log(q_beta_norm / tau)

        # hardening based on alpha and mask0
        q = []
        for i in range(len(alpha)):
            if alpha[i] > 1:
                qi = weak_anchors_prob[i] ** alpha[i]
                qi = qi / qi.sum(0)
            else:
                qi = weak_anchors_prob[i]
            q.append(qi.unsqueeze(0))
        q = torch.cat(q, dim=0)

        output = [q]

        mask = None
        for i, output_i in enumerate(output):
            max_prob, target = torch.max(output_i, dim=1)
            mask = max_prob > ct
            predictions[i].append(torch.masked_select(torch.argmax(output_i, dim=1), mask.squeeze()))
            probs[i].append(F.softmax(output_i, dim=1)[mask])
            sum += torch.sum(mask > 0)
        targets.append(torch.masked_select(batch['target'][mask0], mask.squeeze()))
        if include_neighbors:
            neighbors.append(batch['anchor_neighbors_indices'][mask0][mask])
    predictions = [torch.cat(pred_, dim=0).cpu() for pred_ in predictions]
    probs = [torch.cat(prob_, dim=0).cpu() for prob_ in probs]
    targets = torch.cat(targets, dim=0)

    if include_neighbors:
        neighbors = torch.cat(neighbors, dim=0)
        out = [{'predictions': pred_, 'probabilities': prob_, 'targets': targets, 'neighbors': neighbors} for
               pred_, prob_ in zip(predictions, probs)]

    else:
        out = [{'predictions': pred_, 'probabilities': prob_, 'targets': targets} for pred_, prob_ in
               zip(predictions, probs)]

    if return_features:
        return out, features.cpu()
    else:
        return out, sum


@torch.no_grad()
def get_pseudo_labels(indexs, outputs):
    predictions = []
    predictions = torch.argmax(outputs, dim=1)
    predictions_order = torch.zeros_like(predictions) - 1
    predictions_order[indexs] = predictions
    return predictions_order


@torch.no_grad()
def get_confident_simples_ind(train_dataloader, model, r=0.05):
    model.eval()
    targets, features, indexs = [], [], []
    for i, batch in enumerate(train_dataloader):
        # Forward pass
        input = batch[0].cuda(non_blocking=True)
        with torch.no_grad():
            feature_ = model(input, input, forward_type='embedding')

        features.append(feature_)

    features = torch.cat(features)

    # get confi
    features = torch.cat(features)

    predictions = features.softmax(dim=1)
    sorted, ind = torch.sort(predictions, dim=0, descending=True)
    conf_ind = ind[0:int(predictions.shape[0] * r), :]

    features_conf = features[conf_ind]
    centroids = torch.sum(features_conf, dim=0) / features_conf.shape[0]  # 10*10, each row represents a centroid_i.

    conf_indexs = indexs[conf_ind].reshape(1, -1)  # get the real index. indexs are from batch samples.

    outputs = features.softmax(dim=1)
    val, _ = torch.max(outputs, dim=1)
    # conf_ind = indexs[val > torch.mean(val)]
    return conf_ind


@torch.no_grad()
def sim_evaluate(predictions, criterion):
    # Evaluate model based on Sim loss.

    probs_online, probs_target = predictions['probabilities']
    from losses.losses import SimScanLoss
    loss = criterion(probs_target, probs_online)

    return loss.item()


@torch.no_grad()
def cc_evaluate(predictions, p):
    # Evaluate model based on SCAN loss.
    num_heads = len(predictions)
    output = []

    for head in predictions:
        # Neighbors and anchors
        probs = head['probabilities']
        neighbors = head['neighbors']
        anchors = torch.arange(neighbors.size(0)).view(-1, 1).expand_as(neighbors)

        # Entropy loss
        entropy_loss = entropy(torch.mean(probs, dim=0), input_as_probabilities=True).item()

        # Consistency loss
        similarity = torch.matmul(probs, probs.t())
        neighbors = neighbors.contiguous().view(-1)
        anchors = anchors.contiguous().view(-1)
        similarity = similarity[anchors, neighbors]

        pos_similarity = torch.sum(similarity) / probs.shape[0]
        consistency_loss = -pos_similarity

        # Total loss
        total_loss = - entropy_loss + consistency_loss

        output.append({'entropy': entropy_loss, 'consistency': consistency_loss, 'total_loss': total_loss})

    total_losses = [output_['total_loss'] for output_ in output]
    lowest_loss_head = np.argmin(total_losses)
    lowest_loss = np.min(total_losses)

    return {'scan': output, 'lowest_loss_head': lowest_loss_head, 'lowest_loss': lowest_loss}


@torch.no_grad()
def hungarian_evaluate(subhead_index, all_predictions, class_names=None,
                       compute_purity=True, compute_confusion_matrix=True,
                       confusion_matrix_file=None):
    # Evaluate model based on hungarian matching between predicted cluster assignment and gt classes.
    # This is computed only for the passed subhead index.

    # Hungarian matching
    head = all_predictions[subhead_index]
    targets = head['targets'].cuda()
    predictions = head['predictions'].cuda()
    probs = head['probabilities'].cuda()
    print('prob====================================================')
    print(head['probabilities'])
    print('prob====================================================')
    num_classes = torch.unique(targets).numel()
    num_elems = targets.size(0)

    match = _hungarian_match(predictions, targets, preds_k=num_classes, targets_k=num_classes)
    reordered_preds = torch.zeros(num_elems, dtype=predictions.dtype).cuda()
    for pred_i, target_i in match:
        reordered_preds[predictions == int(pred_i)] = int(target_i)

    # Gather performance metrics
    acc = int((reordered_preds == targets).sum()) / float(num_elems)
    nmi = metrics.normalized_mutual_info_score(targets.cpu().numpy(), predictions.cpu().numpy())
    ari = metrics.adjusted_rand_score(targets.cpu().numpy(), predictions.cpu().numpy())
    ri = metrics.rand_score(targets.cpu().numpy(), predictions.cpu().numpy())

    _, preds_top5 = probs.topk(2, 1, largest=True)
    reordered_preds_top5 = torch.zeros_like(preds_top5)
    for pred_i, target_i in match:
        reordered_preds_top5[preds_top5 == int(pred_i)] = int(target_i)
    correct_top5_binary = reordered_preds_top5.eq(targets.view(-1, 1).expand_as(reordered_preds_top5))
    top5 = float(correct_top5_binary.sum()) / float(num_elems)

    # Compute confusion matrix
    if compute_confusion_matrix:
        confusion_matrix(reordered_preds.cpu().numpy(), targets.cpu().numpy(),
                         class_names, confusion_matrix_file, )

    return {'ACC': acc, 'RI': ri, 'ARI': ari, 'NMI': nmi, 'ACC Top-5': top5, 'hungarian_match': match}


@torch.no_grad()
def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
    # Based on implementation from IIC
    num_samples = flat_targets.shape[0]

    assert (preds_k == targets_k)  # one to one
    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes

    # num_correct is small
    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))

    # return as list of tuples, out_c to gt_c
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))

    return res


@torch.no_grad()
def scan_evaluate(predictions):
    # Evaluate model based on SCAN loss.
    num_heads = len(predictions)
    output = []

    for head in predictions:
        # Neighbors and anchors
        probs = head['probabilities']
        neighbors = head['neighbors']
        anchors = torch.arange(neighbors.size(0)).view(-1, 1).expand_as(neighbors)

        # Entropy loss
        entropy_loss = entropy(torch.mean(probs, dim=0), input_as_probabilities=True).item()

        # Consistency loss
        similarity = torch.matmul(probs, probs.t())

        neighbors = neighbors.contiguous().view(-1)
        anchors = anchors.contiguous().view(-1)

        similarity = similarity[anchors, neighbors]
        ones = torch.ones_like(similarity)
        consistency_loss = F.binary_cross_entropy(similarity, ones).item()

        # Total loss
        total_loss = - entropy_loss + consistency_loss

        output.append({'entropy': entropy_loss, 'consistency': consistency_loss, 'total_loss': total_loss})

    total_losses = [output_['total_loss'] for output_ in output]
    lowest_loss_head = np.argmin(total_losses)
    lowest_loss = np.min(total_losses)

    return {'scan': output, 'lowest_loss_head': lowest_loss_head, 'lowest_loss': lowest_loss}


def kmeans(features, targets, train_or_test):
    num_elems = targets.size(0)
    num_classes = torch.unique(targets).numel()
    kmeans = KMeans(n_clusters=num_classes, n_init=20)
    predicted = kmeans.fit_predict(features.numpy())

    predicted = torch.from_numpy(predicted)
    predicted = predicted.cuda(targets.device)

    match = _hungarian_match(predicted, targets, preds_k=num_classes, targets_k=num_classes)
    reordered_preds = torch.zeros(num_elems, dtype=predicted.dtype).cuda()
    for pred_i, target_i in match:
        reordered_preds[predicted == int(pred_i)] = int(target_i)

    # Gather performance metrics
    acc = int((reordered_preds == targets).sum()) / float(num_elems)
    nmi = metrics.normalized_mutual_info_score(targets.cpu().numpy(), predicted.cpu().numpy())
    ari = metrics.adjusted_rand_score(targets.cpu().numpy(), predicted.cpu().numpy())
    ri = metrics.rand_score(targets.cpu().numpy(), predicted.cpu().numpy())

    print("{} RI: {}, NMI: {}, ARI: {}, ACC: {}".format(train_or_test,ri,nmi,ari,acc))

    return predicted

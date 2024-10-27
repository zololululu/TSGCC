"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8
import math
import numpy as np


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()

    def forward(self, input, target, mask, weight, reduction='mean'):
        if not (mask != 0).any():
            raise ValueError('Mask in MaskedCrossEntropyLoss is all zeros.')
        target = torch.masked_select(target, mask)
        b, c = input.size()
        n = target.size(0)
        input = torch.masked_select(input, mask.view(b, 1)).view(n, c)
        return F.cross_entropy(input, target, weight=weight, reduction=reduction)


class ConfidenceBasedCE(nn.Module):
    def __init__(self, ct1, ct2, threshold, apply_class_balancing):
        super(ConfidenceBasedCE, self).__init__()
        self.loss = MaskedCrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax_func = nn.LogSoftmax(dim=1)
        self.ct1 = ct1
        self.ct2 = ct2
        self.apply_class_balancing = apply_class_balancing
        self.tau = None

    def forward(self, anchors_weak, anchors_strong, neighbors, eta, epoch):
        """
        Loss function during self-labeling

        input: logits for original samples and for its strong augmentations
        output: cross entropy
        """

        # get prob
        weak_anchors_prob = self.softmax(anchors_weak)
        neighbors_prob = neighbors.softmax(2)
        # set ct1mas
        max_prob_0, target_0 = torch.max(weak_anchors_prob, dim=1)
        mask0 = max_prob_0 > self.ct1
        print('weak_anchors_probs--------------------------------------------')
        # print(weak_anchors_prob)
        print(self.ct1)
        print('mask000000000000000000000000000000000000000')
        # mask0是大于第一个阈值的概率
        weak_anchors_prob = weak_anchors_prob[mask0]

        # print(weak_anchors_prob)
        print('weak_mak_probbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
        neighbors_prob = neighbors_prob[mask0]
        anchors_strong = anchors_strong[mask0]
        b, c = weak_anchors_prob.size()

        beta = torch.sum(torch.exp(-torch.norm(weak_anchors_prob.unsqueeze(1) - neighbors_prob, dim=2) ** 2).unsqueeze(
            2) * neighbors_prob, dim=1)
        beta = beta / beta.sum(1).view(-1, 1).expand_as(beta)

        # compute the tau
        q_beta_norm = torch.norm(weak_anchors_prob - beta, dim=1) ** 2
        topk = max(int(eta * b), 1)

        print(weak_anchors_prob)
        print(q_beta_norm)
        print('topkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk')
        topk_min, _ = torch.topk(q_beta_norm, topk, largest=False)
        raise_tau = 1                                                      # tau* 温度系数
        print('raise tau:{}'.format(raise_tau))
        self.tau = topk_min[-1] / torch.exp(torch.tensor([-1.0]).cuda()) * raise_tau
        alpha = -torch.log(q_beta_norm / self.tau)

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

        # set ct2, retrieve target and mask based on soft-label q
        max_prob, target = torch.max(q, dim=1)
        mask = max_prob > self.ct2
        target_masked = torch.masked_select(target, mask.squeeze())
        n = target_masked.size(0)

        # Class balancing weights
        if self.apply_class_balancing:
            idx, counts = torch.unique(target_masked, return_counts=True)
            weight = torch.ones(c).cuda()
            freq = counts.float() / n
            h = 1.02
            weight[idx] = 1 / torch.log(h + freq)

        else:
            weight = None

        # Loss
        input = torch.masked_select(anchors_strong, mask.view(b, 1)).view(n, c)
        input_prob = self.logsoftmax_func(input)
        q_mask = torch.masked_select(q, mask.view(b, 1)).view(n, c)
        print('target_maked_size')
        print(n)
        print(max_prob)
        w_avg = weight.view(1, -1) / torch.sum(weight) * torch.mean(weight)
        loss = -torch.sum(torch.sum(w_avg * q_mask * input_prob, dim=1), dim=0) / n  # add weight

        return loss


class ConfidenceBasedCE_scan(nn.Module):
    def __init__(self, ct1, ct2, threshold, apply_class_balancing):
        super(ConfidenceBasedCE_scan, self).__init__()
        self.loss = MaskedCrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax_func = nn.LogSoftmax(dim=1)
        self.threshold = threshold
        self.ct1 = ct1
        self.ct2 = ct2
        self.apply_class_balancing = apply_class_balancing
        self.tau = None

    def forward(self, anchors_weak, anchors_strong, neighbors, eta, epoch):
        """
        Loss function during self-labeling

        input: logits for original samples and for its strong augmentations
        output: cross entropy
        """
        # Retrieve target and mask based on weakly augmentated anchors
        weak_anchors_prob = self.softmax(anchors_weak)
        max_prob, target = torch.max(weak_anchors_prob, dim=1)
        mask = max_prob > self.threshold
        b, c = weak_anchors_prob.size()
        target_masked = torch.masked_select(target, mask.squeeze())
        n = target_masked.size(0)

        # Inputs are strongly augmented anchors
        input_ = anchors_strong

        # Class balancing weights
        if self.apply_class_balancing:
            idx, counts = torch.unique(target_masked, return_counts=True)
            weight = torch.ones(c).cuda()
            freq = counts.float() / n
            h = 1.02
            weight[idx] = 1 / torch.log(h + freq)

        else:
            weight = None

        # Loss
        loss = self.loss(input_, target, mask, weight=weight, reduction='mean')

        return loss, n


def entropy(x, input_as_probabilities):
    """ 
    Helper function to compute the entropy over the batch 

    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """

    if input_as_probabilities:
        x_ = torch.clamp(x, min=EPS)
        b = x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)

    if len(b.size()) == 2:  # Sample-wise entropy
        return -b.sum(dim=1).mean()
    elif len(b.size()) == 1:  # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' % (len(b.size())))


class SCANLoss(nn.Module):
    def __init__(self, wo1, wo2, wo3, t, u, num_classes, entropy_weight=5.0, weight_t=1.0, alpha=0.1):
        super(SCANLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.bce = None
        self.entropy_weight = entropy_weight  # Default = 2.0
        self.class_num = num_classes
        self.tau_c = t

        self.mask = self.mask_correlated_clusters(self.class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.u = u
        self.temperature = 0.5

        self.wo1 = wo1
        self.wo2 = wo2
        self.wo3 = wo3

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def cluster_contrastive_loss(self, c_i, c_j):
        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        contrastive_loss = self.criterion(logits, labels)
        contrastive_loss /= N
        return contrastive_loss

    def forward(self, anchors, neighbors, epoch, epoch_num):
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]

        output:
            - Loss
        """
        print('SCANLoss forward: ')
        # print('anchors: {}'.format(anchors))
        # print('neighbors.size: {}'.format(neighbors.size()))
        # Softmax
        print('Softmax: ')
        b, c = anchors.size()
        anchors_prob = self.softmax(anchors)
        positives_prob = self.softmax(neighbors)
        print('anchors_prob.size: {}'.format(anchors_prob.size()))
        print('anchors_prob')
        print(anchors_prob)
        print('positives_prob.size: {}'.format(positives_prob.size()))

        # w_ij
        print('calculate w_ij: ')
        # 标准化
        anchors_nl = anchors / torch.norm(anchors, dim=1).view(-1, 1).expand_as(anchors)
        neighbors_nl = neighbors / torch.norm(neighbors, dim=1).view(-1, 1).expand_as(neighbors)
        # 计算w
        w = torch.exp(-torch.norm(anchors_nl - neighbors_nl, dim=1) ** 2 / self.tau_c)
        print('w.size: {}'.format(w.size()))

        # Similarity in output space
        # 计算anchor和与之对应的一个neighbor的相似度
        print('calculate similarity: ')
        similarity = torch.bmm(anchors_prob.view(b, 1, c), positives_prob.view(b, c, 1)).squeeze()
        # print('前十个similarity: {}'.format(similarity[:10]))

        if epoch < 15:
            consistency_loss = -torch.sum(torch.log(similarity), dim=0) / b
        else:
            consistency_loss = -torch.sum(w * torch.log(similarity), dim=0) / b  # add weight

        # cluster contrastive loss
        contrastive_loss = self.cluster_contrastive_loss(anchors_prob, positives_prob)

        # Entropy loss
        entropy_loss = entropy(torch.mean(anchors_prob, 0), input_as_probabilities=True)

        # Total loss

        total_loss = self.wo1 * consistency_loss + self.wo2 * self.u * contrastive_loss - self.wo3 * self.entropy_weight * entropy_loss

        return total_loss, self.wo1 * consistency_loss, self.wo2 * self.u * contrastive_loss, self.wo3 * entropy_loss


class SimCLRLoss(nn.Module):
    # Based on the implementation of SupContrast
    def __init__(self, temperature):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features):
        """
        input:
            - features: hidden feature representation of shape [b, 2, dim]

        output:
            - loss: loss computed according to SimCLR 
        """

        b, n, dim = features.size()
        assert (n == 2)
        mask = torch.eye(b, dtype=torch.float32).cuda()

        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor = features[:, 0]

        # Dot product
        dot_product = torch.matmul(anchor, contrast_features.T) / self.temperature

        # Log-sum trick for numerical stability
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()

        mask = mask.repeat(1, 2)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(b).view(-1, 1).cuda(), 0)
        mask = mask * logits_mask

        # Log-softmax
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Mean log-likelihood for positive
        loss = - ((mask * log_prob).sum(1) / mask.sum(1)).mean()

        return loss

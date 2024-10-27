"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import numpy
import torch
import numpy as np
from utils.utils import AverageMeter, ProgressMeter, get_distant_neighbors


def simclr_train(train_loader, model, criterion, optimizer, epoch):
    """ 
    Train according to the scheme from SimCLR
    https://arxiv.org/abs/2002.05709
    """
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader),
                             [losses],
                             prefix="Epoch: [{}]".format(epoch))

    model.train()

    for i, batch in enumerate(train_loader):
        images = batch['image']
        images_augmented = batch['image_augmented']
        b, c, h, w = images.size()
        input_ = torch.cat([images.unsqueeze(1), images_augmented.unsqueeze(1)], dim=1)
        input_ = input_.view(-1, c, h, w)
        input_ = input_.cuda(non_blocking=True)
        targets = batch['target'].cuda(non_blocking=True)

        output = model(input_).view(b, 2, -1)
        loss = criterion(output)
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 25 == 0:
            progress.display(i)


def scan_train(train_loader, model, criterion, optimizer, epoch, epoch_num, update_cluster_head_only=False,
               writer=None):
    """
    Train w/ SCAN-Loss
    """
    total_losses = AverageMeter('Total Loss', ':.4e')
    consistency_losses = AverageMeter('Consistency Loss', ':.4e')
    contrastive_losses = AverageMeter('Contrastive Loss', ':.4e')
    entropy_losses = AverageMeter('Entropy', ':.4e')
    progress = ProgressMeter(len(train_loader),
                             [total_losses, consistency_losses, contrastive_losses, entropy_losses],
                             prefix="Epoch: [{}]".format(epoch))



    if update_cluster_head_only:
        model.eval()  # No need to update BN
    else:
        model.train()  # Update BN

    cl = []
    cl2 = []
    tl = []

    for i, batch in enumerate(train_loader):
        # Forward pass
        anchors = torch.FloatTensor(batch['anchor']).cuda(non_blocking=True)
        neighbors = torch.FloatTensor(batch['neighbor']).cuda(non_blocking=True)

        # network output
        if update_cluster_head_only:  # Only calculate gradient for backprop of linear layer
            with torch.no_grad():
                anchors_features = model(anchors, forward_pass='backbone')
                neighbors_features = model(neighbors, forward_pass='backbone')
            anchors_output = model(anchors_features, forward_pass='head')
            neighbors_output = model(neighbors_features, forward_pass='head')

        else:  # Calculate gradient for backprop of complete network
            anchors_output = model(anchors)
            neighbors_output = model(neighbors)

        print('anchors_outputs: {}'.format(len(anchors_output)))
        print('neighbors_output: {}'.format(len(neighbors_output)))

        # Loss for every head
        total_loss, consistency_loss, contrastive_loss, entropy_loss = [], [], [], []
        head_count = 1
        for anchors_output_subhead, neighbors_output_subhead in zip(anchors_output, neighbors_output):
            print('Loss for head:{}'.format(head_count))
            total_loss_, consistency_loss_, contrastive_loss_, entropy_loss_ = criterion(anchors_output_subhead,
                                                                                         neighbors_output_subhead,
                                                                                         epoch, epoch_num)
            total_loss.append(total_loss_)
            consistency_loss.append(consistency_loss_)
            contrastive_loss.append(contrastive_loss_)
            entropy_loss.append(entropy_loss_)
            head_count += 1

        # Register the mean loss and backprop the total loss to cover all subheads
        total_losses.update(np.mean([v.item() for v in total_loss]))
        consistency_losses.update(np.mean([v.item() for v in consistency_loss]))
        contrastive_losses.update(np.mean([v.item() for v in contrastive_loss]))
        entropy_losses.update(np.mean([v.item() for v in entropy_loss]))

        total_loss = torch.sum(torch.stack(total_loss, dim=0))
        writer.add_scalar('total_loss', total_loss, global_step=epoch)
        writer.add_scalar('consistency_loss', torch.sum(torch.stack(consistency_loss, dim=0)), global_step=epoch)
        writer.add_scalar('contrastive_loss', torch.sum(torch.stack(contrastive_loss, dim=0)), global_step=epoch)
        writer.add_scalar('entropy_loss', torch.sum(torch.stack(entropy_loss, dim=0)), global_step=epoch)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i == len(train_loader) - 1:
            progress.display(i)

        tl.append(round(total_loss.item(), 3))
        cl.append(round(np.mean([v.item() for v in consistency_loss]), 3))
        cl2.append(round(np.mean([v.item() for v in contrastive_loss]), 3))

    f_total_loss = round(np.mean([v for v in tl]), 3)
    f_consistency_loss = round(np.mean([v for v in cl]), 3)
    f_contrastive_loss = round(np.mean([v for v in cl2]), 3)

    final_loss = [f_total_loss, f_consistency_loss, f_contrastive_loss]

    return final_loss


def selflabel_train(train_loader, model, criterion, optimizer, epoch, epoch_num, eta, ema=None):
    """
    Self-labeling based on confident samples
    """
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader), [losses],
                             prefix="Epoch: [{}]".format(epoch + 1))

    from utils.utils_fish import get_features_train
    outputs, targets = get_features_train(train_loader, model)

    model.train()

    tl = []
    sum = 0
    for i, batch in enumerate(train_loader):
        tss = torch.FloatTensor(batch['ts']).cuda(non_blocking=True)
        tss_augmented = torch.FloatTensor(batch['ts_augmented']).cuda(non_blocking=True)

        neighbors_indices = batch['anchor_neighbors_indices']
        neighbors_output = outputs[neighbors_indices].cuda(non_blocking=True)

        with torch.no_grad():
            output = model(tss)[0]

        output_augmented = model(tss_augmented)[0]

        print(output.shape)

        loss = criterion(output, output_augmented, neighbors_output, eta, epoch)
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ema is not None:  # Apply EMA to update the weights of the network
            ema.update_params(model)
            ema.apply_shadow(model)

        if i == (len(train_loader) - 1):
            progress.display(i + 1)

        tl.append(round(loss.item(), 3))

    f_total_loss = round(np.mean([v for v in tl]), 3)
    final_loss = [f_total_loss]

    return final_loss

# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
import time
from datetime import datetime
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import torch.distributed as dist

from torch.utils.data import DataLoader
from qfa.imagenet_codebase.utils import cross_entropy_loss_with_soft_target


from qfa.cifar100_codebase.conf import settings
from qfa.cifar100_codebase.utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights,build_sub_train_loader

from qfa.elastic_nn.networks import qresnet34
from qfa.elastic_nn.utils import set_activation_statistics, set_running_statistics

def read_setting(id):
    file = open(f'exp/bits_settings/sample{id}.txt')
    bits_setting = file.readline().split()
    bits_setting = [int(i) for i in bits_setting]

def train(net, epoch):

    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        if args.gpu:
            labels = labels.to(device)
            images = images.to(device)
        
        soft_logits = teacher(images)
        soft_label = torch.nn.functional.softmax(soft_logits, dim=1)

        optimizer.zero_grad()

        outputs = net(images)
        loss = cross_entropy_loss_with_soft_target(outputs, soft_label)
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        total = labels.size(0)
        correct = predicted.eq(labels).sum().item()
        if local_rank==1:
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}\tacc: {acc}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * args.b + len(images),
                total_samples=len(cifar100_training_loader.dataset),
                acc = 100.*correct/total
            ))

    finish = time.time()
    if local_rank==1:
        print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(net, epoch=0):

    net.eval()
 
    test_loss = 0.0 # cost function error
    correct = 0.0

    sub_train_loader = build_sub_train_loader(cifar100_training_loader)
    set_running_statistics(net, sub_train_loader)
    for (images, labels) in cifar100_test_loader:

        if args.gpu:
            images = images.to(device)
            labels = labels.to(device)

        
        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
    


    file = open(f'exp/bits_settings/sample{local_rank+args.base}_scratch.txt', 'a+')
    file.write(f'{epoch} {test_loss} {correct.float() / len(cifar100_test_loader.dataset)}\n')
    file.close()

    return correct.float() / len(cifar100_test_loader.dataset)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str,default='resnet34',  help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=256, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('--base', type=int, default=0)
    parser.add_argument('--local_rank', default=-1)
    parser.add_argument('--eval_freq', type = int, default=20)
    args = parser.parse_args()
    
    local_rank = int(args.local_rank)
    dist.init_process_group(backend='nccl')
    read_setting(local_rank+args.base)

    teacher = get_network(args)
    teacher.load_state_dict(torch.load('resnet34_full.pth'))

    net_sample = qresnet34([2,3,4,8])
    net_scratch = qresnet34([2,3,4,8])
    #net_sample.load_state_dict(torch.load('resnet34_quant.pth'))
    #net_scratch.load_state_dict(torch.load('resnet34_quant.pth'))
    seed = int('%d%.3d%.3d' % (local_rank, datetime.now().timetuple().tm_sec, datetime.now().timetuple().tm_min))
    random.seed(seed)
    bits_setting = read_setting(local_rank+args.base)
    
    net_scratch.set_active_subnet(bits_setting)
    #file = open(f'exp/bits_settings/sample{local_rank+args.base}.txt', 'a+')
    #file.write(" ".join([str(i) for i in bits_setting])+'\n')
    #file.close()
    device = torch.device("cuda", local_rank)
    #net_sample.to(device)
    net_scratch.to(device)
    teacher.to(device)
    teacher.eval()
    set_activation_statistics(net_sample)
    set_activation_statistics(net_scratch)

    #data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net_scratch.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    #eval_training(net=net_sample)
    
    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)
        train(net_scratch,epoch)
        if epoch % args.eval_freq == 0:
            acc = eval_training(net_scratch, epoch)

    

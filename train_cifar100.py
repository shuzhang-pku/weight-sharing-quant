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
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import DataLoader
from qfa.imagenet_codebase.utils import cross_entropy_loss_with_soft_target


from qfa.cifar100_codebase.conf import settings
from qfa.cifar100_codebase.utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights,build_sub_train_loader

from qfa.elastic_nn.networks import qresnet34
from qfa.elastic_nn.utils import set_activation_statistics, set_running_statistics

def train(epoch):

    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        if args.gpu:
            labels = labels.to(device)
            images = images.to(device)
        

        optimizer.zero_grad()

        import pdb; pdb.set_trace()

        #uniform
        net.module.set_sandwich_subnet(fix_bit=max(net.module.bits_list))
        outputs = net(images)
        loss = loss_function(outputs,labels)*3
        soft_label = torch.nn.functional.softmax(outputs, dim=1).clone().detach()
        loss.backward(retain_graph=True)


        net.module.set_sandwich_subnet(fix_bit=min(net.module.bits_list))
        outputs = net(images)
        if epoch>settings.MILESTONES[0]:
            loss = cross_entropy_loss_with_soft_target(outputs, soft_label)
        else:
            loss = loss_function(outputs, labels)
        loss.backward(retain_graph=True)

        #random
        subnet_seed = int('%d%.3d%.3d' % (epoch * batch_index, batch_index, 0))
        random.seed(subnet_seed)
        net.module.sample_active_subnet(subnet_seed=subnet_seed)
        outputs = net(images)
        if epoch>settings.MILESTONES[0]:
            loss = cross_entropy_loss_with_soft_target(outputs, soft_label)
        else:
            loss = loss_function(outputs, labels)
        loss.backward()
        

        torch.nn.utils.clip_grad_norm_(net.parameters(), 500)
        optimizer.step()



        _, predicted = outputs.max(1)
        total = labels.size(0)
        correct = predicted.eq(labels).sum().item()
        if local_rank==0:
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}\tacc: {acc}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * args.b + len(images),
                total_samples=len(cifar100_training_loader.dataset),
                acc = 100.*correct/total
            ))

        #update training loss for each iteration
   

        if epoch <= args.warm:
            warmup_scheduler.step()



    finish = time.time()
    if local_rank==0:
        print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(epoch=0, tb=True):

    net.eval()
    net.module.eval()
 
    test_loss = 0.0 # cost function error
    correct = 0.0

    sub_train_loader = build_sub_train_loader(cifar100_training_loader)
    net.module.set_active_subnet(b=8)
    set_running_statistics(net.module, sub_train_loader)
    for (images, labels) in cifar100_test_loader:

        if args.gpu:
            images = images.to(device)
            labels = labels.to(device)

        
        outputs = net.module(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
    


    file = open('result_int8_parallel.txt', 'a+')
    file.write(f'{epoch} {test_loss} {correct.float() / len(cifar100_test_loader.dataset)}\n')
    file.close()

    
    test_loss = 0.0 # cost function error
    correct = 0.0
    
    net.module.set_active_subnet(b=2)
    set_running_statistics(net.module, sub_train_loader)
    for (images, labels) in cifar100_test_loader:

        if args.gpu:
            images = images.to(device)
            labels = labels.to(device)

        
        outputs = net.module(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    file = open('result_int2_parallel.txt', 'a+')
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
    parser.add_argument('--local_rank', default=-1)
    args = parser.parse_args()
    
    local_rank = int(args.local_rank)
    dist.init_process_group(backend='nccl')

    net = qresnet34([2,3,4,8])
    device = torch.device("cuda", local_rank)
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net).to(device)
    net = DDP(net, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
    set_activation_statistics(net)

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
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.4) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)

    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)


    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))


    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)
        if args.resume:
            if epoch <= resume_epoch:
                continue
        dist.barrier()
        train(epoch)
        if local_rank==0 and epoch % 5 == 0:
            acc = eval_training(epoch)
        #start to save best performance model after learning rate decay to 0.01
            if epoch > settings.MILESTONES[2] and best_acc < acc:

                weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
                print('saving weights file to {}'.format(weights_path))
                torch.save(net.module.state_dict(), weights_path)
                best_acc = acc
                continue

            if not epoch % settings.SAVE_EPOCH:

                weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
                print('saving weights file to {}'.format(weights_path))
                torch.save(net.module.state_dict(), weights_path)


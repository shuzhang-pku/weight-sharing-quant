import argparse
import numpy as np
import time
import os
import random

import horovod.torch as hvd
import torch
from qfa.elastic_nn.networks import QResNet, qresnet50
from qfa.imagenet_codebase.run_manager import DistributedImageNetRunConfig
from qfa.imagenet_codebase.run_manager.distributed_run_manager import DistributedRunManager
from qfa.imagenet_codebase.data_providers.base_provider import LoaderConfig
from qfa.elastic_nn.training.progressive_shrinking import load_models
from qfa.elastic_nn.utils import set_activation_statistics

import torchvision
from torchvision import transforms
import torch.optim as optim
from torchvision.models import resnet50
import torch.nn as nn
from qfa.imagenet_codebase.utils import cross_entropy_loss_with_soft_target,cross_entropy_with_label_smoothing

def get_args(phase):
    args = argparse.Namespace()
    args.phase = phase

    args.base_batch_size = 200

    args.opt_type = 'sgd'
    args.momentum = 0.9
    args.no_nesterov = False
    args.weight_decay = 3e-5
    args.label_smoothing = 0.1
    args.no_decay_keys = 'bn#bias#gamma#beta#w_quantizer#a_quantizer#scale#offset'
    args.fp16_allreduce = False

    args.model_init = 'he_fout'
    args.validation_frequency = 1
    args.print_frequency = 10

    args.n_worker = 2
    args.resize_scale = 0.08
    args.distort_color = 'tf'
    args.continuous_size = True
    args.not_sync_distributed_image_size = False

    args.bn_momentum = 0.1
    args.bn_eps = 1e-5
    args.dropout = 0.1

    args.independent_distributed_sampling = False

    args.kd_ratio = 1.0
    args.kd_type = 'ce'

    args.sandwich = True

    args.path = os.path.join('.', 'exp/phase%d' % args.phase)
    if args.sandwich:
        args.dynamic_batch_size = args.total_dynamic_batch_size - 2
    else:
        args.dynamic_batch_size = args.total_dynamic_batch_size
    if args.phase == 0:
        args.model_path = None
        args.n_epochs = 60
        args.base_lr = 2.5e-3
        args.warmup_epochs = 5
        args.warmup_lr = -1
        args.ks_list = '3,5,7'
        args.expand_list = '3,4,6'
        args.depth_list = '2,3,4'
        args.bits_list = '2,3,4,32'
    elif args.phase == 1:
        args.model_path = os.path.join('.', 'exp/phase0/checkpoint/%s/quantize.pth.tar' % str(hvd.rank()))
        args.n_epochs = 120
        args.base_lr = 2.5e-3
        args.warmup_epochs = 5
        args.warmup_lr = -1
        args.ks_list = '3,5,7'
        args.expand_list = '3,4,6'
        args.depth_list = '2,3,4'
        args.bits_list = '2,3,4'

    args.interpolation = 2

    return args


transform = transforms.Compose([
transforms.Resize(224),
transforms.ToTensor(),
transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])


trainset = torchvision.datasets.CIFAR100(root='/home/shuzhangzhong/dataset', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
testset = torchvision.datasets.CIFAR100(root='/home/shuzhangzhong/dataset', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

net = qresnet50()
teacher = resnet50()
fc_inputs = teacher.fc.in_features 
teacher.fc = nn.Sequential(        
    nn.Linear(fc_inputs, 100), 
    nn.LogSoftmax(dim=1)
)
net = net.to('cuda')
#teacher = teacher.to('cuda')
set_activation_statistics(net)
net.load_state_dict(torch.load('resnet50.pth'))
teacher.load_state_dict(torch.load('/home/shuzhangzhong/weight-sharing-quant/resnet50.pth'))
teacher.eval()

optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

def train(epoch):
    print(epoch)
    net.train()
    teacher.eval()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        optimizer.zero_grad()
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        #with torch.no_grad():
        #    soft_label = teacher(inputs)
        output = net(inputs) 
        loss = nn.CrossEntropyLoss()(output,targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = output.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        print(f'epoch:{epoch}',batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    with open('info', 'a') as file_object:
        file_object.write(f"epoch{epoch}, acc{100.*correct/total}\n")



def main(phase):

    for epoch in range(20):

        net.set_sandwich_subnet()
        train(epoch)

        net.sample_active_subnet()
        train(epoch)
    torch.save(net.state_dict(),f'qresnetq.pth')

if __name__ == '__main__':
    main(0)

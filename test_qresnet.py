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
from qfa.elastic_nn.utils import set_activation_statistics,set_running_statistics

import torchvision
from torchvision import transforms
import torch.optim as optim
from torchvision.models import resnet50
import torch.nn as nn
from qfa.imagenet_codebase.utils import cross_entropy_loss_with_soft_target,cross_entropy_with_label_smoothing


transform = transforms.Compose([
transforms.Resize(224),
transforms.ToTensor(),
transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
])


trainset = torchvision.datasets.CIFAR100(root='/home/shuzhangzhong/dataset', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=200, shuffle=True)
testset = torchvision.datasets.CIFAR100(root='/home/shuzhangzhong/dataset', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False)

net = qresnet50()
net = net.to('cuda')




criterion = nn.CrossEntropyLoss()

def test():
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))




net.set_active_subnet(b=32)
set_activation_statistics(net)
net.load_state_dict(torch.load('qresnetq.pth'))
set_running_statistics(net, testloader)
'''
2:62.26%
3:66.94%
4:68.31%
32:67.91%
'''
test()


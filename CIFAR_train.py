# Copyright 2017 Queequeg92.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils

from datetime import datetime
import numpy as np

import os
import sys
import time
import argparse

import models
from torch.autograd import Variable

from utils import mean_cifar10, std_cifar10, mean_cifar100, std_cifar100
from utils import AverageMeter

model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__")
                     and callable(models.__dict__[name]))
#parser.add_argument add parameters
parser = argparse.ArgumentParser(description='PyTorch CIFAR Classification Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='ResNet164',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: ResNet164)')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=128, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--block_number', default=4, type=int,
                    help='number of blocks in each stage')
parser.add_argument('--filter', default=80, type=int,
                    help='number of kerneal in each block')
parser.add_argument('--Group_number', default=4, type=int,
                    help='number of blocks in each stage')
parser.add_argument('--group_number', default=2, type=int,
                    help='number of kerneal in each block')
parser.add_argument('--lr', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--lr_schedule', default=0, type=int,
                    help='learning rate schedule to apply')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, action='store_true', help='nesterov momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--resume', default=False, action='store_true', help='resume from checkpoint')
parser.add_argument('--ckpt_path', default='', type=str, metavar='PATH',
                    help='path to checkpoint (default: none)')



def main():
    global args
    args = parser.parse_args()

    # Data preprocessing.
    print('==> Preparing data......')
    assert (args.dataset == 'cifar10' or args.dataset == 'cifar100'), "Only support cifar10 or cifar100 dataset"
    if args.dataset == 'cifar10':
        print('To train and eval on cifar10 dataset......')
        num_classes = 10
        is_cifar10=True
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            #transform the valuesof pixels [0,255] to the range[0.0,1.0] 
            transforms.ToTensor(),
            #normalize the Tensor with channel=(channel-mean)/std
            transforms.Normalize(mean_cifar10, std_cifar10),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean_cifar10, std_cifar10),
        ])
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=4)
    else:
        print('To train and eval on cifar100 dataset......')
        num_classes = 100
        is_cifar10=False
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean_cifar100, std_cifar100),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean_cifar100, std_cifar100),
        ])
        train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

        test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=4)

    # args.resume=True
    # Model
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(args.ckpt_path), 'Error: checkpoint directory not exists!'
        checkpoint = torch.load(os.path.join(args.ckpt_path,'ckpt.t7'))
        model = checkpoint['model']
        best_acc = checkpoint['best_acc']
        print (best_acc)
        start_epoch = checkpoint['epoch']
        print (start_epoch)
    else:
        print('==> Building model..')
        model = models.__dict__[args.arch](num_classes,args.block_number,args.filter,args.Group_number,args.group_number)
        start_epoch = args.start_epoch
        

    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # Use GPUs if available.
    if torch.cuda.is_available():
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    # Define loss function and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
   	                  momentum=args.momentum,
                          nesterov=args.nesterov,
                          weight_decay=args.weight_decay)

    log_dir = 'logs/' + str(args.block_number)+"_"+str(args.filter)+"_"+str(args.Group_number)+"_"+str(args.group_number)+"_"+args.dataset+"_SRGC-Nets"
    print('block_number:' + str(args.block_number)+"_filter:"+str(args.filter)+"_Group:"+str(args.Group_number)+"_group:"+str(args.group_number)+"_SRGC-Nets")
    
    if not os.path.isdir(log_dir):
            os.mkdir(log_dir)

    best_acc = 0  # best test accuracy

    for epoch in range(start_epoch, args.epochs):
        # Learning rate schedule.
        lr = adjust_learning_rate(optimizer, epoch + 1)

        # Train for one epoch.
        losses_avg, acces_avg=train(train_loader, model, criterion, optimizer , epoch)

        # Eval on test set.
        num_iter = (epoch + 1) * len(train_loader)
        #
        losses_avg,acc = eval(test_loader, model, criterion, epoch, num_iter)
  
        # Save checkpoint.
        print('Saving Checkpoint......')
        
	if torch.cuda.is_available():	
	    state = {
            'model': model,
            'best_acc': best_acc,
            'epoch': epoch,
            }
	else:
	 state = {
            'model': model,
            'best_acc': best_acc,
            'epoch': epoch,
            }
        if not os.path.isdir(os.path.join(log_dir, 'last_ckpt')):
            os.mkdir(os.path.join(log_dir, 'last_ckpt'))
            torch.save(state, os.path.join(log_dir, 'last_ckpt', 'ckpt.t7'))
        if acc > best_acc:
            best_acc = acc
            if not os.path.isdir(os.path.join(log_dir ,'best_ckpt')):
                os.mkdir(os.path.join(log_dir, 'best_ckpt'))
            torch.save(state, os.path.join(log_dir ,'best_ckpt', 'ckpt.t7'))
       
    print (best_acc)
   

def adjust_learning_rate(optimizer, epoch):
    if args.lr_schedule == 0:
        lr = args.lr * ((0.2 ** int(epoch >= 60)) * (0.2 ** int(epoch >= 120)) * (0.2 ** int(epoch >= 160)))
    elif args.lr_schedule == 1:
        lr = args.lr * ((0.1 ** int(epoch >= 150)) * (0.1 ** int(epoch >= 225)))
    elif args.lr_schedule == 2:
        lr = args.lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 120)))
    else:
        raise Exception("Invalid learning rate schedule!")
    
    for param_group in optimizer.param_groups:
        	param_group['lr'] = lr
    return lr


# Training
def train(train_loader, model, criterion, optimizer, epoch):
    print('\nEpoch: %d -> Training' % epoch)
    # Set to eval mode.
    model.train()
    sample_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    end = time.time()
    
    #each batch calculates the values of loss and gradient
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        num_iter = epoch * len(train_loader) + batch_idx
        
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)

        # Compute gradients and do back propagation.
        
        outputs = model(inputs)
        #crossEntrypyLoss Criterion
        #criterion(outputs, targets)loss(x, class) = weight[class] * (-x[class] + log(\sum_j exp(x[j])))
        loss = criterion(outputs, targets)
	
	loss.backward()
        #update the parameters after calcute the gradient with backward()
        optimizer.step()

        losses.update(loss.data[0]*inputs.size(0), inputs.size(0))  
	_, predicted = torch.max(outputs.data, 1)
        correct = predicted.eq(targets.data).cpu().sum()

        acces.update(correct, inputs.size(0))
        # measure elapsed time
        sample_time.update(time.time() - end, inputs.size(0))
        end = time.time()
        
    print('Loss: %.4f | Acc: %.4f%% (%d/%d)' % (losses.avg, 100. * acces.avg, acces.numerator, acces.denominator))
    return losses.avg, acces.avg
    
# Evaluating
def eval(test_loader, model, criterion,  epoch, num_iter):
    print('\nEpoch: %d -> Evaluating' % epoch)
    # Set to eval mode.
    model.eval()
    losses = AverageMeter()
    acces = AverageMeter()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = model(inputs)
        #calculate the losses
        loss = criterion(outputs, targets)

        losses.update(loss.data[0]*inputs.size(0), inputs.size(0))
        _, predicted = torch.max(outputs.data, 1)
        correct = predicted.eq(targets.data).cpu().sum()
        acces.update(correct, inputs.size(0))

        
    print('Loss: %.4f | Acc: %.4f%% (%d/%d)' % (losses.avg, 100. * acces.avg, acces.numerator, acces.denominator))
    
    return losses.avg,acces.avg


if __name__ == '__main__':
    main()



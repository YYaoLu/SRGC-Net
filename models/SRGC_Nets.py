
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.autograd import Variable
import math
import numpy as np
import time

def shuffle_group(x, G,g):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // (G*g)
    
    # reshape
    x = x.view(batchsize, G,g, channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class SRGC_module(nn.Module):
    def __init__(self,planes,kernel_size,stride,bias,Group_number=4,group_number=2):
        super(SRGC_module,self).__init__()
        self.Group_num=Group_number
        self.group_num=group_number
        self.all_groups=self.Group_num*self.group_num
        

        #----first SRGC------------------------------------------------------------------------------------------------------------
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv_group1=nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0, bias=False,groups=self.Group_num)
        self.RGC_1=nn.Conv2d(planes// self.group_num, planes// self.group_num, kernel_size=3, stride=1, padding=1, bias=False,groups=self.Group_num)
        self.PGC_1=nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0, bias=False,groups=self.Group_num)
	
        #----Second SRGC------------------------------------------------------------------------------------------------------------
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv_group2=nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0, bias=False,groups=self.group_num)
        self.RGC_2=nn.Conv2d(planes// self.Group_num, planes// self.Group_num, kernel_size=3, stride=stride, padding=1, bias=False,groups=self.group_num)
        self.PGC_2=nn.Conv2d(planes, planes, kernel_size=1, stride=stride, padding=0, bias=False,groups=self.group_num)
	
        
    def forward(self, x):

        #----first SRGC------------------------------------------------------------------------------------------------------------
        out=F.relu(self.bn1(x))
        out=self.conv_group1(out)
        PGC_1=self.PGC_1(out)
        out=shuffle_group(out, self.Group_num,self.group_num)
        rgc_1=torch.chunk(out,self.group_num,dim=1)
        RGC_1=list()
        for i in range(self.group_num):
            RGC_1.append(self.RGC_1(rgc_1[i]))
        RGC_1=torch.cat(RGC_1,1)
        RGC_1=shuffle_group(RGC_1, self.group_num,self.Group_num)
        SRGC_1=RGC_1+PGC_1

        #----shuffle group------------------------------------------------------------------------------------------------------------
        out=shuffle_group(SRGC_1, self.Group_num,self.group_num)

        #----second SRGC------------------------------------------------------------------------------------------------------------
        out=F.relu(self.bn2(out))
        out=self.conv_group2(out)
        PGC_2=self.PGC_2(out)
        out=shuffle_group(out, self.group_num,self.Group_num)
        rgc_2=torch.chunk(out,self.Group_num,dim=1)
        RGC_2=list()
        for i in range(self.Group_num):
            RGC_2.append(self.RGC_2(rgc_2[i]))
        RGC_2=torch.cat(RGC_2,1)
        RGC_2=shuffle_group(RGC_2, self.Group_num,self.group_num)
        SRGC_2=RGC_2+PGC_2
        return SRGC_2


class SRGC_Block(nn.Module):
    def __init__(self, in_planes, planes, stride=1,Group_number=4,group_number=2):
        super(SRGC_Block, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.point_wise_conv = nn.Conv2d(in_planes,planes,kernel_size=1,stride=1,padding=0,bias=False)
        self.SRGC_module=SRGC_module(planes,kernel_size=3,stride=stride,bias=False,Group_number=Group_number,group_number=group_number)
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self,x):

        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out=self.point_wise_conv(out)
        out=self.SRGC_module(out)
        out+=shortcut
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, filters, num_classes=10,Group_number=4,group_number=2):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.conv1 = conv3x3(3, self.in_planes)
        self.stage1 = self._make_layer(block, filters[0], num_blocks[0], stride=1,Group_number=Group_number,group_number=group_number)
	self.stage2 = self._make_layer(block, filters[1], num_blocks[1], stride=2,Group_number=Group_number,group_number=group_number)
	self.stage3 = self._make_layer(block, filters[2], num_blocks[2], stride=2,Group_number=Group_number,group_number=group_number)
        self.bn = nn.BatchNorm2d(filters[2])
        self.linear = nn.Linear(filters[2], num_classes)
       	
        for m in self.modules():
            #
            if isinstance(m, nn.Conv2d):
                
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride,Group_number,group_number):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, Group_number,group_number))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)	
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = F.relu(self.bn(out))
	out = F.avg_pool2d(out, 8)        
	out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def SRGC_Nets(num_classes=10,block_number=4,filters=80,Group_number=4,group_number=2):
    return ResNet(SRGC_Block, [block_number, block_number, block_number], [filters, filters*2, filters*4], num_classes,Group_number,group_number)


if __name__ == "__main__":
    model = SRGC_Nets(num_classes=10,block_number=4,filters=112,Group_number=4,group_number=4)
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    count=0
    for i in range(100):
        
        x = torch.autograd.Variable(torch.rand(1,3, 32, 32))
        t1 = time.time()
        out = model(x)
        cnt = time.time() - t1
        count+=cnt
    print(count)


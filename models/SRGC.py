
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

from torch.autograd import Variable

import math
#import scipy.io
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class ShuffleBlock(nn.Module):
    def __init__(self,planes,kernel_size,stride,bias,Group_number=4,group_number=2):
        super(ShuffleBlock,self).__init__()
        self.Group_num=Group_number
        self.group_num=group_number
        self.all_groups=self.Group_num*self.group_num

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv_group2=nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0, bias=False,groups=self.Group_num)


        self.conv1_1=nn.Conv2d(planes// self.all_groups, planes// self.all_groups, kernel_size=3, stride=1, padding=1, bias=False)
        if(Group_number>=2):
            self.conv1_2=nn.Conv2d(planes// self.all_groups, planes// self.all_groups, kernel_size=3, stride=1, padding=1, bias=False)
        if(Group_number>=4):
            self.conv1_3=nn.Conv2d(planes// self.all_groups, planes// self.all_groups, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv1_4=nn.Conv2d(planes// self.all_groups, planes// self.all_groups, kernel_size=3, stride=1, padding=1, bias=False)
        if(Group_number>=8):
            self.conv1_5=nn.Conv2d(planes// self.all_groups, planes// self.all_groups, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv1_6=nn.Conv2d(planes// self.all_groups, planes// self.all_groups, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv1_7=nn.Conv2d(planes// self.all_groups, planes// self.all_groups, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv1_8=nn.Conv2d(planes// self.all_groups, planes// self.all_groups, kernel_size=3, stride=1, padding=1, bias=False)
        if(Group_number>=16):
            self.conv1_9=nn.Conv2d(planes// self.all_groups, planes// self.all_groups, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv1_10=nn.Conv2d(planes// self.all_groups, planes// self.all_groups, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv1_11=nn.Conv2d(planes// self.all_groups, planes// self.all_groups, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv1_12=nn.Conv2d(planes// self.all_groups, planes// self.all_groups, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv1_13=nn.Conv2d(planes// self.all_groups, planes// self.all_groups, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv1_14=nn.Conv2d(planes// self.all_groups, planes// self.all_groups, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv1_15=nn.Conv2d(planes// self.all_groups, planes// self.all_groups, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv1_16=nn.Conv2d(planes// self.all_groups, planes// self.all_groups, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv_group3=nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0, bias=False,groups=self.Group_num)
	

        self.bn3 = nn.BatchNorm2d(planes)
        self.conv_group4=nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0, bias=False,groups=self.group_num)
        self.bn4 = nn.BatchNorm2d(planes)
        #G
        self.conv2_1=nn.Conv2d(planes// self.all_groups, planes// self.all_groups, kernel_size=3, stride=stride, padding=1, bias=False)
        if(group_number>=2):
             self.conv2_2=nn.Conv2d(planes// self.all_groups, planes// self.all_groups, kernel_size=3, stride=stride, padding=1, bias=False)        
        if(group_number>=4):
             self.conv2_3=nn.Conv2d(planes// self.all_groups, planes// self.all_groups, kernel_size=3, stride=stride, padding=1, bias=False) 
             self.conv2_4=nn.Conv2d(planes// self.all_groups, planes// self.all_groups, kernel_size=3, stride=stride, padding=1, bias=False)
        if(group_number>=8):
             self.conv2_5=nn.Conv2d(planes// self.all_groups, planes// self.all_groups, kernel_size=3, stride=stride, padding=1, bias=False) 
             self.conv2_6=nn.Conv2d(planes// self.all_groups, planes// self.all_groups, kernel_size=3, stride=stride, padding=1, bias=False) 
             self.conv2_7=nn.Conv2d(planes// self.all_groups, planes// self.all_groups, kernel_size=3, stride=stride, padding=1, bias=False)
             self.conv2_8=nn.Conv2d(planes// self.all_groups, planes// self.all_groups, kernel_size=3, stride=stride, padding=1, bias=False)
        if(group_number>=16):
             self.conv2_9=nn.Conv2d(planes// self.all_groups, planes// self.all_groups, kernel_size=3, stride=stride, padding=1, bias=False) 
             self.conv2_10=nn.Conv2d(planes// self.all_groups, planes// self.all_groups, kernel_size=3, stride=stride, padding=1, bias=False) 
             self.conv2_11=nn.Conv2d(planes// self.all_groups, planes// self.all_groups, kernel_size=3, stride=stride, padding=1, bias=False)
             self.conv2_12=nn.Conv2d(planes// self.all_groups, planes// self.all_groups, kernel_size=3, stride=stride, padding=1, bias=False)
             self.conv2_13=nn.Conv2d(planes// self.all_groups, planes// self.all_groups, kernel_size=3, stride=stride, padding=1, bias=False) 
             self.conv2_14=nn.Conv2d(planes// self.all_groups, planes// self.all_groups, kernel_size=3, stride=stride, padding=1, bias=False) 
             self.conv2_15=nn.Conv2d(planes// self.all_groups, planes// self.all_groups, kernel_size=3, stride=stride, padding=1, bias=False)
             self.conv2_16=nn.Conv2d(planes// self.all_groups, planes// self.all_groups, kernel_size=3, stride=stride, padding=1, bias=False)
             

        self.conv_group5=nn.Conv2d(planes, planes, kernel_size=1, stride=stride, padding=0, bias=False,groups=self.group_num)

    def forward(self, x):
        #A
        #out=self.conv_group1(x)
        #B
        out=F.relu(self.bn2(x))
        out=self.conv_group2(out)

        #divide convolution results into Group_num*group_num groups
        c=torch.chunk(out,self.all_groups,dim=1)
        C=list()

        # Group_num groups convolution.
        for j in xrange(self.group_num):
            c_part=self.conv1_1(c[j])
            C.append(c_part)
        if(self.Group_num>=2):
          for j in xrange(self.group_num):
              c_part=self.conv1_2(c[self.group_num+j])
              C.append(c_part)
        if(self.Group_num>=4):
          for j in xrange(self.group_num):
              c_part=self.conv1_3(c[self.group_num*2+j])
              C.append(c_part)
          for j in xrange(self.group_num):
              c_part=self.conv1_4(c[self.group_num*3+j])
              C.append(c_part)
        if(self.Group_num>=8):
          for j in xrange(self.group_num):
              c_part=self.conv1_5(c[self.group_num*4+j])
              C.append(c_part)
          for j in xrange(self.group_num):
              c_part=self.conv1_6(c[self.group_num*5+j])
              C.append(c_part)
          for j in xrange(self.group_num):
              c_part=self.conv1_7(c[self.group_num*6+j])
              C.append(c_part)
          for j in xrange(self.group_num):
              c_part=self.conv1_8(c[self.group_num*7+j])
              C.append(c_part)
        if(self.Group_num>=16):
          for j in xrange(self.group_num):
              c_part=self.conv1_9(c[self.group_num*8+j])
              C.append(c_part)
          for j in xrange(self.group_num):
              c_part=self.conv1_10(c[self.group_num*9+j])
              C.append(c_part)
          for j in xrange(self.group_num):
              c_part=self.conv1_11(c[self.group_num*10+j])
              C.append(c_part)
          for j in xrange(self.group_num):
              c_part=self.conv1_12(c[self.group_num*11+j])
              C.append(c_part)
          for j in xrange(self.group_num):
              c_part=self.conv1_13(c[self.group_num*12+j])
              C.append(c_part)
          for j in xrange(self.group_num):
              c_part=self.conv1_14(c[self.group_num*13+j])
              C.append(c_part)
          for j in xrange(self.group_num):
              c_part=self.conv1_15(c[self.group_num*14+j])
              C.append(c_part)
          for j in xrange(self.group_num):
              c_part=self.conv1_16(c[self.group_num*15+j])
              C.append(c_part)

        #combine grouping convolution results
        c=torch.cat(C,1)


        d=self.conv_group3(out)

        out=c+d

        #shuffle groups
        e=torch.chunk(out,self.all_groups,dim=1)

        E=list()
        for i in range(self.group_num):
            for j in range(self.Group_num):
                e_part=e[self.group_num*j+i]
		    #print(self.group_num*j+i)
                E.append(e_part)
        
        out=torch.cat(E,1)
        if(out.size(1)/self.Group_num>=40):
	         out=F.relu(self.bn3(out))

        out=self.conv_group4(out)
        if(out.size(1)/self.group_num>=40):
	         out=F.relu(self.bn4(out))

        #divide convolution results into Group_num*group_num groups
        f=torch.chunk(out,self.all_groups,dim=1)

        # group_num groups convolution.
        g=list()
        for j in range(self.Group_num):
           g_part=self.conv2_1(f[j])
           g.append(g_part)
        if(self.group_num>=2):
	         for j in range(self.Group_num):
	           g_part=self.conv2_2(f[j+self.Group_num])
	           g.append(g_part)  
        if(self.group_num>=4):      
           for j in range(self.Group_num):
             g_part=self.conv2_3(f[j+self.Group_num*2])
             g.append(g_part)
           for j in range(self.Group_num):
             g_part=self.conv2_4(f[j+self.Group_num*3])
             g.append(g_part) 
        if(self.group_num>=8):      
           for j in range(self.Group_num):
             g_part=self.conv2_5(f[j+self.Group_num*4])
             g.append(g_part)
           for j in range(self.Group_num):
             g_part=self.conv2_6(f[j+self.Group_num*5])
             g.append(g_part)        
           for j in range(self.Group_num):
             g_part=self.conv2_7(f[j+self.Group_num*6])
             g.append(g_part)
           for j in range(self.Group_num):
             g_part=self.conv2_8(f[j+self.Group_num*7])
             g.append(g_part)  
        if(self.group_num>=16):      
           for j in range(self.Group_num):
             g_part=self.conv2_9(f[j++self.Group_num*8])
             g.append(g_part)
           for j in range(self.Group_num):
             g_part=self.conv2_10(f[j+self.Group_num*9])
             g.append(g_part)        
           for j in range(self.Group_num):
             g_part=self.conv2_11(f[j+self.Group_num*10])
             g.append(g_part)
           for j in range(self.Group_num):
	           g_part=self.conv2_12(f[j+self.Group_num*11])
	           g.append(g_part)  
           for j in range(self.Group_num):
             g_part=self.conv2_13(f[j+self.Group_num*12])
             g.append(g_part)
           for j in range(self.Group_num):
             g_part=self.conv2_14(f[j+self.Group_num*13])
             g.append(g_part)        
           for j in range(self.Group_num):
	           g_part=self.conv2_15(f[j+self.Group_num*14])
	           g.append(g_part)
           for j in range(self.Group_num):
             g_part=self.conv2_16(f[j+self.Group_num*15])
             g.append(g_part)

        #combine grouping convolution results
        G=torch.cat(g,1)
        #H
        H=self.conv_group5(out)
        out=H+G
        return out



class PreActShuffleBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1,numberoflayer=1,stage=1,Group_number=4,group_number=2):
        super(PreActShuffleBlock, self).__init__()
        self.numberoflayer = numberoflayer
        self.Group_num=Group_number
        self.group_num=group_number

        self.bn1 = nn.BatchNorm2d(in_planes)
         #A
        self.conv_group1 = nn.Conv2d(in_planes,planes,kernel_size=1,stride=1,padding=0,bias=False)
        self.shuffle=ShuffleBlock(planes,kernel_size=3,stride=stride,bias=False,Group_number=self.Group_num,group_number=self.group_num)

        self.stage = stage
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(#
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self,x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out=self.conv_group1(out)
        out=self.shuffle(out)
        out+=shortcut
        return out


class SRGCNet(nn.Module):
    def __init__(self, block, num_blocks, filters, num_classes=10,Group_number=4,group_number=2):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.numberoflayer=1
        self.conv1 = conv3x3(3, self.in_planes)
        self.stage1 = self._make_layer(block, filters[0], num_blocks[0], stride=1,stage=1,Group_number=Group_number,group_number=group_number)
        self.stage2 = self._make_layer(block, filters[1], num_blocks[1], stride=2,stage=2,Group_number=Group_number,group_number=group_number)
        self.stage3 = self._make_layer(block, filters[2], num_blocks[2], stride=2,stage=3,Group_number=Group_number,group_number=group_number)
	

        self.bn = nn.BatchNorm2d(filters[2]*block.expansion)
        self.linear = nn.Linear(filters[2]*block.expansion, num_classes)
       	
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride,stage,Group_number,group_number):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
	    #ours
            layers.append(block(self.in_planes, planes, stride,self.numberoflayer,stage,Group_number,group_number))
            self.in_planes = planes * block.expansion
            self.numberoflayer=self.numberoflayer%len(strides)+1
        return nn.Sequential(*layers)

    def forward(self, x,epoch):
        out = self.conv1(x)
	
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)

        out = F.relu(self.bn(out))

        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)

        out = self.linear(out)
        return out


def SRGC(num_classes=10,block_number=4,kernal_number=80,Group_number=4,group_number=4):
    return SRGCNet(PreActShuffleBlock, [block_number, block_number, block_number], [kernal_number, kernal_number*2, kernal_number*4], num_classes,Group_number,group_number)

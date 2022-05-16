'''ResNet in PyTorch.
'''
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable
from torch.nn.parameter import Parameter

from torchvision import datasets
from torchvision import transforms
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR

import os
from datetime import datetime 

import pdb

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class EncBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(EncBlock, self).__init__()
        padding = (kernel_size-1)//2
        self.layers = nn.Sequential()
        self.layers.add_module('Conv', nn.Conv2d(in_planes, out_planes, \
            kernel_size=kernel_size, stride=1, padding=padding, bias=False))
        self.layers.add_module('BatchNorm', nn.BatchNorm2d(out_planes))

    def forward(self, x):
        out = self.layers(x)
        return torch.cat([x,out], dim=1)

class GlobalAveragePooling(nn.Module):
    def __init__(self):
        super(GlobalAveragePooling, self).__init__()

    def forward(self, feat):
        num_channels = feat.size(1)
        return F.avg_pool2d(feat, (feat.size(2), feat.size(3))).view(-1, num_channels)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet, self).__init__()
        
        num_stages = 5 # for the no of blocks
        
        self.in_planes = 64

        blocks = [nn.Sequential() for i in range(num_stages)]

        blocks[0].add_module('Block1_Conv', conv3x3(3,64))
        blocks[0].add_module('Block1_BatchNorm', nn.BatchNorm2d(64))

        blocks[1].add_module('Block1_layer', self._make_layer(block, 64, num_blocks[0], stride=1))

        blocks[2].add_module('Block2_layer', self._make_layer(block, 128, num_blocks[1], stride=2))

        blocks[3].add_module('Block3_layer', self._make_layer(block, 256, num_blocks[2], stride=2))
        # Encoder
        blocks[3].add_module('Block3_Encode', EncBlock(256, 256, 1)) # 256 as out channels as decoder's input is 256-d

        # Decoder
        blocks[4].add_module('Block4_layer', self._make_layer(block, 512, num_blocks[3], stride=2))

        # global average pooling and classifier
        blocks.append(nn.Sequential())
        blocks[-1].add_module('GlobalAveragePooling',  GlobalAveragePooling())

        self._feature_blocks = nn.ModuleList(blocks)
        self.all_feat_names = ['conv'+str(s+1) for s in range(num_stages)] + ['classifier',]
        assert(len(self.all_feat_names) == len(self._feature_blocks))

    def _parse_out_keys_arg(self, out_feat_keys):

    	# By default return the features of the last layer / module.
        out_feat_keys = [self.all_feat_names[-1],] if out_feat_keys is None else out_feat_keys
        #print("out_feat_keys: ", out_feat_keys)

        if len(out_feat_keys) == 0:
            raise ValueError('Empty list of output feature keys.')
        for f, key in enumerate(out_feat_keys):
            if key not in self.all_feat_names:
                raise ValueError('Feature with name {0} does not exist. Existing features: {1}.'.format(key, self.all_feat_names))
            elif key in out_feat_keys[:f]:
                raise ValueError('Duplicate output feature key: {0}.'.format(key))

    	# Find the highest output feature in `out_feat_keys
        max_out_feat = max([self.all_feat_names.index(key) for key in out_feat_keys])
        
        return out_feat_keys, max_out_feat
    
    def forward(self, x, out_feat_keys=None):
        """Forward an image `x` through the network and return the asked output features.
    	Args:
    	  x: input image.
    	  out_feat_keys: a list/tuple with the feature names of the features
                that the function should return. By default the last feature of
                the network is returned.
    	Return:
            out_feats: If multiple output features were asked then `out_feats`
                is a list with the asked output features placed in the same
                order as in `out_feat_keys`. If a single output feature was
                asked then `out_feats` is that output feature (and not a list).
        """
        #pdb.set_trace()
        out_feat_keys, max_out_feat = self._parse_out_keys_arg(out_feat_keys)
        out_feats = [None] * len(out_feat_keys)
        feat = x
        #encode
        for f in range(4): # enc block's no. is 3 (0 indexing) 
            feat = self._feature_blocks[f](feat)
            key = self.all_feat_names[f]
            if key in out_feat_keys:
                out_feats[out_feat_keys.index(key)] = feat
      
        #reparameterize
        # feat = [mu(input to enc block), logvar(from enc block)]
        mu = feat[:,:256] 
        logvar = feat[:, 256:]
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        #print("std.shape: {}, mu.shape: {}".format(std.shape, mu.shape))
        feat = eps.mul(std * 0.001).add_(mu)
      
        #decode
        for f in range(4, max_out_feat+1): # dec block's no. is 4 (0 indexing)
            feat = self._feature_blocks[f](feat)
            key = self.all_feat_names[f]
            if key in out_feat_keys:
                out_feats[out_feat_keys.index(key)] = feat

        out_feats = out_feats[0] if len(out_feats)==1 else out_feats
        return out_feats

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
def ResNet18():
    print("In ResNet18()")
    return ResNet(PreActBlock, [2,2,2,2])

def ResNet34():
    print("In ResNet34()")
    return ResNet(BasicBlock, [3,4,6,3])

class Regressor(nn.Module):
    def __init__(self, resnet_type=34, indim=1024, num_classes=4): #indim = [orig_img_features,transformed_img_features], num_classes is the number of possible transformations
        super(Regressor, self).__init__()
        if resnet_type == 18:
            self.resnet = ResNet18()
        elif resnet_type == 34:
            self.resnet = ResNet34()
        else:
            raise Exception("Supported for only ResNet18 and ResNet34")

        fc1_outdim = 512

        self.fc1 = nn.Linear(indim, fc1_outdim)
        self.fc2 = nn.Linear(fc1_outdim, num_classes)
        self.relu1 = nn.ReLU(inplace=True)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()
            
    def forward(self, x1, x2, out_feat_keys=None):
        x1 = self.resnet(x1, out_feat_keys)
        x2 = self.resnet(x2, out_feat_keys)
        x = torch.cat((x1,x2), dim=1)
        
        x = self.fc1(x)
        x = self.relu1(x)
        penul_feat = x
        x = self.fc2(x)

        return penul_feat, x

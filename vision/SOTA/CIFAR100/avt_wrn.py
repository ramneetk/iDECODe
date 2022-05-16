'''WideResNet in PyTorch.
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

class GlobalAveragePooling(nn.Module):
    def __init__(self):
        super(GlobalAveragePooling, self).__init__()

    def forward(self, feat):
        num_channels = feat.size(1)
        return F.avg_pool2d(feat, (feat.size(2), feat.size(3))).view(-1, num_channels)

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        # self.bn1 = nn.GroupNorm(in_planes//2,in_planes)
        # self.bn1 = nn.Identity()
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        # self.bn2 = nn.GroupNorm(out_planes//2,out_planes)
        # self.bn2 = nn.Identity()
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        if self.equalInOut:
            out = self.relu2(self.bn2(self.conv1(out)))
        else:
            out = self.relu2(self.bn2(self.conv1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        if not self.equalInOut:
            return torch.add(self.convShortcut(x), out)
        else:
            return torch.add(x, out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class EncBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(EncBlock, self).__init__()
        padding = (kernel_size-1)//2
        self.layers = nn.Sequential()
        self.layers.add_module('Conv', nn.Conv2d(in_planes, out_planes, \
            kernel_size=kernel_size, stride=1, padding=padding, bias=False))
        self.layers.add_module('BatchNorm', nn.BatchNorm2d(out_planes))
        # self.layers.add_module('BatchNorm', nn.GroupNorm(out_planes//2,out_planes))
        # self.layers.add_module('Identity', nn.Identity())


    def forward(self, x):
        out = self.layers(x)
        return torch.cat([x,out], dim=1)

class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropRate):
        super(WideResNet, self).__init__()

        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.enc_dim = nChannels[3]

        assert ((depth - 4) % 6 == 0)

        n = (depth - 4) // 6
        block = BasicBlock

        self.num_stages = 4 # for no. of blocks

        blocks = [nn.Sequential() for i in range(self.num_stages)]

        # 1st conv before any network block
        blocks[0].add_module('Block1_Conv', nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False))
        # 1st Network block
        blocks[1].add_module('Block2_NetBlk', NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate))
        # 2nd Network block
        blocks[2].add_module('Block3_NetBlk', NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate))


        # 3rd Network block
        blocks[3].add_module('Block4_NetBlk', NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate))

        # Encoder block
        blocks[3].add_module('Block4_EncBlk', EncBlock(nChannels[3], nChannels[3], 1))

        # global average pooling and classifier
        # blocks.append(nn.Sequential())
        # blocks[-1].add_module('Block4_BN', nn.BatchNorm2d(nChannels[3]))
        # blocks[-1].add_module('Block4_Act', nn.ReLU(inplace=True))

        # global average pooling and classifier
        blocks.append(nn.Sequential())
        blocks[-1].add_module('GlobalAveragePooling',  GlobalAveragePooling())

        self._feature_blocks = nn.ModuleList(blocks)
        self.all_feat_names = ['conv'+str(s+1) for s in range(self.num_stages)] + ['classifier',]
        # self.all_feat_names = ['conv'+str(s+1) for s in range(self.num_stages)]
        assert(len(self.all_feat_names) == len(self._feature_blocks))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _parse_out_keys_arg(self, out_feat_keys):

    	# By default return the features of the last layer / module.
        out_feat_keys = [self.all_feat_names[-1]] if out_feat_keys is None else out_feat_keys
        if len(out_feat_keys) == 0:
            raise ValueError('Empty list of output feature keys.')
        for f, key in enumerate(out_feat_keys):
            if key not in self.all_feat_names:
                raise ValueError('Feature with name {} does not exist. Existing features: {}.'.format(key, self.all_feat_names))
            elif key in out_feat_keys[:f]:
                raise ValueError('Duplicate output feature key: {}.'.format(key))

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
            out_feats: returns the global avg pooled(for applying FC on it) reparameterized encoded space
        """
        #pdb.set_trace()
        # out_feat_keys, _ = self._parse_out_keys_arg(out_feat_keys)
        # out_feats = [None] * len(out_feat_keys)
        feat = x
        #encode
        for f in range(self.num_stages): 
            feat = self._feature_blocks[f](feat)
            # key = self.all_feat_names[f]
      
        #reparameterize
        # feat = [mu(input to enc block), logvar(from enc block)]
        mu = feat[:,:self.enc_dim] 
        logvar = feat[:, self.enc_dim:]
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        #print("std.shape: {}, mu.shape: {}".format(std.shape, mu.shape))
        feat = eps.mul(std * 0.001).add_(mu)
        # if key in out_feat_keys:
        # out_feats[out_feat_keys.index(key)] = feat

        # out_feats = out_feats[0] if len(out_feats)==1 else out_feats
        
        feat = self._feature_blocks[-1](feat)

        return feat

class Regressor(nn.Module):
    def __init__(self, depth=40, widen_factor=2, dropRate=0.3, indim=256, num_classes=4): #indim = [orig_img_features,transformed_img_features], num_classes is the number of possible transformations, num_class = 4 for 4 rotations in buckets of {[0, 90), [90, 180), [180, 270), [270, 360)}
        super(Regressor, self).__init__()

        fc1_outdim = 256

        self.wrn = WideResNet(depth, widen_factor, dropRate)
        self.fc1 = nn.Linear(indim, fc1_outdim)
        self.fc2 = nn.Linear(fc1_outdim, num_classes)

        self.relu1 = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()
            
    def forward(self, x1, x2, out_feat_keys=None):
        x1 = self.wrn(x1, out_feat_keys)
        x2 = self.wrn(x2, out_feat_keys)
        x = torch.cat((x1,x2), dim=1)
        
        x = self.fc1(x)
        x = self.relu1(x)
        penul_feat = x
        x = self.fc2(x)

        return penul_feat, x
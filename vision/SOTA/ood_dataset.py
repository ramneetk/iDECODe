from __future__ import print_function
from PIL import Image, PILLOW_VERSION
import os
import os.path
import numpy as np
import sys
import numbers
import random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch
import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity
import math

from torchvision import datasets as datasets

import torchvision.transforms as transforms

from torch.utils.data import Dataset as Dataset
from torch.utils.data import DataLoader as DataLoader

import torchvision.transforms.functional as F

import pdb

class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], -1

    def __len__(self):
        return self.data_tensor.size(0)

def getAdvCIFAR10(data_root):

    cifar10_adv_data = torch.load(data_root)

    dataset = CustomTensorDataset(cifar10_adv_data)

    return dataset
    
class CIFAR_OOD(data.Dataset):
    """SVHN, Imagenet, LSUN, Places365, CIFAR100, adv_cifar10
    """  
    def __init__(self, dataset_name='SVHN', resample=False, fillcolor=0,
                 transform_pre=None, transform=None, target_transform=None,
                 download=False, data_root=None, rot_bucket_width=10):

        self.transform_pre = transform_pre
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_name = dataset_name
        self.rot_bucket_width = rot_bucket_width
       
        #pdb.set_trace()
        if dataset_name == 'SVHN':

            self.dataset = datasets.SVHN(
                        root='data', split='test', download=True,
                        transform=None)
        
        elif dataset_name == 'CIFAR100':

            self.dataset = datasets.CIFAR100(
                        root='data', train=False, download=True,
                        transform=None)
        
        elif dataset_name == 'Places365':

            self.dataset = datasets.Places365(
                        root='data', split='val', download=True, small=False,
                        transform=None)
        
        elif dataset_name == 'LSUN':
        
            dataroot = os.path.expanduser(os.path.join('data', 'LSUN_resize'))
            self.dataset = datasets.ImageFolder(dataroot, transform=None)
        
        elif dataset_name == 'IMAGENET':
        
            dataroot = os.path.expanduser(os.path.join('data', 'Imagenet_resize'))
            self.dataset = datasets.ImageFolder(dataroot, transform=None)
        
        elif dataset_name == 'adv_cifar10':
            self.dataset = getAdvCIFAR10(data_root=data_root)
        
        self.resample = resample
        self.fillcolor = fillcolor

  
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img1, target  = self.dataset.__getitem__(index)

        if self.dataset_name == 'adv_cifar10':
            # convert tensor data to PIL image
            img1 = F.to_pil_image(img1)

        if self.transform_pre is not None:
            img1 = self.transform_pre(img1)

        #pdb.set_trace()

        # noisy rotations from 4 buckets
        rot_angle1 = 0+random.randint(-self.rot_bucket_width, self.rot_bucket_width) 
        rot_angle2 = 90+random.randint(-self.rot_bucket_width,self.rot_bucket_width)   
        rot_angle3 = 180+random.randint(-self.rot_bucket_width,self.rot_bucket_width)   
        rot_angle4 = 270+random.randint(-self.rot_bucket_width,self.rot_bucket_width)   

        if self.transform_pre is not None:
            img1 = self.transform_pre(img1)           

        img2 = img1.rotate(angle=rot_angle1, resample=self.resample, fillcolor=self.fillcolor)
        img3 = img1.rotate(angle=rot_angle2, resample=self.resample, fillcolor=self.fillcolor)
        img4 = img1.rotate(angle=rot_angle3, resample=self.resample, fillcolor=self.fillcolor)
        img5 = img1.rotate(angle=rot_angle4, resample=self.resample, fillcolor=self.fillcolor)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
            img4 = self.transform(img4)
            img5 = self.transform(img5)

        if self.target_transform is not None:
            target = self.target_transform(target) # target is the class label

        # return img1, img2, coeffs, target
        return img1, img2, img3, img4, img5, target


    def __len__(self):
        return self.dataset.__len__() 
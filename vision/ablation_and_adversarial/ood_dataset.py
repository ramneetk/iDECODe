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
from torchvision.transforms.functional import _get_inverse_affine_matrix
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
    if '.npy' in data_root:
        cifar10_adv_data_np = np.load(data_root)
        cifar10_adv_data_np = cifar10_adv_data_np.transpose((0,3,1,2))
        cifar10_adv_data = torch.tensor(cifar10_adv_data_np)

    else:
        cifar10_adv_data = torch.load(data_root)

    dataset = CustomTensorDataset(cifar10_adv_data)

    return dataset
    
class CIFAR_OOD(data.Dataset):
    """SVHN, Imagenet, LSUN, Places365, CIFAR100, adv_cifar10
    """  
    def __init__(self, dataset_name='SVHN', shift=6, scale=None, resample=False, fillcolor=0, train=False,
                 transform_pre=None, transform=None, target_transform=None, matrix_transform=None,
                 download=False, data_root=None):

        self.transform_pre = transform_pre
        self.transform = transform
        self.target_transform = target_transform
        self.matrix_transform = matrix_transform
        self.dataset_name = dataset_name
        self.train = False  # we require only testdata for OOD
       
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
                        root='data', split='val', download=True, small=True,
                        transform=None)
        
        elif dataset_name == 'LSUN':
        
            dataroot = os.path.expanduser(os.path.join('data', 'LSUN_resize'))
            self.dataset = datasets.ImageFolder(dataroot, transform=None)
        
        elif dataset_name == 'IMAGENET':
        
            dataroot = os.path.expanduser(os.path.join('data', 'Imagenet_resize'))
            self.dataset = datasets.ImageFolder(dataroot, transform=None)
        
        elif dataset_name == 'adv_cifar10':
            self.dataset = getAdvCIFAR10(data_root=data_root)
        
        self.train_data = None    

        #projective transformation
        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale
        
        self.shift = shift

        self.resample = resample
        self.fillcolor = fillcolor
    
    @staticmethod   
    def find_coeffs(pa, pb):
        matrix = []
        for p1, p2 in zip(pa, pb):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
    
        A = np.matrix(matrix, dtype=np.float)
        B = np.array(pb).reshape(8)
    
        res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
        return np.array(res).reshape(8)

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
            #print(type(img1))
            #print(img1.shape)
            img1 = F.to_pil_image(img1)

        if self.transform_pre is not None:
            img1 = self.transform_pre(img1)
              
        # projective transformation on image2        
        width, height = img1.size
        center = (img1.size[0] * 0.5 + 0.5, img1.size[1] * 0.5 + 0.5)
        shift = [float(random.randint(-int(self.shift), int(self.shift))) for ii in range(8)]
        scale = random.uniform(self.scale[0], self.scale[1])
        rotation = random.randint(0,3)
        
        pts = [((0-center[0])*scale+center[0], (0-center[1])*scale+center[1]),
            ((width-center[0])*scale+center[0], (0-center[1])*scale+center[1]),
            ((width-center[0])*scale+center[0], (height-center[1])*scale+center[1]),
            ((0-center[0])*scale+center[0], (height-center[1])*scale+center[1])]        
        pts = [pts[(ii+rotation)%4] for ii in range(4)]
        pts = [(pts[ii][0]+shift[2*ii], pts[ii][1]+shift[2*ii+1]) for ii in range(4)]
        
        coeffs = self.find_coeffs(
            pts,
            [(0, 0), (width, 0), (width, height), (0, height)]
        )
        
        kwargs = {"fillcolor": self.fillcolor} if PILLOW_VERSION[0] == '5' else {}
        img2 = img1.transform((width, height), Image.PERSPECTIVE, coeffs, self.resample, **kwargs)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        if self.target_transform is not None:
            target = self.target_transform(target)
                    
        coeffs = torch.from_numpy(np.array(coeffs, np.float32, copy=False)).view(8, 1, 1)
        
        if self.matrix_transform is not None:
            coeffs = self.matrix_transform(coeffs)

        return img1, img2, coeffs, target


    def __len__(self):
        return self.dataset.__len__() 

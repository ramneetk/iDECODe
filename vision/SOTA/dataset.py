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

from skimage.transform import rotate as skimage_rotate

import pdb

class CIFAR10(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, root, resample=False, fillcolor=0, train=True, transform_pre=None, transform=None,
                 target_transform=None, download=True, class_list=None, rot_bucket_width=10, cal=False, proper_train_size=4500):
        self.root = os.path.expanduser(root)
        self.transform_pre = transform_pre
        self.transform = transform
        self.target_transform = target_transform
        self.train = train 
        self.cal = cal

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # now load the picked numpy arrays
        if self.train or self.cal:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
            
            if class_list != None:
                labels = np.array(self.train_labels)
                indices = []
                new_labels = []
                for i,l in enumerate(labels):
                    if l in class_list:
                        indices.append(i)
                        new_labels.append(l)

                self.train_data = self.train_data[indices]
                self.train_labels = new_labels
            
            if self.train: # proper training set
                self.train_data = self.train_data[0:proper_train_size]
                self.train_labels = self.train_labels[0:proper_train_size]
            else: # calibration set
                self.train_data = self.train_data[proper_train_size:5000]
                self.train_labels = self.train_labels[proper_train_size:5000]

        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            #print(self.test_data.max(),self.test_data.min(),self.test_data.dtype)
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

            if class_list != None:
                labels = np.array(self.test_labels)
                indices = []
                new_labels = []
                for i,l in enumerate(labels):
                    if l in class_list:
                        indices.append(i)
                        new_labels.append(l)

                self.test_data = self.test_data[indices]
                self.test_labels = new_labels

        
        self.resample = resample
        self.fillcolor = fillcolor

        self.rot_bucket_width = rot_bucket_width

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        if self.train or self.cal:
            img1, target = self.train_data[index], self.train_labels[index]
        else:
            img1, target = self.test_data[index], self.test_labels[index]

        # pdb.set_trace()

        # noisy rotations from 4 buckets
        rot_angle1 = 0+random.randint(-self.rot_bucket_width, self.rot_bucket_width) 
        rot_angle2 = 90+random.randint(-self.rot_bucket_width,self.rot_bucket_width)   
        rot_angle3 = 180+random.randint(-self.rot_bucket_width,self.rot_bucket_width)   
        rot_angle4 = 270+random.randint(-self.rot_bucket_width,self.rot_bucket_width)   

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img1 = Image.fromarray(img1)
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
        if self.train or self.cal:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

'''
command to run-
python fnr_at_fixed_epsilon.py --net $path to saved model$ --cuda --gpu 0 --n 5 --trials 5
--cal_set_size_trial $1000/2000$ --file_name $1000/2000.pdf$
'''

from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils

import numpy as np
import matplotlib.pyplot as plt
from pylab import text

from avt_resnet import Regressor

from dataset import CIFAR10
from ood_dataset import CIFAR_OOD
import PIL

import pdb


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='dataset', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--net', default='', help="path load the trained network")
parser.add_argument('--manualSeed', type=int, default=3525, help='manual seed')
parser.add_argument('--shift', type=float, default=4)
parser.add_argument('--shrink', type=float, default=0.8)
parser.add_argument('--enlarge', type=float, default=1.2)
parser.add_argument('--resnetType', type=int, default=34, help='18/34')

parser.add_argument('--gpu', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--file_name', default='2000.pdf', help='final output file')

# OOD detection params
parser.add_argument('--n', type=int, default=20, help='no. of transformations')
parser.add_argument('--trials', type=int, default=5, help='Number of trials')
parser.add_argument('--cal_set_size_trial', type=int, default=4000, help='Cal set size in one trial')
parser.add_argument('--proper_train_size', default=45000, type=int, help='proper training dataset size')

opt = parser.parse_args()
#print(opt)

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:{}".format(opt.gpu) if use_cuda else "cpu")

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
rand_state = np.random.RandomState(seed=opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

net = Regressor(resnet_type=opt.resnetType).to(device)

if opt.cuda:
    net = torch.nn.DataParallel(net, device_ids=[int(opt.gpu)])

if opt.net != '':
    net.load_state_dict(torch.load(opt.net, map_location=device))

in_test_dataset = CIFAR10(root=opt.dataroot, shift=opt.shift, scale=(opt.shrink, opt.enlarge), fillcolor=(128,128,128), download=True, train=False, resample=PIL.Image.BILINEAR,
                           matrix_transform=transforms.Compose([
                               transforms.Normalize((0., 0., 16., 0., 0., 16., 0., 0.), (1., 1., 20., 1., 1., 20., 0.015, 0.015)),
                           ]),
                           transform_pre=None,
                           
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                           ]), cal=False, proper_train_size=opt.proper_train_size)

cal_dataset = CIFAR10(root=opt.dataroot, shift=opt.shift, scale=(opt.shrink, opt.enlarge), fillcolor=(128,128,128), download=True, train=False, resample=PIL.Image.BILINEAR,
                           matrix_transform=transforms.Compose([
                               transforms.Normalize((0., 0., 16., 0., 0., 16., 0., 0.), (1., 1., 20., 1., 1., 20., 0.015, 0.015)),
                           ]),
                           transform_pre=None,

                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                           ]), cal=True, proper_train_size=opt.proper_train_size)


cal_dataloader = torch.utils.data.DataLoader(cal_dataset, batch_size=opt.batchSize,
                                        shuffle=False, num_workers=int(opt.workers))

in_test_dataloader = torch.utils.data.DataLoader(in_test_dataset, batch_size=opt.batchSize,
                                        shuffle=False, num_workers=int(opt.workers))

criterion = nn.MSELoss()

def calc_MSE(test_dataloader):
    mse = []
    net.eval()
    for _, data in enumerate(test_dataloader, 0):
        img1 = data[0].to(device)
        img2 = data[1].to(device)
        matrix = data[2].to(device)
        matrix = matrix.view(-1, 8)
        
        _, _, output_mu, _ = net(img1, img2)
        
        for j in range(output_mu.shape[0]):
            mse.append(criterion(output_mu[j], matrix[j]).item())

    return np.array(mse)

def calc_p_values(n, test_mse, cal_set_mse, is_val_set):

    cal_set_mse_reshaped = cal_set_mse
    cal_set_mse_reshaped = cal_set_mse_reshaped.reshape(1,-1) # cal_set_mse reshaped into row vector

    test_mse_reshaped = test_mse
    test_mse_reshaped = test_mse_reshaped.reshape(-1,1) # test_mse reshaped into column vector

    compare = test_mse_reshaped<=cal_set_mse_reshaped
    p_values = np.sum(compare, axis=1)
    p_values = (p_values+1)/(len(cal_set_mse)+1)
    
    if is_val_set==1:
        np.savez("cifar10_p_values_n{}.npz".format(n), p_values=p_values)
    else:
        np.savez("{}_p_values_n{}.npz".format(opt.ood_dataset,n), p_values=p_values)

    return p_values

def checkOOD(n = opt.n):  

    in_dist_mse_list = []
    cal_set_mse = []

    for _ in range(n):
        in_dist_mse = calc_MSE(in_test_dataloader)
        in_dist_mse_list.append(in_dist_mse)
        cal_mse = calc_MSE(cal_dataloader)
        cal_set_mse.append(cal_mse)

    val_set_mse = in_dist_mse_list

    fnr_list = [] # in-dist samples detected as OODs    
    epsilon_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    cal_set_mse = np.array(cal_set_mse) 
    cal_set_mse = np.transpose(cal_set_mse) 
    val_set_mse = np.array(val_set_mse)
    val_set_mse = np.transpose(val_set_mse)

    cal_set_mse = cal_set_mse[:,:n]
    val_set_mse = val_set_mse[:,:n]

    f_val_set = np.sum(val_set_mse, axis = 1)

    # print("cal_set len:", cal_set_mse.shape[0])

    for _ in range(opt.trials):
        
        indices = rand_state.permutation(cal_set_mse.shape[0])
        cal_set_mse_tmp = cal_set_mse[indices]
        cal_set_mse_tmp = cal_set_mse_tmp[:opt.cal_set_size_trial]
        

        f_cal_set = np.sum(cal_set_mse_tmp, axis = 1)

        val_indist_p_values = calc_p_values(n, f_val_set, f_cal_set, is_val_set=1)

        val_indist_p_values = np.sort(val_indist_p_values)

        tmp_fnr_list = []

        for epsilon in epsilon_list:

            fnr = np.mean(val_indist_p_values<epsilon)

            tmp_fnr_list.append(fnr)
        
        fnr_list.append(tmp_fnr_list)
    
    fnr_list = np.array(fnr_list)

    return fnr_list, np.array(epsilon_list) # shape of fnr_list = opt.tirals X |epsilon_list|
 
if __name__ == "__main__":

    fnr_list, epsilon_list = checkOOD()
    fig, ax = plt.subplots()

    bp = ax.boxplot(fnr_list, meanline=True)
    ax.yaxis.grid(True) # Hide the horizontal gridlines
    ax.xaxis.grid(True) # Show the vertical gridlines    

    # ax.set_xlabel('epsilon', fontsize=16)
    # ax.set_ylabel('FNR', fontsize=16)
    ax.plot(label='calibration set size {}'.format(opt.cal_set_size_trial))
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    plt.yticks([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    plt.yticks(fontsize=14) 
    plt.xticks(fontsize=14) 

    x = [1,2,3,4,5,6,7,8,9,10]
    y = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    ax.plot(x,y, label='FNR=epsilon')
    ax.legend(fontsize=18)

    plt.xlabel('epsilon', fontsize=15)
    plt.ylabel('FNR', fontsize=15)
    plt.savefig(opt.file_name)

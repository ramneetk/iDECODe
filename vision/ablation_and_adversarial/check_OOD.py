'''
command to run with AVT model trained on cifar10 with Resnet34 architecture
python check_OOD.py --cuda --gpu 0 --net $path to saved model$ --n 5 --ood_dataset $IMAGENET/CIFAR100/LSUN/SVHN/Places365/adv_cifar10$ --proper_train_size 45000 --trials 5
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

from avt_resnet import Regressor

from dataset import CIFAR10
from ood_dataset import CIFAR_OOD
import PIL


import pdb

from scipy.integrate import quad_vec
from scipy import integrate


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean in_dist_testue expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='dataset', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--resnetType', type=int, default=34, help='18/34')

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--net', default='', help="path load the trained network")
parser.add_argument('--manualSeed', type=int, default=2535, help='manual seed')
parser.add_argument('--shift', type=float, default=4)
parser.add_argument('--shrink', type=float, default=0.8)
parser.add_argument('--enlarge', type=float, default=1.2)

parser.add_argument('--gpu', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--trials', type=int, default=5, help='no. of trials for taking average for the final results')

# OOD detection params
parser.add_argument('--n', type=int, default=20, help='no. of transformations')
parser.add_argument('--proper_train_size', default=45000, type=int, help='proper training dataset size')
parser.add_argument('--ood_dataset', default='SVHN', help='SVHN/IMAGENET/CIFAR100/LSUN/Places365/adv_cifar10')

# Adv detection params
parser.add_argument('--adv_data_root', default='ResNet34_cifar10/adv_data_ResNet34_cifar10_BIM.pth', help='path to the adversarial examples')

opt = parser.parse_args()
print(opt)

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:{}".format(opt.gpu) if use_cuda else "cpu")

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

net = Regressor(resnet_type=opt.resnetType).to(device)
if opt.cuda:
    net = torch.nn.DataParallel(net, device_ids=[int(opt.gpu)])

if opt.net != '':
    net.load_state_dict(torch.load(opt.net, map_location=device))


ood_test_dataset = CIFAR_OOD(dataset_name=opt.ood_dataset, shift=opt.shift, scale=(opt.shrink, opt.enlarge), fillcolor=(128,128,128), download=True, train=False, resample=PIL.Image.BILINEAR,
                    matrix_transform=transforms.Compose([
                        transforms.Normalize((0., 0., 16., 0., 0., 16., 0., 0.), (1., 1., 20., 1., 1., 20., 0.015, 0.015)),
                    ]),

                    transform_pre=None,

                    transform=transforms.Compose([
                        transforms.Resize((32,32)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ]), data_root=opt.adv_data_root)

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

out_test_dataloader = torch.utils.data.DataLoader(ood_test_dataset, batch_size=opt.batchSize,
                                        shuffle=False, num_workers=int(opt.workers))

criterion = nn.MSELoss()

# pdb.set_trace()

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

def calc_p_values(n, test_mse, cal_set_mse, is_in_dist_test_set):

    cal_set_mse_reshaped = cal_set_mse
    cal_set_mse_reshaped = cal_set_mse_reshaped.reshape(1,-1) # cal_set_mse reshaped into row vector

    test_mse_reshaped = test_mse
    test_mse_reshaped = test_mse_reshaped.reshape(-1,1) # test_mse reshaped into column vector

    compare = test_mse_reshaped<=cal_set_mse_reshaped
    p_values = np.sum(compare, axis=1)
    p_values = (p_values+1)/(len(cal_set_mse)+1)
    
    if is_in_dist_test_set==1:
        np.savez("cifar10_p_values_n{}.npz".format(n), p_values=p_values)
    else:
        np.savez("{}_p_values_n{}.npz".format(opt.ood_dataset,n), p_values=p_values)

    return p_values

def checkOOD(n = opt.n):  

    our_auroc_list = []
    our_tnr_list = []

    baseline_tnr_list = []
    baseline_auroc_list = []
    icad_tnr_list = []
    icad_auroc_list = []

    for trial in range(opt.trials):

        print("Trial: ", trial+1)
        ood_set_mse = []
        in_dist_test_set_mse = []
        cal_set_mse = []

        for iter in range(n):
            in_dist_mse = calc_MSE(in_test_dataloader)
            in_dist_test_set_mse.append(in_dist_mse)
            cal_mse = calc_MSE(cal_dataloader)
            cal_set_mse.append(cal_mse)
            out_dist_mse = calc_MSE(out_test_dataloader)
            ood_set_mse.append(out_dist_mse)

            if iter == 0: # ICAD is iDECODe with n=1
               icad_tnr, icad_auroc = get_icad_results(cal_set_mse, in_dist_test_set_mse, ood_set_mse) 
               icad_auroc_list.append(icad_auroc), icad_tnr_list.append(icad_tnr)

        ########## STEP 1 = for each data point, create n alphas = V(data point) #################
        cal_set_mse = np.array(cal_set_mse) # cal_set_mse = n X |train dataset| - opt.proper_train_size 
        cal_set_mse = np.transpose(cal_set_mse) # cal_set_mse = |train dataset| - opt.proper_train_size  X n
        ood_set_mse = np.array(ood_set_mse) # ood_set_mse = n X |out_dist_test_dataset|
        ood_set_mse = np.transpose(ood_set_mse) # ood_set_mse = |out_dist_test_dataset| X n
        in_dist_test_set_mse = np.array(in_dist_test_set_mse) # in_dist_test_set_mse = n X |in_dist_test_dataset|
        in_dist_test_set_mse = np.transpose(in_dist_test_set_mse) # in_dist_test_set_mse = |in_dist_test_dataset| X n

        # for baseline results
        baseline_tnr, baseline_auroc = get_baseline_results(in_mse=in_dist_test_set_mse, ood_mse=ood_set_mse)
        baseline_tnr_list.append(baseline_tnr)
        baseline_auroc_list.append(baseline_auroc)

        ######## STEP 2 =  Apply F on V(data point), F(t) = summation of all the in_values in t 
        f_cal_set = np.sum(cal_set_mse, axis = 1)
        f_in_dist_test_set = np.sum(in_dist_test_set_mse, axis = 1)
        f_ood_set = np.sum(ood_set_mse, axis = 1)

        ######## STEP 3 = Calculate p-in_values for OOD and in-distribution test set #########
        ood_p_values = calc_p_values(n, f_ood_set, f_cal_set, is_in_dist_test_set=0)
        # calculate p-in_values for in-distribution test in-dist dataset - higher p-in_values for in-dist and lower for OODs
        in_dist_test_p_values = calc_p_values(n, f_in_dist_test_set, f_cal_set, is_in_dist_test_set=1)

        in_dist_test_p_values = np.sort(in_dist_test_p_values)

        epsilon = in_dist_test_p_values[int(len(in_dist_test_p_values)*0.1)] #OOD detection threshold at 90% TPR

        tnr = np.mean(ood_p_values<epsilon)
        tnr = tnr*100.

        au_roc =  getAUROC(n)

        our_tnr_list.append(tnr)
        our_auroc_list.append(au_roc)

    print("Base Score results: AUROC {} +- {} and TNR {} +- {}".format(np.mean(np.array(baseline_auroc_list)), np.std(np.array(baseline_auroc_list)), np.mean(np.array(baseline_tnr_list)), np.std(np.array(baseline_tnr_list))))

    print("ICAD results: AUROC {} +- {} and TNR {} +- {}".format(np.mean(np.array(icad_auroc_list)), np.std(np.array(icad_auroc_list)), np.mean(np.array(icad_tnr_list)), np.std(np.array(icad_tnr_list))))

    return our_auroc_list, our_tnr_list

def get_baseline_results(in_mse, ood_mse):

    in_mse = in_mse[:,0] # using MSE from the first iteration, i.e. n=1
    ood_mse = ood_mse[:,0]
    in_mse = np.sort(in_mse)
    tau = in_mse[int(0.90*len(in_mse))]
    mse_tnr = 100*len(ood_mse[ood_mse>tau])/len(ood_mse)

    mse = np.concatenate((in_mse, ood_mse))
    indist_label = np.zeros((len(in_mse))) # mse for in-dist should be low
    ood_label = np.ones((len(ood_mse)))
    label = np.concatenate((indist_label, ood_label))

    from sklearn.metrics import roc_auc_score
    mse_au_roc = roc_auc_score(label, mse)*100

    return mse_tnr, mse_au_roc

############################ ICAD is iDECODe with n=1 #############################
def get_icad_results(cal_set_mse, in_dist_test_set_mse, ood_set_mse, n=1): 
    ########## STEP 1 = for each data point, create n alphas = V(data point) #################
    cal_set_mse = np.array(cal_set_mse) # cal_set_mse = n X |train dataset| - opt.proper_train_size 
    cal_set_mse = np.transpose(cal_set_mse) # cal_set_mse = |train dataset| - opt.proper_train_size  X n
    ood_set_mse = np.array(ood_set_mse) # ood_set_mse = n X |out_dist_test_dataset|
    ood_set_mse = np.transpose(ood_set_mse) # ood_set_mse = |out_dist_test_dataset| X n
    in_dist_test_set_mse = np.array(in_dist_test_set_mse) # in_dist_test_set_mse = n X |in_dist_test_dataset|
    in_dist_test_set_mse = np.transpose(in_dist_test_set_mse) # in_dist_test_set_mse = |in_dist_test_dataset| X n

    ######## STEP 2 =  Apply F on V(data point), F(t) = summation of all the in_values in t 
    f_cal_set = np.sum(cal_set_mse, axis = 1)
    f_in_dist_test_set = np.sum(in_dist_test_set_mse, axis = 1)
    f_ood_set = np.sum(ood_set_mse, axis = 1)

    ######## STEP 3 = Calculate p-in_values for OOD and in-distribution test set #########
    ood_p_values = calc_p_values(n, f_ood_set, f_cal_set, is_in_dist_test_set=0)
    # calculate p-in_values for in-distribution test in-dist dataset - higher p-in_values for in-dist and lower for OODs
    in_dist_test_p_values = calc_p_values(n, f_in_dist_test_set, f_cal_set, is_in_dist_test_set=1)

    in_dist_test_p_values = np.sort(in_dist_test_p_values)

    epsilon = in_dist_test_p_values[int(len(in_dist_test_p_values)*0.1)] #OOD detection threshold at 90% TPR

    tnr = np.mean(ood_p_values<epsilon)
    tnr = tnr*100.

    au_roc =  getAUROC(n)

    return tnr, au_roc


def getAUROC(n):
    ood_p_values = np.load("{}_p_values_n{}.npz".format(opt.ood_dataset,n))['p_values']
    indist_p_values = np.load("cifar10_p_values_n{}.npz".format(n))['p_values']
    p_values = np.concatenate((indist_p_values, ood_p_values))

    # higher p-in_values for in-dist and lower for OODs
    indist_label = np.ones((in_test_dataset.__len__()))
    ood_label = np.zeros((ood_test_dataset.__len__()))
    label = np.concatenate((indist_label, ood_label))

    from sklearn.metrics import roc_auc_score
    au_roc = roc_auc_score(label, p_values)*100.
    return au_roc
 
if __name__ == "__main__":
    auroc_list, tnr_list = checkOOD()
    print("Our results: AUROC {} +- {} and TNR {} +- {}".format(np.mean(np.array(auroc_list)), np.std(np.array(auroc_list)), np.mean(np.array(tnr_list)), np.std(np.array(tnr_list))))

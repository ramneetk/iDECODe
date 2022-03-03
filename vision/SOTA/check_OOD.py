'''
Assumes models saved in saved_models/class<class_num>.pth

command for running experiments for all 10 classes - 
python check_OOD.py --cuda --dataroot dataset --batchSize 50 --gpu 3 --n 5 --indist_class 10 --proper_train_size 4500 --trials 5

command for running experiment for a single class (ex. 3)
python check_OOD.py --cuda --dataroot dataset --batchSize 50 --gpu 0 --n 5 --net saved_models/class3.pth --ood_dataset cifar_non3_class  --indist_class 3 --proper_train_size 4500 --trials 5
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

from dataset import CIFAR10
import PIL

from avt_wrn import Regressor as wrn_regressor

from ood_dataset import CIFAR_OOD

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
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='dataset', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=512, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--net', default='', help="path to trained model")
parser.add_argument('--manualSeed', type=int, default=2535, help='manual seed')

parser.add_argument('--gpu', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--trials', type=int, default=5, help='no. of trials for taking average for the final results')

# OOD detection params
parser.add_argument('--n', type=int, default=5, help='no of transformations')
parser.add_argument('--proper_train_size', default=4500, type=int, help='proper training dataset size, 4500 for one-class, 45000 for complete dataset')
parser.add_argument('--ood_dataset', default='', help='use cifar_non{}_class.format(in-dist_class) for one-class OOD detection problem, and $SVHN/LSUN/IMAGENET/CIFAR100/Places365$ for complete dataset')

parser.add_argument('--indist_class', default=0, type=int, help='0-9 for 1-class, 10 for running for all classes, -1 for completed dataset')
parser.add_argument('--rot_bucket_width', default=10, type=int)

parser.add_argument('--archi_type', default='wrn', type=str, help='network architechture - wrn/resnet34')

opt = parser.parse_args()
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

in_test_dataset, ood_test_dataset, cal_set, cal_dataloader, in_test_dataloader, out_test_dataloader, net = None, None, None, None, None, None, None

if opt.indist_class != 10 and opt.indist_class != -1: # running one-class detection for a single class
    net = wrn_regressor().to(device) # wide resenet

    if opt.cuda:
        net = torch.nn.DataParallel(net, device_ids=[int(opt.gpu)])

    if opt.net != '':
        net.load_state_dict(torch.load(opt.net, map_location=device))

    cifar10_ood_classes = []
    for i in range(10):
        if i != opt.indist_class:
            cifar10_ood_classes.append(i)

    ood_test_dataset = CIFAR10(root=opt.dataroot, fillcolor=(128,128,128), download=True, resample=PIL.Image.BILINEAR, train=False,
                    
                    transform_pre=None,

                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                    ]), class_list=cifar10_ood_classes, rot_bucket_width=opt.rot_bucket_width, cal=False)

    in_test_dataset = CIFAR10(root=opt.dataroot, fillcolor=(128,128,128), download=True, resample=PIL.Image.BILINEAR, train=False,
                    
                        transform_pre=None,

                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ]), class_list=[opt.indist_class], rot_bucket_width=opt.rot_bucket_width, cal=False)

    cal_dataset = CIFAR10(root=opt.dataroot, train=False, download=True, fillcolor = (128, 128, 128), resample = PIL.Image.BILINEAR,

                            transform_pre=None,
                            
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ]), class_list=[opt.indist_class], cal=True, rot_bucket_width=opt.rot_bucket_width, proper_train_size=opt.proper_train_size)
    

    cal_dataloader = torch.utils.data.DataLoader(cal_dataset, batch_size=opt.batchSize,
                                            shuffle=False, num_workers=int(opt.workers))

    in_test_dataloader = torch.utils.data.DataLoader(in_test_dataset, batch_size=opt.batchSize,
                                            shuffle=False, num_workers=int(opt.workers))

    out_test_dataloader = torch.utils.data.DataLoader(ood_test_dataset, batch_size=opt.batchSize,
                                            shuffle=False, num_workers=int(opt.workers))

elif opt.indist_class == -1: # for complete dataset

    if opt.archi_type == 'resnet34':
        net = resnet_regressor().to(device) # resenet34 for complete dataset
    else:
        net = wrn_regressor().to(device) # wide resenet for class-wise dataset
    
    if opt.cuda:
        net = torch.nn.DataParallel(net, device_ids=[int(opt.gpu)])

    if opt.net != '':
        net.load_state_dict(torch.load(opt.net, map_location=device))

    class_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    cal_dataset = CIFAR10(root=opt.dataroot, train=False, download=True, fillcolor = (128, 128, 128), resample = PIL.Image.BILINEAR,
                            transform_pre=None,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ]), cal=True, class_list=class_list, rot_bucket_width=opt.rot_bucket_width, proper_train_size=opt.proper_train_size, total_train_size=50000)

    in_test_dataset = CIFAR10(root=opt.dataroot, train=False, download=True,
    fillcolor = (128, 128, 128), resample = PIL.Image.BILINEAR,

                            transform_pre=None,

                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ]), cal=False, class_list=class_list, rot_bucket_width=opt.rot_bucket_width, proper_train_size=opt.proper_train_size, total_train_size=50000)

    ood_test_dataset = CIFAR_OOD(dataset_name=opt.ood_dataset,  download=True, fillcolor = (128, 128, 128), resample = PIL.Image.BILINEAR,

                    transform_pre=None,

                    transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ]), rot_bucket_width=opt.rot_bucket_width)
    
    cal_dataloader = torch.utils.data.DataLoader(cal_dataset, batch_size=opt.batchSize,
                                            shuffle=False, num_workers=int(opt.workers))

    in_test_dataloader = torch.utils.data.DataLoader(in_test_dataset, batch_size=opt.batchSize,
                                            shuffle=False, num_workers=int(opt.workers))

    out_test_dataloader = torch.utils.data.DataLoader(ood_test_dataset, batch_size=opt.batchSize,
                                            shuffle=False, num_workers=int(opt.workers))

def calc_CEL(test_dataloader, criterion=nn.CrossEntropyLoss(reduction='none')):
    loss = []
    net.eval()

    counter = 0

    rot1 = torch.zeros((opt.batchSize), dtype=torch.long)
    rot2 = torch.ones((opt.batchSize), dtype=torch.long)
    rot3 = 2*torch.ones((opt.batchSize), dtype=torch.long)
    rot4 = 3*torch.ones((opt.batchSize), dtype=torch.long)
    target_rot = torch.cat((rot1, rot2, rot3, rot4))
    target_rot = target_rot.to(device)

    for i, data in enumerate(test_dataloader, 0):
        img1 = data[0].to(device)
        img2 = data[1].to(device)
        img3 = data[2].to(device)
        img4 = data[3].to(device)
        img5 = data[4].to(device)

        img_org_batch = img1.repeat(4,1,1,1)       
        img_rot_batch = torch.cat((img2,img3,img4,img5), dim=0)      

        _, pred_rot = net(img_org_batch, img_rot_batch)
        
        ce = criterion(pred_rot, target_rot) # 1D ce shape is |batchsizeX4|
        ce = ce.reshape(opt.batchSize,4)

        err = ce # 2D err shape is batchSizeX4
        err = err.mean(1) # 1D err shape is batchSize
        err = err.detach().cpu().numpy()
        
        for i in range(err.shape[0]):
            loss.append(err[i])
        
        counter += 1

    return np.array(loss)

def calc_p_values(n, test_cel, cal_set_cel, is_val_set):

    cal_set_cel_reshaped = cal_set_cel
    cal_set_cel_reshaped = cal_set_cel_reshaped.reshape(1,-1) # cal_set_cel reshaped into row vector

    test_cel_reshaped = test_cel
    test_cel_reshaped = test_cel_reshaped.reshape(-1,1) # test_cel reshaped into column vector

    compare = test_cel_reshaped<=cal_set_cel_reshaped
    p_values = np.sum(compare, axis=1)
    p_values = (p_values+1)/(len(cal_set_cel)+1)
    #print(len(cal_set_cel))
    
    if is_val_set==1:
        np.savez("cifar10_class{}_p_values_n{}.npz".format(opt.indist_class, n), p_values=p_values)
    else:
        np.savez("{}_p_values_n{}.npz".format(opt.ood_dataset,n), p_values=p_values)

    return p_values

def checkOOD_CEL():  

    n = opt.n

    auroc_list = []

    for trial in range(opt.trials):

        print("Trial: ", trial+1)

        ood_set_cel = []
        cal_set_cel = []
        in_dist_cel_list = []

        for _ in range(n):
    
            in_dist_cel = calc_CEL(in_test_dataloader)
            in_dist_cel_list.append(in_dist_cel)
            cal_cel = calc_CEL(cal_dataloader)
            cal_set_cel.append(cal_cel)
            out_dist_cel = calc_CEL(out_test_dataloader)       
            ood_set_cel.append(out_dist_cel)

        ood_set_cel = np.array(ood_set_cel)
        ood_set_cel = np.transpose(ood_set_cel)
        
        ########## STEP 1 = for each data point, create n alphas = V(data point) #################
        cal_set_cel = np.array(cal_set_cel) # cal_set_cel = n X 5000-opt.proper_train_size
        cal_set_cel = np.transpose(cal_set_cel) # cal_set_cel = 5000-opt.proper_train_size X n
        in_dist_cel_list = np.array(in_dist_cel_list)
        in_dist_cel_list = np.transpose(in_dist_cel_list)
        
        np.savez("CIFAR10_class_{}_cal_set_cel.npz".format(opt.indist_class),cal_set_cel=cal_set_cel)

        np.savez("CIFAR10_class{}_in_dist_cel_list.npz".format(opt.indist_class),in_dist_cel_list=in_dist_cel_list) # in_dist_cel_list = 2D array of dim |val set| X n

        np.savez("{}_cel.npz".format(opt.ood_dataset),ood_set_cel=ood_set_cel) # ood_set_cel = 2D array of dim |ood dataset| X n

        ######## STEP 2 =  Apply F on V(data point), F(t) = summation of all the values in t ################
        f_cal_set = np.sum(cal_set_cel, axis = 1)
        f_in_dist_test_set = np.sum(in_dist_cel_list, axis = 1)
        f_ood_set = np.sum(ood_set_cel, axis = 1)

        ######## STEP 3 = Calculate p-values for OOD and in_dist test set #########
        ood_p_values = calc_p_values(n, f_ood_set, f_cal_set, is_val_set=0)
        # calculate p-values for test in-dist dataset
        indist_test_p_values = calc_p_values(n, f_in_dist_test_set, f_cal_set, is_val_set=1)

        au_roc =  getAUROC(n)
        auroc_list.append(au_roc)

    return auroc_list

def getAUROC(n):
    ood_p_values = np.load("{}_p_values_n{}.npz".format(opt.ood_dataset,n))['p_values']
    indist_p_values = np.load("cifar10_class{}_p_values_n{}.npz".format(opt.indist_class, n))['p_values']
    p_values = np.concatenate((indist_p_values, ood_p_values))

    indist_label = np.ones((in_test_dataset.__len__()))
    ood_label = np.zeros((ood_test_dataset.__len__()))
    label = np.concatenate((indist_label, ood_label))

    from sklearn.metrics import roc_auc_score
    au_roc = roc_auc_score(label, p_values)*100
    return au_roc

def runAllExperiments():
    global in_test_dataset, ood_test_dataset, cal_set, cal_dataloader, in_test_dataloader, out_test_dataloader, net
    all_results = []
    for i in range(10):
        
        random.seed(opt.manualSeed)
        torch.manual_seed(opt.manualSeed)

        opt.indist_class = i
        opt.net = "saved_models/class{}.pth".format(i)
        opt.ood_dataset =  "cifar_non{}_class".format(i)
        print(opt)

        net = wrn_regressor().to(device) # wide resenet

        if opt.cuda:
            net = torch.nn.DataParallel(net, device_ids=[int(opt.gpu)])

        if opt.net != '':
            net.load_state_dict(torch.load(opt.net, map_location=device))

        cifar10_ood_classes = []
        for j in range(10):
            if j != opt.indist_class:
                cifar10_ood_classes.append(j)

        ood_test_dataset = CIFAR10(root=opt.dataroot, fillcolor=(128,128,128), download=True, resample=PIL.Image.BILINEAR, train=False,
                        
                        transform_pre=None,

                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ]), class_list=cifar10_ood_classes, rot_bucket_width=opt.rot_bucket_width, cal=False)

        in_test_dataset = CIFAR10(root=opt.dataroot, fillcolor=(128,128,128), download=True, resample=PIL.Image.BILINEAR, train=False,
                        
                            transform_pre=None,

                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ]), class_list=[opt.indist_class], rot_bucket_width=opt.rot_bucket_width, cal=False)


        cal_dataset = CIFAR10(root=opt.dataroot, train=False, download=True, fillcolor = (128, 128, 128), resample = PIL.Image.BILINEAR,
                            transform_pre=transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                            ]),
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ]), cal=True, class_list=[opt.indist_class], rot_bucket_width=opt.rot_bucket_width)
    
        cal_dataloader = torch.utils.data.DataLoader(cal_dataset, batch_size=opt.batchSize,
                                            shuffle=False, num_workers=int(opt.workers))

        in_test_dataloader = torch.utils.data.DataLoader(in_test_dataset, batch_size=opt.batchSize,
                                                shuffle=False, num_workers=int(opt.workers))

        out_test_dataloader = torch.utils.data.DataLoader(ood_test_dataset, batch_size=opt.batchSize,
                                                shuffle=False, num_workers=int(opt.workers))

        auroc_list = checkOOD_CEL()
        print("Class {}: AUROC {} +- {}".format(i, np.mean(np.array(auroc_list)), np.std(np.array(auroc_list))))


if __name__ == "__main__":
    if opt.indist_class == 10:
        runAllExperiments() # for running experiments for all 10 classes
    else: # for running for a specific class
        auroc_list = checkOOD_CEL()
        print("Results: AUROC {} +- {}".format(np.mean(np.array(auroc_list)), np.std(np.array(auroc_list))))

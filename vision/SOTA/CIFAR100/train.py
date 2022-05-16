'''
command to run
python train.py --cuda --outf $output dir for saving the model$ --dataroot dataset --gpu $gpu_num$ --class_num $-1/0-19$ --niter 200 -- train_size $45000/2000$ --archi_type $resnet18/resnet34/wrn$
'''


from __future__ import print_function
import argparse
import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils

from avt_wrn import Regressor as wrn_regressor
from avt_resnet import Regressor as resnet_regressor

from dataset import CIFAR100
import PIL

import pdb

CIFAR100_SUPERCLASS = [
    [4, 31, 55, 72, 95],
    [1, 33, 67, 73, 91],
    [54, 62, 70, 82, 92],
    [9, 10, 16, 29, 61],
    [0, 51, 53, 57, 83],
    [22, 25, 40, 86, 87],
    [5, 20, 26, 84, 94],
    [6, 7, 14, 18, 24],
    [3, 42, 43, 88, 97],
    [12, 17, 38, 68, 76],
    [23, 34, 49, 60, 71],
    [15, 19, 21, 32, 39],
    [35, 63, 64, 66, 75],
    [27, 45, 77, 79, 99],
    [2, 11, 36, 46, 98],
    [28, 30, 44, 78, 93],
    [37, 50, 65, 74, 80],
    [47, 52, 56, 59, 96],
    [8, 13, 48, 58, 90],
    [41, 69, 81, 85, 89],
]

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
parser.add_argument('--batchSize', type=int, default=50)
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
parser.add_argument('--wgtDecay', default=5e-4, type=float)
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--net', default='', help="path to net (to continue training)")
parser.add_argument('--optimizer', default='', help="path to optimizer (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, default=2535, help='manual seed')
parser.add_argument('--debug', default=False, type=str2bool, nargs='?', const=True, help='will call pdb.set_trace() for debugging')
parser.add_argument('--gpu', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--class_num', default=-1, type=int, help='CIFAR100 superclass number (0 to 19) for training 1-class model, -1 is for complete dataset')
parser.add_argument('--m', default=0.1, type=float)
parser.add_argument('--lmbda', default=3, type=float)
parser.add_argument('--reg', default=False, type=str2bool, nargs='?', const=True)
parser.add_argument('--reg_param', default=10., type=float)
parser.add_argument('--rot_bucket_width', default=10, type=int)

parser.add_argument('--train_size', default=2000, type=int, help='proper training dataset size')

parser.add_argument('--archi_type', default='wrn', type=str, help='network architechture - wrn/resnet34')

opt = parser.parse_args()
print(opt)

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:{}".format(opt.gpu) if use_cuda else "cpu")

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.class_num == -1: # for training on complete dataset
    class_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    train_dataset = CIFAR100(root=opt.dataroot, train=True, download=True, fillcolor = (128, 128, 128), resample = PIL.Image.BILINEAR,
                            transform_pre=transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                            ]),
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ]), cal=False, class_list=class_list, rot_bucket_width=opt.rot_bucket_width, proper_train_size=opt.train_size)

    test_dataset = CIFAR100(root=opt.dataroot, train=False, download=True,
    fillcolor = (128, 128, 128), resample = PIL.Image.BILINEAR,

                            transform_pre=None,

                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ]), cal=False, class_list=class_list, rot_bucket_width=opt.rot_bucket_width, proper_train_size=opt.train_size)

else: # for superclass-wise training
    train_dataset = CIFAR100(root=opt.dataroot, train=True, download=True, fillcolor = (128, 128, 128), resample = PIL.Image.BILINEAR,
                            transform_pre=transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                            ]),
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ]), cal=False, class_list=CIFAR100_SUPERCLASS[opt.class_num], rot_bucket_width=opt.rot_bucket_width, proper_train_size=opt.train_size)

    test_dataset = CIFAR100(root=opt.dataroot, train=False, download=True,
    fillcolor = (128, 128, 128), resample = PIL.Image.BILINEAR,

                            transform_pre=None,

                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ]), cal=False, class_list=CIFAR100_SUPERCLASS[opt.class_num], rot_bucket_width=opt.rot_bucket_width, proper_train_size=opt.train_size)

assert train_dataset
assert test_dataset

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize,
                                         shuffle=False, num_workers=int(opt.workers))
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers))

print("device: ", device)

if opt.archi_type == 'resnet34':
    net = resnet_regressor().to(device)
elif opt.archi_type == 'resnet18':
    net = resnet_regressor(resnet_type=18).to(device) # resenet18 for complete dataset
else:
    net = wrn_regressor().to(device) # wide resenet for class-wise dataset

if opt.cuda:
    net = torch.nn.DataParallel(net, device_ids=[int(opt.gpu)])

if opt.net != '':
    net.load_state_dict(torch.load(opt.net))

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=opt.lr, weight_decay=opt.wgtDecay)
#optimizer = optim.Adam(net.parameters())
#optimizer = optim.SGD(net.parameters(), lr=opt.lr, weight_decay=opt.wgtDecay)
#optimizer = optim.SGD(net.parameters(), lr=opt.lr, weight_decay=opt.wgtDecay, momentum=0.9)

def calculateTotalParameters(model):
    count = 0
    for p in model.parameters():
        if p.requires_grad:
            count+=p.numel()
    
    return count

def tc_loss(zs, m):
    means = zs.mean(0).unsqueeze(0)
    res = ((zs.unsqueeze(2) - means.unsqueeze(1)) ** 2).sum(-1)
    pos = torch.diagonal(res, dim1=1, dim2=2)
    offset = torch.diagflat(torch.ones(zs.size(1))).unsqueeze(0).to(device) * 1e6
    neg = (res + offset).min(-1)[0]
    loss = torch.clamp(pos + m - neg, min=0).mean()
    return loss


def evaluate(test_dataloader):
    
    net.eval()

    total_err = 0
    total_ce = 0
    total_tc = 0
    counter = 0

    rot1 = torch.zeros((opt.batchSize), dtype=torch.long)
    rot2 = torch.ones((opt.batchSize), dtype=torch.long)
    rot3 = 2*torch.ones((opt.batchSize), dtype=torch.long)
    rot4 = 3*torch.ones((opt.batchSize), dtype=torch.long)
    target_rot = torch.cat((rot1, rot2, rot3, rot4))
    target_rot = target_rot.to(device)


    with torch.no_grad():

        for _, data in enumerate(test_dataloader, 0):

            img1 = data[0].to(device)
            img2 = data[1].to(device)
            img3 = data[2].to(device)
            img4 = data[3].to(device)
            img5 = data[4].to(device)

            img_org_batch = img1.repeat(4,1,1,1)       
            img_rot_batch = torch.cat((img2,img3,img4,img5), dim=0)      

            penul_feat, pred_rot = net(img_org_batch, img_rot_batch)

            zs = penul_feat.view(opt.batchSize,4,-1)

            tc = tc_loss(zs=zs, m=opt.m)
            ce = criterion(pred_rot, target_rot)

            if opt.reg:
                loss = ce + opt.lmbda * tc + opt.reg_param *(zs*zs).mean()
            else:
                loss = ce + opt.lmbda * tc

            total_err += loss.item()
            total_ce += ce.item()
            total_tc += tc.item()
            counter += 1

    # print("ce loss: {}, tc loss: {}".format(total_ce/counter, total_tc/counter))
    return total_err/counter  

def train():
    best_test_err = float('inf')

    if opt.debug:
        pdb.set_trace()

    rot1 = torch.zeros((opt.batchSize), dtype=torch.long)
    rot2 = torch.ones((opt.batchSize), dtype=torch.long)
    rot3 = 2*torch.ones((opt.batchSize), dtype=torch.long)
    rot4 = 3*torch.ones((opt.batchSize), dtype=torch.long)
    target_rot = torch.cat((rot1, rot2, rot3, rot4))
    target_rot = target_rot.to(device)

    for epoch in range(opt.niter):
        net.train() 
        total_train_loss = 0
        counter = 0
        for _, data in enumerate(train_dataloader, 0):

            net.zero_grad()

            img1 = data[0].to(device)
            img2 = data[1].to(device)
            img3 = data[2].to(device)
            img4 = data[3].to(device)
            img5 = data[4].to(device)

            img_org_batch = img1.repeat(4,1,1,1)       
            img_rot_batch = torch.cat((img2,img3,img4,img5), dim=0)      

            penul_feat, pred_rot = net(img_org_batch, img_rot_batch)

            zs = penul_feat.view(opt.batchSize,4,-1)

            tc = tc_loss(zs=zs, m=opt.m)
            ce = criterion(pred_rot, target_rot)

            if opt.reg:
                loss = ce + opt.lmbda * tc + opt.reg_param *(zs*zs).mean()
            else:
                loss = ce + opt.lmbda * tc
            
            total_train_loss+=loss.item()
            counter+= 1
            loss.backward()
            optimizer.step() 
        # print("Epoch: {}, Training loss: {}".format(epoch, total_train_loss/counter))

        # print("For training data: ")
        train_err = evaluate(train_dataloader)    
        # print("For test data: ")
        test_err  = evaluate(test_dataloader) 

        if best_test_err > test_err:
            torch.save(net.state_dict(), '%s/best_net.pth' % (opt.outf))
            torch.save(optimizer.state_dict(), '%s/best_optimizer.pth' % (opt.outf))  
            best_test_err = test_err  

        print("Epoch: {}, train err: {} test err: {} best test err: {}".format(epoch, train_err, test_err, best_test_err))

        # do checkpointing
        if (epoch+1) % 10 == 0:
            torch.save(net.state_dict(), '%s/net_epoch_%d.pth' % (opt.outf, epoch+1))
            torch.save(optimizer.state_dict(), '%s/optimizer_epoch_%d.pth' % (opt.outf, epoch+1))

# def test_saved_model():
#     test_err  = evaluate(test_dataloader)        
#     print("test err: {}".format(test_err))

if __name__ == "__main__":
    train()

'''
command to run
python train.py --cuda --outf saved_models --dataroot dataset --gpu $gpu_num$ --class_num $0-9$ --niter 200 --train_size 4500 --archi_type wrn
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

from dataset import CIFAR10
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
parser.add_argument('--class_num', default=-1, type=int, help='CIFAR10 class number for training 1-class model')
parser.add_argument('--m', default=0.1, type=float)
parser.add_argument('--lmbda', default=3, type=float)
parser.add_argument('--reg', default=False, type=str2bool, nargs='?', const=True)
parser.add_argument('--reg_param', default=10., type=float)
parser.add_argument('--rot_bucket_width', default=10, type=int)

parser.add_argument('--train_size', default=4500, type=int, help='proper training dataset size')

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


train_dataset = CIFAR10(root=opt.dataroot, train=True, download=True, fillcolor = (128, 128, 128), resample = PIL.Image.BILINEAR,
                        transform_pre=transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                        ]),
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ]), cal=False, class_list=[opt.class_num], rot_bucket_width=opt.rot_bucket_width, proper_train_size=opt.train_size)

test_dataset = CIFAR10(root=opt.dataroot, train=False, download=True,
fillcolor = (128, 128, 128), resample = PIL.Image.BILINEAR,

                        transform_pre=None,

                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ]), cal=False, class_list=[opt.class_num], rot_bucket_width=opt.rot_bucket_width, proper_train_size=opt.train_size)

assert train_dataset
assert test_dataset

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize,
                                         shuffle=False, num_workers=int(opt.workers))
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers))

print("device: ", device)

net = wrn_regressor().to(device) # wide resenet for class-wise dataset

if opt.cuda:
    net = torch.nn.DataParallel(net, device_ids=[int(opt.gpu)])

if opt.net != '':
    net.load_state_dict(torch.load(opt.net))

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=opt.lr, weight_decay=opt.wgtDecay)

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
        print("Epoch: {}, Training loss: {}".format(epoch, total_train_loss/counter))

        # do checkpointing
        #if (epoch+1) % 10 == 0:
            #torch.save(net.state_dict(), '%s/net_epoch_%d.pth' % (opt.outf, epoch+1))
            #torch.save(optimizer.state_dict(), '%s/optimizer_epoch_%d.pth' % (opt.outf, epoch+1))
    
    torch.save(net.state_dict(), '%s/class%d.pth' % (opt.outf, opt.class_num))
    torch.save(optimizer.state_dict(), '%s/optimizer_for_class%d.pth' % (opt.outf, opt.class_num))

def test_saved_model():
    test_err  = evaluate(test_dataloader)        
    print("test err: {}".format(test_err))

if __name__ == "__main__":
    train()

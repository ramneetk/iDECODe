'''
Command to run
python train.py --cuda --outf $output dir$ --gpu 0 --proper_train_size 45000
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

from avt_resnet import Regressor
from dataset import CIFAR10
import PIL

import pdb


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='dataset', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=2600, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--net', default='', help="path to net (to continue training)")
parser.add_argument('--optimizer', default='', help="path to optimizer (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, default=9061, help='manual seed')
parser.add_argument('--shift', type=float, default=4)
parser.add_argument('--shrink', type=float, default=0.8)
parser.add_argument('--enlarge', type=float, default=1.2)
parser.add_argument('--lrMul', type=float, default=10.)
parser.add_argument('--divide', type=float, default=1000.)
parser.add_argument('--resnetType', type=int, default=34, help='18/34')

parser.add_argument('--gpu', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--proper_train_size', type=int, default=45000)

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

train_dataset = CIFAR10(root=opt.dataroot, shift=opt.shift, scale=(opt.shrink, opt.enlarge), fillcolor=(128,128,128), download=True, train=True, resample=PIL.Image.BILINEAR,
                           matrix_transform=transforms.Compose([
                               transforms.Normalize((0., 0., 16., 0., 0., 16., 0., 0.), (1., 1., 20., 1., 1., 20., 0.015, 0.015)),
                           ]),
                           transform_pre=transforms.Compose([
                               transforms.RandomCrop(32, padding=4),
                               transforms.RandomHorizontalFlip(),
                           ]),
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                           ]), cal=False, proper_train_size=opt.proper_train_size)

test_dataset = CIFAR10(root=opt.dataroot, shift=opt.shift, scale=(opt.shrink, opt.enlarge), fillcolor=(128,128,128), download=True, train=False, resample=PIL.Image.BILINEAR,
                           matrix_transform=transforms.Compose([
                               transforms.Normalize((0., 0., 16., 0., 0., 16., 0., 0.), (1., 1., 20., 1., 1., 20., 0.015, 0.015)),
                           ]),
                           transform_pre=transforms.Compose([
                               transforms.RandomCrop(32, padding=4),
                               transforms.RandomHorizontalFlip(),
                           ]),
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                           ]), cal=False, proper_train_size=opt.proper_train_size)


assert train_dataset
assert test_dataset

#pdb.set_trace()

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchSize,
                                         shuffle=False, num_workers=int(opt.workers))
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                         shuffle=False, num_workers=int(opt.workers))

ngpu = int(opt.gpu)

print("DEVICE: ", device)
net = Regressor(resnet_type=opt.resnetType).to(device)
if opt.cuda:
    net = torch.nn.DataParallel(net, device_ids=[int(opt.gpu)])

if opt.net != '':
    net.load_state_dict(torch.load(opt.net))

print(net)

criterion = nn.MSELoss()

# setup optimizer
fc2_params = list(map(id, net.module.fc2.parameters()))
base_params = filter(lambda p: id(p) not in fc2_params, net.parameters())

optimizer = optim.SGD([{'params':base_params}, {'params':net.module.fc2.parameters(), 'lr': opt.lr*opt.lrMul}], lr=opt.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
if opt.optimizer != '':
    optimizer.load_state_dict(torch.load(opt.optimizer))

for epoch in range(opt.niter):
            
    for i, data in enumerate(train_dataloader, 0):
        net.zero_grad()
        img1 = data[0].to(device)
        img2 = data[1].to(device)
        matrix = data[2].to(device)
        matrix = matrix.view(-1, 8)
        
        batch_size = img1.size(0)
        f1, f2, output_mu, output_logvar = net(img1, img2)
        output_logvar = output_logvar / opt.divide
        std_sqr = torch.exp(output_logvar)
        
        err_matrix = criterion(output_mu, matrix)
        err = (torch.sum(output_logvar) + torch.sum((output_mu - matrix)*(output_mu - matrix) / (std_sqr + 1e-4))) / batch_size 
        err.backward()
        optimizer.step()
        
        print('[%d/%d][%d/%d] Loss: %.4f, Loss_matrix: %.4f'
              % (epoch, opt.niter, i, len(train_dataloader),
                 err.item(), err_matrix.item()))
                                                                                    

    # do checkpointing
    if epoch % 100 == 99:
        torch.save(net.state_dict(), '%s/net_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(optimizer.state_dict(), '%s/optimizer_epoch_%d.pth' % (opt.outf, epoch))

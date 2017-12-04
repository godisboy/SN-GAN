import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.utils.data
from torch.nn.modules import conv
from torch.nn.modules.utils import _pair, _triple
import torch.backends.cudnn as cudnn

import random
import argparse
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='train SNDCGAN model')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--gpu_ids', default=[0,1,2,3], help='gpu ids: e.g. 0,1,2, 0,2.')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--batchSize', type=int, default=100, help='with batchSize=1 equivalent to instance normalization.')
parser.add_argument('--label_num', type=int, default=200, help='number of labels.')
opt = parser.parse_args()
print(opt)

dataset = datasets.ImageFolder(root='/media/scw4750/25a01ed5-a903-4298-87f2-a5836dcb6888/AIwalker/dataset/CUB200_object',
                           transform=transforms.Compose([
                               transforms.Scale(64),
                               transforms.CenterCrop(64),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
                                      )

dataloader = torch.utils.data.DataLoader(dataset, batch_size=32,
                                         shuffle=True, num_workers=int(2))
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
    torch.cuda.set_device(opt.gpu_ids[2])

cudnn.benchmark = True

def _l2normalize(v, eps=1e-12):
    return v / (((v**2).sum())**0.5 + eps)

def max_singular_value(W, u=None, Ip=1):
    """
    power iteration for weight parameter
    """
    #xp = W.data
    if u is None:
        u = torch.FloatTensor(1, W.size(0)).normal_(0, 1).cuda()
    _u = u
    for _ in range(Ip):
        #print(_u.size(), W.size())
        _v = _l2normalize(torch.matmul(_u, W.data), eps=1e-12)
        _u = _l2normalize(torch.matmul(_v, torch.transpose(W.data, 0, 1)), eps=1e-12)
    sigma = torch.matmul(torch.matmul(_v, torch.transpose(W.data, 0, 1)), torch.transpose(_u, 0, 1))
    return sigma, _v

class SNConv2d(conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(SNConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, input):
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, _ = max_singular_value(w_mat)
        #print(sigma.size())
        self.weight.data = self.weight.data / sigma
        #print(self.weight.data)
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class _netG(nn.Module):
    def __init__(self, nz, nc, ngf):
        super(_netG, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz+200, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        output = self.main(input)
        return output

nz = opt.nz

G = _netG(nz, 3, 64)
print(G)
save_path = 'log/netG_epoch_199.pth'
G.load_state_dict(torch.load(save_path))

input = torch.FloatTensor(opt.batchSize, 3, 64, 64)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

#fixed label
fix_label = torch.FloatTensor(opt.batchSize)

for i in range(0,100):
    fix_label[i] = i;
    #fix_label[i] = np.random.randint(1,200);

fix = torch.LongTensor(opt.batchSize,1).copy_(fix_label)
fix_onehot = torch.FloatTensor(opt.batchSize, 200)
fix_onehot.zero_()
fix_onehot.scatter_(1, fix, 1)
fix_onehot.view(-1, 200, 1, 1)

fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
fixed_input = torch.cat([fixed_noise, fix_onehot],1)
fixed_input = Variable(fixed_input)

print(fixed_input.size())
criterion = nn.BCELoss()

fill = torch.zeros([200, 200, 64, 64])
for i in range(200):
    fill[i, i, :, :] = 1

if opt.cuda:
    G.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_input = noise.cuda(), fixed_input.cuda()

#for i, data in enumerate(dataloader, 0):
############################
# (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
###########################
# train with real
#real_cpu, labels = data
batch_size = opt.batchSize
#if opt.cuda:
#    real_cpu = real_cpu.cuda()
'''
y = torch.LongTensor(batch_size, 1).copy_(labels)
y_onehot = torch.zeros(batch_size, 200)
y_onehot.scatter_(1, y, 1)
y_onehot.view(-1, batch_size, 1, 1)
y_onehot = Variable(y_onehot.cuda())
'''
# train with fake
noise.resize_(batch_size, 100, 1, 1).normal_(0, 1)
noisev = Variable(noise)
#y_nz = torch.cat([noisev, y_onehot], 1)
fake = G(fixed_input)

vutils.save_image(fake.data,
        '%s/conditional_fake.png' % ('log'),
        normalize=True, nrow=10)





























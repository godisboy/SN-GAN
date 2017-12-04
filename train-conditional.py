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
parser.add_argument('--batchSize', type=int, default=32, help='with batchSize=1 equivalent to instance normalization.')
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
    torch.cuda.set_device(opt.gpu_ids[3])

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

def weight_filler(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class _netG(nn.Module):
    def __init__(self, nz, nc, ngf):
        super(_netG, self).__init__()

        self.convT1 = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )
        self.convT2 = nn.Sequential(
            nn.ConvTranspose2d(200, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )
        self.main = nn.Sequential(
            # input is Z, going into a convolution
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

    def forward(self, input, input_c):
        out1 = self.convT1(input)
        out2 = self.convT2(input_c)
        output = torch.cat([out1, out2], 1)
        output = self.main(output)

        return output

class _netD(nn.Module):
    def __init__(self, nc, ndf):
        super(_netD, self).__init__()

        self.conv1_1 = SNConv2d(nc, ndf/2, 4, 2, 1, bias=False)
        self.conv1_2 = SNConv2d(200, ndf/2, 4, 2, 1, bias=False)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            #SNConv2d(nc, ndf, 4, 2, 1, bias=False),
            #nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            SNConv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            SNConv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            SNConv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            SNConv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            #nn.LeakyReLU(0.2, inplace=True)
            #nn.Softplus()
        )

    def forward(self, input, input_c):
        out1 = self.lrelu(self.conv1_1(input))
        out2 = self.lrelu(self.conv1_2(input_c))
        output = torch.cat([out1, out2], 1)
        output = self.main(output)
        return output.view(-1, 1).squeeze(1)

nz = opt.nz

G = _netG(nz, 3, 64)
SND = _netD(3, 64)
print(G)
print(SND)
G.apply(weight_filler)
SND.apply(weight_filler)

input = torch.FloatTensor(32, 3, 64, 64)
noise = torch.FloatTensor(32, nz, 1, 1)
label = torch.FloatTensor(32)
real_label = 1
fake_label = 0

#fixed label
fix_label = torch.FloatTensor(opt.batchSize)

for i in range(0, 4):
    #label_y = np.random.randint(1,200)
    for j in range(0, 8):
        fix_label[i*8+j] = j
    #fix_label[i] = np.random.randint(1,200);

fix = torch.LongTensor(32,1).copy_(fix_label)
fix_onehot = torch.FloatTensor(opt.batchSize, 200)
fix_onehot.zero_()
fix_onehot.scatter_(1, fix, 1)
fix_onehot = fix_onehot.view(-1, 200, 1, 1)

fixed_noise = torch.FloatTensor(32, nz, 1, 1).normal_(0, 1)
#fixed_input = torch.cat([fixed_noise, fix_onehot],1)
fixed_noise, fix_onehot = Variable(fixed_noise), Variable(fix_onehot)

criterion = nn.BCELoss()

fill = torch.zeros([200, 200, 64, 64])
for i in range(200):
    fill[i, i, :, :] = 1

if opt.cuda:
    G.cuda()
    SND.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise, fix_onehot = noise.cuda(), fixed_noise.cuda(), fix_onehot.cuda()

optimizerG = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerSND = optim.Adam(SND.parameters(), lr=0.0002, betas=(0.5, 0.999))

for epoch in range(200):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        SND.zero_grad()
        real_cpu, labels = data
        batch_size = real_cpu.size(0)
        #if opt.cuda:
        #    real_cpu = real_cpu.cuda()
        y = torch.LongTensor(batch_size, 1).copy_(labels)
        y_onehot = torch.zeros(batch_size, 200)
        y_onehot.scatter_(1, y, 1)
        y_onehot_v = y_onehot.view(batch_size, -1, 1, 1)
        #print(y_onehot_v.size())
        y_onehot_v = Variable(y_onehot_v.cuda())

        y_fill = fill[labels]
        y_fill = Variable(y_fill.cuda())

        input.resize_(real_cpu.size()).copy_(real_cpu)
        label.resize_(batch_size).fill_(real_label)
        inputv = Variable(input)
        labelv = Variable(label)
        output = SND(inputv, y_fill)
        #print(output)
        errD_real = torch.mean(F.softplus(-output).mean())
        #errD_real = criterion(output, labelv)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        noise.resize_(batch_size, 100, 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        #y_nz = torch.cat([noisev, y_onehot], 1)
        fake = G(noisev, y_onehot_v)
        labelv = Variable(label.fill_(fake_label))
        output = SND(fake.detach(), y_fill)
        errD_fake = torch.mean(F.softplus(output))
        #errD_fake = criterion(output, labelv)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerSND.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        G.zero_grad()
        labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
        output = SND(fake, y_fill)
        errG = torch.mean(F.softplus(-output))
        #errG = criterion(output, labelv)
        errG.backward()
        D_G_z2 = output.data.mean()
        optimizerG.step()
        if i % 20 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, 200, i, len(dataloader),
                     errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            vutils.save_image(real_cpu,
                    '%s/real_samples.png' % 'log',
                    normalize=True)
            fake = G(fixed_noise, fix_onehot)
            vutils.save_image(fake.data,
                    '%s/fake_samples_epoch_%03d.png' % ('log', epoch),
                    normalize=True)

    # do checkpointing
torch.save(G.state_dict(), '%s/netG_epoch_%d.pth' % ('log', epoch))
torch.save(SND.state_dict(), '%s/netD_epoch_%d.pth' % ('log', epoch))





























import torch.nn as nn
import torch
from torch.nn.modules import conv, Linear
from torch.nn.modules.utils import _pair
import torch.nn.functional as F

#define a Res Generator
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
    #sigma = torch.sum(_u * torch.transpose(torch.matmul(W.data, torch.transpose(_v, 0, 1)), 0, 1), 1)
    return sigma, _u

class SNConv2d(conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(SNConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)
        self.u = None

    def forward(self, input):
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, _u = max_singular_value(w_mat, self.u)
        self.u = _u
        self.weight.data = self.weight.data / sigma
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class SNLinear(Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(SNLinear, self).__init__(in_features, out_features, bias)
        self.u = None
    def forward(self, input):
        w_mat = self.weight
        sigma, _u = max_singular_value(w_mat, self.u)
        self.u = _u
        self.weight.data = self.weight.data / sigma
        return F.linear(input, self.weight, self.bias)


class _netG(nn.Module):
    def __init__(self, nz, nc, ngf):
        super(_netG, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=True),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=True),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 3, 1, 1, bias=True),
            nn.Tanh()
            # state size. (nc) x 32 x 32
        )

    def forward(self, input):
        output = self.main(input)
        return output

class _netD(nn.Module):
    def __init__(self, nc, ndf):
        super(_netD, self).__init__()

        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            #SNConv2d()
            SNConv2d(nc, ndf, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            SNConv2d(ndf, ndf, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf) x 1 x 32
            SNConv2d(ndf, ndf * 2, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            SNConv2d(ndf * 2, ndf * 2, 4, 2, 1, bias=True),
            #nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf*2) x 16 x 16
            SNConv2d(ndf * 2, ndf * 4, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            SNConv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=True),
            #nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf*4) x 8 x 8
            SNConv2d(ndf * 4, ndf * 8, 3, 1, 1, bias=True),
            #nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. (ndf*8) x 4 x 4
            #SNConv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            #nn.Sigmoid()
        )
        self.snlinear = SNLinear(ndf * 8 * 4 * 4, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(output.size(0), -1)
        output = self.snlinear(output)
        return output.view(-1, 1).squeeze(1)
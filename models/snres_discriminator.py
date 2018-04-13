import torch.nn as nn
from src.snlayers.snconv2d import SNConv2d
from src.snlayers.snlinear import SNLinear
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, use_BN = False, downsample=False):
        super(ResBlock, self).__init__()
        #self.conv1 = SNConv2d(n_dim, n_out, kernel_size=3, stride=2)
        hidden_channels = in_channels
        self.downsample = downsample

        self.resblock = self.make_res_block(in_channels, out_channels, hidden_channels, use_BN, downsample)
        self.residual_connect = self.make_residual_connect(in_channels, out_channels)
    def make_res_block(self, in_channels, out_channels, hidden_channels, use_BN, downsample):
        model = []
        if use_BN:
            model += [nn.BatchNorm2d(in_channels)]

        model += [nn.ReLU()]
        model += [SNConv2d(in_channels, hidden_channels, kernel_size=3, padding=1)]
        model += [nn.ReLU()]
        model += [SNConv2d(hidden_channels, out_channels, kernel_size=3, padding=1)]
        if downsample:
            model += [nn.AvgPool2d(2)]
        return nn.Sequential(*model)
    def make_residual_connect(self, in_channels, out_channels):
        model = []
        model += [SNConv2d(in_channels, out_channels, kernel_size=1, padding=0)]
        if self.downsample:
            model += [nn.AvgPool2d(2)]
            return nn.Sequential(*model)
        else:
            return nn.Sequential(*model)

    def forward(self, input):
        return self.resblock(input) + self.residual_connect(input)

class OptimizedBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OptimizedBlock, self).__init__()
        self.res_block = self.make_res_block(in_channels, out_channels)
        self.residual_connect = self.make_residual_connect(in_channels, out_channels)
    def make_res_block(self, in_channels, out_channels):
        model = []
        model += [SNConv2d(in_channels, out_channels, kernel_size=3, padding=1)]
        model += [nn.ReLU()]
        model += [SNConv2d(out_channels, out_channels, kernel_size=3, padding=1)]
        model += [nn.AvgPool2d(2)]
        return nn.Sequential(*model)
    def make_residual_connect(self, in_channels, out_channels):
        model = []
        model += [SNConv2d(in_channels, out_channels, kernel_size=1, padding=0)]
        model += [nn.AvgPool2d(2)]
        return nn.Sequential(*model)
    def forward(self, input):
        return self.res_block(input) + self.residual_connect(input)

class SNResDiscriminator(nn.Module):
    def __init__(self, ndf=64, ndlayers=4):
        super(SNResDiscriminator, self).__init__()
        self.res_d = self.make_model(ndf, ndlayers)
        self.fc = nn.Sequential(SNLinear(ndf*16, 1), nn.Sigmoid())
    def make_model(self, ndf, ndlayers):
        model = []
        model += [OptimizedBlock(3, ndf)]
        tndf = ndf
        for i in range(ndlayers):
            model += [ResBlock(tndf, tndf*2, downsample=True)]
            tndf *= 2
        model += [nn.ReLU()]
        return nn.Sequential(*model)
    def forward(self, input):
        out = self.res_d(input)
        out = F.avg_pool2d(out, out.size(3), stride=1)
        out = out.view(-1, 1024)
        return self.fc(out)

# naiveresnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class NoiseLayer(nn.Module):
    def __init__(self, in_planes, out_planes, level):
        super(NoiseLayer, self).__init__()
        self.noise = nn.Parameter(torch.Tensor(0), requires_grad=False).to(device)

        self.level = level
        self.layers = nn.Sequential(
            nn.ReLU(True),
            nn.BatchNorm2d(in_planes),
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1),
        )

    def forward(self, x):
        if self.noise.numel() == 0:
            self.noise.resize_(x.data[0].shape).uniform_()
            self.noise = (2 * self.noise - 1) * self.level

        y = torch.add(x, self.noise)
        z = self.layers(y)
        return z

class NoiseBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, shortcut=None, level=0.2):
        super(NoiseBasicBlock, self).__init__()
        self.layers = nn.Sequential(
            NoiseLayer(in_planes, planes, level),
            nn.MaxPool2d(stride, stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            NoiseLayer(planes, planes, level),
            nn.BatchNorm2d(planes),
        )
        self.shortcut = shortcut

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y += residual
        y = F.relu(y)
        return y

class NoiseBottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, shortcut=None, level=0.2):
        super(NoiseBottleneck, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            NoiseLayer(planes, planes, level),
            nn.MaxPool2d(stride, stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 4),
        )
        self.shortcut = shortcut

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y += residual
        y = F.relu(y)
        return y

class NoiseResNet(nn.Module):
    def __init__(self, block, nblocks, nchannels, nfilters, nclasses, pool, level):
        super(NoiseResNet, self).__init__()
        self.in_planes = nfilters
        self.pre_layers = nn.Sequential(
            nn.Conv2d(nchannels,nfilters,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(nfilters),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.layer1 = self._make_layer(block, 1*nfilters, nblocks[0], level=level)
        self.layer2 = self._make_layer(block, 2*nfilters, nblocks[1], stride=2, level=level)
        self.layer3 = self._make_layer(block, 4*nfilters, nblocks[2], stride=2, level=level)
        self.layer4 = self._make_layer(block, 8*nfilters, nblocks[3], stride=2, level=level)
        self.avgpool = nn.AvgPool2d(pool, stride=1)
        self.linear = nn.Linear(8*nfilters*block.expansion, nclasses)

    def _make_layer(self, block, planes, nblocks, stride=1, level=0.2):
        shortcut = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            shortcut = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, shortcut, level=level))
        self.in_planes = planes * block.expansion
        for i in range(1, nblocks):
            layers.append(block(self.in_planes, planes, level=level))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.pre_layers(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = self.avgpool(x5)
        x7 = x6.view(x6.size(0), -1)
        x8 = self.linear(x7)
        return x8

def noiseresnet18(nchannels, nfilters, nclasses, pool=7, level=0.1):
    return NoiseResNet(NoiseBasicBlock, [2,2,2,2], nchannels=nchannels, nfilters=nfilters, nclasses=nclasses, pool=pool, level=level)

def noiseresnet34(nchannels, nfilters, nclasses, pool=7, level=0.1):
    return NoiseResNet(NoiseBasicBlock, [3,4,6,3], nchannels=nchannels, nfilters=nfilters, nclasses=nclasses, pool=pool, level=level)

def noiseresnet50(nchannels, nfilters, nclasses, pool=7, level=0.1):
    return NoiseResNet(NoiseBottleneck, [3,4,6,3], nchannels=nchannels, nfilters=nfilters, nclasses=nclasses, pool=pool, level=level)

def noiseresnet101(nchannels, nfilters, nclasses, pool=7, level=0.1):
    return NoiseResNet(NoiseBottleneck, [3,4,23,3], nchannels=nchannels, nfilters=nfilters, nclasses=nclasses, pool=pool, level=level)

def noiseresnet152(nchannels, nfilters, nclasses, pool=7, level=0.1):
    return NoiseResNet(NoiseBottleneck, [3,8,36,3], nchannels=nchannels, nfilters=nfilters, nclasses=nclasses, pool=pool, level=level)

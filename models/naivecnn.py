# hourglass.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class NoiseLayer(nn.Module):
    def __init__(self, in_planes, out_planes, level):
        super(NoiseLayer, self).__init__()
        
        self.noise = torch.randn(1,in_planes,1,1)
        self.level = level
        self.layers = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_planes),
        )

    def forward(self, x):
        tmp1 = x.data.shape
        tmp2 = self.noise.shape
        
        if (tmp1[1] != tmp2[1]) or (tmp1[2] != tmp2[2]) or (tmp1[3] != tmp2[3]):
            self.noise = (2*torch.rand(x.data.shape)-1)*self.level
            self.noise = self.noise.cuda()

        x.data = x.data + self.noise
        x = self.layers(x)
        return x

class NoiseModel(nn.Module):
    def __init__(self, nblocks, nlayers, nchannels, nfilters, nclasses, level):
        super(NoiseModel, self).__init__()

        self.num = nfilters
        self.level = level
        
        layers = []
        layers.append(NoiseLayer(3, nfilters, self.level))
        for i in range(1, nlayers):
            layers.append(self._make_layer(nfilters, nfilters, nblocks, self.level))
            layers.append(nn.MaxPool2d(2,2))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(self.num, nclasses)

    def _make_layer(self, in_planes, out_planes, nblocks, level):
        layers = []
        for i in range(nblocks):
            layers.append(NoiseLayer(in_planes, out_planes, level))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.num)
        x = self.classifier(x)
        return x
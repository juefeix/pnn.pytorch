# model.py

import math
from torch import nn
import models
import losses

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

class Model:

    def __init__(self, args):
        self.cuda = args.cuda
        self.nfilters = args.nfilters
        self.nclasses = args.nclasses
        self.nchannels = args.nchannels
        self.nblocks = args.nblocks
        self.nlayers = args.nlayers
        self.level = args.level
        self.nchannels = args.nchannels
        self.net_type = args.net_type
        self.avgpool = args.avgpool

    def setup(self, checkpoints):
        model = getattr(models, self.net_type)(self.nchannels, self.nfilters, self.nclasses, self.avgpool, self.level)
        criterion = losses.Classification()

        if checkpoints.latest('resume') == None:
            model.apply(weights_init)
        else:
            tmp = checkpoints.load(checkpoints.latest('resume'))
            model.load_state_dict(tmp)

        if self.cuda:
            model = model.cuda()
            criterion = criterion.cuda()

        return model, criterion
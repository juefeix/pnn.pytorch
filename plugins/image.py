# image.py

import os
import math
import torchvision.utils as vutils

class Image:

    def __init__(self, path, ext='png'):
        if os.path.isdir(path) == False:
            os.makedirs(path)
        self.path = path
        self.names = []
        self.ext = ext
        self.iteration = 1
        self.num = 0

    def register(self, modules):
        # here modules is assumed to be a list
        self.num = self.num + len(modules)
        for tmp in modules:
            self.names.append(tmp)

    def update(self, modules):
        # here modules is assumed to be a list
        for i in range(self.num):
            name = os.path.join(self.path, '%s_%03d.png' % (self.names[i], self.iteration))
            nrow = math.ceil(math.sqrt(modules[i].size(0)))
            vutils.save_image(modules[i],name, nrow=nrow, padding=0, normalize=True,scale_each=True)
        self.iteration = self.iteration + 1
# checkpoints.py

import os
import torch

class Checkpoints:
    def __init__(self,args):
        self.dir_save = args.save
        self.dir_load = args.resume

        if os.path.isdir(self.dir_save) == False:
            os.makedirs(self.dir_save)

    def latest(self, name):
        if name == 'resume':
            if self.dir_load == None:
                return None
            else:
                return self.dir_load

    def save(self, epoch, model, best):
        if best == True:
            torch.save(model.state_dict(), '%s/model_epoch_%d.pth' % (self.dir_save, epoch))

        return None
            
    def load(self, filename):
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            model = torch.load(filename)
        else:
            print("=> no checkpoint found at '{}'".format(filename))

        return model
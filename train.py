# train.py

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import plugins
import math


class Trainer():
    def __init__(self, args, model, criterion):

        self.args = args
        self.model = model
        self.criterion = criterion

        self.port = args.port
        self.dir_save = args.save

        self.cuda = args.cuda
        self.nepochs = args.nepochs
        self.nclasses = args.nclasses
        self.nchannels = args.nchannels
        self.batch_size = args.batch_size
        self.resolution_high = args.resolution_high
        self.resolution_wide = args.resolution_wide

        self.lr = args.learning_rate
        self.momentum = args.momentum
        self.adam_beta1 = args.adam_beta1
        self.adam_beta2 = args.adam_beta2
        self.weight_decay = args.weight_decay
        self.optim_method = args.optim_method

        # Felix added
        self.dataset_train_name = args.dataset_train

        parameters = filter(lambda p: p.requires_grad, model.parameters())
        if self.optim_method == 'Adam':
            self.optimizer = optim.Adam(parameters, lr=self.lr, betas=(self.adam_beta1, self.adam_beta2), weight_decay=self.weight_decay)
        elif self.optim_method == 'RMSprop':
            self.optimizer = optim.RMSprop(parameters, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        elif self.optim_method == 'SGD':
            self.optimizer = optim.SGD(parameters, lr=self.lr,  momentum=self.momentum, weight_decay=self.weight_decay, nesterov=True)
        else:
            raise(Exception("Unknown Optimization Method"))

        # for classification
        self.label = torch.zeros(self.batch_size).long()
        self.input = torch.zeros(self.batch_size,self.nchannels,self.resolution_high,self.resolution_wide)

        if args.cuda:
            self.label = self.label.cuda()
            self.input = self.input.cuda()

        self.input = Variable(self.input)
        self.label = Variable(self.label)

        # logging training 
        self.log_loss_train = plugins.Logger(args.logs, 'TrainLogger.txt')
        self.params_loss_train = ['Loss','Accuracy']
        self.log_loss_train.register(self.params_loss_train)

        # logging testing 
        self.log_loss_test = plugins.Logger(args.logs, 'TestLogger.txt')
        self.params_loss_test = ['Loss','Accuracy']
        self.log_loss_test.register(self.params_loss_test)

        # monitor training
        self.monitor_train = plugins.Monitor()
        self.params_monitor_train = ['Loss','Accuracy']
        self.monitor_train.register(self.params_monitor_train)

        # monitor testing
        self.monitor_test = plugins.Monitor()
        self.params_monitor_test = ['Loss','Accuracy']
        self.monitor_test.register(self.params_monitor_test)

        # visualize training
        self.visualizer_train = plugins.Visualizer(self.port, 'Train')
        self.params_visualizer_train = {
        'Loss':{'dtype':'scalar','vtype':'plot'},
        'Accuracy':{'dtype':'scalar','vtype':'plot'},
        }
        self.visualizer_train.register(self.params_visualizer_train)

        # visualize testing
        self.visualizer_test = plugins.Visualizer(self.port, 'Test')
        self.params_visualizer_test = {
        'Loss':{'dtype':'scalar','vtype':'plot'},
        'Accuracy':{'dtype':'scalar','vtype':'plot'},
        }
        self.visualizer_test.register(self.params_visualizer_test)

        # display training progress
        self.print_train = '[%d/%d][%d/%d] '
        for item in self.params_loss_train:
            self.print_train = self.print_train + item + " %.4f "

        # display testing progress
        self.print_test = '[%d/%d][%d/%d] '
        for item in self.params_loss_test:
            self.print_test = self.print_test + item + " %.4f "

        self.evalmodules = []
        
        self.giterations = 0
        self.losses_test = {}
        self.losses_train = {}
        # print(self.model)

    def learning_rate(self, epoch):
        # training schedule
        # for CIFAR10
        ## return self.lr * ((0.1 ** int(epoch >= 60)) * (0.1 ** int(epoch >= 120))* (0.1 ** int(epoch >= 160)))
        # Felix added
        if self.dataset_train_name == 'CIFAR10':
            return self.lr * ((0.1 ** int(epoch >= 60)) * (0.1 ** int(epoch >= 90))* (0.1 ** int(epoch >= 120)))
        elif self.dataset_train_name == 'CIFAR100':
            return self.lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 120))* (0.1 ** int(epoch >= 160)))
        elif self.dataset_train_name == 'MNIST':
            return self.lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 120))* (0.1 ** int(epoch >= 160)))
        elif self.dataset_train_name == 'FRGC':
            return self.lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 120))* (0.1 ** int(epoch >= 160)))
        elif self.dataset_train_name == 'ImageNet':
            decay = math.floor((epoch - 1) / 30)
            return self.lr * math.pow(0.1, decay)

        # return self.lr

    def get_optimizer(self, epoch, optimizer):
        lr = self.learning_rate(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer

    # not sure if this is working as it should
    def model_eval(self):
        self.model.eval()
        for m in self.model.modules():
            for i in range(len(self.evalmodules)):
                if isinstance(m, self.evalmodules[i]):
                    m.train()

    def model_train(self):
        self.model.train()

    def train(self, epoch, dataloader):
        self.monitor_train.reset()
        data_iter = iter(dataloader)

        self.input.volatile = False
        self.label.volatile = False

        self.optimizer = self.get_optimizer(epoch+1, self.optimizer)

        # switch to train mode
        self.model_train()

        i = 0
        while i < len(dataloader):

            ############################
            # Update network
            ############################

            input,label = data_iter.next()
            i += 1

            batch_size = input.size(0)
            if batch_size == self.batch_size:
                self.input.data.resize_(input.size()).copy_(input)
                self.label.data.resize_(label.size()).copy_(label)

                output = self.model(self.input)
                loss = self.criterion(output,self.label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # this is for classfication
                pred = output.data.max(1)[1]

                acc = float(pred.eq(self.label.data).cpu().sum()*100.0) / float(batch_size)
                self.losses_train['Accuracy'] = float(acc)
                self.losses_train['Loss'] = float(loss.data[0])
                self.monitor_train.update(self.losses_train, batch_size)
                print(self.print_train % tuple([epoch, self.nepochs, i, len(dataloader)] + [self.losses_train[key] for key in self.params_monitor_train]))

        loss = self.monitor_train.getvalues()
        self.log_loss_train.update(loss)
        self.visualizer_train.update(loss)
        return self.monitor_train.getvalues('Accuracy')

    def test(self, epoch, dataloader):
        self.monitor_test.reset()
        data_iter = iter(dataloader)

        self.input.volatile = True
        self.label.volatile = True

        # switch to eval mode
        self.model_eval()

        i = 0
        while i < len(dataloader):

            ############################
            # Evaluate Network
            ############################

            input,label = data_iter.next()
            i += 1

            batch_size = input.size(0)
            if batch_size == self.batch_size:
                self.input.data.resize_(input.size()).copy_(input)
                self.label.data.resize_(label.size()).copy_(label)

                self.model.zero_grad()
                output = self.model(self.input)
                loss = self.criterion(output,self.label)

                # this is for classification
                pred = output.data.max(1)[1]
                acc = float(pred.eq(self.label.data).cpu().sum()*100.0) / float(batch_size)
                self.losses_test['Accuracy'] = float(acc)
                self.losses_test['Loss'] = float(loss.data[0])
                self.monitor_test.update(self.losses_test, batch_size)
                print(self.print_test % tuple([epoch, self.nepochs, i, len(dataloader)] + [self.losses_test[key] for key in self.params_monitor_test]))

        loss = self.monitor_test.getvalues()
        self.log_loss_test.update(loss)
        self.visualizer_test.update(loss)
        return self.monitor_test.getvalues('Accuracy')

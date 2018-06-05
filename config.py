# config.py
import os
import datetime
import argparse

result_path = "results/"
result_path = os.path.join(result_path, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S/'))

parser = argparse.ArgumentParser(description='Your project title goes here')

# ======================== Data Setings ============================================
parser.add_argument('--dataset-test', type=str, default='CIFAR10', metavar='', help='name of training dataset')
parser.add_argument('--dataset-train', type=str, default='CIFAR10', metavar='', help='name of training dataset')
parser.add_argument('--split_test', type=float, default=None, metavar='', help='percentage of test dataset to split')
parser.add_argument('--split_train', type=float, default=None, metavar='', help='percentage of train dataset to split')
parser.add_argument('--dataroot', type=str, default='../../data', metavar='', help='path to the data')
parser.add_argument('--save', type=str, default=result_path +'Save', metavar='', help='save the trained models here')
parser.add_argument('--logs', type=str, default=result_path +'Logs', metavar='', help='save the training log files here')
parser.add_argument('--resume', type=str, default=None, metavar='', help='full path of models to resume training')
parser.add_argument('--nclasses', type=int, default=10, metavar='', help='number of classes for classification')
parser.add_argument('--input-filename-test', type=str, default=None, metavar='', help='input test filename for filelist and folderlist')
parser.add_argument('--label-filename-test', type=str, default=None, metavar='', help='label test filename for filelist and folderlist')
parser.add_argument('--input-filename-train', type=str, default=None, metavar='', help='input train filename for filelist and folderlist')
parser.add_argument('--label-filename-train', type=str, default=None, metavar='', help='label train filename for filelist and folderlist')
parser.add_argument('--loader-input', type=str, default=None, metavar='', help='input loader')
parser.add_argument('--loader-label', type=str, default=None, metavar='', help='label loader')

# ======================== Network Model Setings ===================================
parser.add_argument('--nblocks', type=int, default=10, metavar='', help='number of blocks in each layer')
parser.add_argument('--nlayers', type=int, default=6, metavar='', help='number of layers')
parser.add_argument('--nchannels', type=int, default=3, metavar='', help='number of input channels')
parser.add_argument('--nfilters', type=int, default=64, metavar='', help='number of filters in each layer')
parser.add_argument('--avgpool', type=int, default=1, metavar='', help='set to 7 for imagenet and 1 for cifar10')
parser.add_argument('--level', type=float, default=0.1, metavar='', help='noise level for uniform noise')
parser.add_argument('--resolution-high', type=int, default=32, metavar='', help='image resolution height')
parser.add_argument('--resolution-wide', type=int, default=32, metavar='', help='image resolution width')
parser.add_argument('--ndim', type=int, default=None, metavar='', help='number of feature dimensions')
parser.add_argument('--nunits', type=int, default=None, metavar='', help='number of units in hidden layers')
parser.add_argument('--dropout', type=float, default=None, metavar='', help='dropout parameter')
parser.add_argument('--net-type', type=str, default='noiseresnet18', metavar='', help='type of network')
parser.add_argument('--length-scale', type=float, default=None, metavar='', help='length scale')
parser.add_argument('--tau', type=float, default=None, metavar='', help='Tau')

# ======================== Training Settings =======================================
parser.add_argument('--cuda', type=bool, default=True, metavar='', help='run on gpu')
parser.add_argument('--ngpu', type=int, default=1, metavar='', help='number of gpus to use')
parser.add_argument('--batch-size', type=int, default=64, metavar='', help='batch size for training')
parser.add_argument('--nepochs', type=int, default=500, metavar='', help='number of epochs to train')
parser.add_argument('--niters', type=int, default=None, metavar='', help='number of iterations at test time')
parser.add_argument('--epoch-number', type=int, default=None, metavar='', help='epoch number')
parser.add_argument('--nthreads', type=int, default=20, metavar='', help='number of threads for data loading')
parser.add_argument('--manual-seed', type=int, default=1, metavar='', help='manual seed for randomness')
parser.add_argument('--port', type=int, default=8097, metavar='', help='port for visualizing training at http://localhost:port')

# ======================== Hyperparameter Setings ==================================
parser.add_argument('--optim-method', type=str, default='Adam', metavar='', help='the optimization routine ')
parser.add_argument('--learning-rate', type=float, default=1e-3, metavar='', help='learning rate')
parser.add_argument('--learning-rate-decay', type=float, default=None, metavar='', help='learning rate decay')
parser.add_argument('--momentum', type=float, default=0.9, metavar='', help='momentum')
parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='', help='weight decay')
parser.add_argument('--adam-beta1', type=float, default=0.9, metavar='', help='Beta 1 parameter for Adam')
parser.add_argument('--adam-beta2', type=float, default=0.999, metavar='', help='Beta 2 parameter for Adam')

args = parser.parse_args()
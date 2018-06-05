# main.py

import torch
import random
import torchvision
from model import Model
from config import parser
from dataloader import Dataloader
from checkpoints import Checkpoints
from train import Trainer
import utils

# parse the arguments
args = parser.parse_args()
random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
utils.saveargs(args)

# initialize the checkpoint class
checkpoints = Checkpoints(args)

# Create Model
models = Model(args)
model, criterion = models.setup(checkpoints)

# Data Loading
dataloader = Dataloader(args)
loader_train, loader_test = dataloader.create()

# The trainer handles the training loop and evaluation on validation set
trainer = Trainer(args, model, criterion)
loss_train = trainer.visualize(loader_train)
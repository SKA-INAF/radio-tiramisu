
# coding: utf-8

# ## Dependencies

import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from models import tiramisu
from datasets import camvid
from datasets import joint_transforms
import utils.imgs
import utils.training as train_utils
import datetime


CAMVID_PATH = Path('./data/')
RESULTS_PATH = Path('.results/')
WEIGHTS_PATH = Path('.weights/')
RESULTS_PATH.mkdir(exist_ok=True)
WEIGHTS_PATH.mkdir(exist_ok=True)
batch_size = 20

normalize = transforms.Normalize(mean=camvid.mean, std=camvid.std)


test_dset = camvid.CamVid(
    CAMVID_PATH, 'test', joint_transform=None,
    transform=transforms.Compose([
        transforms.Resize([132, 132]),
        transforms.ToTensor(),
        normalize
    ]),
    target_transform=transforms.Compose([
          #transforms.Resize([132, 132]),
          camvid.LabelToLongTensor(),
    ]))
test_loader = torch.utils.data.DataLoader(
    test_dset, batch_size=batch_size, shuffle=False)


LR = 1e-4
LR_DECAY = 0.995
DECAY_EVERY_N_EPOCHS = 1
N_EPOCHS = 1000
torch.cuda.manual_seed(0)


model = tiramisu.FCDenseNet67(n_classes=4).cuda()
model.apply(train_utils.weights_init)
optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, weight_decay=1e-4)
criterion = nn.NLLLoss(weight=camvid.class_weight.cuda()).cuda()


train_utils.load_weights(model, str(WEIGHTS_PATH)+'/latest.th')
train_utils.test(model, test_loader, criterion, epoch=1)  


train_utils.view_sample_predictions(model, test_loader, 0, n=10)


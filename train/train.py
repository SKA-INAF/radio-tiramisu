
# coding: utf-8

# ## Dependencies

# In[1]:

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
import wandb
from torch.utils.tensorboard import SummaryWriter
# ## CamVid
# 
# Clone this repository which holds the CamVid dataset
# ```
# git clone https://github.com/alexgkendall/SegNet-Tutorial
# ```

# In[2]:

# Init wandb logger 
wandb.init(project='astro_tiramisu')
config = wandb.config
config.learning_rate = 0.01

CAMVID_PATH = Path('./data/')
RESULTS_PATH = Path('.results/')
WEIGHTS_PATH = Path('.weights/')
RESULTS_PATH.mkdir(exist_ok=True)
WEIGHTS_PATH.mkdir(exist_ok=True)
batch_size = 20


# In[3]:


normalize = transforms.Normalize(mean=camvid.mean, std=camvid.std)
train_joint_transformer = transforms.Compose([
    #joint_transforms.JointRandomCrop(224), # commented for fine-tuning
    joint_transforms.JointRandomHorizontalFlip()
    ])
train_dset = camvid.CamVid(CAMVID_PATH, 'train',
      joint_transform=train_joint_transformer,
      transform=transforms.Compose([
          transforms.Resize([132, 132]),
          transforms.ToTensor(),
          normalize,
    ]),
    target_transform=transforms.Compose([
          #transforms.Resize([132, 132]),
          camvid.LabelToLongTensor(),
    ]))
train_loader = torch.utils.data.DataLoader(
    train_dset, batch_size=batch_size, shuffle=True)

val_dset = camvid.CamVid(
    CAMVID_PATH, 'val', joint_transform=None,
    transform=transforms.Compose([
        transforms.Resize([132, 132]),
        transforms.ToTensor(),
        normalize
    ]),
    target_transform=transforms.Compose([
          #transforms.Resize([132, 132]),
          camvid.LabelToLongTensor(),
    ]))
    
val_loader = torch.utils.data.DataLoader(
    val_dset, batch_size=batch_size, shuffle=False)

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


# In[4]:


print("Train: %d" %len(train_loader.dataset.imgs))
print("Val: %d" %len(val_loader.dataset.imgs))
print("Test: %d" %len(test_loader.dataset.imgs))
print("Classes: %d" % len(train_loader.dataset.classes))

inputs, targets = next(iter(train_loader))
print("Inputs: ", inputs.size())
print("Targets: ", targets.size())

#utils.imgs.view_image(inputs[0])
#utils.imgs.view_annotated(targets[0])

#train_utils.save_mask(train_loader)

# ## Train

# In[5]:


LR = 1e-4
LR_DECAY = 0.995
DECAY_EVERY_N_EPOCHS = 1
N_EPOCHS = 1000
torch.cuda.manual_seed(0)


# In[6]:


model = tiramisu.FCDenseNet67(n_classes=4).cuda()
model.apply(train_utils.weights_init)
optimizer = torch.optim.RMSprop(model.parameters(), lr=LR, weight_decay=1e-4)
criterion = nn.NLLLoss(weight=camvid.class_weight.cuda()).cuda()
#criterion = nn.NLLLoss().cuda()


# In[7]:
writer = SummaryWriter()
output_file = open("log.txt", 'a')
output_file.write('Date: ' + str(datetime.datetime.now()) + '\n')
'''
for epoch in range(1, N_EPOCHS+1):
    since = time.time()
    
    ### Train ###
    #trn_loss, trn_err = train_utils.train(
    #    model, train_loader, optimizer, criterion, epoch)
    # trn_acc = 1-trn_err
    trn_loss, trn_metrics = train_utils.train(model, train_loader, optimizer, criterion, epoch)    
    writer.add_scalar("Loss/train", trn_loss, epoch)
    for class_name in train_loader.dataset.classes:
        for metric_name in trn_metrics[class_name]:
            writer.add_scalar(metric_name + "/train_" + class_name, trn_metrics[class_name][metric_name], epoch)

    #print('Epoch {:d}\nTrain - Loss: {:.4f}, Acc: {:.4f}'.format(epoch, trn_loss, trn_acc))    
    print('Epoch {:d}\nTrain - Loss: {:.4f}'.format(epoch, trn_loss))
    print('Per class metrics: ')
    for i, class_name in enumerate(train_loader.dataset.classes):
        print('\t {}: \tAcc: {:.4f}, \tIoU: {:.4f}, \tSensitivity: {:.4f}, \tSpecificity: {:.4f}, \tPrecision: {:.4f}, \tDice: {:.4f}, \tObject Precision: {:.4f}, \tObject Recall: {:.4f}'.format(class_name, trn_metrics[class_name]['accuracy'], trn_metrics[class_name]['iou'], trn_metrics[class_name]['sensitivity'], trn_metrics[class_name]['specificity'], trn_metrics[class_name]['precision'], trn_metrics[class_name]['dice'], trn_metrics[class_name]['obj_precision'], trn_metrics[class_name]['obj_recall']))
    #output_file.write('Epoch {:d}\nTrain - Loss: {:.4f}, Acc: {:.4f}'.format(epoch, trn_loss, trn_acc)+ '\n')
    output_file.write('Epoch {:d}\nTrain - Loss: {:.4f}'.format(epoch, trn_loss)+ '\n')
    output_file.write('Per class metrics: \n')
    for i, class_name in enumerate(train_loader.dataset.classes):
        output_file.write('\t {}: \tAcc: {:.4f}, \tIoU: {:.4f}, \tSensitivity: {:.4f}, \tSpecificity: {:.4f}, \tPrecision: {:.4f}, \tDice: {:.4f}, \tObject Precision: {:.4f}, \tObject Recall: {:.4f}\n'.format(class_name, trn_metrics[class_name]['accuracy'], trn_metrics[class_name]['iou'], trn_metrics[class_name]['sensitivity'], trn_metrics[class_name]['specificity'], trn_metrics[class_name]['precision'], trn_metrics[class_name]['dice'], trn_metrics[class_name]['obj_precision'], trn_metrics[class_name]['obj_recall']))
    time_elapsed = time.time() - since  
    print('Train Time {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    output_file.write('Train Time {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)+ '\n')
    
    ### Val ###
    #val_loss, val_err = train_utils.test(model, val_loader, criterion, epoch)    
    #val_acc = 1 - val_err
    val_loss, test_metrics = train_utils.test(model, val_loader, criterion, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    for class_name in test_loader.dataset.classes:
        for metric_name in trn_metrics[class_name]:
            writer.add_scalar(metric_name + "/val_" + class_name, trn_metrics[class_name][metric_name], epoch)
    #print('Val - Loss: {:.4f} | Acc: {:.4f}'.format(val_loss, val_acc))
    print('Val - Loss: {:.4f}'.format(val_loss))
    print('Per class metrics: ')
    for i, class_name in enumerate(test_loader.dataset.classes):
        print('\t {}: \tAcc: {:.4f}, \tIoU: {:.4f}, \tSensitivity: {:.4f}, \tSpecificity: {:.4f}, \tPrecision: {:.4f}, \tDice: {:.4f}, \tObject Precision: {:.4f}, \tObject Recall: {:.4f}'.format(class_name, test_metrics[class_name]['accuracy'], test_metrics[class_name]['iou'], test_metrics[class_name]['sensitivity'], test_metrics[class_name]['specificity'], test_metrics[class_name]['precision'], test_metrics[class_name]['dice'], test_metrics[class_name]['obj_precision'], test_metrics[class_name]['obj_recall']))
    
    #output_file.write('Val - Loss: {:.4f} | Acc: {:.4f}'.format(val_loss, val_acc)+ '\n')
    output_file.write('Val - Loss: {:.4f}'.format(val_loss)+ '\n')
    output_file.write('Per class metrics: \n')
    for i, class_name in enumerate(val_loader.dataset.classes):
        output_file.write('\t {}: \tAcc: {:.4f}, \tIoU: {:.4f}, \tSensitivity: {:.4f}, \tSpecificity: {:.4f}, \tPrecision: {:.4f}, \tDice: {:.4f}, \tObject Precision: {:.4f}, \tObject Recall: {:.4f}\n'.format(class_name, test_metrics[class_name]['accuracy'], test_metrics[class_name]['iou'], test_metrics[class_name]['sensitivity'], test_metrics[class_name]['specificity'], test_metrics[class_name]['precision'], test_metrics[class_name]['dice'], test_metrics[class_name]['obj_precision'], test_metrics[class_name]['obj_recall']))
    time_elapsed = time.time() - since  
    print('Total Time {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
    output_file.write('Total Time {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60)+ '\n')
    train_utils.view_sample_predictions(model, val_loader, epoch, n=5, writer=writer)
    
    #train_utils.save_predictions(im, targ, pred, epoch)
    ### Checkpoint ###    
    val_acc = np.nanmean([test_metrics[class_name]['accuracy'] for class_name in val_loader.dataset.classes])
    train_utils.save_weights(model, epoch, val_loss, val_acc)

    ### Adjust Lr ###
    train_utils.adjust_learning_rate(LR, LR_DECAY, optimizer, epoch, DECAY_EVERY_N_EPOCHS)

output_file.close()

'''
# ## Test

# In[8]:

train_utils.load_weights(model, str(WEIGHTS_PATH)+'/weights-244-0.019-0.922.pth')
test_loss, test_metrics = train_utils.test(model, test_loader, criterion, epoch=1)  
writer.add_scalar("Loss/test", test_loss, 1)
for class_name in test_loader.dataset.classes:
    for metric_name in test_metrics[class_name]:
        wandb.log({f'test_{class_name}/{metric_name}' : test_metrics[class_name][metric_name]})
        writer.add_scalar(metric_name + "/test_" + class_name, test_metrics[class_name][metric_name], 1)


# In[9]:


train_utils.view_sample_predictions(model, test_loader, 1, 100, None)


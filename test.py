
# coding: utf-8

# ## Dependencies

from datasets.rg_data import AstroDataLoaders
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms

import wandb
from models import tiramisu
from datasets import camvid
import utils.training as train_utils


DATA_PATH = Path('./data/')
RESULTS_PATH = Path('.results/')
WEIGHTS_PATH = Path('./weights/')
RESULTS_PATH.mkdir(exist_ok=True)
WEIGHTS_PATH.mkdir(exist_ok=True)
batch_size = 20

normalize = transforms.Normalize(mean=camvid.mean, std=camvid.std)

data_loader = AstroDataLoaders()
test_loader = data_loader.get_test_loader()

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
val_loss, test_metrics = train_utils.test(model, test_loader, criterion, epoch=1)  

print('Val - Loss: {:.4f}'.format(val_loss))
print('Per class metrics: ')
for i, class_name in enumerate(test_loader.dataset.classes):
    print('\t {}: \tAcc: {:.4f}, \tIoU: {:.4f}, \tSensitivity: {:.4f}, \tPrecision: {:.4f}, \tDice: {:.4f}, \tObject Precision: {:.4f}, \tObject Recall: {:.4f}'.format(class_name, test_metrics[class_name]['accuracy'], test_metrics[class_name]['iou'], test_metrics[class_name]['recall'], test_metrics[class_name]['precision'], test_metrics[class_name]['dice'], test_metrics[class_name]['object_precision'], test_metrics[class_name]['object_recall']))


train_utils.view_sample_predictions(model, test_loader, 0, n=10)

train_utils.load_weights(model, str(WEIGHTS_PATH)+args.resume)
test_loss, test_metrics = train_utils.test(model, test_loader, criterion, epoch=1)  
writer.add_scalar("Loss/test", test_loss, 1)

for class_name in test_loader.dataset.classes:
    for metric_name in test_metrics[class_name]:
        wandb.log({f'test_{class_name}/{metric_name}' : test_metrics[class_name][metric_name]})
        writer.add_scalar(metric_name + "/test_" + class_name, test_metrics[class_name][metric_name], 1)

train_utils.view_sample_predictions(model, test_loader, 1, 100, None)
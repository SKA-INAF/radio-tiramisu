import os
import sys
import math
import string
import random
import shutil
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
from . import imgs as img_utils

RESULTS_PATH = './output/'
WEIGHTS_PATH = './weights/'


def save_weights(model, epoch, loss, acc): #err):
    accuracy = sum(acc.values()) / len(acc)
    weights_fname = 'weights-%d-%.3f-%.3f.pth' % (epoch, loss, accuracy)
    weights_fpath = os.path.join(WEIGHTS_PATH, weights_fname)
    torch.save({
            'startEpoch': epoch,
            'loss':loss,
            #'error': err,
            'accuracy': accuracy,
            'state_dict': model.state_dict()
        }, weights_fpath)
    shutil.copyfile(weights_fpath, WEIGHTS_PATH+'latest.th')

def load_weights(model, fpath):
    print("loading weights '{}'".format(fpath))
    weights = torch.load(fpath, map_location=torch.device('cpu'))
    startEpoch = weights['startEpoch']
    model.load_state_dict(weights['state_dict'])
    print("loaded weights (lastEpoch {}, loss {}, error {})"
          .format(startEpoch-1, weights['loss'], weights['accuracy']))
    return startEpoch

def get_predictions(output_batch):
    bs,c,h,w = output_batch.size()
    tensor = output_batch.data
    values, indices = tensor.cpu().max(1)
    indices = indices.view(bs,h,w)
    return indices

def error(preds, targets):
    assert preds.size() == targets.size()
    bs,h,w = preds.size()
    n_pixels = bs*h*w

    incorrect = preds.ne(targets).cpu().sum()
    err = incorrect/n_pixels
    return round(float(err),5)


def compute_accuracy(preds, targets, classes):
    accuracy = {}

    for i, class_name in enumerate(classes):
        intersection = preds == targets # returns true where the prediction matches the ground truth
        current_class = preds == i # isolates the class of interest
        correct = torch.where(torch.logical_and(intersection, current_class), 1., 0.)
        gt = torch.where(targets == i, 1., 0.)

        batch_accuracies = correct.sum((1,2)) / gt.sum((1,2))
        not_nan_mask = batch_accuracies.eq(batch_accuracies)
        accuracy[class_name] = batch_accuracies[not_nan_mask].mean()
    
    return accuracy


def compute_iou(preds, targets, classes):
    iou = {}

    for i, class_name in enumerate(classes):
        intersection = preds == targets # returns true where the prediction matches the ground truth
        current_class = preds == i # isolates the class of interest
        correct = torch.where(torch.logical_and(intersection, current_class), 1., 0.)
        gt = torch.where(targets == i, 1., 0.)
        union = torch.logical_or(current_class, gt)

        batch_iou = correct.sum((1,2)) / union.sum((1,2))
        not_nan_mask = batch_iou.eq(batch_iou)
        iou[class_name] = batch_iou[not_nan_mask].mean()
    
    return iou

def compute_performance(preds, targets):
    assert preds.size() == targets.size()
    tp = targets.mul(preds).eq(1).sum().item()
    fp = targets.eq(0).long().mul(preds).eq(1).sum().item()
    fn = preds.eq(0).long().mul(targets).eq(1).sum().item()
    tn = targets.eq(0).long().mul(preds).eq(0).sum().item()
    return tp, fp, fn, tn

def train(model, trn_loader, optimizer, criterion, epoch):
    model.train()
    trn_loss = 0
    trn_acc = {class_name: [] for class_name in trn_loader.dataset.classes}
    for idx, data in enumerate(trn_loader):
        inputs = Variable(data[0].cuda())
        targets = Variable(data[1].cuda())

        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        trn_loss += loss.data.item()
        pred = get_predictions(output)
        #trn_error += error(pred, targets.data.cpu())
        tmp_acc = compute_accuracy(pred, targets.data.cpu(), trn_loader.dataset.classes) # Type: dict
        for i, class_name in enumerate(tmp_acc):
            if tmp_acc[class_name] != 0:
                trn_acc[class_name].append(tmp_acc[class_name])

    trn_loss /= len(trn_loader)
    #trn_error /= len(trn_loader)
    trn_acc = {class_name: np.mean(trn_acc[class_name]) for class_name in trn_acc}
    #return trn_loss, trn_error
    return trn_loss, trn_acc

def test(model, test_loader, criterion, epoch=1):
    model.eval()
    test_loss = 0
    test_acc = {class_name: [] for class_name in test_loader.dataset.classes}
    for data, target in test_loader:
        with torch.no_grad():
            #data = Variable(data.cuda(), volatile=True)
            #target = Variable(target.cuda())
            data = data.cpu()
            target = target.cpu()
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = get_predictions(output)
            #test_error += error(pred, target.data.cpu())
            #test_acc += compute_accuracy(pred, target.data.cpu(), test_loader.dataset.classes)
            tmp_acc = compute_accuracy(pred, target.data.cpu(), test_loader.dataset.classes) # Type: dict
            for i, class_name in enumerate(tmp_acc):
                if tmp_acc[class_name] != 0:
                    test_acc[class_name].append(tmp_acc[class_name])
    test_loss /= len(test_loader)
    #test_error /= len(test_loader)
    test_acc = {class_name: np.mean(test_acc[class_name]) for class_name in test_acc}
    #return test_loss, test_acc
    return test_loss, test_acc

def adjust_learning_rate(lr, decay, optimizer, cur_epoch, n_epochs):
    """Sets the learning rate to the initially
        configured `lr` decayed by `decay` every `n_epochs`"""
    new_lr = lr * (decay ** (cur_epoch // n_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform(m.weight)
        m.bias.data.zero_()

def predict(model, input_loader, n_batches=1):
    input_loader.batch_size = 1
    predictions = []
    model.eval()
    for input, target in input_loader:
        data = Variable(input.cuda(), volatile=True)
        label = Variable(target.cuda())
        output = model(data)
        pred = get_predictions(output)
        predictions.append([input,target,pred])
    return predictions

def view_sample_predictions(model, loader, epoch, n):
    inputs, targets = next(iter(loader))
    data = Variable(inputs.cpu(), volatile=True)
    label = Variable(targets.cpu())
    output = model(data)
    pred = get_predictions(output)
    batch_size = inputs.size(0)
    imgs_to_save = []
    output_path=".\output"
    for i in range(min(n, batch_size)):
        im = img_utils.view_image(inputs[i])
        t = img_utils.view_annotated(targets[i], False)
        p = img_utils.view_annotated(pred[i], False)
        save_predictions(im, t, p, epoch, i+1)

def save_predictions(im, targ, pred, epoch, idx):
    im = torch.tensor(im).permute(2,0,1)
    targ = torch.tensor(targ).permute(2,0,1)
    pred = torch.tensor(pred).permute(2,0,1)

    imgs_to_save = []
    imgs_to_save.append(im.cpu())
    t = targ.cpu()
    imgs_to_save.append(t)
    np_pred = pred.cpu()
    imgs_to_save.append(np_pred)
    imgs_fname = 'output-'+str(epoch)+'_'+str(idx)+'.png'
    imgs_fpath = os.path.join(str(RESULTS_PATH), imgs_fname)
    save_image(imgs_to_save, imgs_fpath, 3)

def save_mask(loader):   
    i=0 
    for im, target in iter(loader):
        for t in target:
            
            imgs_fname = loader.dataset.imgs[i].split("\\")[2]
            imgs_fpath = os.path.join("./mask_train", imgs_fname)
            t= img_utils.view_annotated(t,False)
            targ = torch.tensor(t).permute(2,0,1)
            save_image(targ.cpu(), imgs_fpath, 3)
            i=i+1
        

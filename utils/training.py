import os
import sys
import math
import string
import random
import shutil
import numpy as np

import wandb
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
from scipy import ndimage
from utils import imgs as img_utils

classes = ['Void', 'Sidelobe', 'Source', 'Galaxy']


RESULTS_PATH = '.results/'
WEIGHTS_PATH = '.weights/'

metric_values = ['union', 'tp', 'fp', 'fn', 'obj_tp', 'obj_fp', 'obj_fn']
metric_names = ['accuracy', 'iou', 'precision', 'recall', 'dice', 'obj_precision', 'obj_recall']


def save_weights(model, epoch, loss, acc): #err):
    weights_fname = 'weights-%d-%.3f-%.3f.pth' % (epoch, loss, acc)
    weights_fpath = os.path.join(WEIGHTS_PATH, weights_fname)
    torch.save({
            'startEpoch': epoch,
            'loss':loss,
            #'error': err,
            'accuracy': acc,
            'state_dict': model.state_dict()
        }, weights_fpath)
    shutil.copyfile(weights_fpath, WEIGHTS_PATH + 'latest.pth')

def load_weights(model, fpath, device="cuda"):
    print("loading weights '{}'".format(fpath))
    if device == 'cpu':
        weights = torch.load(fpath, map_location=torch.device('cpu'))
    else:
        weights = torch.load(fpath)
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


def compute_union(preds, targets, class_name, class_id):
    total_union = {}
    current_class = torch.where(preds == class_id, 1.,0.) # isolates the class of interest
    gt = torch.where(targets == class_id, 1., 0.)
    union = torch.where(torch.logical_or(current_class, gt), 1., 0.)

    total_union = union.sum().item()
    
    return total_union


def compute_object_confusion_matrix(preds, targets, class_name, class_id, threshold=0.5):

    tp = 0
    fp = 0
    fn = 0

    for pred, target in zip(preds, targets):

        gt = torch.where(target == class_id, 1., 0.)
        current_class = torch.where(pred == class_id, 1., 0.) # isolates the class of interest
        pred_objects, nr_pred_objects = ndimage.label(current_class)
        target_objects, nr_target_objects = ndimage.label(gt)

        for pred_idx in range(nr_pred_objects):
            current_obj_pred = torch.where(torch.from_numpy(pred_objects == pred_idx), 1., 0.)

            obj_iou = get_obj_iou(nr_target_objects, target_objects, current_obj_pred)
            if nr_target_objects != 0:
                if obj_iou >= threshold:
                    tp += 1
                else: 
                    fp += 1

        if nr_target_objects > nr_pred_objects:
            fn += (nr_target_objects - nr_pred_objects)
    
    return tp, fp, fn


def get_obj_iou(nr_target_objects, target_objects, current_obj_pred):
    obj_ious = []
    for target_idx in range(nr_target_objects):
        current_obj_target = torch.from_numpy(target_objects == target_idx)
        intersection = torch.where(torch.logical_and(current_obj_pred, current_obj_target), 1., 0.)
        union = torch.where(torch.logical_or(current_obj_pred, current_obj_target), 1., 0.)

        obj_ious.append(intersection.sum() / union.sum())
    if len(obj_ious) > 0:
        return np.nanmax(obj_ious).item()
    else:
        return 0 

def compute_confusion_matrix(preds, targets, class_name, class_id):

    assert preds.size() == targets.size()
    # intersection = preds == targets # returns true where the prediction matches the ground truth
    current_class = preds == class_id # isolates the class of interest
    gt = targets == class_id
    # correct = torch.where(torch.logical_and(intersection, current_class), 1., 0.)

    tp = gt.mul(current_class).eq(1).sum().item()
    fp = gt.eq(0).long().mul(current_class).eq(1).sum().item()
    fn = current_class.eq(0).long().mul(gt).eq(1).sum().item()
    # tn = gt.eq(0).long().mul(current_class).eq(0).sum().item()

    return tp, fp, fn

def division(x,y):
    return x / y if y else 0

def compute_final_metrics(metrics, eps=1e-6):
    final_metrics = {}

    final_metrics['accuracy']       =   division(metrics['tp'], (metrics['tp'] + metrics['fn']))
    final_metrics['iou']            =   division(metrics['tp'], metrics['union'])
    final_metrics['recall']    =   division(metrics['tp'], (metrics['tp'] + metrics['fn']))
    final_metrics['precision']      =   division(metrics['tp'], (metrics['tp'] + metrics['fp']))
    final_metrics['dice']           =   division(metrics['tp'], (metrics['tp'] + metrics['fp'] + metrics['fn']))
    final_metrics['obj_precision']  =   division(metrics['obj_tp'], (metrics['obj_tp'] + metrics['obj_fp']))
    final_metrics['obj_recall']     =   division(metrics['obj_tp'], (metrics['obj_tp'] + metrics['obj_fn']))

    return final_metrics

def compute_batch_metrics(union, tp, fp, fn):

    accuracy       =   division(tp, tp + fn)
    iou            =   division(tp, union)
    precision      =   division(tp, tp + fp)
    recall         =   division(tp, tp + fn)
    dice           =   division(tp, tp + fp + fn)

    return accuracy, iou, precision, recall, dice

def compute_batch_obj_metrics(obj_tp, obj_fp, obj_fn):

    obj_precision  =   division(obj_tp, obj_tp + obj_fp)
    obj_recall     =   division(obj_tp, obj_tp + obj_fn)

    return obj_precision, obj_recall

def wandb_plot_metrics(metrics, split):
    for class_name in classes[1:]:
        wandb.log({split + '_' + class_name + '/' + metric_name: metrics[class_name][metric_name] for metric_name in metric_names})

def train(model, trn_loader, optimizer, criterion, epoch):
    model.train()
    trn_loss = 0

    trn_metrics = {class_name: {metric_name: 0. for metric_name in metric_names} for class_name in classes}
    batch_metrics = {class_name: {metric_name: [] for metric_name in metric_names} for class_name in classes}

    for idx, data in enumerate(trn_loader):

        if idx == 100:
            break
        inputs, targets = data
        inputs = inputs.cuda()
        targets = targets.cuda()

        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        trn_loss += loss.data.item()
        preds = get_predictions(output)

        # Skipping Background class in metric computation (i + 1)
        for i, class_name in enumerate(classes[1:]): 
            union = compute_union(preds, targets.data.cpu(), class_name, i + 1) 
            if union == 0:
                # There is no object with that class, skipping...
                continue

            tp, fp, fn = compute_confusion_matrix(preds, targets.data.cpu(), class_name, i + 1)
            obj_tp, obj_fp, obj_fn = compute_object_confusion_matrix(preds, targets.data.cpu(), class_name, i + 1)

            accuracy, iou, precision, recall, dice = compute_batch_metrics(union, tp, fp, fn)
            obj_precision, obj_recall = compute_batch_obj_metrics(obj_tp, obj_fp, obj_fn)

            batch_metrics[class_name]['accuracy'].append(accuracy)
            batch_metrics[class_name]['iou'].append(iou)
            batch_metrics[class_name]['precision'].append(precision)
            batch_metrics[class_name]['recall'].append(recall)
            batch_metrics[class_name]['dice'].append(dice)
            batch_metrics[class_name]['obj_precision'].append(obj_precision)
            batch_metrics[class_name]['obj_recall'].append(obj_recall)

            # batch_metrics = {class_name: compute_final_metrics(batch_metrics[class_name]) for class_name in classes}
            # for metric_val in metric_values:
            #     trn_metrics[class_name][metric_val] += tmp_values[metric_val][class_name]

    trn_loss /= len(trn_loader)

    for class_name in classes[1:]:
        trn_metrics[class_name] = {metric_name: np.mean(batch_metrics[class_name][metric_name]) for metric_name in metric_names}
    wandb_plot_metrics(trn_metrics, 'train')
    
    return trn_loss, trn_metrics

def test(model, test_loader, criterion, epoch=1):
    model.eval()
    test_loss = 0
    test_metrics = {class_name: {metric_name: 0. for metric_name in metric_names} for class_name in classes}
    batch_metrics = {class_name: {metric_name: [] for metric_name in metric_names} for class_name in classes}

    for data, target in test_loader:
        with torch.no_grad():
            data = data.cuda()
            targets = target.cuda()
            output = model(data)
            test_loss += criterion(output, targets).item()
            preds = get_predictions(output)

            # Skipping Background class in metric computation (i + 1)
            for i, class_name in enumerate(classes[1:]): 
                union = compute_union(preds, targets.data.cpu(), class_name, i + 1) 
                if union == 0:
                    # There is no object with that class, skipping...
                    continue

                tp, fp, fn = compute_confusion_matrix(preds, targets.data.cpu(), class_name, i + 1)
                obj_tp, obj_fp, obj_fn = compute_object_confusion_matrix(preds, targets.data.cpu(), class_name, i + 1)

                accuracy, iou, precision, recall, dice = compute_batch_metrics(union, tp, fp, fn)
                obj_precision, obj_recall = compute_batch_obj_metrics(obj_tp, obj_fp, obj_fn)

                batch_metrics[class_name]['accuracy'].append(accuracy)
                batch_metrics[class_name]['iou'].append(iou)
                batch_metrics[class_name]['precision'].append(precision)
                batch_metrics[class_name]['recall'].append(recall)
                batch_metrics[class_name]['dice'].append(dice)
                batch_metrics[class_name]['obj_precision'].append(obj_precision)
                batch_metrics[class_name]['obj_recall'].append(obj_recall)
            
    test_loss /= len(test_loader)

    for class_name in classes[1:]:
        test_metrics[class_name] = {metric_name: np.mean(batch_metrics[class_name][metric_name]) for metric_name in metric_names}
    wandb_plot_metrics(test_metrics, 'test')
    return test_loss, test_metrics

def adjust_learning_rate(lr, decay, optimizer, cur_epoch, n_epochs):
    """Sets the learning rate to the initially
        configured `lr` decayed by `decay` every `n_epochs`"""
    new_lr = lr * (decay ** (cur_epoch // n_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight)
        m.bias.data.zero_()

def predict(model, input_loader, n_batches=1):
    input_loader.batch_size = 1
    predictions = []
    model.eval()
    for data, target in input_loader:
        data = data.cuda()
        target = target.cuda()
        output = model(data)
        pred = get_predictions(output)
        predictions.append([data, target, pred])
    return predictions

def view_sample_predictions(model, loader, epoch, n, writer):
    inputs, targets = next(iter(loader))
    data = Variable(inputs.cuda(), volatile=True)
    label = Variable(targets.cuda())
    output = model(data)
    pred = get_predictions(output)
    batch_size = inputs.size(0)
    imgs_to_save = []
    output_path=".\output"
    for i in range(min(n, batch_size)):
        im = img_utils.view_image(inputs[i])
        t = img_utils.view_annotated(targets[i], False)
        p = img_utils.view_annotated(pred[i], False)
        save_predictions(im, t, p, epoch, i+1, writer)

def save_predictions(im, targ, pred, epoch, idx, writer):
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
    # writer.add_image(imgs_fname, imgs_to_save)
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
        
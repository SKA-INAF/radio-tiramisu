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
from . import imgs as img_utils


RESULTS_PATH = '.results/'
WEIGHTS_PATH = '.weights/'

metric_values = ['union', 'tp', 'fp', 'tn', 'fn', 'obj_tp', 'obj_fp', 'obj_fn']
metrics = ['accuracy', 'iou', 'precision', 'sensitivity', 'specificity', 'dice', 'obj_precision', 'obj_recall']


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
    shutil.copyfile(weights_fpath, WEIGHTS_PATH + 'latest.th')

def load_weights(model, fpath):
    print("loading weights '{}'".format(fpath))
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


def compute_batch_metrics(preds, targets, classes):
    total_union = {}
    
    for i, class_name in enumerate(classes):
        intersection = torch.where(preds == targets, 1., 0.)  # returns true where the prediction matches the ground truth
        current_class = torch.where(preds == i, 1.,0.) # isolates the class of interest
        correct = torch.where(torch.logical_and(intersection, current_class), 1., 0.)
        gt = torch.where(targets == i, 1., 0.)
        union = torch.where(torch.logical_or(current_class, gt), 1., 0.)

        # batch_accuracies = correct.sum((1,2)) / gt.sum((1,2))
        # not_nan_mask = batch_accuracies.eq(batch_accuracies)
        total_union[class_name] = union.sum()
    
    return total_union


def compute_object_confusion_matrix(preds, targets, classes, threshold=0.5):
    tp = {class_name: 0. for class_name in classes}
    fp = {class_name: 0. for class_name in classes}
    fn = {class_name: 0. for class_name in classes}

    
    #img = preds[0]
    #labeled, nr_objects = ndimage.label(img)# > threshold) 
    for pred, target in zip(preds, targets):
        for i, class_name in enumerate(classes):
            
            #intersection = pred == target # returns true where the prediction matches the ground truth
            gt = torch.where(target == i, 1., 0.)
            current_class = torch.where(pred == i, 1., 0.) # isolates the class of interest
            pred_objects, nr_pred_objects = ndimage.label(current_class)
            target_objects, nr_target_objects = ndimage.label(gt)

            
            for pred_idx in range(nr_pred_objects):
                current_obj_pred = torch.where(torch.from_numpy(pred_objects == pred_idx), 1., 0.)

                obj_iou = get_obj_iou(nr_target_objects, target_objects, current_obj_pred)
                if nr_target_objects != 0:
                    if obj_iou >= threshold:
                        tp[class_name] += 1
                    else: 
                        fp[class_name] += 1

            if nr_target_objects > nr_pred_objects:
                fn[class_name] += (nr_target_objects - nr_pred_objects)

            # if (tp + fp) != 0:
            #     image_precision = tp / (tp + fp)
            # else: 
            #     image_precision = math.nan
            # 
            # if (tp + fn) != 0:
            #     image_recall = tp / (tp + fn)
            # else:
            #     image_recall = math.nan
 
            # obj_precision[class_name].append(image_precision)
            # obj_recall[class_name].append(image_recall)

    #total_tp = {class_name: tp[class_name].sum() for class_name in classes}
    #total_fp = {class_name: fp[class_name].sum() for class_name in classes}
    #total_fn = {class_name: fn[class_name].sum() for class_name in classes}
    
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

def compute_confusion_matrix(preds, targets, classes):
    tp = {class_name: 0 for class_name in classes}
    tn = {class_name: 0 for class_name in classes}
    fp = {class_name: 0 for class_name in classes}
    fn = {class_name: 0 for class_name in classes}

    for i, class_name in enumerate(classes):
        assert preds.size() == targets.size()
        intersection = preds == targets # returns true where the prediction matches the ground truth
        current_class = preds == i # isolates the class of interest
        gt = targets == i
        correct = torch.where(torch.logical_and(intersection, current_class), 1., 0.)

        tp[class_name] = gt.mul(current_class).eq(1).sum().item()
        fp[class_name] = gt.eq(0).long().mul(current_class).eq(1).sum().item()
        fn[class_name] = current_class.eq(0).long().mul(gt).eq(1).sum().item()
        tn[class_name] = gt.eq(0).long().mul(current_class).eq(0).sum().item()

    return tp, fp, fn, tn

def compute_final_metrics(metrics):
    final_metrics = {}

    final_metrics['accuracy'] = metrics['tp'] / (metrics['tp'] + metrics['fn'])
    final_metrics['iou'] = metrics['tp'] / metrics['union']
    final_metrics['sensitivity'] = metrics['tp'] / (metrics['tp'] + metrics['fn'])
    final_metrics['specificity'] = metrics['tn'] / (metrics['tn'] + metrics['fp'])
    final_metrics['precision'] = metrics['tp'] / (metrics['tp'] + metrics['fp'])
    final_metrics['dice'] = metrics['tp'] / (metrics['tp'] + metrics['fp'] + metrics['fn'])
    final_metrics['obj_precision'] = metrics['obj_tp'] / (metrics['obj_tp'] + metrics['obj_fp'])
    final_metrics['obj_recall'] = metrics['obj_tp'] / (metrics['obj_tp'] + metrics['obj_fn'])

    return final_metrics

def wandb_plot_metrics(metrics, split):
    for class_name in trn_loader.dataset.classes:
        wandb.log({split + '_' + class_name + '/' + metric_val: metrics[class_name][metric_val] for metric_val in metric_values})

def train(model, trn_loader, optimizer, criterion, epoch):
    model.train()
    trn_loss = 0
    correct = {class_name: 0 for class_name in trn_loader.dataset.classes}
    gt = {class_name: 0 for class_name in trn_loader.dataset.classes}
    correct = {class_name: 0 for class_name in trn_loader.dataset.classes}


    
    trn_metrics = {class_name: {metric_val: 0. for metric_val in metric_values} for class_name in trn_loader.dataset.classes}
    tmp_values = {metric_val: {} for metric_val in metric_values}


    for idx, data in enumerate(trn_loader):
        inputs = Variable(data[0].cuda())
        targets = Variable(data[1].cuda())

        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        trn_loss += loss.data.item()
        preds = get_predictions(output)
        #trn_error += error(pred, targets.data.cpu())
        tmp_values['union'] = compute_batch_metrics(preds, targets.data.cpu(), trn_loader.dataset.classes) # Type: dict
        tmp_values['tp'], tmp_values['fp'], tmp_values['fn'], tmp_values['tn'] = compute_confusion_matrix(preds, targets.data.cpu(), trn_loader.dataset.classes)
        tmp_values['obj_tp'], tmp_values['obj_fp'], tmp_values['obj_fn'] = compute_object_confusion_matrix(preds, targets.data.cpu(), trn_loader.dataset.classes)
        for i, class_name in enumerate(trn_loader.dataset.classes):
            for metric_val in metric_values:
                trn_metrics[class_name][metric_val] += tmp_values[metric_val][class_name]

    trn_loss /= len(trn_loader)
    #trn_error /= len(trn_loader)
    trn_metrics = {class_name: compute_final_metrics(trn_metrics[class_name]) for class_name in trn_loader.dataset.classes}
    wandb_plot_metrics(trn_metrics, 'train')
    
    #return trn_loss, trn_error
    return trn_loss, trn_metrics

def test(model, test_loader, criterion, epoch=1):
    model.eval()
    test_loss = 0
    test_metrics = {class_name: {metric_val: 0. for metric_val in metric_values} for class_name in test_loader.dataset.classes}
    tmp_values = {metric_val: {} for metric_val in metric_values}
    for data, target in test_loader:
        with torch.no_grad():
            #data = Variable(data.cuda(), volatile=True)
            #target = Variable(target.cuda())
            data = data.cuda()
            targets = target.cuda()
            output = model(data)
            test_loss += criterion(output, targets).item()
            preds = get_predictions(output)
            #test_error += error(pred, target.data.cpu())
            #test_acc += compute_accuracy(pred, target.data.cpu(), test_loader.dataset.classes)
            
            tmp_values['union'] = compute_batch_metrics(preds, targets.data.cpu(), test_loader.dataset.classes) # Type: dict
            tmp_values['tp'], tmp_values['fp'], tmp_values['fn'], tmp_values['tn'] = compute_confusion_matrix(preds, targets.data.cpu(), test_loader.dataset.classes)
            tmp_values['obj_tp'], tmp_values['obj_fp'], tmp_values['obj_fn'] = compute_object_confusion_matrix(preds, targets.data.cpu(), test_loader.dataset.classes)
            for i, class_name in enumerate(test_loader.dataset.classes):
                for metric_val in metric_values:
                    test_metrics[class_name][metric_val] += tmp_values[metric_val][class_name]
    test_loss /= len(test_loader)
    #test_error /= len(test_loader)
    test_metrics = {class_name: compute_final_metrics(test_metrics[class_name]) for class_name in test_loader.dataset.classes}
    wandb_plot_metrics(test_metrics, 'test')
    #return test_loss, test_acc
    return test_loss, test_metrics

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
        
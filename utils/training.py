import os
from utils.metrics import *
import shutil
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision.utils import save_image
import torch.nn.functional as F
import cv2
from utils import imgs as img_utils
from datasets.rg_masks import CLASSES



# classes = ['Void', 'Sidelobe', 'Source', 'Galaxy']


RESULTS_PATH = '.results/'
WEIGHTS_PATH = 'weights/'

metric_values = ['union', 'tp', 'fp', 'fn', 'obj_tp', 'obj_fp', 'obj_fn']
metric_names = ['accuracy', 'iou', 'precision', 'recall', 'dice', 'obj_precision', 'obj_recall']


def save_weights(model, epoch, loss, acc, weights_path="weights"): #err):
    weights_fname = 'weights-%d-%.3f-%.3f.pth' % (epoch, loss, acc)
    weights_fpath = os.path.join(weights_path, weights_fname)
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


def train(model, trn_loader, optimizer, criterion, epoch, device='cuda'):
    model.train()
    trn_loss = 0

    trn_metrics = {class_name: {metric_name: 0. for metric_name in metric_names} for class_name in CLASSES}
    batch_metrics = {class_name: {metric_name: [] for metric_name in metric_names} for class_name in CLASSES}

    for data in tqdm(trn_loader, desc="Training"):

        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        trn_loss += loss.data.item()
        preds = get_predictions(output)

        # Skipping Background class in metric computation (i + 1)
        for i, class_name in enumerate(CLASSES[1:]): 
            union = compute_union(preds, targets.data.cpu(), i + 1) 
            if union == 0:
                # There is no object with that class, skipping...
                continue

            tp, fp, fn, tn = compute_confusion_matrix(preds, targets.data.cpu(), i + 1)
            obj_tp, obj_fp, obj_fn = compute_object_confusion_matrix(preds, targets.data.cpu(), i + 1)

            accuracy, iou, precision, recall, dice = compute_batch_metrics(union, tp, fp, fn, tn)
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

    for class_name in CLASSES[1:]:
        trn_metrics[class_name] = {metric_name: np.mean(batch_metrics[class_name][metric_name]) for metric_name in metric_names}
        trn_metrics[class_name]['f1-score'] = \
            (2 * trn_metrics[class_name]['precision'] * trn_metrics[class_name]['recall']) / \
                (trn_metrics[class_name]['precision'] + trn_metrics[class_name]['recall'])
        trn_metrics[class_name]['obj_f1-score'] = \
            (2 * trn_metrics[class_name]['obj_precision'] * trn_metrics[class_name]['obj_recall']) / \
                (trn_metrics[class_name]['obj_precision'] + trn_metrics[class_name]['obj_recall'])
    
    return trn_loss, trn_metrics

def test(model, test_loader, criterion, epoch=1, device='cuda'):
    model.eval()
    test_loss = 0
    total_metrics = {metric_name: 0. for metric_name in metric_names}
    test_metrics = {class_name: {metric_name: 0. for metric_name in metric_names} for class_name in CLASSES}
    batch_metrics = {class_name: {metric_name: [] for metric_name in metric_names} for class_name in CLASSES}

    skipped = 0
    for batch in tqdm(test_loader, desc="Testing"):
        with torch.no_grad():
            data, target = batch
            data = data.to(device)
            targets = target.to(device)
            output = model(data)
            # if (target < 0).any():
            #     skipped += 1
            #     print(f"Skipping, count: {skipped}")
            #     continue
            # test_loss += criterion(output, targets).item()
            preds = get_predictions(output)

            # Skipping Background class in metric computation (i + 1)
            for i, class_name in enumerate(CLASSES[1:]): 
                union = compute_union(preds, targets.data.cpu(), i + 1) 
                if union == 0:
                    # There is no object with that class, skipping...
                    continue

                tp, fp, fn, tn = compute_confusion_matrix(preds, targets.data.cpu(), i + 1)
                obj_tp, obj_fp, obj_fn = compute_object_confusion_matrix(preds, targets.data.cpu(), i + 1)

                accuracy, iou, precision, recall, dice = compute_batch_metrics(union, tp, fp, fn, tn)
                obj_precision, obj_recall = compute_batch_obj_metrics(obj_tp, obj_fp, obj_fn)

                batch_metrics[class_name]['accuracy'].append(accuracy)
                batch_metrics[class_name]['iou'].append(iou)
                batch_metrics[class_name]['precision'].append(precision)
                batch_metrics[class_name]['recall'].append(recall)
                batch_metrics[class_name]['dice'].append(dice)
                batch_metrics[class_name]['obj_precision'].append(obj_precision)
                batch_metrics[class_name]['obj_recall'].append(obj_recall)
            
    test_loss /= len(test_loader)

    for class_name in CLASSES[1:]:
        test_metrics[class_name] = {metric_name: np.mean(batch_metrics[class_name][metric_name]) for metric_name in metric_names}
        test_metrics[class_name]['f1-score'] = \
            (2 * test_metrics[class_name]['precision'] * test_metrics[class_name]['recall']) / \
                (test_metrics[class_name]['precision'] + test_metrics[class_name]['recall'])
        test_metrics[class_name]['obj_f1-score'] = \
            (2 * test_metrics[class_name]['obj_precision'] * test_metrics[class_name]['obj_recall']) / \
                (test_metrics[class_name]['obj_precision'] + test_metrics[class_name]['obj_recall'])
                
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

def predict(model, input_loader, n_batches=1, device="cuda"):
    input_loader.batch_size = 1
    predictions = []
    model.eval()
    for data, target in input_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        pred = get_predictions(output)
        predictions.append([data, target, pred])
    return predictions

def view_sample_predictions(model, loader, epoch, n, writer, device='cuda'):
    inputs, targets = next(iter(loader))
    data = inputs.to(device)
    label = targets.to(device)
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
    save_image(imgs_to_save, imgs_fpath)

def save_mask(loader):   
    i=0 
    for im, target in iter(loader):
        for t in target:
            
            imgs_fname = loader.dataset.imgs[i].split("\\")[2]
            imgs_fpath = os.path.join("./mask_train", imgs_fname)
            t= img_utils.view_annotated(t,False)
            targ = torch.tensor(t).permute(2,0,1)
            save_image(targ.cpu(), imgs_fpath)
            i=i+1
        
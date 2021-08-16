import torch
import numpy as np
from scipy import ndimage


def compute_union(preds, targets, class_id):
    total_union = {}
    current_class = torch.where(preds == class_id, 1.,0.) # isolates the class of interest
    gt = torch.where(targets == class_id, 1., 0.)
    union = torch.where(torch.logical_or(current_class, gt), 1., 0.)

    total_union = union.sum().item()
    
    return total_union


def compute_object_confusion_matrix(preds, targets, class_id, threshold=0.5):

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

def compute_confusion_matrix(preds, targets, class_id):

    assert preds.size() == targets.size()
    current_class = preds == class_id # isolates the class of interest
    gt = targets == class_id

    tp = gt.mul(current_class).eq(1).sum().item()
    fp = gt.eq(0).long().mul(current_class).eq(1).sum().item()
    fn = current_class.eq(0).long().mul(gt).eq(1).sum().item()
    tn = current_class.eq(0).long().mul(gt).eq(0).sum().item()

    return tp, fp, fn, tn

def division(x,y):
    return x / y if y else 0

def compute_final_metrics(metrics, eps=1e-6):
    final_metrics = {}

    final_metrics['accuracy']       =   division(metrics['tp'], (metrics['tp'] + metrics['fn']))
    final_metrics['iou']            =   division(metrics['tp'], metrics['union'])
    final_metrics['recall']         =   division(metrics['tp'], (metrics['tp'] + metrics['fn']))
    final_metrics['precision']      =   division(metrics['tp'], (metrics['tp'] + metrics['fp']))
    final_metrics['dice']           =   division(metrics['tp'], (metrics['tp'] + metrics['fp'] + metrics['fn']))
    final_metrics['obj_precision']  =   division(metrics['obj_tp'], (metrics['obj_tp'] + metrics['obj_fp']))
    final_metrics['obj_recall']     =   division(metrics['obj_tp'], (metrics['obj_tp'] + metrics['obj_fn']))

    return final_metrics

def compute_batch_metrics(union, tp, fp, fn, tn):

    # TODO IoU and Dice are the same metric, remove?

    accuracy       =   division(tp + tn, tp + fp + tn + fn)
    iou            =   division(tp, union)
    precision      =   division(tp, tp + fp)
    recall         =   division(tp, tp + fn)
    dice           =   division(tp, tp + fp + fn)

    return accuracy, iou, precision, recall, dice

def compute_batch_obj_metrics(obj_tp, obj_fp, obj_fn):

    obj_precision  =   division(obj_tp, obj_tp + obj_fp)
    obj_recall     =   division(obj_tp, obj_tp + obj_fn)

    return obj_precision, obj_recall

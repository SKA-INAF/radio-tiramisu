import torch
import numpy as np
from scipy import ndimage


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
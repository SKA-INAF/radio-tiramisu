import torch


def compute_union(preds, targets, class_id):
    total_union = {}
    # isolates the class of interest
    current_class = torch.where(preds == class_id, 1., 0.)
    gt = torch.where(targets == class_id, 1., 0.)
    union = torch.where(torch.logical_or(current_class, gt), 1., 0.)

    total_union = union.sum().item()

    return total_union


def compute_confusion_matrix(preds, targets, class_id):

    assert preds.size() == targets.size()
    current_class = preds == class_id  # isolates the class of interest
    gt = targets == class_id

    tp = gt.mul(current_class).eq(1).sum().item()
    fp = gt.eq(0).long().mul(current_class).eq(1).sum().item()
    fn = current_class.eq(0).long().mul(gt).eq(1).sum().item()
    tn = current_class.eq(0).long().mul(gt).eq(0).sum().item()

    return tp, fp, fn, tn


def division(x, y):
    return x / y if y else 0


def compute_final_metrics(metrics, eps=1e-6):
    final_metrics = {}

    final_metrics['accuracy'] = division(
        metrics['tp'], (metrics['tp'] + metrics['fn']))
    final_metrics['iou'] = division(metrics['tp'], metrics['union'])
    final_metrics['recall'] = division(
        metrics['tp'], (metrics['tp'] + metrics['fn']))
    final_metrics['precision'] = division(
        metrics['tp'], (metrics['tp'] + metrics['fp']))
    final_metrics['dice'] = division(
        metrics['tp'], (metrics['tp'] + metrics['fp'] + metrics['fn']))
    final_metrics['obj_precision'] = division(
        metrics['obj_tp'], (metrics['obj_tp'] + metrics['obj_fp']))
    final_metrics['obj_recall'] = division(
        metrics['obj_tp'], (metrics['obj_tp'] + metrics['obj_fn']))

    return final_metrics


def compute_batch_metrics(union, tp, fp, fn, tn):

    accuracy = division(tp + tn, tp + fp + tn + fn)
    iou = division(tp, union)
    precision = division(tp, tp + fp)
    recall = division(tp, tp + fn)

    return accuracy, iou, precision, recall


def compute_batch_obj_metrics(obj_tp, obj_fp, obj_fn):

    obj_precision = division(obj_tp, obj_tp + obj_fp)
    obj_recall = division(obj_tp, obj_tp + obj_fn)

    return obj_precision, obj_recall

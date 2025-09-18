import torch
import numpy as np

# ==== Métricas ====
def dice_score(pred, target, num_classes):
    dice = []
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    for c in range(num_classes):
        pred_c = (pred == c).astype(np.uint8)
        target_c = (target == c).astype(np.uint8)
        inter = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        if union > 0:
            dice.append(2.0 * inter / union)
    return np.mean(dice)

def iou_score(pred, target, num_classes):
    ious = []
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    for c in range(num_classes):
        pred_c = (pred == c).astype(np.uint8)
        target_c = (target == c).astype(np.uint8)
        inter = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum() - inter
        if union > 0:
            ious.append(inter / union)
    return np.mean(ious)

from math import isnan
import torch

def compute_iou_binary(mask_bin: torch.Tensor, gt_mask: torch.Tensor, mean=True, eps=False):
    """
    Compute IoU.

    Args:
        mask_bin (torch.Tensor): Shape (B, H, W) or (B, 1, H, W)
        gt_mask (torch.Tensor): Shape (B, H, W) or (B, 1, H, W)

    Returns:
        torch.Tensor: IoU (scalar)

    """
    if len(mask_bin.shape) == 4:
        mask_bin = mask_bin.squeeze(1)
    if len(gt_mask.shape) == 4:
        gt_mask = gt_mask.squeeze(1)

    if eps is True:
        eps = 1e-5
    else:
        eps = False
    
    device = mask_bin.device

    B = mask_bin.shape[0]
    IoU_per_image = []

    for b in range(B):
        # True Positives
        true_positives = torch.sum(mask_bin[b, :, :] * gt_mask[b, :, :])
        # False Positives
        false_positives = torch.sum(mask_bin[b, :, :] * (1-gt_mask[b, :, :]))
        # False Negatives
        false_negatives = torch.sum((1-mask_bin[b, :, :]) * gt_mask[b, :, :])
        #IOU
        Intersection_area = true_positives
        Predicted_area = true_positives + false_positives
        Real_area = true_positives + false_negatives
        Union_area = Predicted_area + Real_area - Intersection_area
        
        if eps is False:
            iou = (Intersection_area) / (Union_area)
        else:
            iou = (Intersection_area + eps) / (Union_area + eps)
        IoU_per_image.append(iou)

    if mean is True:
        IoU = torch.mean(torch.as_tensor(IoU_per_image, device=device), dim=0)
    else:
        IoU = torch.as_tensor(IoU_per_image, device=device)

    return IoU

def compute_F1(mask_bin, gt_mask, eps=False):
    """
    Computer F1 score.

    Args:
        mask_bin (torch.Tensor): Shape (B, H, W) or (B, 1, H, W)
        gt_mask (torch.Tensor): Shape (B, H, W) or (B, 1, H, W)

    Return:
        F1 socre (scalar)

    """
    if len(mask_bin.shape) == 4:
        mask_bin = mask_bin.squeeze(1)
    if len(gt_mask.shape) == 4:
        gt_mask = gt_mask.squeeze(1)

    if eps is True:
        eps = 1e-5
    else:
        eps = False

    # True Positives
    true_positives = torch.sum(mask_bin * gt_mask)
    # False Positives
    false_positives = torch.sum(mask_bin * (1-gt_mask))
    # False Negatives
    false_negatives = torch.sum((1-mask_bin) * gt_mask)
    # Precision
    precision = true_positives / (true_positives + false_positives)
    # Recall
    recall = true_positives / (true_positives + false_negatives)
    # F1 Score
    if eps is False:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = (2 * precision * recall + eps) / (precision + recall + eps)

    return f1

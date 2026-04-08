"""
utils/losses.py — Enhanced loss function factory used by all 3 train scripts.

Supports:
  - weighted_cross_entropy: Standard weighted CE
  - focal: Focal loss for hard example mining
  - ce_dice: Combined CE + Dice loss (best for imbalanced segmentation)
  - ce_focal_dice: All three combined for maximum performance

Bug 8 fix: class_weights explicitly cast to float32, not float64.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced segmentation.
    Applies (1 - p_t)^gamma weighting to down-weight easy examples.
    """

    def __init__(self, gamma=2.0, weight=None, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Args:
            logits:  [B, C, H, W] raw logits
            targets: [B, H, W] int64 ground truth class IDs
        Returns:
            scalar focal loss
        """
        C = logits.shape[1]

        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction='none',
        )

        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        targets_unsqueeze = targets.unsqueeze(1).clamp(0, C - 1)
        p_t = probs.gather(1, targets_unsqueeze).squeeze(1)

        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * ce_loss

        valid = targets != self.ignore_index
        if valid.any():
            return loss[valid].mean()
        return loss.mean()


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for segmentation.
    Complements CE by directly optimizing the IoU-like Dice metric.
    Especially helps with rare/small classes that CE underweights.
    """

    def __init__(self, n_classes=10, smooth=1.0, ignore_index=255):
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Args:
            logits:  [B, C, H, W] raw logits
            targets: [B, H, W] int64 ground truth
        Returns:
            scalar dice loss (1 - mean_dice)
        """
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]

        # Create valid mask
        valid = (targets != self.ignore_index).unsqueeze(1)  # [B, 1, H, W]

        # One-hot encode targets
        targets_clamped = targets.clamp(0, self.n_classes - 1)
        targets_onehot = F.one_hot(targets_clamped, self.n_classes)  # [B, H, W, C]
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        # Apply valid mask
        probs = probs * valid.float()
        targets_onehot = targets_onehot * valid.float()

        # Per-class Dice
        dims = (0, 2, 3)  # sum over batch, H, W
        intersection = (probs * targets_onehot).sum(dims)
        cardinality = probs.sum(dims) + targets_onehot.sum(dims)

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        return 1.0 - dice_per_class.mean()


def lovasz_softmax_flat(probas, labels, ignore_index=255):
    """
    Computes the Lovasz-Softmax loss.
    """
    C = probas.size(1)
    losses = []
    valid = (labels != ignore_index)
    probas = probas[valid]
    labels = labels[valid]
    
    if probas.numel() == 0:
        return probas.sum() * 0.
        
    for c in range(C):
        fg = (labels == c).float() 
        if fg.sum() == 0:
            continue
        class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        
        p = len(fg_sorted)
        gts = fg_sorted.sum()
        intersection = gts - fg_sorted.cumsum(0)
        union = gts + (1 - fg_sorted).cumsum(0)
        jaccard = 1. - intersection / union
        
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
            
        losses.append(torch.dot(errors_sorted, jaccard))
        
    return sum(losses) / len(losses) if len(losses) > 0 else probas.sum() * 0.

class LovaszLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        probas = F.softmax(logits, dim=1)
        probas = probas.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
        targets = targets.view(-1)
        return lovasz_softmax_flat(probas, targets, self.ignore_index)

class CombinedLoss(nn.Module):
    """
    Combined loss: alpha * CE/Focal + beta * Dice.
    Best practice for semantic segmentation — CE drives per-pixel learning,
    Dice optimizes the region-level metric directly.
    """

    def __init__(self, ce_loss, dice_loss, ce_weight=1.0, dice_weight=0.5):
        super().__init__()
        self.ce_loss = ce_loss
        self.dice_loss = dice_loss
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.ce_weight * ce + self.dice_weight * dice


def get_loss_fn(config, device):
    """
    Factory function: create loss from config.

    Supported types:
      - "weighted_cross_entropy": Standard weighted CE
      - "focal": Focal loss only
      - "ce_dice": Weighted CE + Dice (recommended)
      - "ce_focal_dice": Focal + Dice (maximum performance)

    Bug 8 fix: weights explicitly float32.
    """
    loss_cfg = config["loss"]
    ignore_index = loss_cfg["ignore_index"]
    n_classes = config["n_classes"]

    # Class weights — force float32 (Bug 8)
    weights = torch.tensor(loss_cfg["class_weights"], dtype=torch.float32).to(device)

    loss_type = loss_cfg["type"]

    if loss_type == "weighted_cross_entropy":
        return nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index)

    elif loss_type == "focal":
        return FocalLoss(
            gamma=loss_cfg["focal_gamma"],
            weight=weights,
            ignore_index=ignore_index,
        )

    elif loss_type == "ce_dice":
        ce = nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index, label_smoothing=0.1)
        dice = DiceLoss(n_classes=n_classes, ignore_index=ignore_index)
        return CombinedLoss(ce, dice, ce_weight=1.0, dice_weight=0.5)

    elif loss_type == "ce_focal_dice":
        focal = FocalLoss(
            gamma=loss_cfg["focal_gamma"],
            weight=weights,
            ignore_index=ignore_index,
        )
        lovasz = LovaszLoss(ignore_index=ignore_index)
        # 0.5 weight for Lovasz is standard, as it produces large gradients
        return CombinedLoss(focal, lovasz, ce_weight=1.0, dice_weight=0.5)

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

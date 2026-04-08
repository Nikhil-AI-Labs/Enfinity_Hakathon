"""
utils/metrics.py — All evaluation metric functions, shared by all 3 model pipelines.

Key design decisions:
  - compute_iou / compute_dice: per-batch, returns (mean, per_class_list) — used in training loops
  - compute_iou_from_confusion: correct full-dataset IoU from accumulated confusion matrix (Bug 6 fix)
  - Confusion matrix accumulation avoids OOM on large datasets (costs only 10×10 = 100 longs)
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ===========================================================================
# Per-batch metrics (used during training epoch loops for quick feedback)
# ===========================================================================

def compute_iou(pred_logits, labels, n_classes=10, ignore_index=255):
    """
    Compute IoU per class from a single batch of logits.

    Args:
        pred_logits: [B, C, H, W] raw logits
        labels:      [B, H, W] int64 ground truth
    Returns:
        (mean_iou: float, class_iou: list of floats/NaN)
    """
    pred = torch.argmax(pred_logits, dim=1).view(-1)
    target = labels.view(-1)

    iou_per_class = []
    for c in range(n_classes):
        if c == ignore_index:
            continue
        pred_c = pred == c
        target_c = target == c
        intersection = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()
        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append((intersection / union).item())

    return float(np.nanmean(iou_per_class)), iou_per_class


def compute_dice(pred_logits, labels, n_classes=10, smooth=1e-6):
    """
    Compute Dice coefficient per class from a single batch.

    Args:
        pred_logits: [B, C, H, W] raw logits
        labels:      [B, H, W] int64 ground truth
    Returns:
        (mean_dice: float, class_dice: list of floats)
    """
    pred = torch.argmax(pred_logits, dim=1).view(-1)
    target = labels.view(-1)

    dice_per_class = []
    for c in range(n_classes):
        pred_c = pred == c
        target_c = target == c
        intersection = (pred_c & target_c).sum().float()
        dice = (2.0 * intersection + smooth) / (pred_c.sum().float() + target_c.sum().float() + smooth)
        dice_per_class.append(dice.item())

    return float(np.mean(dice_per_class)), dice_per_class


def compute_pixel_accuracy(pred_logits, labels):
    """Fraction of pixels where argmax(pred) == label."""
    pred = torch.argmax(pred_logits, dim=1)
    return float((pred == labels).float().mean().item())


# ===========================================================================
# Full-dataset metrics via confusion matrix (Bug 6 fix — memory-safe)
# ===========================================================================

def build_confusion_matrix(pred, label, n_classes=10):
    """
    Accumulate into a confusion matrix from pre-argmaxed predictions.

    Args:
        pred:  [B, H, W] int64 predicted class IDs
        label: [B, H, W] int64 ground truth class IDs
        n_classes: number of classes

    Returns:
        conf: [n_classes, n_classes] int64 confusion matrix
    """
    mask = (label >= 0) & (label < n_classes)
    conf = torch.bincount(
        n_classes * label[mask].long() + pred[mask].long(),
        minlength=n_classes ** 2,
    ).reshape(n_classes, n_classes)
    return conf


def compute_iou_from_confusion(conf):
    """
    Compute per-class IoU and mean IoU from a confusion matrix.

    Args:
        conf: [n_classes, n_classes] confusion matrix (accumulated across dataset)

    Returns:
        (mean_iou: float, class_iou: list of floats)
    """
    diag = conf.diagonal().float()
    row_sum = conf.sum(dim=1).float()
    col_sum = conf.sum(dim=0).float()
    union = row_sum + col_sum - diag + 1e-6
    iou_per_class = diag / union

    class_iou = iou_per_class.tolist()
    # Only average over classes that are actually present (union > threshold)
    valid = (row_sum + col_sum) > 0
    if valid.any():
        mean_iou = iou_per_class[valid].mean().item()
    else:
        mean_iou = 0.0
    return mean_iou, class_iou


def compute_dice_from_confusion(conf):
    """Compute per-class Dice from confusion matrix."""
    diag = conf.diagonal().float()
    row_sum = conf.sum(dim=1).float()
    col_sum = conf.sum(dim=0).float()
    dice_per_class = (2 * diag + 1e-6) / (row_sum + col_sum + 1e-6)

    class_dice = dice_per_class.tolist()
    valid = (row_sum + col_sum) > 0
    if valid.any():
        mean_dice = dice_per_class[valid].mean().item()
    else:
        mean_dice = 0.0
    return mean_dice, class_dice


# ===========================================================================
# Saving results
# ===========================================================================

def save_metrics_to_file(results_dict, filepath, class_names=None):
    """
    Write a formatted .txt file with IoU, Dice, Pixel Accuracy.

    results_dict must contain: mean_iou, mean_dice, mean_pixel_acc, class_iou, class_dice
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Mean IoU:            {results_dict['mean_iou']:.4f}\n")
        f.write(f"Mean Dice:           {results_dict['mean_dice']:.4f}\n")
        f.write(f"Mean Pixel Accuracy: {results_dict['mean_pixel_acc']:.4f}\n")
        f.write("=" * 60 + "\n\n")

        if class_names and 'class_iou' in results_dict:
            f.write("Per-Class IoU:\n")
            f.write("-" * 40 + "\n")
            best_iou, best_name = -1, ""
            worst_iou, worst_name = 2, ""
            for i, name in enumerate(class_names):
                iou_val = results_dict['class_iou'][i]
                dice_val = results_dict['class_dice'][i] if 'class_dice' in results_dict else float('nan')
                iou_str = f"{iou_val:.4f}" if not np.isnan(iou_val) else "N/A"
                dice_str = f"{dice_val:.4f}" if not np.isnan(dice_val) else "N/A"
                f.write(f"  {name:<20}: IoU={iou_str}  Dice={dice_str}\n")
                if not np.isnan(iou_val):
                    if iou_val > best_iou:
                        best_iou, best_name = iou_val, name
                    if iou_val < worst_iou:
                        worst_iou, worst_name = iou_val, name
            f.write("\n")
            f.write(f"Best class:  {best_name} ({best_iou:.4f})\n")
            f.write(f"Worst class: {worst_name} ({worst_iou:.4f})\n")

    print(f"Saved evaluation metrics to {filepath}")


def save_iou_bar_chart(class_iou, class_names, color_palette, filepath):
    """
    Matplotlib bar chart of per-class IoU with class-specific colors.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    n = len(class_names)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [np.array(color_palette[i]) / 255.0 for i in range(n)]
    valid_iou = [v if not np.isnan(v) else 0 for v in class_iou]
    mean_iou = float(np.nanmean(class_iou))

    bars = ax.bar(range(n), valid_iou, color=colors, edgecolor='black', linewidth=0.8)
    ax.set_xticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel("IoU", fontsize=11)
    ax.set_title(f"Per-Class IoU  (Mean: {mean_iou:.4f})", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=mean_iou, color='red', linestyle='--', linewidth=1.5, label=f'Mean IoU = {mean_iou:.4f}')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved per-class IoU chart to {filepath}")


# ===========================================================================
# Training history plots — 4 standard plots
# ===========================================================================

def save_training_plots(history, output_dir):
    """Save 4 training metric plots: loss+acc, iou, dice, all combined."""
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Loss + Pixel Accuracy
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history['train_loss'], label='Train', linewidth=1.5)
    axes[0].plot(history['val_loss'], label='Val', linewidth=1.5)
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['train_pixel_acc'], label='Train', linewidth=1.5)
    axes[1].plot(history['val_pixel_acc'], label='Val', linewidth=1.5)
    axes[1].set_title('Pixel Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150)
    plt.close()

    # Plot 2: IoU
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history['train_iou'], label='Train IoU', linewidth=1.5)
    axes[0].set_title('Train IoU')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('IoU')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['val_iou'], label='Val IoU', linewidth=1.5, color='tab:orange')
    axes[1].set_title('Validation IoU')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('IoU')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iou_curves.png'), dpi=150)
    plt.close()

    # Plot 3: Dice
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history['train_dice'], label='Train Dice', linewidth=1.5)
    axes[0].set_title('Train Dice')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Dice')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['val_dice'], label='Val Dice', linewidth=1.5, color='tab:orange')
    axes[1].set_title('Validation Dice')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dice_curves.png'), dpi=150)
    plt.close()

    # Plot 4: All 4 metrics — 2×2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, key, title in zip(
        axes.flat,
        ['loss', 'iou', 'dice', 'pixel_acc'],
        ['Loss', 'IoU', 'Dice Score', 'Pixel Accuracy'],
    ):
        ax.plot(history[f'train_{key}'], label='Train', linewidth=1.5)
        ax.plot(history[f'val_{key}'], label='Val', linewidth=1.5)
        ax.set_title(f'{title} vs Epoch')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_metrics_curves.png'), dpi=150)
    plt.close()

    print(f"Saved all training plots to {output_dir}")


def save_history_to_file(history, output_dir):
    """Save per-epoch training history to a formatted text file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "evaluation_metrics.txt")

    with open(filepath, "w") as f:
        f.write("TRAINING RESULTS\n")
        f.write("=" * 100 + "\n\n")

        f.write("Final Metrics:\n")
        f.write(f"  Final Train Loss:     {history['train_loss'][-1]:.4f}\n")
        f.write(f"  Final Val Loss:       {history['val_loss'][-1]:.4f}\n")
        f.write(f"  Final Train IoU:      {history['train_iou'][-1]:.4f}\n")
        f.write(f"  Final Val IoU:        {history['val_iou'][-1]:.4f}\n")
        f.write(f"  Final Train Dice:     {history['train_dice'][-1]:.4f}\n")
        f.write(f"  Final Val Dice:       {history['val_dice'][-1]:.4f}\n")
        f.write(f"  Final Train Accuracy: {history['train_pixel_acc'][-1]:.4f}\n")
        f.write(f"  Final Val Accuracy:   {history['val_pixel_acc'][-1]:.4f}\n")
        f.write("=" * 100 + "\n\n")

        f.write("Best Results:\n")
        f.write(f"  Best Val IoU:      {max(history['val_iou']):.4f} (Epoch {int(np.argmax(history['val_iou'])) + 1})\n")
        f.write(f"  Best Val Dice:     {max(history['val_dice']):.4f} (Epoch {int(np.argmax(history['val_dice'])) + 1})\n")
        f.write(f"  Best Val Accuracy: {max(history['val_pixel_acc']):.4f} (Epoch {int(np.argmax(history['val_pixel_acc'])) + 1})\n")
        f.write(f"  Lowest Val Loss:   {min(history['val_loss']):.4f} (Epoch {int(np.argmin(history['val_loss'])) + 1})\n")
        f.write("=" * 100 + "\n\n")

        f.write("Per-Epoch History:\n")
        f.write("-" * 110 + "\n")
        headers = ['Epoch', 'TrLoss', 'VaLoss', 'TrIoU', 'VaIoU', 'TrDice', 'VaDice', 'TrAcc', 'VaAcc']
        f.write("{:<7} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}\n".format(*headers))
        f.write("-" * 110 + "\n")

        for i in range(len(history['train_loss'])):
            f.write("{:<7} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}\n".format(
                i + 1,
                history['train_loss'][i], history['val_loss'][i],
                history['train_iou'][i], history['val_iou'][i],
                history['train_dice'][i], history['val_dice'][i],
                history['train_pixel_acc'][i], history['val_pixel_acc'][i],
            ))

    print(f"Saved training history to {filepath}")

"""
utils/augmentations.py — Utility for synchronized spatial transforms.

Color augmentations (jitter, grayscale) are handled in dataset.py since they apply
only to images. This file is reserved for spatial augmentations that need sync
between image and mask.
"""

import torch


def get_synchronized_flip(prob=0.5):
    """
    Returns a bool: whether to apply horizontal flip.
    Caller generates flip decision ONCE then applies to both image and mask.

    Args:
        prob: probability of flipping
    Returns:
        bool — True if should flip
    """
    return torch.rand(1).item() < prob

"""
utils/dataset.py — Shared MaskDataset class used by ALL 3 model pipelines.

Enhanced augmentation pipeline:
  - Synchronized spatial: Random HFlip, Random Rotation (applied to BOTH image & mask)
  - Image-only: ColorJitter, RandomGrayscale, GaussianBlur
  - Mask resize: NEAREST interpolation (prevents class-value artifacts)
  - output_size: override for SegFormer (512×512) vs DINOv2 (266×476)
"""

import os
import torch
import numpy as np
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import random


class MaskDataset(Dataset):
    """
    Unified dataset for offroad segmentation.

    Args:
        data_dir (str): Root folder containing Color_Images/ and Segmentation/
        config (dict): Loaded config.json dictionary
        augment (bool): Whether to apply training augmentations
        output_size (tuple, optional): Override (H, W) — e.g. (512, 512) for SegFormer
        return_filename (bool): If True, also returns the filename (for test scripts)
    """

    def __init__(self, data_dir, config, augment=False, output_size=None, return_filename=False):
        self.image_dir = os.path.join(data_dir, "Color_Images")
        self.masks_dir = os.path.join(data_dir, "Segmentation")
        self.config = config
        self.augment = augment
        self.return_filename = return_filename

        # Default output size from config; SegFormer overrides to (512, 512)
        if output_size is not None:
            self.output_size = output_size
        else:
            self.output_size = (config["image_height"], config["image_width"])

        # Build value_map: raw pixel value (int) → class ID (int)
        self.value_map = {int(k): int(v) for k, v in config["classes"]["value_map"].items()}

        # Sorted file list for reproducibility
        self.data_ids = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        # Augmentation config
        aug_cfg = config["augmentation"]
        self.color_jitter = T.ColorJitter(
            brightness=aug_cfg["color_jitter_brightness"],
            contrast=aug_cfg["color_jitter_contrast"],
            saturation=aug_cfg["color_jitter_saturation"],
            hue=aug_cfg["color_jitter_hue"],
        )
        self.random_grayscale = T.RandomGrayscale(p=aug_cfg["random_grayscale_prob"])
        self.hflip_prob = aug_cfg["random_hflip_prob"]

        # Enhanced augmentation params
        self.rotation_degrees = aug_cfg.get("random_rotation_degrees", 10)
        self.gaussian_blur_prob = aug_cfg.get("gaussian_blur_prob", 0.2)
        self.gaussian_blur_kernel = aug_cfg.get("gaussian_blur_kernel", 5)

        # Normalization
        self.normalize = T.Normalize(
            mean=aug_cfg["normalize_mean"],
            std=aug_cfg["normalize_std"],
        )

    @staticmethod
    def convert_mask(mask_pil, value_map):
        """Map raw pixel values (0,100,200,...,10000) to class IDs 0–9.

        Handles both grayscale (2D) and RGB-encoded grayscale (3D) masks.
        Falcon platform sometimes saves masks as 3-channel PNGs where
        all channels are identical — we extract channel 0.
        """
        arr = np.array(mask_pil)

        # Handle 3-channel RGB masks (all channels identical in Falcon format)
        if arr.ndim == 3:
            arr = arr[:, :, 0]

        arr = arr.astype(np.int32)
        new_arr = np.zeros_like(arr, dtype=np.uint8)
        for raw_value, class_id in value_map.items():
            new_arr[arr == raw_value] = class_id
        return Image.fromarray(new_arr)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        mask_path = os.path.join(self.masks_dir, data_id)

        # Load
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = self.convert_mask(mask, self.value_map)

        # --- Synchronized augmentation ---
        if self.augment:
            # 1. Random horizontal flip — SAME decision for both image and mask
            if torch.rand(1).item() < self.hflip_prob:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

            # 2. Random rotation — SAME angle for both (NEAREST for mask)
            if self.rotation_degrees > 0:
                angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
                image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
                mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, fill=0)

            # 3. Color jitter — image ONLY
            image = self.color_jitter(image)

            # 4. Random grayscale — image ONLY
            image = self.random_grayscale(image)

            # 5. Gaussian blur — image ONLY (helps with generalization)
            if torch.rand(1).item() < self.gaussian_blur_prob:
                image = image.filter(ImageFilter.GaussianBlur(radius=self.gaussian_blur_kernel // 2))

        # --- Resize ---
        # BILINEAR for image, NEAREST for mask (critical — prevents class interpolation)
        image = image.resize((self.output_size[1], self.output_size[0]), Image.BILINEAR)
        mask = mask.resize((self.output_size[1], self.output_size[0]), Image.NEAREST)

        # --- To tensor + normalize ---
        image = T.ToTensor()(image)          # [3, H, W] float32 in [0, 1]
        image = self.normalize(image)        # ImageNet normalization

        # Direct numpy→torch: avoids ToTensor()'s /255 → *255 roundtrip
        # which can corrupt class IDs through floating-point rounding
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))  # [H, W] int64, class IDs 0–9

        if self.return_filename:
            return image, mask, data_id
        return image, mask

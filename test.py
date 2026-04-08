"""
DINOv2 ViT-L + DPT: Inference & Evaluation Script with Enhanced TTA

Supports 7-augmentation Test-Time Augmentation (original, H-flip, V-flip,
HV-flip, and 3 multi-scales), with softmax probability averaging.
"""

import sys, os

import json
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.dataset import MaskDataset
from utils.metrics import (
    build_confusion_matrix, compute_iou_from_confusion, compute_dice_from_confusion,
    save_metrics_to_file, save_iou_bar_chart,
)
from model import DPTDecoder


def mask_to_color(mask_np, color_palette):
    h, w = mask_np.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, rgb in enumerate(color_palette):
        color[mask_np == cid] = rgb
    return color


def denormalize(img_tensor, mean, std):
    img = img_tensor.cpu().numpy()
    img = np.moveaxis(img, 0, -1)
    img = img * np.array(std) + np.array(mean)
    return np.clip(img, 0, 1)


def predict_tta(imgs, backbone, decoder, layer_indices, H, W, tta_scales=[0.75, 1.0, 1.25, 1.5]):
    """
    Enhanced Test-Time Augmentation with 7+ augmentations:
    - Original
    - Horizontal flip
    - Vertical flip
    - Horizontal + Vertical flip
    - Multi-scale (0.75x, 1.25x, 1.5x)
    
    Uses softmax probability averaging for better calibration.
    """
    prob_outputs = []
    
    # Helper function to process at a given scale
    def process_scale(imgs_input, target_h, target_w):
        feats = backbone.get_intermediate_layers(imgs_input, n=layer_indices, return_class_token=False)
        logits = decoder(imgs_input, feats)
        # Decoder already outputs at input resolution, resize to target
        logits_resized = F.interpolate(logits, size=(target_h, target_w), mode='bilinear', align_corners=False)
        return F.softmax(logits_resized, dim=1)
    
    # 1. Original (no augmentation)
    prob_outputs.append(process_scale(imgs, H, W))
    
    # 2. Horizontal Flip
    imgs_hflip = torch.flip(imgs, dims=[3])
    probs_hflip = process_scale(imgs_hflip, H, W)
    probs_hflip = torch.flip(probs_hflip, dims=[3])
    prob_outputs.append(probs_hflip)
    
    # 3. Vertical Flip
    imgs_vflip = torch.flip(imgs, dims=[2])
    probs_vflip = process_scale(imgs_vflip, H, W)
    probs_vflip = torch.flip(probs_vflip, dims=[2])
    prob_outputs.append(probs_vflip)
    
    # 4. Horizontal + Vertical Flip
    imgs_hvflip = torch.flip(imgs, dims=[2, 3])
    probs_hvflip = process_scale(imgs_hvflip, H, W)
    probs_hvflip = torch.flip(probs_hvflip, dims=[2, 3])
    prob_outputs.append(probs_hvflip)
    
    # 5-7. Multi-scale augmentations
    for scale in tta_scales:
        if scale == 1.0:
            continue  # Already processed as original
        
        # Round to multiples of 14 for ViT compatibility
        H_scaled = int((H * scale) // 14) * 14
        W_scaled = int((W * scale) // 14) * 14
        
        # Skip if dimensions are invalid
        if H_scaled < 14 or W_scaled < 14:
            continue
            
        imgs_scaled = F.interpolate(imgs, size=(H_scaled, W_scaled), mode='bilinear', align_corners=False)
        probs_scaled = process_scale(imgs_scaled, H, W)
        prob_outputs.append(probs_scaled)
    
    # Average all probability maps
    avg_probs = torch.mean(torch.stack(prob_outputs), dim=0)
    
    # Convert back to logits for consistency (optional, but helps with numerical stability)
    return torch.log(avg_probs + 1e-8)


def main():
    _DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Test Model 4: DINOv2 ViT-L + DPT")
    parser.add_argument("--config", default=os.path.join(_DIR, "config.json"))
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation")
    parser.add_argument("--no-tta", dest="tta", action="store_false", help="Disable TTA (faster)")
    parser.add_argument("--tta_scales", type=str, default="0.75,1.0,1.25,1.5", 
                        help="Comma-separated TTA scales (e.g., '0.75,1.0,1.25')")
    parser.set_defaults(tta=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Model 4 Test] Device: {device}")
    print(f"[Model 4 Test] TTA: {'ENABLED' if args.tta else 'DISABLED'}")

    H, W = cfg["image_height"], cfg["image_width"]
    TH, TW = H // cfg["patch_size"], W // cfg["patch_size"]
    n_classes = cfg["n_classes"]
    layer_indices = cfg["backbone"]["intermediate_layers"]
    
    # Parse TTA scales
    tta_scales = [float(s.strip()) for s in args.tta_scales.split(',')]
    print(f"[Model 4 Test] TTA Scales: {tta_scales}")

    base_output = cfg["paths"].get("output_base", os.path.join(_DIR, "output"))
    model_path = args.model_path or os.path.join(base_output, "checkpoints", "dinov2_dpt_full_lora.pth")
    data_dir = args.data_dir or cfg["paths"]["test_dir"]
    output_dir = args.output_dir or os.path.join(_DIR, "output", "predictions")

    masks_dir = os.path.join(output_dir, "masks")
    masks_color_dir = os.path.join(output_dir, "masks_color")
    comparisons_dir = os.path.join(output_dir, "comparisons")
    for d in [masks_dir, masks_color_dir, comparisons_dir]:
        os.makedirs(d, exist_ok=True)

    print("Loading DINOv2 ViT-L backbone...")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    
    # Inject LoRA BEFORE loading weights (critical for state_dict compatibility)
    print("Injecting LoRA into backbone...")
    from model import inject_lora
    inject_lora(backbone, rank=8, target_blocks=6)
    
    backbone.to(device)

    dummy = torch.randn(1, 3, H, W).to(device)
    with torch.no_grad():
        test_feats = backbone.get_intermediate_layers(dummy, n=layer_indices, return_class_token=False)
    in_channels = test_feats[0].shape[-1]
    del dummy, test_feats

    print(f"Loading DPT decoder from {model_path}...")
    decoder = DPTDecoder(
        in_channels=in_channels, num_classes=n_classes, TH=TH, TW=TW
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    backbone.load_state_dict(checkpoint['backbone'], strict=False)
    decoder.load_state_dict(checkpoint['decoder'])
    
    # Set to eval mode AFTER loading (Bug 5 fix)
    backbone.eval()
    decoder.eval()
    print("Model loaded and set to eval mode")

    test_dataset = MaskDataset(data_dir, cfg, augment=False, return_filename=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg["training"]["batch_size"],
                             shuffle=False, num_workers=cfg["training"]["num_workers"],
                             pin_memory=True)
    print(f"Test samples: {len(test_dataset)}")

    class_names = cfg["classes"]["names"]
    color_palette = cfg["classes"]["color_palette"]
    aug_cfg = cfg["augmentation"]

    conf_matrix = torch.zeros(n_classes, n_classes, dtype=torch.long)
    total_correct, total_pixels = 0, 0
    sample_count = 0
    
    # Inference time tracking
    total_inference_time = 0.0
    num_batches = 0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing", unit="batch")
        for imgs, labels, filenames in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            # Measure inference time
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()
            
            with autocast(enabled=cfg["training"]["mixed_precision"]):
                if args.tta:
                    outputs = predict_tta(imgs, backbone, decoder, layer_indices, H, W, tta_scales)
                else:
                    feats = backbone.get_intermediate_layers(imgs, n=layer_indices, return_class_token=False)
                    outputs = decoder(imgs, feats)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            batch_time = time.time() - start_time
            total_inference_time += batch_time
            num_batches += 1
            
            # Update progress bar with timing info
            avg_time_per_batch = total_inference_time / num_batches
            avg_time_per_image = avg_time_per_batch / imgs.shape[0]
            pbar.set_postfix({
                'batch_time': f'{batch_time:.3f}s',
                'avg_img_time': f'{avg_time_per_image:.3f}s'
            })

            pred = torch.argmax(outputs, dim=1)
            conf_matrix += build_confusion_matrix(pred.cpu(), labels.cpu(), n_classes)
            total_correct += (pred == labels).sum().item()
            total_pixels += labels.numel()

            for i in range(imgs.shape[0]):
                fname = filenames[i]
                base = os.path.splitext(fname)[0]
                pred_np = pred[i].cpu().numpy().astype(np.uint8)

                Image.fromarray(pred_np).save(os.path.join(masks_dir, f"{base}_pred.png"))
                color_mask = mask_to_color(pred_np, color_palette)
                Image.fromarray(color_mask).save(os.path.join(masks_color_dir, f"{base}_pred_color.png"))

                if sample_count < args.num_samples:
                    img_vis = denormalize(imgs[i], aug_cfg["normalize_mean"], aug_cfg["normalize_std"])
                    gt_color = mask_to_color(labels[i].cpu().numpy().astype(np.uint8), color_palette)

                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                    axes[0].imshow(img_vis); axes[0].set_title("Input"); axes[0].axis("off")
                    axes[1].imshow(gt_color); axes[1].set_title("Ground Truth"); axes[1].axis("off")
                    axes[2].imshow(mask_to_color(pred_np, color_palette)); axes[2].set_title("Prediction"); axes[2].axis("off")
                    plt.suptitle(f"Sample: {fname}")
                    plt.tight_layout()
                    plt.savefig(os.path.join(comparisons_dir, f"comparison_{sample_count}.png"),
                                dpi=150, bbox_inches='tight')
                    plt.close()

                sample_count += 1

    mean_iou, class_iou = compute_iou_from_confusion(conf_matrix)
    mean_dice, class_dice = compute_dice_from_confusion(conf_matrix)
    pixel_acc = total_correct / max(total_pixels, 1)
    
    # Calculate timing statistics
    avg_time_per_batch = total_inference_time / num_batches
    avg_time_per_image = total_inference_time / len(test_dataset)
    fps = len(test_dataset) / total_inference_time

    print(f"\n{'='*70}")
    print(f"RESULTS — Model 4 (DINOv2 + DPT)")
    print(f"{'='*70}")
    print(f"  TTA Mode:       {'ENABLED' if args.tta else 'DISABLED'}")
    print(f"  Mean IoU:       {mean_iou:.4f}")
    print(f"  Mean Dice:      {mean_dice:.4f}")
    print(f"  Pixel Accuracy: {pixel_acc:.4f}")
    print(f"\n{'='*70}")
    print(f"INFERENCE TIMING")
    print(f"{'='*70}")
    print(f"  Total Time:           {total_inference_time:.2f}s")
    print(f"  Avg Time per Batch:   {avg_time_per_batch:.3f}s")
    print(f"  Avg Time per Image:   {avg_time_per_image:.3f}s")
    print(f"  Throughput (FPS):     {fps:.2f}")
    print(f"{'='*70}\n")
    
    print("Per-Class IoU:")
    for i, name in enumerate(class_names):
        print(f"  {name:<20}: IoU={class_iou[i]:.4f}")

    results = {
        'mean_iou': mean_iou, 'mean_dice': mean_dice, 'mean_pixel_acc': pixel_acc,
        'class_iou': class_iou, 'class_dice': class_dice,
    }
    save_metrics_to_file(results, os.path.join(output_dir, "evaluation_metrics.txt"), class_names)
    save_iou_bar_chart(class_iou, class_names, color_palette,
                       os.path.join(output_dir, "per_class_iou.png"))
    
    # Save timing information
    timing_file = os.path.join(output_dir, "inference_timing.txt")
    with open(timing_file, "w") as f:
        f.write("INFERENCE TIMING STATISTICS\n")
        f.write("=" * 60 + "\n")
        f.write(f"TTA Mode:              {'ENABLED' if args.tta else 'DISABLED'}\n")
        if args.tta:
            f.write(f"TTA Scales:            {tta_scales}\n")
        f.write(f"Device:                {device}\n")
        f.write(f"Total Images:          {len(test_dataset)}\n")
        f.write(f"Batch Size:            {cfg['training']['batch_size']}\n")
        f.write(f"Total Inference Time:  {total_inference_time:.2f}s\n")
        f.write(f"Avg Time per Batch:    {avg_time_per_batch:.3f}s\n")
        f.write(f"Avg Time per Image:    {avg_time_per_image:.3f}s\n")
        f.write(f"Throughput (FPS):      {fps:.2f}\n")
        f.write("=" * 60 + "\n")
    print(f"Timing statistics saved to {timing_file}")

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()

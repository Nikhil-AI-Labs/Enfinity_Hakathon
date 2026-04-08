"""
DINOv2 ViT-L + DPT Decoder: Training Script

Two-phase training: Phase 1 warms up the DPT head with frozen DINOv2,
Phase 2 injects LoRA into the top-6 backbone blocks for joint fine-tuning.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys

import json
import argparse
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

warnings.filterwarnings("ignore", message="xFormers is not available")

from utils.dataset import MaskDataset
from utils.metrics import (
    build_confusion_matrix, compute_iou_from_confusion, compute_dice_from_confusion,
    save_metrics_to_file, save_iou_bar_chart, save_training_plots, save_history_to_file,
)
from utils.losses import get_loss_fn
from model import DPTDecoder

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


class DINOv2DPTPipeline(nn.Module):
    def __init__(self, backbone, decoder, H, W, layer_indices):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.H = H
        self.W = W
        self.layer_indices = layer_indices
        self.phase1 = True

    def forward(self, imgs):
        if self.phase1:
            with torch.no_grad():
                feats = self.backbone.get_intermediate_layers(
                    imgs, n=self.layer_indices, return_class_token=False
                )
        else:
            feats = self.backbone.get_intermediate_layers(
                imgs, n=self.layer_indices, return_class_token=False
            )

        logits = self.decoder(imgs, feats)
        return F.interpolate(logits, size=(self.H, self.W),
                             mode='bilinear', align_corners=False)


def get_pipe(m):
    return m.module if isinstance(m, nn.DataParallel) else m


def main():
    _DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Train Model 4: DINOv2 ViT-L + DPT")
    parser.add_argument("--config", default=os.path.join(_DIR, "config.json"))
    parser.add_argument("--train_dir", default=None, help="Override training data directory")
    parser.add_argument("--val_dir", default=None, help="Override validation data directory")
    parser.add_argument("--resume_phase2", action="store_true", help="Skip Phase 1 and load weights")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    print(f"[Model 4 — DINOv2+DPT] Device: {device} | GPUs: {n_gpus}")

    H, W = cfg["image_height"], cfg["image_width"]
    TH, TW = H // cfg["patch_size"], W // cfg["patch_size"]
    n_classes = cfg["n_classes"]
    layer_indices = cfg["backbone"]["intermediate_layers"]
    print(f"Image: {H}x{W} → Tokens: {TH}x{TW}")
    print(f"Intermediate layers: {layer_indices}")

    output_dir = cfg["paths"].get("output_base", os.path.join(_DIR, "output"))
    output_dir = os.path.join(output_dir, "checkpoints")
    train_stats_dir = os.path.join(_DIR, "output", "training_stats")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(train_stats_dir, exist_ok=True)

    print("Loading DINOv2 ViT-L backbone...")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    backbone.eval()
    backbone.to(device)
    for p in backbone.parameters():
        p.requires_grad = False

    dummy = torch.randn(1, 3, H, W).to(device)
    with torch.no_grad():
        test_feats = backbone.get_intermediate_layers(dummy, n=layer_indices, return_class_token=False)
    in_channels = test_feats[0].shape[-1]
    del dummy, test_feats

    decoder = DPTDecoder(
        in_channels=in_channels, num_classes=n_classes, TH=TH, TW=TW
    )
    pipeline = DINOv2DPTPipeline(backbone, decoder, H, W, layer_indices).to(device)
    pipeline.phase1 = True

    if n_gpus > 1:
        pipeline = nn.DataParallel(pipeline)
        print(f"DataParallel enabled across {n_gpus} GPUs")

    train_dir = args.train_dir or cfg["paths"]["train_dir"]
    val_dir = args.val_dir or cfg["paths"]["val_dir"]
    effective_batch = cfg["training"]["batch_size"] * max(n_gpus, 1)

    train_dataset = MaskDataset(train_dir, cfg, augment=True)
    val_dataset = MaskDataset(val_dir, cfg, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=effective_batch,
                              shuffle=True, num_workers=cfg["training"]["num_workers"],
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=effective_batch,
                            shuffle=False, num_workers=cfg["training"]["num_workers"],
                            pin_memory=True)
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Batch: {effective_batch}")

    tcfg = cfg["training"]
    n_epochs = tcfg["n_epochs"]
    phase1_epochs = tcfg["phase1_epochs"]
    lr_head = tcfg["lr_head"]
    lr_backbone = tcfg["lr_backbone_finetune"]
    patience = tcfg["early_stopping_patience"]
    grad_clip = tcfg.get("gradient_clip_norm", 1.0)

    loss_fct = get_loss_fn(cfg, device)
    
    aug_cfg = cfg["augmentation"]
    color_palette = cfg["classes"]["color_palette"]

    optimizer = torch.optim.AdamW(
        get_pipe(pipeline).decoder.parameters(),
        lr=lr_head, weight_decay=tcfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase1_epochs)
    scaler = GradScaler(enabled=tcfg["mixed_precision"])

    history = {k: [] for k in [
        'train_loss', 'val_loss', 'train_iou', 'val_iou',
        'train_dice', 'val_dice', 'train_pixel_acc', 'val_pixel_acc'
    ]}
    best_val_iou = 0.0
    patience_counter = 0

    print(f"\n{'='*70}")
    print(f"Training: {n_epochs} epochs (Phase 1: {phase1_epochs}, Phase 2: {n_epochs - phase1_epochs})")
    print(f"Loss: {cfg['loss']['type']} | Grad clip: {grad_clip}")
    print(f"{'='*70}\n")

    # --- RESUME LOGIC ---
    start_epoch = 0
    if args.resume_phase2:
        print("\n--- Resuming from Phase 2 ---")
        save_path = os.path.join(output_dir, "dinov2_dpt_full_lora.pth")
        checkpoint = torch.load(save_path, map_location=device, weights_only=True)
        get_pipe(pipeline).backbone.load_state_dict(checkpoint['backbone'], strict=False)
        get_pipe(pipeline).decoder.load_state_dict(checkpoint['decoder'])
        start_epoch = phase1_epochs
        best_val_iou = 0.0 # Last baseline from your trace
    # --------------------

    for epoch in range(start_epoch, n_epochs):
        if epoch == phase1_epochs:
            print(f"\n{'='*70}")
            print("PHASE 2: Injecting LoRA into Backbone (DPT)")
            print(f"{'='*70}\n")

            pipe = get_pipe(pipeline)
            pipe.phase1 = False

            from model import inject_lora
            lora_params = inject_lora(pipe.backbone, rank=8, target_blocks=6)

            # --- THE FIX: Push new LoRA layers to the GPU ---
            pipeline.to(device)
            # ------------------------------------------------

            optimizer = torch.optim.AdamW([
                {"params": lora_params, "lr": 1e-4}, 
                {"params": pipe.decoder.parameters(), "lr": 1e-4},
            ], weight_decay=tcfg["weight_decay"])

            import math
            accum_steps = tcfg.get("accumulation_steps", 1)
            steps_per_epoch = math.ceil(len(train_loader) / accum_steps)
            total_steps = (n_epochs - phase1_epochs) * steps_per_epoch

            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr=[1e-4, 1e-4], 
                total_steps=total_steps,
                pct_start=0.1, 
                anneal_strategy='cos'
            )
            patience_counter = 0
            best_val_iou = 0.0

        if epoch < phase1_epochs:
            pipeline.train()
            get_pipe(pipeline).backbone.eval()
        else:
            pipeline.train()

        train_losses = []
        train_conf = torch.zeros(n_classes, n_classes, dtype=torch.long)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]", leave=False)
        for batch_idx, (imgs, labels) in enumerate(pbar):
            imgs, labels = imgs.to(device), labels.to(device)

            with autocast(enabled=tcfg["mixed_precision"]):
                outputs = pipeline(imgs)
                loss = loss_fct(outputs, labels)

            accum_steps = tcfg.get("accumulation_steps", 1)
            loss = loss / accum_steps
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in pipeline.parameters() if p.requires_grad], grad_clip
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                if epoch >= phase1_epochs:
                    scheduler.step()

            train_losses.append(loss.item() * accum_steps)
            with torch.no_grad():
                pred = torch.argmax(outputs, dim=1)
                train_conf += build_confusion_matrix(pred.cpu(), labels.cpu(), n_classes)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        pipeline.eval()
        val_losses = []
        val_conf = torch.zeros(n_classes, n_classes, dtype=torch.long)
        val_pixel_correct, val_pixel_total = 0, 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Val]", leave=False)
            saved_plot = False
            for imgs, labels in pbar:
                imgs, labels = imgs.to(device), labels.to(device)
                with autocast(enabled=tcfg["mixed_precision"]):
                    outputs = pipeline(imgs)
                    loss = loss_fct(outputs, labels)
                val_losses.append(loss.item())
                pred = torch.argmax(outputs, dim=1)
                
                if not saved_plot:
                    img_vis = denormalize(imgs[0], aug_cfg["normalize_mean"], aug_cfg["normalize_std"])
                    gt_color = mask_to_color(labels[0].cpu().numpy().astype(np.uint8), color_palette)
                    pred_color = mask_to_color(pred[0].cpu().numpy().astype(np.uint8), color_palette)
                    
                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                    axes[0].imshow(img_vis); axes[0].set_title("Input Color Image"); axes[0].axis("off")
                    axes[1].imshow(gt_color); axes[1].set_title("Ground Truth"); axes[1].axis("off")
                    axes[2].imshow(pred_color); axes[2].set_title("Model Prediction"); axes[2].axis("off")
                    plt.suptitle(f"Epoch {epoch+1} Validation Sample")
                    plt.tight_layout()
                    plt.savefig(os.path.join(train_stats_dir, f"epoch_{epoch+1}_sample.png"), dpi=150, bbox_inches='tight')
                    plt.close()
                    saved_plot = True

                val_conf += build_confusion_matrix(pred.cpu(), labels.cpu(), n_classes)
                val_pixel_correct += (pred == labels).sum().item()
                val_pixel_total += labels.numel()

        train_iou, _ = compute_iou_from_confusion(train_conf)
        train_dice, _ = compute_dice_from_confusion(train_conf)
        val_iou, val_class_iou = compute_iou_from_confusion(val_conf)
        val_dice, val_class_dice = compute_dice_from_confusion(val_conf)
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        val_pixel_acc = val_pixel_correct / max(val_pixel_total, 1)
        train_pixel_acc = float(train_conf.diagonal().sum()) / max(float(train_conf.sum()), 1)

        for k, v in [('train_loss', train_loss), ('val_loss', val_loss),
                      ('train_iou', train_iou), ('val_iou', val_iou),
                      ('train_dice', train_dice), ('val_dice', val_dice),
                      ('train_pixel_acc', train_pixel_acc), ('val_pixel_acc', val_pixel_acc)]:
            history[k].append(v)

        phase = "P1" if epoch < phase1_epochs else "P2"
        lr_cur = optimizer.param_groups[0]['lr']
        print(f"[{phase}] Epoch {epoch+1}/{n_epochs} | "
              f"TrLoss={train_loss:.4f} VaLoss={val_loss:.4f} | "
              f"TrIoU={train_iou:.4f} VaIoU={val_iou:.4f} | "
              f"VaDice={val_dice:.4f} VaAcc={val_pixel_acc:.4f} | lr={lr_cur:.2e}")

        # Scheduler steps organically managed block above if Phase 2
        if epoch < phase1_epochs:
            scheduler.step()

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            patience_counter = 0
            save_path = os.path.join(output_dir, "dinov2_dpt_full_lora.pth")
            torch.save({
                'backbone': get_pipe(pipeline).backbone.state_dict(),
                'decoder': get_pipe(pipeline).decoder.state_dict()
            }, save_path)
            print(f"  ✓ Best val IoU: {best_val_iou:.4f} — saved")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  ✗ Early stopping at epoch {epoch+1}")
                break

    save_training_plots(history, train_stats_dir)
    save_history_to_file(history, train_stats_dir)
    class_names = cfg["classes"]["names"]
    color_palette = cfg["classes"]["color_palette"]
    save_iou_bar_chart(val_class_iou, class_names, color_palette,
                       os.path.join(train_stats_dir, "per_class_iou.png"))
    results = {
        'mean_iou': val_iou, 'mean_dice': val_dice, 'mean_pixel_acc': val_pixel_acc,
        'class_iou': val_class_iou, 'class_dice': val_class_dice,
    }
    save_metrics_to_file(results, os.path.join(train_stats_dir, "final_evaluation.txt"), class_names)

    print(f"\n{'='*70}")
    print(f"Training complete! Best Val IoU: {best_val_iou:.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

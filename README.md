# 🌿 DINOv2 + DPT Offroad Semantic Segmentation

<p align="center">
  <img src="results/training/all_metrics_curves.png" alt="Training Curves" width="750"/>
</p>

<p align="center">
  <a href="#architecture">Architecture</a> •
  <a href="#results">Results</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#dataset">Dataset</a> •
  <a href="#training">Training</a> •
  <a href="#inference">Inference</a> •
  <a href="#project-structure">Structure</a>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white"/>
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white"/>
  <img alt="DINOv2" src="https://img.shields.io/badge/Backbone-DINOv2%20ViT--L%2F14-4267B2?logo=meta&logoColor=white"/>
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green"/>
</p>

---

## Overview

This repository contains the **final and best-performing model** from an Offroad Semantic Segmentation project — a **DINOv2 ViT-L/14 + Dense Prediction Transformer (DPT)** with **LoRA fine-tuning** for efficient domain adaptation.

The model segments unstructured off-road scenes into **10 semantic classes**, handling challenging terrain such as dense vegetation, dry grass, rocks, logs, and open landscape.

### 🏆 Performance Highlights

| Split | Mean IoU | Mean Dice | Pixel Accuracy |
|-------|----------|-----------|----------------|
| **Validation** | **66.76%** | **78.95%** | **87.14%** |
| **Test** | **30.81%** | **37.85%** | **66.61%** |

> The test set contains a highly skewed class distribution (several near-zero classes like Flowers, Logs, Ground Clutter), which compresses the mean IoU. Per-class breakdown in the [Results](#results) section below.

---

## Architecture

<p align="center">
  <img src="results/training/epoch_samples/epoch_40_sample.png" alt="Model Prediction — Epoch 40" width="750"/>
  <br><em>Model output at the final epoch (40): Input | Ground Truth | Prediction</em>
</p>

### Design Principles

The architecture is purpose-built for off-road terrain parsing, bridging two key capabilities:
1. **Rich semantic context** from DINOv2's powerful self-supervised representations
2. **Pixel-precise boundaries** via multi-scale DPT feature fusion and a shallow CNN branch

```
Input Image (952×532)
       │
       ├──────────────────────────────┐
       │                              │
  DINOv2 ViT-L/14                ShallowCNN
  (layers 5, 11, 17, 23)          (2×stride-2 Conv)
       │                              │
  [B, TH×TW, 1024] × 4          [B, 64, H/4, W/4]
       │                              │
  Reshape to spatial grids           │
  + 1×1 Projection → 256             │
       │                              │
  DPT Top-Down Fusion                │
  (FeatureFusionBlocks)              │
       │                              │
  Bilinear upsample to H/4 ──────────┤
                                      │
                              Cat([256+64, H/4, W/4])
                                      │
                              Refinement Conv (256→128→64)
                                      │
                              Classifier Conv (64→10)
                                      │
                        Bilinear upsample → (H, W)
                                      │
                              Logits [B, 10, 952, 532]
```

### Key Modules

#### 1. DINOv2 ViT-L/14 Backbone
- Pre-trained via self-supervised DINO v2 objective on LVD-142M dataset
- Provides semantically rich patch embeddings at 1/14th input resolution
- Features extracted from **4 intermediate layers** `[5, 11, 17, 23]` to capture multi-scale context

#### 2. LoRA Fine-tuning (Parameter-Efficient)
- **Low-Rank Adaptation** injected into the QKV projections of the **last 6 transformer blocks**
- Rank-8 matrices (A and B) added in parallel — only ~300K extra parameters
- Keeps frozen backbone knowledge intact while adapting to off-road domain
- Training only: LoRA is injected after Phase 1 warmup

```python
# LoRA update equation
output = W_frozen(x) + (α/r) * B(A(x))
# where W_frozen is frozen, A ∈ R^{d×r}, B ∈ R^{r×d}, r=8, α=16
```

#### 3. DPT Decoder
- **Dense Prediction Transformer** decoder with top-down feature fusion
- Uses `FeatureFusionBlock` = `ResidualConvUnit` + bilinear upsample + skip connection add
- Fuses the deepest layer first, then progressively incorporates shallower features

#### 4. Shallow CNN Branch
- Two stride-2 ConvBNGELU blocks operating directly on the raw RGB image
- Captures **fine-grained texture and edge information** (grass strands, rock crevices) at H/4 resolution
- Concatenated with the DPT output before final classification

### Two-Phase Training Strategy

```
Phase 1 (Epochs 1–10): Head Warmup
  ├── DINOv2 backbone: FULLY FROZEN (no_grad)
  ├── Only DPT decoder + ShallowCNN train
  ├── Optimizer: AdamW, lr=2e-4
  └── Scheduler: CosineAnnealingLR

Phase 2 (Epochs 11–40): Joint Fine-tuning
  ├── LoRA layers injected into top 6 backbone blocks
  ├── Backbone LoRA + Decoder trained jointly
  ├── Optimizer: AdamW with two param groups (lr=1e-4 each)
  └── Scheduler: OneCycleLR (pct_start=0.1, cosine anneal)
```

---

## Results

### Training Curves

<p align="center">
  <img src="results/training/training_curves.png" alt="Loss Curves" width="48%"/>
  <img src="results/training/iou_curves.png" alt="IoU Curves" width="48%"/>
</p>
<p align="center">
  <img src="results/training/dice_curves.png" alt="Dice Curves" width="48%"/>
  <img src="results/training/val_per_class_iou.png" alt="Val Per-Class IoU" width="48%"/>
</p>

### Per-Epoch Validation Summary (Best Results)

| Metric | Best Value | Epoch |
|--------|-----------|-------|
| Val IoU | **0.6676** | 40 |
| Val Dice | **0.7895** | 40 |
| Val Accuracy | **0.8715** | 39 |
| Val Loss | **0.3757** | 40 |

### Test Set — Per-Class Breakdown

<p align="center">
  <img src="results/test/test_per_class_iou.png" alt="Test Per-Class IoU" width="600"/>
</p>

| Class | IoU | Dice |
|-------|-----|------|
| Trees | 0.4673 | 0.6369 |
| Lush Bushes | 0.0007 | 0.0014 |
| Dry Grass | **0.4890** | **0.6568** |
| Dry Bushes | 0.3775 | 0.5481 |
| Ground Clutter | 0.0000 | 0.0000 |
| Flowers | 0.0000 | 0.0000 |
| Logs | 0.0000 | 0.0000 |
| Rocks | 0.0716 | 0.1336 |
| Landscape | 0.6866 | 0.8142 |
| **Sky** | **0.9886** | **0.9943** |

> **Note:** Ground Clutter, Flowers, and Logs have near-zero or zero IoU on the test split due to extreme class imbalance — these classes have very few pixels in the test set. Sky (99%) and Landscape (69%) demonstrate the model's core capability.

### Inference Performance

| Metric | Value |
|--------|-------|
| Test Images | 1,002 |
| TTA Mode | ENABLED (7 augmentations) |
| TTA Scales | 0.75×, 1.0×, 1.25×, 1.5× |
| Avg Time per Image | **1.311s** |
| Throughput | **0.76 FPS** |
| Device | CUDA |

> TTA involves 7 augmented forward passes (original + H-flip + V-flip + HV-flip + 3 scales), which increases accuracy at the cost of speed.

### Qualitative Comparisons

<p align="center">
  <img src="results/test/comparisons/comparison_0.png" alt="Comparison 0" width="750"/>
  <br><em>Sample 1: Input | Ground Truth | Prediction</em>
</p>
<p align="center">
  <img src="results/test/comparisons/comparison_1.png" alt="Comparison 1" width="750"/>
  <br><em>Sample 2: Input | Ground Truth | Prediction</em>
</p>
<p align="center">
  <img src="results/test/comparisons/comparison_2.png" alt="Comparison 2" width="750"/>
  <br><em>Sample 3: Input | Ground Truth | Prediction</em>
</p>

> 10 total comparison images available in [`results/test/comparisons/`](results/test/comparisons/)

---

## Quick Start

### Prerequisites

```bash
# Python 3.9+ and CUDA-capable GPU recommended
pip install -r requirements.txt
```

**requirements.txt includes:**
- `torch >= 2.0.0`
- `torchvision >= 0.15.0`
- `numpy >= 1.21.0`
- `Pillow >= 9.0.0`
- `tqdm >= 4.64.0`
- `matplotlib >= 3.5.0`

---

## Dataset

### Structure

The model expects the following dataset directory layout:

```
dataset/
├── train/
│   ├── Color_Images/        # RGB .png input images
│   └── Segmentation/        # Mask .png files (same filename)
├── val/
│   ├── Color_Images/
│   └── Segmentation/
└── test/
    ├── Color_Images/
    └── Segmentation/
```

### Segmentation Classes

| Raw Pixel Value | Class ID | Class Name | Color |
|----------------|----------|------------|-------|
| 100 | 0 | Trees | 🟢 Forest Green |
| 200 | 1 | Lush Bushes | 🟩 Bright Green |
| 300 | 2 | Dry Grass | 🟫 Tan |
| 500 | 3 | Dry Bushes | 🟤 Brown |
| 550 | 4 | Ground Clutter | 🫒 Olive |
| 600 | 5 | Flowers | 🌸 Pink |
| 700 | 6 | Logs | 🍂 Dark Brown |
| 800 | 7 | Rocks | ⬜ Gray |
| 7100 | 8 | Landscape | 🔶 Sienna |
| 10000 | 9 | Sky | 🔵 Sky Blue |

### Update `config.json` Paths

```json
"paths": {
    "train_dir": "/path/to/your/train",
    "val_dir":   "/path/to/your/val",
    "test_dir":  "/path/to/your/test",
    "output_base": "/path/to/output"
}
```

---

## Training

### Full Training (Phase 1 → Phase 2)

```bash
python train.py --config config.json
```

This runs the complete 2-phase training pipeline:
- **Phase 1** (10 epochs): Backbone frozen, decoder warm-up
- **Phase 2** (30 epochs): LoRA injected, joint optimization

The best checkpoint is saved as `dinov2_dpt_full_lora.pth` in the output directory.

### Resume from Phase 2

If Phase 1 is already complete, skip it and go straight to Phase 2 fine-tuning:

```bash
python train.py --config config.json --resume_phase2
```

### Custom Data Directories

```bash
python train.py \
  --config config.json \
  --train_dir /custom/path/to/train \
  --val_dir /custom/path/to/val
```

### Key Hyperparameters (`config.json`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `image_width` | 952 | Must be divisible by 14 (ViT patch size) |
| `image_height` | 532 | Must be divisible by 14 |
| `batch_size` | 4 | + gradient accumulation ×4 = effective batch 16 |
| `n_epochs` | 40 | 10 phase1 + 30 phase2 |
| `lr_head` | 2e-4 | Phase 1 decoder LR |
| `loss.type` | `ce_focal_dice` | Combined CE + Focal + Dice loss |
| `backbone.intermediate_layers` | [5, 11, 17, 23] | Multi-scale ViT layer indices |

---

## Inference

### Standard Inference (No TTA)

```bash
python test.py \
  --config config.json \
  --model_path /path/to/dinov2_dpt_full_lora.pth \
  --data_dir /path/to/test \
  --output_dir ./output \
  --no-tta
```

### Inference with Test-Time Augmentation (Recommended)

```bash
python test.py \
  --config config.json \
  --model_path /path/to/dinov2_dpt_full_lora.pth \
  --output_dir ./output \
  --tta \
  --tta_scales "0.75,1.0,1.25,1.5"
```

**TTA applies 7 augmented forward passes:**
1. Original image
2. Horizontal flip → unflip
3. Vertical flip → unflip
4. H+V flip → unflip
5. Scale 0.75× → resize back
6. Scale 1.25× → resize back
7. Scale 1.5× → resize back

Softmax probabilities are averaged across all 7 passes before taking argmax.

### Output Structure

```
output/
├── masks/                   # Raw class-ID masks (uint8 PNG, values 0–9)
├── masks_color/             # RGB colorized segmentation masks
├── comparisons/             # Side-by-side: Input | GT | Prediction
├── evaluation_metrics.txt   # Mean IoU, Dice, per-class breakdown
├── per_class_iou.png        # Bar chart visualization
└── inference_timing.txt     # FPS, per-image/batch timing statistics
```

---

## Project Structure

```
DPT_Offroad_Segmentation/
│
├── model.py                    # All model definitions
│   ├── LoRAQKV                 # Low-Rank Adaptation module
│   ├── inject_lora()           # Helper to inject LoRA into backbone
│   ├── ShallowCNN              # High-res edge/texture extractor
│   ├── ResidualConvUnit        # Pre-activation residual block
│   ├── FeatureFusionBlock      # DPT skip-connection fusion
│   └── DPTDecoder              # Full decoder pipeline
│
├── train.py                    # Training pipeline
│   └── DINOv2DPTPipeline       # Backbone + Decoder wrapper for DataParallel
│
├── test.py                     # Inference + evaluation with TTA
│   └── predict_tta()           # 7-augmentation TTA function
│
├── config.json                 # All hyperparameters and paths
├── requirements.txt            # Python dependencies
│
├── utils/                      # Shared utilities
│   ├── dataset.py              # MaskDataset (loads images + masks)
│   ├── losses.py               # CE + Focal + Dice combined loss
│   ├── metrics.py              # Confusion matrix, IoU, Dice, plots
│   └── augmentations.py        # Training augmentation pipeline
│
└── results/
    ├── metrics/
    │   ├── val_evaluation_metrics.txt    # Per-epoch validation history
    │   ├── test_evaluation_metrics.txt   # Final test set metrics
    │   └── inference_timing.txt          # Timing and FPS statistics
    │
    ├── training/
    │   ├── training_curves.png           # Train/val loss over epochs
    │   ├── iou_curves.png               # Train/val IoU over epochs
    │   ├── dice_curves.png              # Train/val Dice over epochs
    │   ├── all_metrics_curves.png       # All 4 metrics combined
    │   ├── val_per_class_iou.png        # Per-class IoU bar chart (val)
    │   └── epoch_samples/
    │       ├── epoch_5_sample.png       # Visual progress checkpoints
    │       ├── epoch_10_sample.png
    │       ├── epoch_39_sample.png
    │       └── epoch_40_sample.png
    │
    └── test/
        ├── test_per_class_iou.png       # Per-class IoU bar chart (test)
        └── comparisons/
            ├── comparison_0.png         # 10 side-by-side test samples
            ├── comparison_1.png
            └── ...
```

---

## Loss Function

The model uses a **combined CE + Focal + Dice loss** for handling class imbalance:

```python
loss = 0.4 * CrossEntropyLoss(class_weighted) 
     + 0.3 * FocalLoss(gamma=2.0)
     + 0.3 * DiceLoss()
```

**Class weights** (higher = rarer class gets more loss emphasis):
```
Trees: 1.0 | Lush Bushes: 1.5 | Dry Grass: 1.3 | Dry Bushes: 1.8
Ground Clutter: 2.5 | Flowers: 3.0 | Logs: 3.5 | Rocks: 2.0
Landscape: 0.7 | Sky: 0.6
```

---

## Reproducing Results

1. Install requirements: `pip install -r requirements.txt`
2. Update `config.json` paths to point to your dataset
3. Run training:
   ```bash
   python train.py --config config.json
   ```
4. Run evaluation with TTA:
   ```bash
   python test.py --config config.json --model_path dinov2_dpt_full_lora.pth --tta
   ```

**Hardware used for training:** Kaggle GPU (T4 / P100)  
**Approx. training time:** ~3–4 hours for 40 epochs

---

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.

The DINOv2 backbone is subject to Meta's [DINOv2 License](https://github.com/facebookresearch/dinov2/blob/main/LICENSE).

---

## Citation

If you use this work, please cite:

```bibtex
@misc{offroad-dpt-2024,
  title   = {DINOv2 + DPT with LoRA for Offroad Semantic Segmentation},
  year    = {2024},
  url     = {https://github.com/YOUR_USERNAME/DPT_Offroad_Segmentation}
}
```

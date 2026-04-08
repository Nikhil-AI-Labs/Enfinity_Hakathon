import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- LoRA & ShallowCNN Tools ---
class ConvBNGELU(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, padding=0, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.GELU()
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class ShallowCNN(nn.Module):
    """Extracts raw edge/texture features at H/4 resolution directly from the image."""
    def __init__(self, out_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNGELU(3, 32, 3, stride=2, padding=1),
            ConvBNGELU(32, out_channels, 3, stride=2, padding=1)
        )
    def forward(self, x): return self.net(x)

class LoRAQKV(nn.Module):
    """Low-Rank Adaptation for DINOv2's attention projections."""
    def __init__(self, original_linear, rank=8, alpha=16):
        super().__init__()
        self.original = original_linear
        self.original.weight.requires_grad = False
        if self.original.bias is not None: self.original.bias.requires_grad = False
        self.lora_A = nn.Linear(original_linear.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, original_linear.out_features, bias=False)
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
    def forward(self, x):
        return self.original(x) + self.scale * self.lora_B(self.lora_A(x))

def inject_lora(backbone, rank=8, target_blocks=6):
    blocks = list(backbone.blocks)
    lora_params = []
    for block in blocks[-target_blocks:]:
        block.attn.qkv = LoRAQKV(block.attn.qkv, rank=rank)
        lora_params += list(block.attn.qkv.lora_A.parameters())
        lora_params += list(block.attn.qkv.lora_B.parameters())
    return lora_params

# --- DPT Blocks ---
class ResidualConvUnit(nn.Module):
    """Pre-activation residual block used in DPT fusion."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.relu(self.bn1(x))
        out = self.conv1(out)
        out = self.relu(self.bn2(out))
        out = self.conv2(out)
        return out + x

class FeatureFusionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.out_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.res_conv = ResidualConvUnit(channels)
        
    def forward(self, x, skip=None):
        out = self.res_conv(x)
        if skip is not None:
            # Interpolate out to match the skip connection's dynamic spatial size
            out = F.interpolate(out, size=skip.shape[2:], mode='bilinear', align_corners=False)
            out = out + self.out_conv(skip)
        return out

class DPTDecoder(nn.Module):
    def __init__(self, in_channels=1024, num_classes=10, TH=19, TW=34, feat_dim=256):
        super().__init__()
        self.TH, self.TW = TH, TW
        
        self.shallow_cnn = ShallowCNN(out_channels=64)
        
        # Reassemble logic (Project 1024 -> 256)
        self.projects = nn.ModuleList([nn.Conv2d(in_channels, feat_dim, 1, bias=False) for _ in range(4)])
        
        # DPT Fusion Blocks
        self.fusion4 = FeatureFusionBlock(feat_dim)
        self.fusion3 = FeatureFusionBlock(feat_dim)
        self.fusion2 = FeatureFusionBlock(feat_dim)
        self.fusion1 = FeatureFusionBlock(feat_dim)

        # Refinement & Classification
        self.refinement = nn.Sequential(
            ConvBNGELU(feat_dim + 64, 128, 3, padding=1),
            ConvBNGELU(128, 64, 3, padding=1)
        )
        self.classifier = nn.Conv2d(64, num_classes, 1)

    def forward(self, x_raw, backbone_features):
        B, _, H, W = x_raw.shape
        
        # 1. Pixel-level details
        image_skip = self.shallow_cnn(x_raw)  # Dynamic output shape
        
        # 2. Reshape ViT tokens - DYNAMIC TH/TW computation from input image size
        TH_dynamic, TW_dynamic = H // 14, W // 14
        feats = [f.reshape(B, TH_dynamic, TW_dynamic, -1).permute(0, 3, 1, 2) for f in backbone_features]
        feats = [proj(f) for proj, f in zip(self.projects, feats)]
        
        # 3. DPT Top-Down Fusion (Deep to Shallow)
        # Deepest layer acts as the base
        out = self.fusion4(feats[3]) 
        out = self.fusion3(out, skip=feats[2])
        out = self.fusion2(out, skip=feats[1])
        out = self.fusion1(out, skip=feats[0])
        
        # 4. Align DPT features with ShallowCNN spatial grid
        out_up = F.interpolate(out, size=image_skip.shape[2:], mode='bilinear', align_corners=False)
        
        # 5. Concat & Refine
        fused = torch.cat([out_up, image_skip], dim=1) # [B, 256+64, H/4, W/4]
        refined = self.refinement(fused)
        logits = self.classifier(refined)
        
        return F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)

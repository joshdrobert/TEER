"""Deep learning segmentation engine for the mitral valve complex."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .exceptions import SegmentationInferenceError
from .schemas import SegmentationConfig


class ConvBlock3D(nn.Module):
    """Residual-style 3D convolutional block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.skip = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.gelu(x + residual)


class AttentionGate3D(nn.Module):
    """Lightweight additive attention gate for skip connections."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.theta = nn.Conv3d(channels, channels, kernel_size=1)
        self.phi = nn.Conv3d(channels, channels, kernel_size=1)
        self.psi = nn.Conv3d(channels, 1, kernel_size=1)

    def forward(self, skip: torch.Tensor, gating: torch.Tensor) -> torch.Tensor:
        attn = torch.sigmoid(self.psi(F.gelu(self.theta(skip) + self.phi(gating))))
        return skip * attn


class TemporalSpatialTransformer(nn.Module):
    """Map temporal context into a spatial latent bias volume."""

    def __init__(self, latent_dim: int, channels: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, channels),
        )

    def forward(self, x: torch.Tensor, phase_scalar: torch.Tensor) -> torch.Tensor:
        bias = self.encoder(phase_scalar[:, None]).view(x.shape[0], x.shape[1], 1, 1, 1)
        return x + bias


class AttentionUNet3D(nn.Module):
    """3D Attention U-Net tailored to thin leaflet structures."""

    def __init__(self, config: SegmentationConfig) -> None:
        super().__init__()
        base = config.base_channels
        self.enc1 = ConvBlock3D(config.in_channels, base)
        self.enc2 = ConvBlock3D(base, base * 2)
        self.enc3 = ConvBlock3D(base * 2, base * 4)
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = ConvBlock3D(base * 4, base * 8)
        self.temporal = TemporalSpatialTransformer(config.temporal_latent_dim, base * 8)
        self.up3 = nn.ConvTranspose3d(base * 8, base * 4, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose3d(base * 2, base, kernel_size=2, stride=2)
        self.attn3 = AttentionGate3D(base * 4)
        self.attn2 = AttentionGate3D(base * 2)
        self.attn1 = AttentionGate3D(base)
        self.dec3 = ConvBlock3D(base * 8, base * 4)
        self.dec2 = ConvBlock3D(base * 4, base * 2)
        self.dec1 = ConvBlock3D(base * 2, base)
        self.head = nn.Conv3d(base, config.out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, phase_scalar: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bottleneck = self.temporal(self.bottleneck(self.pool(e3)), phase_scalar)
        d3 = self.up3(bottleneck)
        d3 = torch.cat([d3, self.attn3(e3, d3)], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, self.attn2(e2, d2)], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, self.attn1(e1, d1)], dim=1)
        d1 = self.dec1(d1)
        return self.head(d1)


@dataclass
class SegmentationBatch:
    """Container for model-ready batch tensors."""

    image: torch.Tensor
    phase_scalar: torch.Tensor
    target: torch.Tensor


class MitralSegmentationEngine:
    """Inference and loss utilities for the segmentation model."""

    def __init__(self, config: SegmentationConfig, device: str = "cpu") -> None:
        self.config = config
        self.device = torch.device(device)
        self.model = AttentionUNet3D(config).to(self.device)

    def infer(self, image: torch.Tensor, phase_scalar: torch.Tensor) -> torch.Tensor:
        """Run segmentation inference and return logits."""
        try:
            self.model.eval()
            with torch.no_grad():
                return self.model(image.to(self.device), phase_scalar.to(self.device))
        except RuntimeError as exc:
            raise SegmentationInferenceError("Segmentation inference failed.") from exc

    def compute_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Blend categorical cross-entropy with continuous Dice loss."""
        ce = F.cross_entropy(logits, target)
        probs = torch.softmax(logits, dim=1)
        target_one_hot = F.one_hot(target, num_classes=logits.shape[1]).permute(0, 4, 1, 2, 3).float()
        dims = tuple(range(2, probs.ndim))
        intersection = torch.sum(probs * target_one_hot, dim=dims)
        denominator = torch.sum(probs + target_one_hot, dim=dims)
        dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
        return 0.5 * ce + 0.5 * dice

    def training_step(self, batch: SegmentationBatch) -> Dict[str, torch.Tensor]:
        """Single-step training interface for downstream trainers."""
        self.model.train()
        logits = self.model(batch.image.to(self.device), batch.phase_scalar.to(self.device))
        loss = self.compute_loss(logits, batch.target.to(self.device))
        return {"loss": loss, "logits": logits}

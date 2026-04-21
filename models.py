#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Speckle Video Recognition Models
================================

Two model architectures:
  1. cnn_pool  —  Frame-level CNN (ResNet18) + Temporal Average Pooling  [default / recommended]
  2. r3d       —  3D ResNet-18 (R3D)

For speckle data (one video per class, limited data), cnn_pool is more stable
because it reuses ImageNet pretrained weights and only needs to learn temporal
aggregation and the classification head.
"""

import torch
import torch.nn as nn
from torchvision import models


class CNNPoolModel(nn.Module):
    """
    Frame-level CNN + Temporal Average Pooling

    Pipeline:
      1. Pretrained ResNet18 extracts 512-D features per frame
      2. Temporal average pooling over T frames in a clip → 512-D
      3. Dropout + FC → 26-way classification

    Advantage: Leverages ImageNet pretrained weights; converges stably with small data.
    """

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()

        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
        except AttributeError:
            backbone = models.resnet18(pretrained=pretrained)

        feat_dim = backbone.fc.in_features  # 512
        # Remove final FC layer, keep avgpool output (B, 512, 1, 1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W) — batch of video clips
        Returns:
            logits: (B, num_classes)
        """
        B, T, C, H, W = x.shape

        x = x.reshape(B * T, C, H, W)
        features = self.backbone(x)               # (B*T, 512, 1, 1)
        features = features.reshape(B, T, -1)     # (B, T, 512)

        pooled = features.mean(dim=1)             # (B, 512) — temporal avg pool
        pooled = self.dropout(pooled)

        return self.fc(pooled)


class R3DModel(nn.Module):
    """
    3D ResNet-18 (R3D)

    Uses 3D convolutions to model both spatial and temporal patterns.
    In principle it can capture speckle temporal dynamics but needs more data to train well.
    """

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()

        try:
            weights = models.video.R3D_18_Weights.DEFAULT if pretrained else None
            self.model = models.video.r3d_18(weights=weights)
        except AttributeError:
            self.model = models.video.r3d_18(pretrained=pretrained)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W) — batch of video clips
        Returns:
            logits: (B, num_classes)
        """
        # R3D expects (B, C, T, H, W) format
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return self.model(x)


def get_model(model_type: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Factory function: create model by name."""
    if model_type == "cnn_pool":
        return CNNPoolModel(num_classes, pretrained)
    elif model_type == "r3d":
        return R3DModel(num_classes, pretrained)
    else:
        raise ValueError(f"Unknown model type: {model_type}, choices: cnn_pool, r3d")

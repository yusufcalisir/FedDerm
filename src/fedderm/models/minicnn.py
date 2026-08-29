"""
fedderm/models/minicnn.py
-------------------------
Compact CNN baseline for 28x28 (or small-resolution) medical image classification.

Architecture rationale
----------------------
DermaMNIST images are 28x28 px. Standard ResNet-18 applies a 7x7 stride-2 conv
followed by a 3x3 maxpool (stride 2) in its stem, reducing a 28x28 input to
6x6 before the first residual block, and to 1x1 by the final avg-pool -- too
aggressive. A custom shallow CNN avoids this and is more appropriate:

  Input 3x28x28
  -> Block 1: Conv 3x3, BN, ReLU, Conv 3x3, BN, ReLU           -> 32x28x28
  -> MaxPool 2x2                                                 -> 32x14x14
  -> Block 2: Conv 3x3, BN, ReLU, Conv 3x3, BN, ReLU           -> 64x14x14
  -> MaxPool 2x2                                                 -> 64x7x7
  -> Block 3: Conv 3x3 (pad=1), BN, ReLU                        -> 128x7x7
  -> Block 4: Conv 3x3 (pad=1), BN, ReLU                        -> 256x7x7
  -> Global Average Pooling                                      -> 256
  -> FC 256->128, ReLU, Dropout(0.4)
  -> FC 128->num_classes

~1.3 M parameters. Trains in ~2-3 min/epoch on a modern CPU for DermaMNIST.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBnRelu(nn.Sequential):
    """Conv2d -> BatchNorm2d -> ReLU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=padding, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class MiniCNN(nn.Module):
    """Compact CNN for 28x28 or small-resolution multi-class image classification.

    Args:
        num_classes: Number of output classes (7 for DermaMNIST).
        in_channels: Number of input channels (3 for RGB).
        dropout:     Dropout probability in the classifier head.
    """

    def __init__(
        self,
        num_classes: int = 7,
        in_channels: int = 3,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 3x28x28 -> 32x28x28
            ConvBnRelu(in_channels, 32),
            ConvBnRelu(32, 32),
            nn.MaxPool2d(2),               # -> 32x14x14

            # Block 2: 32x14x14 -> 64x14x14
            ConvBnRelu(32, 64),
            ConvBnRelu(64, 64),
            nn.MaxPool2d(2),               # -> 64x7x7

            # Block 3: 64x7x7 -> 128x7x7
            ConvBnRelu(64, 128),

            # Block 4: 128x7x7 -> 256x7x7
            ConvBnRelu(128, 256),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)   # -> 256x1x1

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_model(num_classes: int = 7, dropout: float = 0.4) -> MiniCNN:
    """Factory function that returns a MiniCNN instance."""
    return MiniCNN(num_classes=num_classes, dropout=dropout)

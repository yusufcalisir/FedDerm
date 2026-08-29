"""
fedderm/models/minicnn.py
-------------------------
Compact CNN baseline for 28x28 (or small-resolution) medical image classification.

Architecture rationale
----------------------
DermaMNIST images are 28x28 px. Standard ResNet-18 applies a 7x7 stride-2 conv
followed by a 3x3 maxpool (stride 2) in its stem, reducing a 28x28 input to
6x6 before the first residual block, and to 1x1 by the final avg-pool -- too
aggressive. A custom shallow CNN avoids this and is more appropriate.

Normalization: GroupNorm (num_groups=8) throughout instead of BatchNorm2d.

Two reasons:
  1. **Federated compatibility**: Under FedAvg, BatchNorm's running_mean and
     running_var are computed locally per client on highly non-IID data and
     then naively averaged across clients. This is known to cause catastrophic
     performance collapse (see FedBN, ICLR 2021). GroupNorm has no running
     statistics -- normalization is computed per-sample per-group at both
     train and eval time, so there are no buffers to mishandle across clients.
  2. **Opacus (DP-SGD) compatibility**: Opacus requires per-sample gradient
     computation, which is incompatible with BatchNorm's batch-level statistics
     when batch_size > 1. GroupNorm works correctly with Opacus out of the box.

Architecture:
  Input 3x28x28
  -> Block 1: Conv 3x3, GN(8), ReLU, Conv 3x3, GN(8), ReLU  -> 32x28x28
  -> MaxPool 2x2                                               -> 32x14x14
  -> Block 2: Conv 3x3, GN(8), ReLU, Conv 3x3, GN(8), ReLU  -> 64x14x14
  -> MaxPool 2x2                                               -> 64x7x7
  -> Block 3: Conv 3x3 (pad=1), GN(8), ReLU                  -> 128x7x7
  -> Block 4: Conv 3x3 (pad=1), GN(8), ReLU                  -> 256x7x7
  -> Global Average Pooling                                    -> 256
  -> FC 256->128, ReLU, Dropout(0.4)
  -> FC 128->num_classes

~469k parameters. Trains in ~2-3 min/epoch on a modern CPU for DermaMNIST.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Default number of groups for GroupNorm.
# Must divide into every channel size used (32, 64, 128, 256).
_GN_GROUPS = 8


from typing import cast


class ConvGnRelu(nn.Sequential):
    """Conv2d -> GroupNorm -> ReLU block.

    Replaces the previous ConvBnRelu (BatchNorm2d) for federated compatibility
    and Opacus (DP-SGD) per-sample gradient compatibility.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
        num_groups: int = _GN_GROUPS,
    ) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      padding=padding, stride=stride, bias=False),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=False),
        )


class MiniCNN(nn.Module):
    """Compact CNN for 28x28 or small-resolution multi-class image classification.

    Uses GroupNorm throughout for federated learning and Opacus compatibility.

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
            ConvGnRelu(in_channels, 32),
            ConvGnRelu(32, 32),
            nn.MaxPool2d(2),               # -> 32x14x14

            # Block 2: 32x14x14 -> 64x14x14
            ConvGnRelu(32, 64),
            ConvGnRelu(64, 64),
            nn.MaxPool2d(2),               # -> 64x7x7

            # Block 3: 64x7x7 -> 128x7x7
            ConvGnRelu(64, 128),

            # Block 4: 128x7x7 -> 256x7x7
            ConvGnRelu(128, 256),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)   # -> 256x1x1

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return cast(torch.Tensor, self.classifier(x))


def build_model(num_classes: int = 7, dropout: float = 0.4) -> MiniCNN:
    """Factory function that returns a MiniCNN instance with GroupNorm."""
    return MiniCNN(num_classes=num_classes, dropout=dropout)

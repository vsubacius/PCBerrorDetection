"""The distillation teacher: a ResNet-18 adapted to the tile task.

Used only on the PC/Colab side, to produce the soft targets the KD runs
distil from. It is never quantized and never runs on the chip, so its cost
does not matter -- only that it is meaningfully *better* than the student,
since distillation can only transfer what the teacher knows.

Adapted from the standard ResNet-18 in three ways:

* **Input channels follow the frozen input mode** (1 for A, 2 for B and C).
  The plan's earlier wording fixed the teacher at 2 channels, which would
  have let a KD run distil from a teacher that saw inputs the student never
  gets. Same channels, same information.
* **Six sigmoid logits**, not a 1000-way softmax, because tiles carry
  independent multi-label defect classes.
* **No pretrained weights.** ImageNet initialisation is of little use for a
  binarized 1-channel PCB tile, and in this environment
  ``download.pytorch.org`` is unreachable anyway, so the teacher trains from
  scratch. Written out here rather than imported from torchvision, which is
  not a dependency of this repo.

Unlike the student, the teacher sees the tile at **full 160x160 resolution**:
the standard 7x7 stride-2 stem plus 3x3 max-pool gives a 40x40 feature map,
the same spatial size the student reaches, but computed from every pixel
rather than from a 4x-pooled copy. That extra resolution is a large part of
why it should outperform the student, and therefore of what there is to
distil.

Caveat worth keeping in view: 11M parameters trained on 20,000 tiles drawn
from 800 images -- which are themselves only 7 physical boards -- will overfit
hard. Check that the teacher actually beats the student on validation before
reading anything into a KD result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

from deeppcb import NUM_CLASSES  # noqa: E402


class BasicBlock(nn.Module):
    """The two-convolution residual block of ResNet-18/34."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample: nn.Module | None = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet18Teacher(nn.Module):
    """ResNet-18 with a configurable input stem and a 6-way multi-label head.

    Returns raw logits of shape ``(N, num_classes)``, matching the student so
    the distillation loss can compare them directly.
    """

    layer_channels = (64, 128, 256, 512)
    layer_blocks = (2, 2, 2, 2)

    def __init__(self, in_channels: int = 1, num_classes: int = NUM_CLASSES,
                 dropout: float = 0.0):
        super().__init__()
        if in_channels not in (1, 2):
            raise ValueError(f"in_channels must be 1 or 2, got {in_channels}")
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        layers = []
        channels = 64
        for index, (out_channels, blocks) in enumerate(
                zip(self.layer_channels, self.layer_blocks)):
            stride = 1 if index == 0 else 2
            for block in range(blocks):
                layers.append(BasicBlock(channels, out_channels,
                                         stride if block == 0 else 1))
                channels = out_channels
        self.layers = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Linear(channels, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)
        # Zero-init the last BN of each block, so every residual branch starts
        # as an identity. Standard ResNet trick; it makes a deep model from
        # scratch much easier to train, which matters here with no pretraining.
        for module in self.modules():
            if isinstance(module, BasicBlock):
                nn.init.zeros_(module.bn2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layers(x)
        return self.head(self.dropout(self.pool(x).flatten(1)))


def build_teacher(in_channels: int = 1, **kwargs) -> ResNet18Teacher:
    """The teacher factory ``train.py`` calls when a KD run loads a checkpoint."""
    return ResNet18Teacher(in_channels=in_channels, **kwargs)

"""Three model factories used by VisionMind.

1. CustomCNN     - 3-block CNN built from scratch (no pretraining)
2. build_resnet18 - ImageNet-pretrained ResNet18 with a new classifier head
3. build_mobilenet_v2 - ImageNet-pretrained MobileNetV2 with a new head

All models output logits over CIFAR10_CLASSES (10 classes). Final
softmax is left to the caller (CrossEntropyLoss applies it during training,
``inference.py`` applies it explicitly for ensembling).
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 10


class CustomCNN(nn.Module):
    """A simple, course-aligned CNN built from scratch.

    Architecture: three convolutional blocks (Conv -> BN -> ReLU -> Conv ->
    BN -> ReLU -> MaxPool) with channel widths 32/64/128, followed by a
    GlobalAvgPool and a linear classifier. Dropout(0.3) before the head.

    Total params: ~315k, well under 1M.
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()

        def block(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 32),
            block(32, 64),
            block(64, 128),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


def build_resnet18(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """ResNet18 with ImageNet weights and a new fully-connected head."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_mobilenet_v2(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """MobileNetV2 with ImageNet weights and a new classifier head."""
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


ModelName = Literal["custom_cnn", "resnet18", "mobilenet_v2"]


def build_model(name: ModelName, num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """Factory: build a model by short name."""
    if name == "custom_cnn":
        return CustomCNN(num_classes)
    if name == "resnet18":
        return build_resnet18(num_classes, pretrained)
    if name == "mobilenet_v2":
        return build_mobilenet_v2(num_classes, pretrained)
    raise ValueError(f"Unknown model name: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

"""CIFAR-10 data pipeline: train / val / test loaders with augmentation.

The CIFAR-10 dataset is auto-downloaded by torchvision on first run
(~170 MB) into the ``data/`` directory. The dataset itself is NOT
committed to the repository (see .gitignore).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# Per-channel mean / std computed from CIFAR-10 train set
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def build_transforms(train: bool, image_size: int = 32) -> transforms.Compose:
    """Build the torchvision transform pipeline.

    Training uses light augmentation (random crop with padding + horizontal
    flip). Evaluation uses only normalization. Both resize to ``image_size``
    so the same pipeline works for from-scratch CNNs and ImageNet-pretrained
    backbones (call with image_size=224 for the latter).
    """
    ops = []
    if image_size != 32:
        ops.append(transforms.Resize(image_size))
    if train:
        ops.append(transforms.RandomCrop(image_size, padding=4))
        ops.append(transforms.RandomHorizontalFlip())
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD))
    return transforms.Compose(ops)


def get_dataloaders(
    data_dir: str | Path = "data",
    batch_size: int = 128,
    num_workers: int = 2,
    image_size: int = 32,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader) for CIFAR-10."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    train_tf = build_transforms(train=True, image_size=image_size)
    eval_tf = build_transforms(train=False, image_size=image_size)

    full_train = datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=True,
        transform=train_tf,
    )
    test_set = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=True,
        transform=eval_tf,
    )

    val_size = int(len(full_train) * val_fraction)
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(
        full_train, [train_size, val_size], generator=generator
    )
    # Validation should use the eval transform; rebuild the underlying dataset
    val_set.dataset = datasets.CIFAR10(  # type: ignore[attr-defined]
        root=str(data_dir), train=True, download=False, transform=eval_tf
    )

    common = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def class_distribution(loader: DataLoader) -> dict[str, int]:
    """Count samples per class in a loader (for quick EDA)."""
    counts = {c: 0 for c in CIFAR10_CLASSES}
    for _, labels in loader:
        for lab in labels.tolist():
            counts[CIFAR10_CLASSES[lab]] += 1
    return counts

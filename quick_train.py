"""One-time helper to produce the demo checkpoint + bundled images.

Run ONCE on your machine:

    python quick_train.py

It will:
  1. Auto-download CIFAR-10 (~170 MB, cached afterwards)
  2. Train a small CustomCNN for 5 epochs on CPU/GPU (~3-5 min CPU)
  3. Save the checkpoint to models/quick_cnn.pt  (~1.2 MB)
  4. Save 5 hand-picked test images as PNG into demo_images/
     so the live demo can run completely offline.

Subsequent runs of demo.py / start.bat just use the bundled checkpoint
and the bundled images — no training, no downloads.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "_cifar"
MODELS = ROOT / "models"
DEMO_IMG = ROOT / "demo_images"
MODELS.mkdir(parents=True, exist_ok=True)
DEMO_IMG.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck")


class CustomCNN(nn.Module):
    """Small CNN — same family as src/model.py::CustomCNN, ~315k params."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cls = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, num_classes))

    def forward(self, x):
        return self.cls(self.pool(self.features(x)))


def get_loaders(batch_size: int = 128):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_set = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=train_tf)
    test_set  = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=eval_tf)
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=0),
        test_set,
    )


def epoch(model, loader, criterion, optimizer=None) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = total_correct = total = 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
    return total_loss / total, total_correct / total


def save_demo_images(test_set):
    """Pick one clear example per class, save as 128×128 PNG (4× upscale)."""
    selected: dict[int, int] = {}
    for idx in range(len(test_set)):
        _, y = test_set[idx]
        if y not in selected:
            selected[y] = idx
        if len(selected) == 10:
            break
    targets = [0, 1, 3, 5, 8, 9]   # airplane, automobile, cat, dog, ship, truck
    raw = datasets.CIFAR10(root=DATA_DIR, train=False, download=False)
    for cls_idx in targets:
        i = selected[cls_idx]
        pil_img, _ = raw[i]
        pil_big = pil_img.resize((128, 128), Image.NEAREST)
        out = DEMO_IMG / f"{cls_idx:02d}_{CLASSES[cls_idx]}.png"
        pil_big.save(out)
        print(f"  saved {out.name}")


def main() -> None:
    print("=" * 60)
    print(f"  quick_train.py — device: {DEVICE}")
    print("=" * 60)

    train_loader, test_loader, test_set_eval = get_loaders(batch_size=128)
    raw_test = datasets.CIFAR10(root=DATA_DIR, train=False, download=False)

    model = CustomCNN(num_classes=10).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optim = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optim, T_max=5)

    print("\n  training 5 epochs ...")
    t0 = time.time()
    for ep in range(1, 6):
        tl, ta = epoch(model, train_loader, criterion, optim)
        vl, va = epoch(model, test_loader, criterion)
        scheduler.step()
        print(f"  epoch {ep}/5  train_loss={tl:.4f} train_acc={ta:.3f}  "
              f"test_loss={vl:.4f} test_acc={va:.3f}")

    elapsed = time.time() - t0
    print(f"\n  total training time: {elapsed/60:.1f} min")

    ckpt = MODELS / "quick_cnn.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"  saved checkpoint: {ckpt}  ({ckpt.stat().st_size / 1024:.1f} KB)")

    print("\n  saving 6 demo PNGs to demo_images/ ...")
    save_demo_images(raw_test)
    print("\nDone. You can now run demo.py.")


if __name__ == "__main__":
    main()

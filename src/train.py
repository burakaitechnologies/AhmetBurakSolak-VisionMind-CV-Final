"""Reusable training loop for the three VisionMind models.

Usage from the CLI:

    python -m src.train --model resnet18 --epochs 15 --batch-size 128

The function ``train_model`` is also imported directly by the training
notebook so the same code path is used in both contexts.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .data_processing import get_dataloaders
from .model import ModelName, build_model, count_parameters


def _epoch_pass(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> Tuple[float, float]:
    """One pass over ``loader``. If ``optimizer`` is given, also trains."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    correct = 0
    total = 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(inputs)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += inputs.size(0)

    return total_loss / total, correct / total


def train_model(
    model_name: ModelName,
    epochs: int = 15,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    image_size: int | None = None,
    data_dir: str = "data",
    save_dir: str = "models",
    device: str | None = None,
) -> Dict[str, List[float]]:
    """Train one model end-to-end and save the best checkpoint by val accuracy.

    Returns a dict with per-epoch ``train_loss``, ``train_acc``,
    ``val_loss``, ``val_acc`` plus final ``test_acc``.
    """
    if image_size is None:
        # Pretrained ImageNet backbones expect 224; the from-scratch CNN
        # works at the native CIFAR resolution.
        image_size = 32 if model_name == "custom_cnn" else 224

    device_ = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
    )

    model = build_model(model_name).to(device_)
    print(f"[{model_name}] trainable params: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    best_val = 0.0
    best_ckpt = save_path / f"{model_name}_best.pt"

    start = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _epoch_pass(model, train_loader, device_, criterion, optimizer)
        va_loss, va_acc = _epoch_pass(model, val_loader, device_, criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        improved = va_acc > best_val
        if improved:
            best_val = va_acc
            torch.save(model.state_dict(), best_ckpt)

        print(
            f"epoch {epoch:02d}/{epochs}  "
            f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f}  "
            f"va_loss={va_loss:.4f} va_acc={va_acc:.4f}"
            + ("  *" if improved else "")
        )

    # Final test pass with the best checkpoint
    model.load_state_dict(torch.load(best_ckpt, map_location=device_))
    _, test_acc = _epoch_pass(model, test_loader, device_, criterion)
    history["test_acc"] = [test_acc]
    elapsed = time.time() - start
    print(f"[{model_name}] best_val={best_val:.4f}  test_acc={test_acc:.4f}  "
          f"time={elapsed/60:.1f} min  ckpt={best_ckpt}")

    # Persist history alongside the checkpoint
    (save_path / f"{model_name}_history.json").write_text(json.dumps(history, indent=2))
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a VisionMind model.")
    parser.add_argument("--model", choices=["custom_cnn", "resnet18", "mobilenet_v2"], default="resnet18")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--save-dir", default="models")
    args = parser.parse_args()

    train_model(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        image_size=args.image_size,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()

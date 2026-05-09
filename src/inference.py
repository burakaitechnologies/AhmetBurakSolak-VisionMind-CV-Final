"""Single-image and batch inference, including the soft-vote ensemble.

CLI usage:

    # Predict the class of one image with all three models + ensemble
    python -m src.inference --image path/to/photo.jpg --explain

The ``--explain`` flag also writes a Grad-CAM heatmap next to the input
image (uses ``src.explainability``).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .data_processing import CIFAR10_CLASSES, CIFAR10_MEAN, CIFAR10_STD
from .model import ModelName, build_model

DEFAULT_CHECKPOINTS: Dict[ModelName, str] = {
    "custom_cnn": "models/custom_cnn_best.pt",
    "resnet18": "models/resnet18_best.pt",
    "mobilenet_v2": "models/mobilenet_v2_best.pt",
}


def _preprocess(image: Image.Image, image_size: int) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return tf(image).unsqueeze(0)


def load_models(
    checkpoints: Dict[ModelName, str] | None = None,
    device: str | torch.device = "cpu",
) -> Dict[ModelName, torch.nn.Module]:
    """Load the trained checkpoints for every available model."""
    checkpoints = checkpoints or DEFAULT_CHECKPOINTS
    device_ = torch.device(device)
    loaded: Dict[ModelName, torch.nn.Module] = {}
    for name, path in checkpoints.items():
        if not Path(path).exists():
            print(f"[inference] skipping {name}: no checkpoint at {path}")
            continue
        model = build_model(name).to(device_)
        model.load_state_dict(torch.load(path, map_location=device_))
        model.eval()
        loaded[name] = model
    if not loaded:
        raise FileNotFoundError("No checkpoints found. Train at least one model first.")
    return loaded


def predict(
    image: Image.Image | str | Path,
    models_: Dict[ModelName, torch.nn.Module],
    device: str | torch.device = "cpu",
) -> Dict[str, Dict]:
    """Run all loaded models on an image; return per-model + ensemble results."""
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    device_ = torch.device(device)

    out: Dict[str, Dict] = {}
    probs_list: List[torch.Tensor] = []
    for name, model in models_.items():
        image_size = 32 if name == "custom_cnn" else 224
        x = _preprocess(image, image_size).to(device_)
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu()
        probs_list.append(probs)
        idx = int(torch.argmax(probs).item())
        out[name] = {
            "class": CIFAR10_CLASSES[idx],
            "confidence": float(probs[idx].item()),
            "probs": {c: float(p.item()) for c, p in zip(CIFAR10_CLASSES, probs)},
        }

    # Soft-vote ensemble (only meaningful when 2+ models present)
    if len(probs_list) >= 2:
        ensemble_probs = torch.stack(probs_list).mean(dim=0)
        idx = int(torch.argmax(ensemble_probs).item())
        out["ensemble"] = {
            "class": CIFAR10_CLASSES[idx],
            "confidence": float(ensemble_probs[idx].item()),
            "probs": {c: float(p.item()) for c, p in zip(CIFAR10_CLASSES, ensemble_probs)},
        }
    return out


def _format_result(result: Dict[str, Dict]) -> str:
    lines = []
    for name, info in result.items():
        head = f"{name:>14s}: {info['class']:<10s} ({info['confidence']*100:5.2f}%)"
        lines.append(head)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="VisionMind single-image inference.")
    parser.add_argument("--image", required=True, help="Path to an input image (any common format).")
    parser.add_argument("--explain", action="store_true", help="Also produce a Grad-CAM heatmap.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    models_ = load_models(device=args.device)
    image = Image.open(args.image).convert("RGB")
    result = predict(image, models_, device=args.device)

    print(_format_result(result))

    if args.explain:
        from .explainability import save_gradcam_overlay

        out_path = Path(args.image).with_name(f"gradcam_{Path(args.image).stem}.png")
        # Use the strongest single model for the overlay (ResNet18 if present)
        explain_model_name: ModelName = (
            "resnet18" if "resnet18" in models_ else next(iter(models_.keys()))
        )
        save_gradcam_overlay(
            image=image,
            model=models_[explain_model_name],
            model_name=explain_model_name,
            out_path=out_path,
            device=args.device,
        )
        print(f"[explain] Grad-CAM saved to {out_path}")


if __name__ == "__main__":
    main()

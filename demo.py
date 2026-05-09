"""VisionMind — Live Demo
=============================
Click-and-record demo for the recorded 3-5 minute video.

NO internet required. NO training required. Everything ships with the
repo: the trained checkpoint (models/quick_cnn.pt) and 6 hand-picked
CIFAR-10 test images (demo_images/*.png).

Pipeline (per image):
  1. Load image, normalize, run through the trained CustomCNN
  2. Show predicted class + confidence + full softmax distribution
  3. Compute Grad-CAM heatmap from the last conv block
  4. Display 3-panel figure: input · Grad-CAM · overlay
  5. Save the figure into demo_outputs/ for the video editor
  6. Wait for the user to close the window before continuing

Author: Ahmet Burak Solak — ITAI 1378 Final Project (Spring 2026)
Repo:   https://github.com/burakaitechnologies/AhmetBurakSolak-VisionMind-CV-Final
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).parent
DEMO_IMG = ROOT / "demo_images"
OUT_DIR = ROOT / "demo_outputs"
CHECKPOINT = ROOT / "models" / "quick_cnn.pt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


# ----------------------------------------------------------------------- #
# Tiny CNN — same as the project's CustomCNN
# ----------------------------------------------------------------------- #
class CustomCNN(nn.Module):
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


# ----------------------------------------------------------------------- #
# Grad-CAM
# ----------------------------------------------------------------------- #
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model.eval()
        self.target = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._fwd = target_layer.register_forward_hook(self._save_activation)
        self._bwd = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _m, _i, output): self.activations = output.detach()
    def _save_gradient(self, _m, _gi, grad_out): self.gradients = grad_out[0].detach()

    def remove(self) -> None:
        self._fwd.remove(); self._bwd.remove()

    def __call__(self, x: torch.Tensor) -> Tuple[np.ndarray, int, float, torch.Tensor]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1).squeeze(0)
        idx = int(torch.argmax(probs).item())
        logits[0, idx].backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam).squeeze().detach().cpu().numpy()
        if cam.max() - cam.min() > 1e-8:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
        return cam, idx, float(probs[idx].item()), probs.cpu()


# ----------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------- #
PREPROCESS = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


def overlay(rgb: np.ndarray, heat: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    h, w = rgb.shape[:2]
    heat_resized = np.array(
        Image.fromarray((heat * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    ).astype(np.float32) / 255.0
    cmap = mpl_cm.get_cmap("jet") if hasattr(mpl_cm, "get_cmap") else plt.colormaps["jet"]
    colored = cmap(heat_resized)[..., :3]
    return np.clip((1 - alpha) * rgb + alpha * colored, 0, 1)


def load_model() -> nn.Module:
    if not CHECKPOINT.exists():
        print(f"[ERROR] Checkpoint not found at {CHECKPOINT}")
        print("        Run 'python quick_train.py' once to produce it.")
        sys.exit(1)
    model = CustomCNN().to(DEVICE)
    state = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


# ----------------------------------------------------------------------- #
# Per-image demo
# ----------------------------------------------------------------------- #
def run_one(image_path: Path, model: nn.Module, idx: int, total: int) -> None:
    pil_image = Image.open(image_path).convert("RGB")
    display_size = (256, 256)
    rgb = np.array(pil_image.resize(display_size, Image.NEAREST)).astype(np.float32) / 255.0
    x = PREPROCESS(pil_image).unsqueeze(0).to(DEVICE)

    print()
    print("=" * 70)
    print(f"  Image {idx + 1} / {total}: {image_path.name}")
    print("=" * 70)

    cam = GradCAM(model, model.features[-1])
    try:
        heat, pred_idx, conf, probs = cam(x)
    finally:
        cam.remove()

    probs_np = probs.detach().numpy()
    sorted_idx_np = probs_np.argsort()[::-1]
    print("\n  Top-3 predictions:")
    for i in range(3):
        cls = CLASSES[int(sorted_idx_np[i])]
        p = float(probs_np[sorted_idx_np[i]])
        marker = " <- prediction" if i == 0 else ""
        print(f"    {cls:<12s}  {p*100:5.2f}%{marker}")

    blended = overlay(rgb, heat)

    fig = plt.figure(figsize=(14, 5.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.4])

    ax0 = fig.add_subplot(gs[0, 0]); ax0.imshow(rgb); ax0.axis("off")
    ax0.set_title("Input image", fontsize=12, fontweight="bold")

    ax1 = fig.add_subplot(gs[0, 1]); ax1.imshow(heat, cmap="jet"); ax1.axis("off")
    ax1.set_title("Grad-CAM heatmap", fontsize=12, fontweight="bold")

    ax2 = fig.add_subplot(gs[0, 2]); ax2.imshow(blended); ax2.axis("off")
    ax2.set_title(f"Prediction: {CLASSES[pred_idx]}\n({conf*100:.1f}% confidence)",
                  fontsize=12, fontweight="bold")

    ax3 = fig.add_subplot(gs[0, 3])
    bars = ax3.barh(CLASSES, probs_np, color="#3b82f6")
    bars[pred_idx].set_color("#22c55e")
    ax3.set_xlim(0, 1.0); ax3.invert_yaxis()
    ax3.set_xlabel("Probability"); ax3.set_title("Class probabilities",
                                                  fontsize=12, fontweight="bold")
    for b, v in zip(bars, probs_np):
        ax3.text(min(v + 0.02, 0.97), b.get_y() + b.get_height()/2,
                 f"{v*100:.1f}%", va="center", fontsize=8)
    ax3.grid(axis="x", alpha=0.3)

    fig.suptitle(
        f"VisionMind — Live Demo  ({idx + 1}/{total})  ·  ITAI 1378 Final Project  ·  Ahmet Burak Solak",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()

    out_path = OUT_DIR / f"demo_{image_path.stem}.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"\n  saved {out_path.relative_to(ROOT)}")

    print("\n  >>> CLOSE the figure window when ready for the next image. <<<")
    plt.show()


def main() -> None:
    print()
    print("============================================================")
    print("    VisionMind - Live Demo (offline, no internet needed)    ")
    print("    ITAI 1378 Final Project - Ahmet Burak Solak             ")
    print("============================================================")
    print(f"  device:       {DEVICE}")
    print(f"  checkpoint:   {CHECKPOINT.name}")
    print(f"  demo images:  {DEMO_IMG.name}/")
    print()

    images = sorted(p for p in DEMO_IMG.iterdir()
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not images:
        print(f"[ERROR] No demo images found in {DEMO_IMG}/")
        print("        Run 'python quick_train.py' once to produce them.")
        sys.exit(1)
    print(f"  {len(images)} demo image(s) ready")

    print("\nLoading trained model (CustomCNN, 5 epochs on CIFAR-10) ...")
    model = load_model()
    print("  model loaded.")

    print("\nRunning demo. A figure will pop up for each image — close it to advance.")
    print("Each figure is saved into demo_outputs/ for the video editor.")
    time.sleep(1.0)

    for i, path in enumerate(images):
        run_one(path, model, i, len(images))

    print()
    print("============================================================")
    print(f"  Demo complete. {len(images)} stills saved to demo_outputs/.")
    print("  Repo: github.com/burakaitechnologies/AhmetBurakSolak-VisionMind-CV-Final")
    print("============================================================")


if __name__ == "__main__":
    main()

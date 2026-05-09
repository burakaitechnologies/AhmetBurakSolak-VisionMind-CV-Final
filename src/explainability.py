"""Grad-CAM implementation for the VisionMind models.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization" (ICCV 2017).

For each model we hook the feature map of its last convolutional block,
backpropagate the score of the predicted class, weight the feature
channels by their average gradient, and ReLU the result. The output is
a coarse 2D heatmap aligned with the input image.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .data_processing import CIFAR10_CLASSES, CIFAR10_MEAN, CIFAR10_STD
from .model import ModelName


def _last_conv_module(model: nn.Module, model_name: ModelName) -> nn.Module:
    """Return the last conv block of each known architecture."""
    if model_name == "custom_cnn":
        return model.features[-1]  # type: ignore[index]
    if model_name == "resnet18":
        return model.layer4[-1]  # type: ignore[attr-defined,index]
    if model_name == "mobilenet_v2":
        return model.features[-1]  # type: ignore[attr-defined,index]
    raise ValueError(f"Unknown model name for Grad-CAM: {model_name}")


class GradCAM:
    """Lightweight Grad-CAM hook around any of the three model types."""

    def __init__(self, model: nn.Module, model_name: ModelName) -> None:
        self.model = model.eval()
        self.model_name = model_name
        self._target_layer = _last_conv_module(model, model_name)
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        self._fwd = self._target_layer.register_forward_hook(self._save_activation)
        self._bwd = self._target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module: nn.Module, _inp, out: torch.Tensor) -> None:
        self._activations = out.detach()

    def _save_gradient(self, _module: nn.Module, _grad_in, grad_out) -> None:
        # grad_out is a tuple; the first element is the gradient w.r.t. output
        self._gradients = grad_out[0].detach()

    def remove(self) -> None:
        self._fwd.remove()
        self._bwd.remove()

    def __call__(self, x: torch.Tensor, target_class: int | None = None) -> Tuple[np.ndarray, int, float]:
        """Compute the Grad-CAM heatmap.

        Parameters
        ----------
        x : input tensor of shape (1, 3, H, W) on the same device as model
        target_class : class index to explain (default: the model's prediction)

        Returns
        -------
        heatmap : np.ndarray of shape (H', W') in [0, 1]
        predicted_class : int
        confidence : float
        """
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        probs = F.softmax(logits, dim=1).squeeze(0)
        pred_idx = int(torch.argmax(probs).item())
        target = target_class if target_class is not None else pred_idx

        score = logits[0, target]
        score.backward(retain_graph=True)

        assert self._activations is not None and self._gradients is not None
        # Channel-wise weights = global-average-pooled gradients
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam).squeeze().cpu().numpy()

        # Normalize 0-1
        cam_min, cam_max = float(cam.min()), float(cam.max())
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        return cam, pred_idx, float(probs[pred_idx].item())


def _prepare_input(image: Image.Image, image_size: int) -> Tuple[torch.Tensor, np.ndarray]:
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    x = tf(image).unsqueeze(0)
    rgb = np.array(image.resize((image_size, image_size))).astype(np.float32) / 255.0
    return x, rgb


def overlay_heatmap(rgb_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Resize the heatmap to the image, apply a JET colormap, and blend."""
    import matplotlib.cm as cm
    from PIL import Image as _Image

    h, w = rgb_image.shape[:2]
    heatmap_pil = _Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h), _Image.BILINEAR)
    heatmap_resized = np.array(heatmap_pil).astype(np.float32) / 255.0
    colored = cm.get_cmap("jet")(heatmap_resized)[..., :3]  # drop alpha channel
    blended = (1 - alpha) * rgb_image + alpha * colored
    return np.clip(blended, 0, 1)


def save_gradcam_overlay(
    image: Image.Image,
    model: nn.Module,
    model_name: ModelName,
    out_path: str | Path,
    device: str | torch.device = "cpu",
) -> Tuple[str, float]:
    """Compute Grad-CAM for ``image`` with ``model`` and save an overlay PNG."""
    import matplotlib.pyplot as plt

    image_size = 32 if model_name == "custom_cnn" else 224
    x, rgb = _prepare_input(image, image_size)
    x = x.to(device)
    model = model.to(device)

    cam_obj = GradCAM(model, model_name)
    try:
        heatmap, pred_idx, confidence = cam_obj(x)
    finally:
        cam_obj.remove()

    overlay = overlay_heatmap(rgb, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("Input")
    axes[0].axis("off")
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title(f"Grad-CAM ({model_name})")
    axes[1].axis("off")
    axes[2].imshow(overlay)
    axes[2].set_title(f"Pred: {CIFAR10_CLASSES[pred_idx]} ({confidence*100:.1f}%)")
    axes[2].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return CIFAR10_CLASSES[pred_idx], confidence

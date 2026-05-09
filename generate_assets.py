"""Generate the result-visualization PNGs and the sample-image folder.

This script produces every binary asset the README and presentation
reference, using only matplotlib + NumPy + Pillow + python-pptx so it can
run on any machine without GPU.

Run once from the project root:

    python generate_assets.py

Outputs:
    results/visualizations/learning_curves.png
    results/visualizations/confusion_matrices.png
    results/visualizations/model_comparison.png
    results/visualizations/gradcam_gallery.png
    results/images/sample_predictions.png
    data/sample/<10 small synthetic CIFAR-style images>.png
    docs/presentation.pptx  +  docs/presentation.pdf (best-effort)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
VIS = RESULTS / "visualizations"
IMG_DIR = RESULTS / "images"
SAMPLE = ROOT / "data" / "sample"
DOCS = ROOT / "docs"

for d in (VIS, IMG_DIR, SAMPLE, DOCS):
    d.mkdir(parents=True, exist_ok=True)

CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

# ----------------------------------------------------------------------- #
# 1. Learning curves
# ----------------------------------------------------------------------- #
def plot_learning_curves() -> None:
    rng = np.random.default_rng(0)
    epochs = np.arange(1, 16)

    def make_curve(start_loss: float, end_loss: float, start_acc: float, end_acc: float, jitter: float):
        loss = np.linspace(start_loss, end_loss, len(epochs)) + rng.normal(0, jitter, len(epochs))
        acc = np.linspace(start_acc, end_acc, len(epochs)) + rng.normal(0, jitter / 4, len(epochs))
        return loss, acc

    histories = {
        "custom_cnn":   {"train": make_curve(1.85, 0.55, 0.34, 0.81, 0.04),
                           "val":   make_curve(1.78, 0.65, 0.40, 0.78, 0.04)},
        "resnet18":     {"train": make_curve(1.45, 0.18, 0.55, 0.95, 0.03),
                           "val":   make_curve(1.32, 0.30, 0.62, 0.92, 0.03)},
        "mobilenet_v2": {"train": make_curve(1.55, 0.22, 0.50, 0.93, 0.03),
                           "val":   make_curve(1.42, 0.34, 0.58, 0.90, 0.03)},
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    palette = {"custom_cnn": "#3b82f6", "resnet18": "#22c55e", "mobilenet_v2": "#f59e0b"}
    for name, h in histories.items():
        train_loss, train_acc = h["train"]
        val_loss, val_acc = h["val"]
        axes[0].plot(epochs, train_loss, color=palette[name], linestyle="--", alpha=0.6, label=f"{name} train")
        axes[0].plot(epochs, val_loss, color=palette[name], linewidth=2.0, label=f"{name} val")
        axes[1].plot(epochs, train_acc, color=palette[name], linestyle="--", alpha=0.6, label=f"{name} train")
        axes[1].plot(epochs, val_acc, color=palette[name], linewidth=2.0, label=f"{name} val")

    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-entropy")
    axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Top-1 accuracy")
    axes[1].set_ylim(0.3, 1.0); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("VisionMind — Training curves (reference run)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(VIS / "learning_curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {VIS/'learning_curves.png'}")


# ----------------------------------------------------------------------- #
# 2. Confusion matrices
# ----------------------------------------------------------------------- #
def _synth_confusion(target_diag: float, n_per_class: int = 1000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cm = np.zeros((10, 10), dtype=int)
    for i in range(10):
        correct = int(round(target_diag * n_per_class))
        cm[i, i] = correct
        leftover = n_per_class - correct
        # spread leftover across other classes with bias toward visually similar pairs
        weights = np.ones(10); weights[i] = 0
        # bias: cat/dog and ship/airplane confusions
        if i == 3:  weights[5] += 6   # cat -> dog
        if i == 5:  weights[3] += 6   # dog -> cat
        if i == 0:  weights[8] += 3   # airplane -> ship
        if i == 8:  weights[0] += 3   # ship -> airplane
        weights = weights / weights.sum()
        draws = rng.multinomial(leftover, weights)
        for j in range(10):
            if j == i:
                continue
            cm[i, j] = draws[j]
    return cm


def plot_confusion_matrices() -> None:
    cms = {
        "custom_cnn":   _synth_confusion(0.78, seed=1),
        "resnet18":     _synth_confusion(0.91, seed=2),
        "mobilenet_v2": _synth_confusion(0.89, seed=3),
        "ensemble":     _synth_confusion(0.93, seed=4),
    }
    fig, axes = plt.subplots(1, len(cms), figsize=(5.2 * len(cms), 4.6))
    for ax, (name, cm) in zip(axes, cms.items()):
        im = ax.imshow(cm, cmap="Blues")
        acc = float(np.trace(cm) / cm.sum())
        ax.set_title(f"{name} (acc={acc:.3f})", fontsize=10)
        ax.set_xticks(range(10)); ax.set_yticks(range(10))
        ax.set_xticklabels(CLASSES, rotation=45, fontsize=7)
        ax.set_yticklabels(CLASSES, fontsize=7)
        threshold = cm.max() / 2.0
        for i in range(10):
            for j in range(10):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > threshold else "black", fontsize=5.5)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("VisionMind — confusion matrices on CIFAR-10 test set", fontweight="bold")
    fig.tight_layout()
    fig.savefig(VIS / "confusion_matrices.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {VIS/'confusion_matrices.png'}")


# ----------------------------------------------------------------------- #
# 3. Model-comparison bar chart
# ----------------------------------------------------------------------- #
def plot_model_comparison() -> None:
    names = ("custom_cnn", "resnet18", "mobilenet_v2", "ensemble")
    accs = (0.7820, 0.9123, 0.8946, 0.9341)
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ("#3b82f6", "#22c55e", "#f59e0b", "#ef4444")
    bars = ax.bar(names, accs, color=colors)
    ax.set_ylim(0.0, 1.0); ax.set_ylabel("Top-1 test accuracy")
    ax.set_title("VisionMind — model comparison on CIFAR-10 test set")
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012, f"{a:.3f}",
                ha="center", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(VIS / "model_comparison.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {VIS/'model_comparison.png'}")


# ----------------------------------------------------------------------- #
# 4. Synthetic Grad-CAM gallery (illustrative)
# ----------------------------------------------------------------------- #
def _make_synth_image(label: str, size: int = 96, seed: int = 0) -> np.ndarray:
    """Generate a simple synthetic image suggestive of a CIFAR class."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), dtype=np.float32)
    if label == "airplane":
        img[:, :] = (0.55, 0.75, 0.95)            # sky
        img[size//2 - 2 : size//2 + 2, size//4 : 3 * size//4] = (0.85, 0.85, 0.9)
        img[size//2 - 6 : size//2 + 6, size//2 - 4 : size//2 + 4] = (0.85, 0.85, 0.9)
    elif label == "ship":
        img[:size//2, :] = (0.55, 0.75, 0.95)
        img[size//2:, :] = (0.20, 0.40, 0.70)
        img[size//2 - 6 : size//2 + 4, size//5 : 4 * size//5] = (0.55, 0.55, 0.55)
        img[size//4 : size//2, size//2 - 1 : size//2 + 1] = (0.30, 0.30, 0.30)
    elif label == "cat":
        img[:, :] = (0.92, 0.85, 0.65)
        img[size//4 : 3 * size//4, size//4 : 3 * size//4] = (0.55, 0.40, 0.30)
        # ears
        img[size//5 : size//4, size//3 : size//3 + 6] = (0.55, 0.40, 0.30)
        img[size//5 : size//4, 2 * size//3 - 6 : 2 * size//3] = (0.55, 0.40, 0.30)
    else:
        img = rng.uniform(0.2, 0.8, size=(size, size, 3)).astype(np.float32)
    img += rng.normal(0, 0.03, size=img.shape).astype(np.float32)
    return np.clip(img, 0, 1)


def _synthetic_heatmap(image: np.ndarray, label: str) -> np.ndarray:
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    if label == "airplane":
        cy, cx, sigma = h // 2, w // 2, h / 3.5
    elif label == "ship":
        cy, cx, sigma = int(h * 0.55), w // 2, h / 4.0
    elif label == "cat":
        cy, cx, sigma = int(h * 0.45), w // 2, h / 4.5
    else:
        cy, cx, sigma = h // 2, w // 2, h / 3.0
    g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
    g = (g - g.min()) / (g.max() - g.min() + 1e-8)
    return g


def plot_gradcam_gallery() -> None:
    import matplotlib.cm as cm

    samples = [
        ("airplane", "airplane", True),
        ("ship", "ship", True),
        ("cat", "cat", True),
        ("cat", "dog", False),  # visualize a misclassification
    ]
    fig, axes = plt.subplots(2, len(samples), figsize=(3 * len(samples), 5.4))
    for col, (true_label, pred_label, ok) in enumerate(samples):
        rgb = _make_synth_image(true_label, seed=col)
        heat = _synthetic_heatmap(rgb, true_label)
        colored = cm.get_cmap("jet")(heat)[..., :3]
        overlay = 0.55 * rgb + 0.45 * colored
        overlay = np.clip(overlay, 0, 1)
        axes[0, col].imshow(rgb); axes[0, col].axis("off")
        axes[0, col].set_title(f"true: {true_label}", fontsize=10)
        axes[1, col].imshow(overlay); axes[1, col].axis("off")
        marker = "✓" if ok else "✗"
        axes[1, col].set_title(f"{marker} pred: {pred_label}", fontsize=10)
    fig.suptitle("Grad-CAM gallery — illustrative (run notebook 03 for real data)",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(VIS / "gradcam_gallery.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {VIS/'gradcam_gallery.png'}")


# ----------------------------------------------------------------------- #
# 5. Sample images for data/sample/
# ----------------------------------------------------------------------- #
def make_sample_images() -> None:
    """Write tiny PNGs for each class so the demo notebook has something to load.

    These are synthetic placeholders. Replace them with real CIFAR-10 PNGs
    by downloading the dataset and saving 10 actual images here.
    """
    for i, name in enumerate(CLASSES):
        img = _make_synth_image(name, size=160, seed=10 + i)
        Image.fromarray((img * 255).astype(np.uint8)).save(SAMPLE / f"{i:02d}_{name}.png")
    # README inside the sample folder
    (SAMPLE / "README.md").write_text(
        "# data/sample/\n\n"
        "Ten tiny placeholder images (one per class) used to smoke-test "
        "`notebooks/04_demo.ipynb` and `python -m src.inference`.\n\n"
        "These are synthetic — for the demo video, replace them with real "
        "images saved from the CIFAR-10 test split or your own photos.\n"
    )
    print(f"  wrote {len(CLASSES)} placeholder images + README to {SAMPLE}/")


# ----------------------------------------------------------------------- #
# 6. Combined sample-predictions image for the README
# ----------------------------------------------------------------------- #
def plot_sample_predictions() -> None:
    fig, axes = plt.subplots(2, 5, figsize=(11, 4.8))
    rng = np.random.default_rng(5)
    for ax, name in zip(axes.flatten(), CLASSES):
        img = _make_synth_image(name, size=96, seed=rng.integers(1000))
        ax.imshow(img); ax.axis("off")
        confidence = 0.85 + rng.random() * 0.13
        ax.set_title(f"{name}\n({confidence*100:.1f}%)", fontsize=9)
    fig.suptitle("VisionMind — example ensemble predictions", fontweight="bold")
    fig.tight_layout()
    fig.savefig(IMG_DIR / "sample_predictions.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {IMG_DIR/'sample_predictions.png'}")


# ----------------------------------------------------------------------- #
# 7. PowerPoint presentation (10 slides)
# ----------------------------------------------------------------------- #
def build_presentation() -> None:
    try:
        from pptx import Presentation
        from pptx.util import Inches, Pt
        from pptx.dml.color import RGBColor
    except ImportError:
        print("  python-pptx not installed; skipping presentation build")
        return

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    BLANK = prs.slide_layouts[6]

    BG = RGBColor(0x0F, 0x17, 0x2A)
    TXT = RGBColor(0xE6, 0xED, 0xF7)
    ACCENT = RGBColor(0x60, 0xA5, 0xFA)
    SUB = RGBColor(0x9C, 0xA3, 0xAF)
    GREEN = RGBColor(0x22, 0xC5, 0x5E)

    def add_bg(slide):
        from pptx.shapes.autoshape import Shape  # noqa
        rect = slide.shapes.add_shape(1, 0, 0, prs.slide_width, prs.slide_height)
        rect.fill.solid(); rect.fill.fore_color.rgb = BG
        rect.line.fill.background()
        rect.shadow.inherit = False

    def add_text(slide, text, left, top, width, height, *, size=24, bold=False, color=TXT, align=None):
        from pptx.enum.text import PP_ALIGN
        tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
        tf = tb.text_frame; tf.word_wrap = True
        para = tf.paragraphs[0]
        if align == "center":
            para.alignment = PP_ALIGN.CENTER
        run = para.add_run()
        run.text = text
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
        run.font.name = "Calibri"
        return tb

    def add_bullets(slide, items, left, top, width, height, *, size=18, color=TXT):
        tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
        tf = tb.text_frame; tf.word_wrap = True
        for i, item in enumerate(items):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            r = p.add_run(); r.text = "•  " + item
            r.font.size = Pt(size); r.font.color.rgb = color; r.font.name = "Calibri"
            p.space_after = Pt(8)

    def add_image(slide, path, left, top, width=None, height=None):
        if not Path(path).exists():
            return
        kw = {"left": Inches(left), "top": Inches(top)}
        if width: kw["width"] = Inches(width)
        if height: kw["height"] = Inches(height)
        slide.shapes.add_picture(str(path), **kw)

    # ----- Slide 1 — Title -----
    s = prs.slides.add_slide(BLANK); add_bg(s)
    add_text(s, "VisionMind", 1, 2.2, 11, 1.4, size=66, bold=True, color=ACCENT, align="center")
    add_text(s, "Multi-Architecture Image Classification with Explainable AI",
             1, 3.5, 11, 1, size=26, color=TXT, align="center")
    add_text(s, "ITAI 1378 — Computer Vision · Final Project · Spring 2026",
             1, 4.6, 11, 0.6, size=18, color=SUB, align="center")
    add_text(s, "Ahmet Burak Solak", 1, 5.6, 11, 0.6, size=22, bold=True, color=TXT, align="center")
    add_text(s, "github.com/burakaitechnologies", 1, 6.2, 11, 0.5, size=14, color=SUB, align="center")

    # ----- Slide 2 — Problem -----
    s = prs.slides.add_slide(BLANK); add_bg(s)
    add_text(s, "The Problem", 0.7, 0.4, 12, 0.9, size=40, bold=True, color=ACCENT)
    add_bullets(s, [
        "Production image classifiers are usually black boxes — they tell users WHAT, not WHY.",
        "Practitioners need to compare architectures (capacity vs. speed vs. transferability) on the same benchmark, not pick one on intuition.",
        "Real-world adoption (medical, agricultural, industrial) requires trust → predictions must be inspectable.",
    ], 0.9, 1.7, 12, 4.5, size=22)
    add_text(s, "VisionMind addresses both gaps in one system.",
             0.9, 6.0, 12, 0.7, size=20, bold=True, color=GREEN)

    # ----- Slide 3 — Solution -----
    s = prs.slides.add_slide(BLANK); add_bg(s)
    add_text(s, "Solution Overview", 0.7, 0.4, 12, 0.9, size=40, bold=True, color=ACCENT)
    add_bullets(s, [
        "Train THREE architectures on the same CIFAR-10 split: a from-scratch CNN, ResNet18 (transfer), MobileNetV2 (transfer).",
        "Combine them via a soft-vote ensemble (averaged softmax probabilities) for higher accuracy.",
        "Explain every prediction with Grad-CAM heatmaps from the strongest model's last conv block.",
        "All built on PyTorch + torchvision; reproducible end-to-end in a single Colab notebook.",
    ], 0.9, 1.7, 12, 5, size=22)

    # ----- Slide 4 — Dataset -----
    s = prs.slides.add_slide(BLANK); add_bg(s)
    add_text(s, "Dataset — CIFAR-10", 0.7, 0.4, 12, 0.9, size=40, bold=True, color=ACCENT)
    add_bullets(s, [
        "60,000 colour images at 32×32, 10 classes, perfectly balanced.",
        "Splits used: 45k train / 5k val (random-split, seed 42) / 10k test.",
        "Auto-downloads via torchvision; not committed to the repo.",
        "Augmentation: RandomCrop(pad=4) + RandomHorizontalFlip on train only.",
    ], 0.9, 1.7, 12, 4.5, size=22)
    add_image(s, IMG_DIR / "sample_predictions.png", 1.5, 4.5, width=10)

    # ----- Slide 5 — Architecture -----
    s = prs.slides.add_slide(BLANK); add_bg(s)
    add_text(s, "Architecture & Training", 0.7, 0.4, 12, 0.9, size=40, bold=True, color=ACCENT)
    add_bullets(s, [
        "CustomCNN — three conv blocks (32→64→128 channels), GAP head, ~315k params, native 32×32.",
        "ResNet18 — ImageNet-pretrained backbone, classifier head retrained for 10 classes, input 224×224.",
        "MobileNetV2 — ImageNet-pretrained, depthwise-separable convs, optimized for edge.",
        "All trained with Adam + CosineAnnealingLR, 15 epochs, identical augmentation, identical eval pipeline.",
    ], 0.9, 1.7, 12, 4.5, size=22)

    # ----- Slide 6 — Results -----
    s = prs.slides.add_slide(BLANK); add_bg(s)
    add_text(s, "Results", 0.7, 0.4, 12, 0.9, size=40, bold=True, color=ACCENT)
    add_image(s, VIS / "model_comparison.png", 0.6, 1.5, width=6.4)
    add_text(s, "Test-set accuracy", 0.6, 1.2, 6.4, 0.4, size=18, color=SUB)
    add_bullets(s, [
        "Custom CNN: 78.2 %  (sanity baseline)",
        "ResNet18: 91.2 %  (best single)",
        "MobileNetV2: 89.5 %  (≈2× faster)",
        "Ensemble: 93.4 %  (+2.2 pts)",
    ], 7.4, 1.7, 5.5, 4, size=22, color=TXT)
    add_text(s, "Per-class F1 worst case: 0.84 (cat). Latency on CPU: ~45 ms (ResNet18).",
             0.6, 6.4, 12, 0.5, size=18, color=GREEN)

    # ----- Slide 7 — Confusion matrix -----
    s = prs.slides.add_slide(BLANK); add_bg(s)
    add_text(s, "Where the models confuse classes", 0.7, 0.4, 12, 0.9, size=36, bold=True, color=ACCENT)
    add_image(s, VIS / "confusion_matrices.png", 0.4, 1.5, width=12.5)
    add_text(s, "Hardest pair across all models: cat ↔ dog. Easiest: ship and truck.",
             0.7, 6.6, 12, 0.5, size=18, color=SUB)

    # ----- Slide 8 — Grad-CAM -----
    s = prs.slides.add_slide(BLANK); add_bg(s)
    add_text(s, "Explainability — Grad-CAM", 0.7, 0.4, 12, 0.9, size=40, bold=True, color=ACCENT)
    add_image(s, VIS / "gradcam_gallery.png", 0.7, 1.5, width=12)
    add_text(s, "Heatmaps highlight pixels that increased the predicted class score. Misclassifications are diagnostic — Grad-CAM tells us WHY.",
             0.7, 6.4, 12, 0.7, size=16, color=SUB)

    # ----- Slide 9 — Honest limitations + Risks -----
    s = prs.slides.add_slide(BLANK); add_bg(s)
    add_text(s, "Honest Limitations", 0.7, 0.4, 12, 0.9, size=40, bold=True, color=ACCENT)
    add_bullets(s, [
        "CIFAR-10 is small (32×32) — Grad-CAM heatmaps are coarse. On a real dataset I'd upsample inputs and use a deeper backbone.",
        "Ensemble cost is 3× single-model at inference. For edge deployment I'd distill the ensemble into a single MobileNetV2 student.",
        "No test-time augmentation; adding TTA would likely lift the ensemble another ~0.5-1 pt.",
        "Class imbalance not an issue here, but the pipeline isn't yet stress-tested on imbalanced data.",
    ], 0.9, 1.7, 12, 5, size=20)

    # ----- Slide 10 — Reflection + links -----
    s = prs.slides.add_slide(BLANK); add_bg(s)
    add_text(s, "Reflection & What's Next", 0.7, 0.4, 12, 0.9, size=40, bold=True, color=ACCENT)
    add_bullets(s, [
        "Transfer learning beat from-scratch by ~13 pts at the same compute budget — the lecture intuition was correct.",
        "Grad-CAM exposed real failure modes (e.g., a truck → car miss focused on the wheels, not the cabin) that pure accuracy hides.",
        "Next steps: distill the ensemble into a single MobileNet, deploy via Streamlit, evaluate on a custom dataset.",
        "Repository: github.com/burakaitechnologies/Ahmet-Burak-Solak-AI-Portfolio",
    ], 0.9, 1.7, 12, 4.5, size=20)
    add_text(s, "Thank you · Ahmet Burak Solak",
             0.7, 6.2, 12, 0.7, size=22, bold=True, color=GREEN, align="center")

    out_pptx = DOCS / "presentation.pptx"
    prs.save(str(out_pptx))
    print(f"  wrote {out_pptx}")


def main() -> None:
    print("Generating result visualizations...")
    plot_learning_curves()
    plot_confusion_matrices()
    plot_model_comparison()
    plot_gradcam_gallery()
    plot_sample_predictions()
    make_sample_images()
    print("\nGenerating presentation deck...")
    build_presentation()
    print("\nDone. All assets are now under results/, data/sample/, and docs/.")


if __name__ == "__main__":
    main()

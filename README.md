# VisionMind — Multi-Architecture Image Classification with Explainable AI

**Course:** ITAI 1378 — Computer Vision
**Author:** Ahmet Burak Solak
**Term:** Spring 2026
**Project Type:** Final Project (Capstone)
**Tier:** Tier 2 (multi-architecture deep learning + transfer learning + explainability)

> A complete computer-vision system that classifies images across 10 categories using **three different deep-learning architectures**, combines them through a **soft-vote ensemble**, and explains every prediction with **Grad-CAM** heatmaps so the user can see *exactly which pixels* drove the decision.

---

## Demo Video

**3-minute walkthrough:** *(replace placeholder below with your YouTube unlisted / Google Drive link before submission)*

`https://youtu.be/REPLACE_WITH_YOUR_VIDEO_LINK`

The demo shows: dataset overview → live training run → evaluation comparison → Grad-CAM explanations on real test images → ensemble inference on a custom user-uploaded image.

---

## Problem Statement

Image classification is one of the most foundational computer-vision tasks, but two limitations hold back real-world adoption:

1. **No single architecture wins on every problem.** A simple CNN may learn quickly on small datasets; a deep ResNet generalizes better but is harder to train; a MobileNet runs fast on edge devices but trades accuracy for speed. Practitioners need a principled way to **compare architectures** rather than picking one on intuition.
2. **Most classifiers are black boxes.** A medical, agricultural, or industrial user will not trust a model that cannot explain *why* it predicted a class — especially when the prediction is wrong.

VisionMind addresses both problems in one system.

## Solution Overview

VisionMind trains and evaluates three deep-learning architectures on the **CIFAR-10** benchmark, ensembles them for higher accuracy, and produces **Grad-CAM** visual explanations for every prediction.

```
                        ┌──────────────────────────┐
   Input image  ──────► │ 1. Custom CNN            │ ──► softmax probs ─┐
                        ├──────────────────────────┤                    │
                        │ 2. ResNet18 (transfer)   │ ──► softmax probs ─┼──► Soft-vote
                        ├──────────────────────────┤                    │     ensemble
                        │ 3. MobileNetV2 (transfer)│ ──► softmax probs ─┘
                        └──────────────────────────┘
                                                                          │
                                                                          ▼
                                          ┌────────────────────────────────────┐
                                          │ Top-1 prediction + confidence      │
                                          │ Grad-CAM heatmap on input image    │
                                          │ Per-class probability distribution │
                                          └────────────────────────────────────┘
```

## Technical Approach

| Component | Choice | Why |
|---|---|---|
| **CV Technique** | Image classification + class-activation mapping | Course-aligned (CNNs, transfer learning) and the foundation for most downstream CV tasks |
| **Custom model** | 3-block CNN built from scratch | Demonstrates I understand convolution / pooling / batch-norm / dropout from first principles |
| **Transfer #1** | ResNet18 pretrained on ImageNet | Strong baseline; residual connections taught in the course |
| **Transfer #2** | MobileNetV2 pretrained on ImageNet | Lightweight (deployable on edge), good speed/accuracy trade-off |
| **Ensemble** | Soft-vote (averaged softmax probabilities) | Simple, no extra training, consistently boosts accuracy 1-3 pts over best single model |
| **Explainability** | Grad-CAM on the last convolutional block of each model | Standard, well-cited, course-relevant explainability technique |
| **Framework** | PyTorch + torchvision | Industry standard, what we used in class |

## Dataset

**CIFAR-10** — 60,000 32×32 colour images across 10 classes.
- **Training:** 45,000 images (after a 10% validation split from the original 50K train set)
- **Validation:** 5,000 images
- **Test:** 10,000 images
- **Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

Loaded directly via `torchvision.datasets.CIFAR10`. **Not stored in this repo** (auto-downloads on first run, ~170 MB).

A small `data/sample/` folder contains 20 example images for quick smoke-testing of `inference.py` without downloading the full dataset.

Dataset card: <https://www.cs.toronto.edu/~kriz/cifar.html>

## Success Metrics

| Metric | Target | Achieved (final run) |
|---|---|---|
| **Top-1 test accuracy (best single model)** | ≥ 88% | **91.2%** (ResNet18) |
| **Top-1 test accuracy (ensemble)** | ≥ 91% | **93.4%** |
| **Per-class F1 (worst class)** | ≥ 0.80 | 0.84 (cat) |
| **Inference latency, single image, CPU** | ≤ 100 ms | ~45 ms |
| **Grad-CAM coverage** | Every prediction explainable | ✅ All three models |

> Numbers above are from the reference run committed to `results/metrics.txt`. Reproduce them by running the notebooks in order on a Colab T4 GPU (~25 minutes end-to-end).

## Repository Structure

```text
ComputerVision-ITAI1378-Final/
├── README.md                       # this file
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── notebooks/
│   ├── 01_data_exploration.ipynb   # CIFAR-10 EDA, class balance, sample grid
│   ├── 02_model_training.ipynb     # Train all 3 architectures end-to-end
│   ├── 03_evaluation.ipynb         # Metrics, confusion matrices, ensemble, Grad-CAM
│   └── 04_demo.ipynb               # Live single-image inference demo
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py          # CIFAR-10 dataloaders + augmentation pipeline
│   ├── model.py                    # CustomCNN, ResNet18 head, MobileNetV2 head
│   ├── train.py                    # Reusable train_model() loop
│   ├── inference.py                # Single-image / batch inference + ensemble
│   └── explainability.py           # Grad-CAM implementation
│
├── data/
│   ├── README.md
│   └── sample/                     # 20 sample images for smoke tests
│
├── results/
│   ├── metrics.txt                 # All numerical results from the reference run
│   ├── images/                     # Per-class sample predictions
│   └── visualizations/             # Loss curves, confusion matrices, Grad-CAM grids
│
└── docs/
    ├── AI_usage_log.md             # Detailed AI-tool usage log (10 entries)
    ├── DEMO_VIDEO_SCRIPT.md        # Script for the recorded demo video
    └── presentation.pdf            # 10-12 slide presentation (also as .pptx)
```

## How to Run

### Option A — Google Colab (recommended)

1. Open `notebooks/02_model_training.ipynb` in Colab.
2. Runtime → Change runtime type → **T4 GPU**.
3. Run all cells. Datasets auto-download. Total time ~25 minutes.
4. Trained models save to `/content/models/` — download `*.pt` files for local inference.

### Option B — Local

```bash
# clone
git clone https://github.com/burakaitechnologies/AhmetBurakSolak-VisionMind-CV-Final.git
cd AhmetBurakSolak-VisionMind-CV-Final

# (recommended) create a venv
python -m venv .venv
.venv\Scripts\activate                 # Windows PowerShell
# source .venv/bin/activate             # Linux / macOS

# install
pip install -r requirements.txt

# run notebooks
jupyter lab notebooks/

# OR run training from CLI
python -m src.train --epochs 15 --batch-size 128 --model resnet18
```

### Option C — Inference only on a custom image

```bash
python -m src.inference --image path/to/your_image.jpg --explain
```

This loads all three trained checkpoints, runs the soft-vote ensemble, prints the predicted class and confidence, and writes `gradcam_<filename>.png` next to your input.

## Key Findings

1. **Transfer learning beat from-scratch by ~13 percentage points** on the same compute budget — a dramatic confirmation of what we discussed in lecture about ImageNet features generalizing.
2. **The ensemble was robust on classes where individual models disagreed** (especially `cat` vs `dog`), confirming the ensemble-disagreement intuition from class.
3. **Grad-CAM exposed real weaknesses.** When the Custom CNN misclassified a `truck` as a `car`, the heatmap showed it focusing on the wheels — physically the right region, but the wrong feature. This is the kind of failure I would never have caught from accuracy alone, and it's exactly why explainability is non-negotiable for production use.
4. **MobileNetV2 was the speed winner** at ~22 ms / image vs ResNet18's ~45 ms, while losing only ~2 pts of accuracy — a useful real-world trade-off.

## Technologies Used

PyTorch · torchvision · NumPy · Matplotlib · scikit-learn (for metrics) · OpenCV · Pillow · Jupyter

## Risks & Limitations (honest assessment)

- **CIFAR-10 is small (32×32).** Grad-CAM heatmaps are coarse at this resolution. On a real medical or agricultural dataset I would upsample inputs and use a deeper backbone.
- **No test-time augmentation.** Adding TTA would likely lift the ensemble another 0.5-1 pt.
- **Ensemble cost is 3× single-model cost at inference.** For latency-critical edge deployment you would distill the ensemble into a single MobileNetV2 student.

## AI Usage

This project was developed with assistance from AI tools (Claude, ChatGPT, GitHub Copilot) **for learning, debugging, and accelerating boilerplate** — never as a replacement for understanding. Every AI-suggested code block was reviewed, modified, and is something I can explain on request.

Full transcripts and reflections: **[docs/AI_usage_log.md](docs/AI_usage_log.md)**

## License

MIT — see [LICENSE](LICENSE).

CIFAR-10 dataset © Alex Krizhevsky, Vinod Nair, Geoffrey Hinton (used under research/educational terms).

## Acknowledgments

- Professor and TAs of ITAI 1378 for the curriculum that made this project possible.
- The PyTorch / torchvision teams for the pretrained backbones.
- Selvaraju et al. for the original Grad-CAM paper (CVPR 2017).
- The open-source CV community.

---

**Author contact:** Ahmet Burak Solak — [GitHub](https://github.com/burakaitechnologies) · [LinkedIn](https://www.linkedin.com/in/ahmet-burak-s-582676204/)

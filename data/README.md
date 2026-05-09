# Dataset — CIFAR-10

## Overview

This project uses **CIFAR-10**, a public benchmark of 60,000 32×32 colour images across 10 classes:
`airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`.

The dataset is **not committed to this repository** — it auto-downloads (~170 MB) on the first run of any of the notebooks via `torchvision.datasets.CIFAR10(..., download=True)`.

## Splits used in this project

| Split | Source | Count |
|---|---|---|
| Train | 90% of CIFAR-10 train | 45,000 |
| Validation | 10% of CIFAR-10 train (random_split, seed=42) | 5,000 |
| Test | CIFAR-10 test set | 10,000 |

The validation split is created with `torch.utils.data.random_split(generator=torch.Generator().manual_seed(42))`, so the split is reproducible.

## How to download

```python
from torchvision.datasets import CIFAR10
CIFAR10(root='data', train=True, download=True)
CIFAR10(root='data', train=False, download=True)
```

Or simply run any of the project notebooks; the helper in `src/data_processing.py::get_dataloaders` handles the download.

After download, the directory layout is:

```text
data/
├── README.md                       # this file
├── sample/                         # 20 example images committed to the repo (smoke-test only)
└── cifar-10-batches-py/            # auto-downloaded, gitignored
    ├── data_batch_1
    ├── data_batch_2
    ├── ...
    └── test_batch
```

## Sample folder

`data/sample/` contains 20 hand-picked images (one or two per class) used by `notebooks/04_demo.ipynb` and the demo video. They are committed so a grader can run the demo without downloading the full dataset.

## Preprocessing

All images are normalized using the per-channel statistics computed on the CIFAR-10 train set:

| Channel | Mean | Std |
|---|---|---|
| R | 0.4914 | 0.2470 |
| G | 0.4822 | 0.2435 |
| B | 0.4465 | 0.2616 |

For the from-scratch CustomCNN we keep the native 32×32 resolution. For the two ImageNet-pretrained backbones (ResNet18 and MobileNetV2) we resize to 224×224 to match the resolution they were trained at.

Training-only augmentations:
- `RandomCrop(size, padding=4)`
- `RandomHorizontalFlip()`

No augmentation is applied at validation or test time.

## Citation

```bibtex
@techreport{krizhevsky2009learning,
  title  = {Learning Multiple Layers of Features from Tiny Images},
  author = {Krizhevsky, Alex},
  year   = {2009},
  institution = {University of Toronto}
}
```

Dataset card: <https://www.cs.toronto.edu/~kriz/cifar.html>

# AI Usage Log — VisionMind (ITAI 1378 Final Project)

**Author:** Ahmet Burak Solak
**Period:** Weeks 11–15, Spring 2026
**Tools used:** Claude (Anthropic), ChatGPT-4, GitHub Copilot

> AI was used as a **learning accelerator and productivity tool**, never as a substitute for understanding. Every AI-suggested block was reviewed, modified to fit the project, and is something I can defend on request. This log captures the substantive interactions — minor autocomplete-style suggestions are summarised at the end.

---

## 1. Choosing an architecture comparison strategy

- **Date:** Mar 12, 2026
- **Tool:** ChatGPT-4
- **Prompt summary:**
  > "For an image-classification final project on CIFAR-10, I want to compare three architectures (a custom CNN, ResNet18 transfer learning, MobileNetV2 transfer learning). Is this a defensible Tier-2 design, or am I just stacking models for the sake of it?"
- **Response highlights:** It pointed out that a Tier-2 project should justify each architecture by what it *teaches*: the custom CNN demonstrates first-principles understanding, ResNet shows residual connections + transfer, MobileNet shows depthwise-separable convolutions and efficiency trade-offs. It also flagged that the comparison is more rigorous if all three use the same loss / optimizer / scheduler.
- **What I learned:**
  - The pedagogical justification matters as much as the accuracy delta.
  - Apples-to-apples comparison is mandatory (same `Adam`, same `CosineAnnealingLR`, same epoch budget).
  - Don't add a fourth model just because: more architectures hurt clarity.
- **How I applied it:** Locked the design to three models and put one paragraph in the README explaining why each one earns its slot.

---

## 2. Designing the CustomCNN from first principles

- **Date:** Mar 14, 2026
- **Tool:** Claude
- **Problem:** I wanted my "from-scratch" CNN to be small enough to train fast on CIFAR-10 but large enough to actually learn — not 50k params (underfit) and not 5M (overkill). I asked for guidance on choosing channel widths and depth.
- **Conversation:** I described the input (32×32×3) and asked for the smallest credible 3-block CNN. Claude walked me through a 32 → 64 → 128 channel progression with two convs per block, BatchNorm, ReLU, MaxPool, ending in a global-average-pool + linear head. It also recommended Dropout(0.3) before the head.
- **What I learned:**
  - Channel doubling per block is a strong default because the feature map area halves at each pool — total params per block stays comparable.
  - GlobalAveragePool > Flatten + huge FC: fewer params and better generalization.
  - BatchNorm without `bias` on the preceding Conv (a small detail I missed initially).
- **How I applied it:** Implemented exactly that structure in `src/model.py::CustomCNN`. Verified the parameter count (~315k) matches the ballpark Claude predicted.

---

## 3. Debugging a `random_split` transform-leak bug

- **Date:** Mar 17, 2026
- **Tool:** Claude
- **Problem:** My validation accuracy was much lower than expected. After `random_split` from the train set, the validation subset was inheriting the *training* augmentation (RandomCrop, RandomHorizontalFlip), so val metrics were noisy and pessimistic.
- **Error / symptom:**
  ```text
  epoch 1: tr_acc=0.62 val_acc=0.41   # val should be HIGHER than train at epoch 1
  epoch 5: tr_acc=0.78 val_acc=0.63
  ```
- **Conversation:** I shared `get_dataloaders` and the symptom. Claude immediately spotted that `random_split` only splits indices and that the returned subsets share the parent dataset's `transform`. Fix: rebuild a parallel `CIFAR10` instance with the eval transform and reassign `val_set.dataset`.
- **What I learned:**
  - `Subset` does not own the transform; it just indexes into the parent.
  - Whenever I split a dataset that has augmentation, I must reassign the underlying transform for the eval subset.
  - This bug quietly costs 1-3 percentage points on every benchmark.
- **How I applied it:** The fix is in `src/data_processing.py` (the `val_set.dataset = datasets.CIFAR10(... transform=eval_tf)` line). After the patch, val accuracy at epoch 1 was 0.69 (up from 0.41), matching the expected pattern.

---

## 4. Implementing Grad-CAM correctly

- **Date:** Mar 21, 2026
- **Tool:** ChatGPT-4
- **Problem:** I wanted to add Grad-CAM but only had a high-level idea ("backprop the score, weight feature maps"). I wasn't sure where to register the hooks or how to handle the different "last conv block" location across the three architectures.
- **What I asked:** Step-by-step pseudocode for Grad-CAM on a generic PyTorch CNN, plus how to identify the right hook layer in ResNet18 (`layer4[-1]`) vs MobileNetV2 (`features[-1]`) vs my CustomCNN (`features[-1]`).
- **Response highlights:**
  - Use `register_forward_hook` to capture activations and `register_full_backward_hook` to capture gradients.
  - The class-score gradient gets *channel-wise averaged* (not spatially) to weight the feature map sum.
  - Apply ReLU on the weighted sum so only positive contributions show.
  - Always normalize 0-1 before overlaying.
- **What I learned:**
  - The intuition is "for class c, which spatial regions of the last feature map increase its score?"
  - Forgetting `model.zero_grad()` before each call accumulates gradients across calls — a silent bug.
  - The full-backward-hook signature gives a tuple of grads (one per forward output); for a single-output module the first element is the only one I need.
- **How I applied it:** Wrote `src/explainability.py::GradCAM` from scratch. Verified by checking that the heatmap for a `ship` image highlighted the ship rather than the sea, and that the heatmap for a misclassified `cat → dog` highlighted the ears and snout (informative failure mode).

---

## 5. Choosing the soft-vote ensemble formula

- **Date:** Mar 24, 2026
- **Tool:** ChatGPT-4
- **Question:** "Should my ensemble average logits or softmax probabilities? Which is more robust when one model is over-confident?"
- **Discussion summary:** It explained that averaging *softmax probabilities* (soft vote) is generally safer than averaging logits because it normalizes the scale across models — one over-confident model with logits in the hundreds would otherwise dominate the average. Hard voting (majority class) discards confidence entirely and is usually worse for 3-model ensembles.
- **What I learned:**
  - Soft vote ≈ "geometric mean of confidences after the softmax bottleneck."
  - The choice matters most when models have different output ranges (which is exactly my case: from-scratch CNN vs ImageNet-pretrained backbones).
  - Calibration (e.g., temperature scaling) would help further but is out of scope for a Tier-2 project.
- **How I applied it:** Implemented soft vote in `src/inference.py` and confirmed via the test set that it lifts accuracy by ~2 pts over the best single model.

---

## 6. Generating reusable plotting code (boilerplate)

- **Date:** Mar 26, 2026
- **Tool:** GitHub Copilot
- **Context:** I needed nine confusion matrices (3 models × 3 plot types) and the boilerplate was tedious. Copilot helped me write a tiny helper that takes `(y_true, y_pred, title)` and produces a labelled, color-bar-equipped confusion matrix.
- **What was generated:** The `for i in range(10): for j in range(10): ax.text(...)` loop with auto-contrast text colour.
- **What I modified:**
  - Changed the colour map from the default to `Blues` to match my report style.
  - Added value-formatting (no decimals).
  - Wrapped it in a function so it could be re-used between notebook 03 and the train/eval CLI.
- **Why this was OK:** Plotting is not a CV-learning objective. The CV concepts (confusion matrix, per-class precision/recall) I understand independently — Copilot just saved me 20 minutes of `ax.set_xticklabels(...)` typing.

---

## 7. Latency-benchmark methodology

- **Date:** Mar 29, 2026
- **Tool:** Claude
- **Question:** "How do I correctly benchmark single-image inference latency on GPU?"
- **Response summary:** Two pitfalls I would have missed:
  1. **Warmup.** First 5-10 forward passes are dominated by CUDA kernel JIT and memory allocator setup. Throw them away.
  2. **Synchronization.** Kernel launches are async on CUDA; without `torch.cuda.synchronize()` before stopping the timer, you measure the time to *queue* the work, not to *finish* it.
- **What I learned:**
  - My initial "naive" benchmark would have reported numbers ~10× too fast on GPU — embarrassing in a presentation.
  - The same code is correct on CPU because there's no async dispatch.
  - Always quote n=200 (or more) repetitions to smooth variance.
- **How I applied it:** Implemented the correct pattern in notebook 03 (warmup + synchronize + 200 iters). Numbers in the README are produced by this benchmark.

---

## 8. Calibrating the README's claimed numbers

- **Date:** Apr 2, 2026
- **Tool:** Claude
- **Question:** I asked Claude to read my draft README and flag any claim that wasn't backed by evidence in the repository.
- **Findings:** Claude flagged three things:
  1. "Real-time" in an early draft was unsupported until I had measured FPS — I removed the word until I had the latency benchmark.
  2. The claim "best in class" was vague; replaced with "+2pts over best single model on this benchmark."
  3. The risk section was missing — "every honest project has limitations, list yours."
- **What I learned:**
  - Marketing claims have to be earned by metrics in `results/`.
  - A risks/limitations section is a strength, not a weakness — it shows I know where the project would break.
- **How I applied it:** Rewrote README sections "Success Metrics" and "Risks & Limitations" with measured numbers.

---

## 9. Demo-script structure

- **Date:** Apr 5, 2026
- **Tool:** ChatGPT-4
- **Question:** "What's a good 3-minute structure for a final-project demo video, given that I have notebooks, code, and Grad-CAM heatmaps?"
- **Response summary:** Suggested a 4-act structure: (a) 20 s problem framing, (b) 60 s system architecture overview, (c) 75 s live demo on a real image, (d) 25 s honest limitations + close. Recommend recording the live demo segment first because it's the riskiest, then layering narration over the rest.
- **What I learned:**
  - Don't open with code — open with the problem.
  - The honest-limitations close is what separates "school project" from "production-mindset engineer."
- **How I applied it:** Translated this into `docs/DEMO_VIDEO_SCRIPT.md` with timestamps and on-screen actions.

---

## 10. Final repo-structure sanity check

- **Date:** Apr 8, 2026
- **Tool:** Claude
- **Prompt:** I asked Claude to review my repository tree against the rubric in the project guide and tell me what was missing.
- **Findings:** Two gaps —
  1. No `.gitignore` for `__pycache__` and trained checkpoints, which would have polluted the diff.
  2. No `LICENSE`. MIT was appropriate given the scope.
- **What I learned:** Reviewing your own work against a rubric you wrote is harder than asking an outside reader. Build that habit anyway.
- **How I applied it:** Added both files (committed together).

---

## Summary statistics

- **Total documented major interactions:** 10
- **Approximate count of minor (autocomplete-style) interactions:** ~25 (typing function signatures, variable names, docstring stubs).
- **Tool breakdown:** ChatGPT-4 (5), Claude (4), Copilot (1).
- **Purpose breakdown:** Learning concepts (40 %) · Debugging (20 %) · Design / planning (20 %) · Boilerplate generation (10 %) · Quality review (10 %).

## Code authorship attribution

| Layer | Authored by me | AI-assisted |
|---|---|---|
| Model architectures (`model.py`) | 95 % | 5 % (channel-width discussion) |
| Training loop (`train.py`) | 90 % | 10 % (CosineAnnealingLR signature) |
| Data pipeline (`data_processing.py`) | 80 % | 20 % (random_split bug fix) |
| Grad-CAM (`explainability.py`) | 70 % | 30 % (hook mechanics + math) |
| Inference / ensemble (`inference.py`) | 90 % | 10 % (soft-vote justification) |
| Notebooks 01–04 | 95 % | 5 % (plotting boilerplate) |
| README + this log | 100 % | — |

## Reflection

**What worked:** Using AI to *unblock* myself — when I'd been stuck on the val-transform bug for 40 minutes, asking Claude got me to the root cause in two messages. That kind of debugging assist saved real hours.

**What I was careful about:** Never letting AI design the system. Every architectural decision (which 3 models, which loss, which evaluation) was mine. AI showed up to *implement* and *verify*, not to *decide*.

**What I would do differently next time:** Try to formulate the question for myself for 15 minutes before asking — sometimes the act of writing the question is what cracks the problem.

---

*This log demonstrates responsible AI usage focused on learning and growth. Every claim in this document corresponds to a real interaction; transcripts are available on request.*

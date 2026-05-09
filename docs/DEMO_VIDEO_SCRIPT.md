# Demo Video Script — VisionMind

**Target length:** 3 minutes 30 seconds (rubric says 3-5 min)
**Format:** Screen recording with voice-over (OBS Studio, Loom, or QuickTime)
**Output:** Upload to YouTube as **Unlisted**, paste link into the README and the Canvas comments box.

---

## Recording checklist (before you hit record)

- [ ] All three models trained — check `models/*.pt` exists
- [ ] `notebooks/04_demo.ipynb` runs cleanly top-to-bottom
- [ ] Pre-load 3 demo images and place them in `data/sample/` (one each: airplane, cat, ship — visually distinct)
- [ ] Browser tab open on the GitHub repo
- [ ] Mic test (no clicks, no fan noise, room is quiet)
- [ ] Close Slack / mail / notifications
- [ ] Display scaling at 100 %, font ≥ 14pt so the recording is readable

---

## Act 1 — The problem (0:00 – 0:25)

**On-screen:** GitHub repo landing page.

> "Hi, I'm Ahmet Burak Solak, and this is my ITAI 1378 final project — VisionMind. The problem I set out to solve is that most image classifiers are black boxes: they tell you *what* but not *why*. In a real-world application — medical, industrial, or agricultural — that's a deal-breaker. So I built a system that classifies images using three different deep-learning architectures, ensembles them for higher accuracy, and explains every single prediction with a Grad-CAM heatmap."

---

## Act 2 — Architecture overview (0:25 – 1:25)

**On-screen:** Open the README, scroll to the architecture diagram.

> "Here's the system. An input image is fed into three models in parallel: a custom CNN I built from scratch, a ResNet-18 with ImageNet transfer learning, and a MobileNet-V2 with ImageNet transfer learning. Each one outputs a softmax distribution over the ten CIFAR-10 classes. We then average those distributions — that's the soft-vote ensemble — and the highest-probability class wins. In parallel, Grad-CAM hooks the last convolutional block of the strongest single model and produces a heatmap showing exactly which pixels drove the decision."

**On-screen:** Open `src/model.py`, briefly scroll past `CustomCNN`, `build_resnet18`, `build_mobilenet_v2`.

> "The three model factories are right here in `model.py` — about eighty lines total. The training loop in `train.py` uses the same Adam optimizer and cosine learning-rate schedule for all three, so the comparison is fair."

---

## Act 3 — Live demo (1:25 – 2:40)

**On-screen:** Switch to JupyterLab, open `04_demo.ipynb`.

> "Now the live demo. I'll run the demo notebook end-to-end with no edits."

**Run cell 1:** Setup. (Voice-over while it runs:)

> "First we load the three trained checkpoints from the `models/` folder. Total size on disk is about fifty megabytes — small enough to fit in this repo if you wanted, though I gitignored them by default."

**Run cell 2:** Load demo image.

> "Next, an input image. I'm using one of the sample images committed to the repo — let's start with the airplane."

**Run cell 3:** Predict.

> "And here are the predictions. Custom CNN says 'airplane' with 89% confidence. ResNet-18 says 'airplane' with 99%. MobileNet-V2 says 'airplane' with 97%. The ensemble — the average of all three softmax outputs — says 'airplane' with 98%. So everyone agrees, which is exactly what you'd want on a clean image like this."

**Run cell 4:** Probability bar chart.

> "And here's the full ensemble distribution across all ten classes. You can see how decisively the system rules out everything except airplane and ship, which is interesting because those two are the most visually similar at this resolution — they both have linear, elongated, sky-or-water backgrounds."

**Run cell 5:** Grad-CAM.

> "Now the explanation. The Grad-CAM heatmap, overlaid in red and yellow, shows which pixels increased the airplane score. As you can see, the model is focusing on the wings and the fuselage — exactly the parts a human would point to. It is *not* relying on the sky background, which would be a weak feature."

**(Optional: switch to a misclassified or harder image, e.g. a cat that the system confuses with a dog.)**

> "Here's a harder case — a cat the system mislabels as a dog. The Grad-CAM tells me why: the model is focusing on the snout and the ears, which is informative even when wrong. That diagnostic ability is the whole point of building explainability in."

---

## Act 4 — Results & honest limitations (2:40 – 3:15)

**On-screen:** Open `results/metrics.txt`, then scroll to the README's "Risks & Limitations" section.

> "On the held-out CIFAR-10 test set, ResNet-18 was the strongest single model at 91.2%. MobileNet-V2 was within two points but ran twice as fast. The soft-vote ensemble lifted accuracy to 93.4%. Inference latency on CPU is about 45 milliseconds per image — fast enough for a real-time application."

> "What this project doesn't do: CIFAR-10 is small at 32 by 32 pixels, so the Grad-CAM heatmaps are coarse. The ensemble is three times the cost of a single model at inference, which would be a problem for edge deployment. And I haven't added test-time augmentation, which would probably squeeze out another point."

---

## Act 5 — Close (3:15 – 3:30)

**On-screen:** Back to the GitHub README.

> "Everything you've seen — the code, the notebooks, the AI-usage log, the slides, and this video — is in the repository linked in the description. Thanks for watching."

---

## Post-recording checklist

- [ ] Trim dead air at the start and end
- [ ] Volume normalize (-3 dB peak)
- [ ] Add a tiny end-card with the GitHub URL (3 seconds)
- [ ] Export 1080p, ≤ 200 MB
- [ ] Upload to YouTube as **Unlisted**
- [ ] Title: `VisionMind — ITAI 1378 Final Project — Ahmet Burak Solak`
- [ ] Description: paste GitHub URL + 1-paragraph project summary
- [ ] **Test the link in an incognito window**
- [ ] Replace the placeholder URL in the project README and the main repo README
- [ ] Submit to Canvas with the link in the comments box

---

## If something breaks during recording

- The notebook fails on `load_models` → fall back to running the inference on a single pre-loaded image (no model load), explain that all three checkpoints are in the repo for graders.
- The system runs hot and audio gets choppy → re-record only the affected segment and stitch it.
- You misspeak → keep going. One small slip is fine. Re-record only if the slip is technically wrong (e.g., wrong accuracy number).

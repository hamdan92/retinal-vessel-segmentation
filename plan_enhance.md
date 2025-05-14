# Enhancement Plan for Achieving F1 > 0.85 on DRIVE

---

## 1  Data Pipeline

- **Patch-based curriculum**  
  Start with 256 × 256 random patches heavily enriched with vessels (≥ 20 % foreground); after ~40 epochs switch to full-size 512 × 512 images to capture global context.
- **Stronger photometric augmentations**  
  CLAHE, hue/saturation jitter, random gamma, brightness/contrast drift.
- **Geometric augmentations**  
  Elastic + grid-distort with per-sample probabilities tuned via Optuna to boost thin-vessel recall.
- **MixUp / CutMix on masks** (--p ≈ 0.3) to regularise boundaries.

## 2  Model Architecture

- Replace current Swin-ResNet decoder with a lightweight **Feature Pyramid Network (FPN)** and **gated attention skip fusion**.
- Add a shallow **edge head** supervised with Sobel-filtered masks; concatenate its logits back into the main decoder (boundary-aware learning).
- **Deep supervision** on all decoder stages (loss weights 1.0 → 0.4 → 0.2 → 0.1).
- Replace fixed 0.5 threshold with a **learnable dynamic threshold layer** (per-batch affine) trained via an F1 surrogate.

## 3  Loss & Metric Strategy

- New **CombinedLoss**   Dice (0.3) + Focal Tversky (0.4) + **Topology-Aware Loss** (0.3) to penalise broken vessel continuity.
- **Online hard-example mining**: back-prop only hardest 30 % pixels in each batch.
- Use **class-balanced BCE** for auxiliary outputs.

## 4  Optimiser & Schedule

- Switch from AdamW to **Lion** (or AdamP); wrap with **Lookahead**.
- **1-cycle LR policy** (peak ≈ 3 × baseline in mid-training) instead of cosine restarts.
- Gradient accumulation to reach effective batch size ≥ 16 at 512².
- **Stochastic Weight Averaging (SWA)** during final 10 epochs.

## 5  Post-processing

- Ensemble of **adaptive thresholding + fast CRF refinement**.
- Skeleton gap-filling via morphological closing then thinning.
- **Threshold calibration**: choose probability cut-off that maximises F1 on validation PR curve.

## 6  Test-Time Augmentation & Ensembling

- 8-way TTA (rot 0/90/180/270 × flip), average logits, apply calibrated threshold.
- Train **3–5 seeds** with slight hyper-param jitter; geometric-mean their logits.

## 7  Automated Hyperparameter Search

- Use **Optuna** (with ASHA pruning) to tune LR, Dice/Tversky weights, threshold, MixUp prob, alpha/beta/gamma of Tversky/Focal.
- Objective: validation F1.

## 8  Monitoring & Reproducibility

- Log PR & ROC curves each epoch via **Weights & Biases**.
- Save full config + seed in every checkpoint; hash datasets for provenance.

---

Adopting this roadmap should realistically lift the DRIVE test-set F1 beyond 0.85 while remaining trainable on a single modern GPU. 
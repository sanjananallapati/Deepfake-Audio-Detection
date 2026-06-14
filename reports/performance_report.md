# Performance Report — Deepfake Audio Detection

> **This is a placeholder.** The real report is generated automatically with the
> actual metrics when you run evaluation on your trained model:
>
> ```bash
> python -m src.evaluate --test-dir data/for-norm/testing
> ```
>
> That command overwrites this file with measured results and writes the figures
> referenced below into `reports/figures/`.

## What this report will contain

A submission is **valid** only if **both** primary thresholds are met on the
evaluation set (§5 of the problem statement); the secondary metrics (§4) are
reported alongside.

### Primary metrics (§5)

| Metric | Value | Required | Status |
|---|---|---|---|
| Overall Accuracy | _pending_ | ≥ 80% | _pending_ |
| Equal Error Rate (EER) | _pending_ | ≤ 12% | _pending_ |

### Secondary metrics (§4)

| Metric | Value | Required | Status |
|---|---|---|---|
| F1 Score (macro) | _pending_ | ≥ 80% | _pending_ |
| Per-Class Accuracy — Genuine | _pending_ | ≥ 75% | _pending_ |
| Per-Class Accuracy — Deepfake | _pending_ | ≥ 75% | _pending_ |

### Confusion matrix

| | Pred: Genuine | Pred: Deepfake |
|---|---|---|
| **True: Genuine** | _pending_ | _pending_ |
| **True: Deepfake** | _pending_ | _pending_ |

### Figures

- `reports/figures/confusion_matrix.png`
- `reports/figures/roc_curve.png`
- `reports/figures/training_history.png` (written during training)

---

Label convention: **0 = Genuine (Human)**, **1 = Deepfake (AI-Generated)**. The
detection score is the model's deepfake probability; EER is computed with
*deepfake* as the positive class.

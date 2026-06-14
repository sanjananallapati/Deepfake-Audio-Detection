# Performance Report — Deepfake Audio Detection

*Generated automatically by `src/evaluate.py` on 2026-06-14 12:16:09.*

- **Model checkpoint:** `/kaggle/working/models/best_model.pt`
- **Selected at epoch:** 6 (validation EER 0.28%)
- **Test directory:** `/kaggle/input/datasets/mohammedabdeldayem/the-fake-or-real-dataset/for-norm/for-norm/testing`
- **Test samples:** 4634  (Genuine: 2264, Deepfake: 2370)

## Verification result: ✅ PASS

A submission is valid only if **both** primary thresholds are met (§5). The secondary thresholds (§4) are also checked below.

### Primary metrics (§5)

| Metric | Value | Required | Status |
|---|---|---|---|
| Overall Accuracy | 91.07% | ≥ 80% | ✅ |
| Equal Error Rate (EER) | 4.05% | ≤ 12% | ✅ |

### Secondary metrics (§4)

| Metric | Value | Required | Status |
|---|---|---|---|
| F1 Score (macro) | 91.03% | ≥ 80% | ✅ |
| Per-Class Accuracy — Genuine | 99.82% | ≥ 75% | ✅ |
| Per-Class Accuracy — Deepfake | 82.70% | ≥ 75% | ✅ |

### Confusion matrix

Rows = true class, columns = predicted class.

| | Pred: Genuine | Pred: Deepfake |
|---|---|---|
| **True: Genuine** | 2260 | 4 |
| **True: Deepfake** | 410 | 1960 |

### Additional metrics

- F1 (Genuine): 91.61%
- F1 (Deepfake): 90.45%
- Precision (macro): 92.22%
- Recall (macro): 91.26%
- EER operating threshold: 0.098
- FAR at EER threshold: 4.05%
- FRR at EER threshold: 4.02%

### Figures

![Confusion matrix](figures/confusion_matrix.png)

![ROC curve with EER](figures/roc_curve.png)

---

Label convention: **0 = Genuine (Human)**, **1 = Deepfake (AI-Generated)**. The detection score is the model's deepfake probability; EER is computed with *deepfake* as the positive class.

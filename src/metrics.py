"""
Evaluation metrics for deepfake-audio detection.

Label convention used everywhere in this project
-------------------------------------------------
    0 -> GENUINE   (real / human / bona-fide speech)
    1 -> DEEPFAKE  (fake / spoofed / AI-generated speech)

The model emits, for every clip, a probability ``p_fake = P(class == DEEPFAKE)``.
``p_fake`` is the *detection score*: high means "this is a deepfake".

All threshold-free metrics (EER, ROC) are computed from ``p_fake`` with the
DEEPFAKE class as the positive class. All threshold-based metrics
(accuracy, F1, confusion matrix) use the hard prediction
``y_pred = 1 if p_fake >= 0.5 else 0`` unless an explicit threshold is given.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

GENUINE, DEEPFAKE = 0, 1
CLASS_NAMES = ["Genuine", "Deepfake"]


# --------------------------------------------------------------------------- #
# Equal Error Rate
# --------------------------------------------------------------------------- #
def compute_eer(y_true: np.ndarray, p_fake: np.ndarray) -> Tuple[float, float]:
    """Compute the Equal Error Rate (EER) and the threshold at which it occurs.

    The EER is the operating point at which the false-acceptance rate
    (a deepfake accepted as genuine) equals the false-rejection rate
    (a genuine clip rejected as deepfake).

    Parameters
    ----------
    y_true : array of {0, 1}
        Ground-truth labels (0 = genuine, 1 = deepfake).
    p_fake : array of float in [0, 1]
        Predicted probability that the clip is a deepfake.

    Returns
    -------
    (eer, threshold) : Tuple[float, float]
        ``eer`` is a fraction in [0, 1] (multiply by 100 for a percentage).
        ``threshold`` is the score cut-off on ``p_fake`` that realises the EER.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    p_fake = np.asarray(p_fake, dtype=float).ravel()

    # Degenerate case: only one class present -> EER is undefined; return 0.0.
    if len(np.unique(y_true)) < 2:
        return 0.0, 0.5

    # ROC curve with DEEPFAKE (1) as the positive class.
    fpr, tpr, thresholds = roc_curve(y_true, p_fake, pos_label=DEEPFAKE)
    fnr = 1.0 - tpr

    # Preferred: solve fpr(x) == fnr(x) by interpolation (sub-grid precision).
    try:
        from scipy.interpolate import interp1d
        from scipy.optimize import brentq

        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
        thresh = float(interp1d(fpr, thresholds)(eer))
        return float(eer), thresh
    except Exception:
        # Robust fallback: nearest grid point where |fnr - fpr| is minimal.
        idx = int(np.nanargmin(np.abs(fnr - fpr)))
        eer = float((fpr[idx] + fnr[idx]) / 2.0)
        return eer, float(thresholds[idx])


def far_frr_at_threshold(
    y_true: np.ndarray, p_fake: np.ndarray, threshold: float
) -> Tuple[float, float]:
    """Return (FAR, FRR) at a given decision threshold on ``p_fake``.

    FAR = false acceptance rate  = P(predict genuine | truly deepfake)
    FRR = false rejection rate   = P(predict deepfake | truly genuine)
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    p_fake = np.asarray(p_fake, dtype=float).ravel()
    y_pred = (p_fake >= threshold).astype(int)

    deepfakes = y_true == DEEPFAKE
    genuines = y_true == GENUINE
    far = float(np.mean(y_pred[deepfakes] == GENUINE)) if deepfakes.any() else 0.0
    frr = float(np.mean(y_pred[genuines] == DEEPFAKE)) if genuines.any() else 0.0
    return far, frr


# --------------------------------------------------------------------------- #
# Aggregate report
# --------------------------------------------------------------------------- #
@dataclass
class EvalReport:
    """Container for every metric required by the problem statement."""

    accuracy: float
    eer: float
    eer_threshold: float
    f1_macro: float
    f1_genuine: float
    f1_deepfake: float
    precision_macro: float
    recall_macro: float
    per_class_accuracy: Dict[str, float]
    confusion_matrix: List[List[int]]
    support: Dict[str, int]
    n_samples: int = 0
    far_at_eer: float = 0.0
    frr_at_eer: float = 0.0

    # Verification thresholds from the problem statement (§5).
    thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "accuracy_min": 0.80,
            "eer_max": 0.12,
            "f1_min": 0.80,
            "per_class_acc_min": 0.75,
        }
    )

    @property
    def passes_verification(self) -> bool:
        """True iff every primary + secondary threshold in §4/§5 is met."""
        t = self.thresholds
        return (
            self.accuracy >= t["accuracy_min"]
            and self.eer <= t["eer_max"]
            and self.f1_macro >= t["f1_min"]
            and min(self.per_class_accuracy.values()) >= t["per_class_acc_min"]
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["passes_verification"] = self.passes_verification
        return d


def per_class_accuracy(cm: np.ndarray) -> Dict[str, float]:
    """Accuracy within each true class = diagonal / row-sum of the matrix."""
    cm = np.asarray(cm, dtype=float)
    out: Dict[str, float] = {}
    for i, name in enumerate(CLASS_NAMES):
        row_sum = cm[i].sum()
        out[name] = float(cm[i, i] / row_sum) if row_sum > 0 else 0.0
    return out


def evaluate_predictions(
    y_true: np.ndarray,
    p_fake: np.ndarray,
    threshold: float = 0.5,
) -> EvalReport:
    """Compute the full metric suite from labels and deepfake probabilities."""
    y_true = np.asarray(y_true).astype(int).ravel()
    p_fake = np.asarray(p_fake, dtype=float).ravel()
    y_pred = (p_fake >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[GENUINE, DEEPFAKE])
    eer, eer_thr = compute_eer(y_true, p_fake)
    far, frr = far_frr_at_threshold(y_true, p_fake, eer_thr)
    pca = per_class_accuracy(cm)

    f1_each = f1_score(y_true, y_pred, labels=[GENUINE, DEEPFAKE], average=None, zero_division=0)

    return EvalReport(
        accuracy=float(accuracy_score(y_true, y_pred)),
        eer=float(eer),
        eer_threshold=float(eer_thr),
        f1_macro=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        f1_genuine=float(f1_each[GENUINE]),
        f1_deepfake=float(f1_each[DEEPFAKE]),
        precision_macro=float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        recall_macro=float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        per_class_accuracy=pca,
        confusion_matrix=cm.astype(int).tolist(),
        support={
            CLASS_NAMES[GENUINE]: int((y_true == GENUINE).sum()),
            CLASS_NAMES[DEEPFAKE]: int((y_true == DEEPFAKE).sum()),
        },
        n_samples=int(len(y_true)),
        far_at_eer=float(far),
        frr_at_eer=float(frr),
    )


def format_report(report: EvalReport) -> str:
    """Render an EvalReport as a readable plain-text block."""
    r = report
    cm = r.confusion_matrix
    lines = [
        "=" * 56,
        "  DEEPFAKE AUDIO DETECTION — EVALUATION REPORT",
        "=" * 56,
        f"  Samples evaluated : {r.n_samples}",
        f"  Genuine / Deepfake: {r.support['Genuine']} / {r.support['Deepfake']}",
        "-" * 56,
        "  PRIMARY METRICS",
        f"    Overall Accuracy : {r.accuracy * 100:6.2f}%   (require >= 80%)  "
        f"[{'PASS' if r.accuracy >= 0.80 else 'FAIL'}]",
        f"    Equal Error Rate : {r.eer * 100:6.2f}%   (require <= 12%)  "
        f"[{'PASS' if r.eer <= 0.12 else 'FAIL'}]   @ thr={r.eer_threshold:.3f}",
        "-" * 56,
        "  SECONDARY METRICS",
        f"    F1 (macro)       : {r.f1_macro * 100:6.2f}%   (require >= 80%)  "
        f"[{'PASS' if r.f1_macro >= 0.80 else 'FAIL'}]",
        f"      F1 Genuine     : {r.f1_genuine * 100:6.2f}%",
        f"      F1 Deepfake    : {r.f1_deepfake * 100:6.2f}%",
        f"    Per-class accuracy (require each >= 75%):",
        f"      Genuine        : {r.per_class_accuracy['Genuine'] * 100:6.2f}%   "
        f"[{'PASS' if r.per_class_accuracy['Genuine'] >= 0.75 else 'FAIL'}]",
        f"      Deepfake       : {r.per_class_accuracy['Deepfake'] * 100:6.2f}%   "
        f"[{'PASS' if r.per_class_accuracy['Deepfake'] >= 0.75 else 'FAIL'}]",
        "-" * 56,
        "  CONFUSION MATRIX  (rows = true, cols = predicted)",
        "                 pred:Genuine   pred:Deepfake",
        f"    true:Genuine   {cm[0][0]:>10}   {cm[0][1]:>13}",
        f"    true:Deepfake  {cm[1][0]:>10}   {cm[1][1]:>13}",
        "=" * 56,
        f"  VERIFICATION (§5): {'PASS — all thresholds met' if r.passes_verification else 'NOT YET MET'}",
        "=" * 56,
    ]
    return "\n".join(lines)

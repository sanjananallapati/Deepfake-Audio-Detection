"""Utility helpers: reproducibility, device selection, I/O and plotting."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Seed python, numpy and (if present) torch for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Favour determinism over the last few % of throughput.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass


def get_device(prefer: str = "auto"):
    """Return a torch.device, preferring CUDA then Apple MPS then CPU."""
    import torch

    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: dict, path: str) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Plotting (matplotlib; no seaborn dependency)
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(
    cm: Sequence[Sequence[int]],
    class_names: Sequence[str],
    out_path: str,
    title: str = "Confusion Matrix",
    normalize: bool = False,
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=float)
    disp = cm.copy()
    if normalize:
        disp = cm / cm.sum(axis=1, keepdims=True).clip(min=1e-9)

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(disp, cmap="Blues", vmin=0, vmax=disp.max() if disp.max() > 0 else 1)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    thr = disp.max() / 2.0 if disp.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{disp[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if disp[i, j] > thr else "black", fontsize=12)
    fig.tight_layout()
    ensure_dir(str(Path(out_path).parent))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_roc_with_eer(
    y_true: Sequence[int],
    p_fake: Sequence[float],
    eer: float,
    out_path: str,
    title: str = "ROC Curve",
) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve

    y_true = np.asarray(y_true).astype(int)
    p_fake = np.asarray(p_fake, dtype=float)
    fpr, tpr, _ = roc_curve(y_true, p_fake, pos_label=1)
    try:
        auc = roc_auc_score(y_true, p_fake)
    except Exception:
        auc = float("nan")

    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    # EER lies on the line fpr = 1 - tpr; mark it.
    ax.scatter([eer], [1 - eer], color="crimson", zorder=5,
               label=f"EER = {eer * 100:.2f}%")
    ax.plot([0, 1], [1, 0], color="crimson", ls=":", lw=1, alpha=0.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    ensure_dir(str(Path(out_path).parent))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_history(history: Dict[str, List[float]], out_path: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history.get("train_loss", [])) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(epochs, history.get("train_loss", []), label="train")
    axes[0].plot(epochs, history.get("val_loss", []), label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()

    axes[1].plot(epochs, [a * 100 for a in history.get("val_acc", [])], color="green")
    axes[1].axhline(80, ls="--", color="grey", alpha=0.7)
    axes[1].set_title("Validation accuracy (%)"); axes[1].set_xlabel("epoch")

    axes[2].plot(epochs, [e * 100 for e in history.get("val_eer", [])], color="crimson")
    axes[2].axhline(12, ls="--", color="grey", alpha=0.7)
    axes[2].set_title("Validation EER (%)"); axes[2].set_xlabel("epoch")

    fig.tight_layout()
    ensure_dir(str(Path(out_path).parent))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

"""
Training entry point.

Usage
-----
    python -m src.train                         # uses defaults / config.yaml
    python -m src.train --config config.yaml --epochs 40 --batch-size 64
    python -m src.train --train-dir data/for-norm/training \
                        --val-dir   data/for-norm/validation

What it does
------------
* Scans the training directory (and a validation directory if given, otherwise
  carves a stratified validation split off the training data).
* Trains :class:`~src.model.SpecNetCNN` on log-mel features with Adam +
  ``ReduceLROnPlateau`` and optional inverse-frequency class weights.
* After every epoch it measures validation loss, accuracy and **EER**, and
  keeps the checkpoint with the **lowest validation EER** (ties broken by the
  higher accuracy) — EER is the metric the problem statement is most strict on.
* Stops early when val-EER has not improved for ``early_stopping_patience``
  epochs.
* Saves a **self-contained** checkpoint to ``models/best_model.pt`` bundling the
  weights *and* the exact audio/feature/model settings *and* the label map, so
  evaluation and inference never depend on an external config file.
* Writes the training history to ``reports/history.json`` and a learning-curve
  figure to ``reports/figures/training_history.png``.

Mixed-precision (AMP) is enabled automatically only when a CUDA GPU is present.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from .config import Config
from .dataset import (
    build_dataset_class,
    class_distribution,
    class_weights,
    scan_dataset,
    stratified_split,
)
from .metrics import CLASS_NAMES, compute_eer
from .model import build_model
from .utils import (
    count_parameters,
    ensure_dir,
    get_device,
    plot_history,
    save_json,
    seed_everything,
)


# --------------------------------------------------------------------------- #
# One epoch of training / one pass of evaluation
# --------------------------------------------------------------------------- #
def _train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_amp):
    import torch

    model.train()
    running_loss, n_correct, n_total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * x.size(0)
        n_correct += (logits.argmax(1) == y).sum().item()
        n_total += x.size(0)

    return running_loss / max(n_total, 1), n_correct / max(n_total, 1)


@np.errstate(all="ignore")
def _evaluate(model, loader, criterion, device):
    """Return (loss, accuracy, eer) plus raw (y_true, p_fake) for the split."""
    import torch

    model.eval()
    running_loss, n_total = 0.0, 0
    all_p_fake: List[float] = []
    all_true: List[int] = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            running_loss += loss.item() * x.size(0)
            n_total += x.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1]  # P(deepfake)
            all_p_fake.extend(probs.detach().cpu().numpy().tolist())
            all_true.extend(y.detach().cpu().numpy().tolist())

    y_true = np.asarray(all_true, dtype=int)
    p_fake = np.asarray(all_p_fake, dtype=float)
    if n_total:
        y_pred = (p_fake >= 0.5).astype(int)
        acc = float((y_pred == y_true).mean())
        eer, _ = compute_eer(y_true, p_fake)
    else:
        acc, eer = 0.0, 1.0
    return running_loss / max(n_total, 1), acc, float(eer), y_true, p_fake


# --------------------------------------------------------------------------- #
# Main training routine
# --------------------------------------------------------------------------- #
def train(cfg: Config, train_samples=None, val_samples=None) -> Dict:
    """Train the detector.

    ``train_samples`` / ``val_samples`` may be passed in (already-scanned
    ``[(path, label), …]`` lists) to train on a custom mix of data — e.g.
    several Fake-or-Real variants combined for diversity. When omitted they are
    scanned from ``cfg.paths.train_dir`` / ``val_dir`` as before.
    """
    import torch
    from torch.utils.data import DataLoader

    seed_everything(cfg.train.seed)
    device = get_device("auto")
    use_amp = bool(cfg.train.amp and device.type == "cuda")
    print(f"[train] device={device}  amp={use_amp}")

    # ---- data ---------------------------------------------------------- #
    if train_samples is None:
        train_samples = scan_dataset(cfg.paths.train_dir)
    if val_samples is None:
        if cfg.paths.val_dir and Path(cfg.paths.val_dir).exists():
            val_samples = scan_dataset(cfg.paths.val_dir)
        else:
            print(f"[train] no val_dir -> stratified {cfg.train.val_split:.0%} split of train")
            train_samples, val_samples = stratified_split(
                train_samples, cfg.train.val_split, seed=cfg.train.seed
            )

    print(f"[train] train samples: {dict(class_distribution(train_samples))} "
          f"(total {len(train_samples)})")
    print(f"[train] val   samples: {dict(class_distribution(val_samples))} "
          f"(total {len(val_samples)})")

    AudioSpoofDataset = build_dataset_class()
    train_ds = AudioSpoofDataset(
        train_samples, cfg.audio, cfg.features, cfg.specaugment, cfg.waveaugment,
        train=True,
    )
    # No augmentation on the validation split — it must measure clean performance.
    val_ds = AudioSpoofDataset(
        val_samples, cfg.audio, cfg.features, None, None, train=False
    )
    wave_on = cfg.waveaugment.enabled
    print(f"[train] augmentation: waveform={'on' if wave_on else 'off'}, "
          f"specaugment={'on' if cfg.specaugment.enabled else 'off'}")

    pin = device.type == "cuda"
    persist = cfg.train.num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.train.num_workers, pin_memory=pin, drop_last=False,
        persistent_workers=persist,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.train.num_workers, pin_memory=pin,
        persistent_workers=persist,
    )

    # ---- model / optim ------------------------------------------------- #
    model = build_model(cfg.model).to(device)
    print(f"[train] model={cfg.model.name}  params={count_parameters(model):,}")

    if cfg.train.use_class_weights:
        w = class_weights(train_samples, cfg.model.num_classes)
        weight_tensor = torch.tensor(w, dtype=torch.float32, device=device)
        print(f"[train] class weights: {w.tolist()}")
    else:
        weight_tensor = None

    criterion = torch.nn.CrossEntropyLoss(
        weight=weight_tensor, label_smoothing=cfg.train.label_smoothing
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.train.scheduler_factor,
        patience=cfg.train.scheduler_patience,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- loop ---------------------------------------------------------- #
    history: Dict[str, List[float]] = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_eer": []
    }
    best_eer, best_acc, best_state = float("inf"), -1.0, None
    best_epoch, epochs_no_improve = 0, 0

    for epoch in range(1, cfg.train.epochs + 1):
        tr_loss, tr_acc = _train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, use_amp
        )
        va_loss, va_acc, va_eer, _, _ = _evaluate(model, val_loader, criterion, device)
        scheduler.step(va_loss)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["val_eer"].append(va_eer)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[epoch {epoch:02d}/{cfg.train.epochs}] "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.3f} | "
              f"val_loss={va_loss:.4f} acc={va_acc:.3f} EER={va_eer*100:.2f}% | lr={lr_now:.2e}")

        improved = (va_eer < best_eer - 1e-6) or (
            abs(va_eer - best_eer) <= 1e-6 and va_acc > best_acc
        )
        if improved:
            best_eer, best_acc, best_epoch = va_eer, va_acc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            print(f"          ^ new best (val EER={best_eer*100:.2f}%, acc={best_acc:.3f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.train.early_stopping_patience:
                print(f"[train] early stopping at epoch {epoch} "
                      f"(no val-EER improvement for {epochs_no_improve} epochs)")
                break

    if best_state is None:  # degenerate (e.g. 0 epochs) — keep current weights
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = len(history["val_eer"])

    # ---- save checkpoint (self-contained) ------------------------------ #
    ensure_dir(cfg.paths.model_dir)
    checkpoint = {
        "format_version": 1,
        "model_state_dict": best_state,
        "model_config": asdict(cfg.model),
        "audio_config": asdict(cfg.audio),
        "feature_config": asdict(cfg.features),
        "class_names": CLASS_NAMES,
        "label_map": {0: CLASS_NAMES[0], 1: CLASS_NAMES[1]},
        "best_epoch": best_epoch,
        "val_eer": best_eer,
        "val_accuracy": best_acc,
        "history": history,
    }
    torch.save(checkpoint, cfg.paths.model_path)
    print(f"[train] saved best checkpoint -> {cfg.paths.model_path} "
          f"(epoch {best_epoch}, val EER={best_eer*100:.2f}%, val acc={best_acc:.3f})")

    # ---- artefacts ----------------------------------------------------- #
    ensure_dir(cfg.paths.reports_dir)
    save_json(history, str(Path(cfg.paths.reports_dir) / "history.json"))
    try:
        fig_path = str(Path(cfg.paths.figures_dir) / "training_history.png")
        plot_history(history, fig_path)
        print(f"[train] learning curves -> {fig_path}")
    except Exception as e:  # plotting must never break a finished run
        print(f"[train] (warning) could not plot history: {e}")

    return checkpoint


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the deepfake-audio detector.")
    p.add_argument("--config", default="config.yaml",
                   help="YAML config to overlay on defaults (optional).")
    p.add_argument("--train-dir", default=None, help="Override paths.train_dir.")
    p.add_argument("--val-dir", default=None,
                   help="Override paths.val_dir (omit to auto-split train).")
    p.add_argument("--model-dir", default=None, help="Override paths.model_dir.")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-class-weights", action="store_true",
                   help="Disable inverse-frequency class weighting.")
    return p


def main(argv=None) -> None:
    args = build_argparser().parse_args(argv)
    cfg = Config.load(args.config)

    if args.train_dir is not None:
        cfg.paths.train_dir = args.train_dir
    if args.val_dir is not None:
        cfg.paths.val_dir = args.val_dir
    if args.model_dir is not None:
        cfg.paths.model_dir = args.model_dir
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.num_workers is not None:
        cfg.train.num_workers = args.num_workers
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.no_class_weights:
        cfg.train.use_class_weights = False

    train(cfg)


if __name__ == "__main__":
    main()

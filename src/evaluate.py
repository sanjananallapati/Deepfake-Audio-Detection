"""
Evaluate a trained checkpoint on a labelled test set.

Usage
-----
    python -m src.evaluate
    python -m src.evaluate --test-dir data/for-norm/testing --model models/best_model.pt

It scans the test directory, runs batched inference, computes the full metric
suite required by the problem statement (accuracy, EER, F1, per-class accuracy,
confusion matrix), prints a readable report, saves figures (confusion matrix +
ROC-with-EER) under ``reports/figures/``, writes machine-readable
``reports/metrics.json``, and (re)generates the human-readable
``reports/performance_report.md`` with the **real numbers** and an explicit
PASS / FAIL against the §5 verification thresholds.
"""
from __future__ import annotations

import argparse
import datetime as _dt
from pathlib import Path
from typing import Tuple

import numpy as np

from .config import Config
from .dataset import build_dataset_class, class_distribution, scan_dataset
from .metrics import CLASS_NAMES, EvalReport, evaluate_predictions, format_report
from .predict import ModelBundle, load_bundle
from .utils import (
    ensure_dir,
    plot_confusion_matrix,
    plot_roc_with_eer,
    save_json,
)


def _infer_over_dir(bundle: ModelBundle, test_dir: str, batch_size: int,
                    num_workers: int) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Run the model over every labelled file under ``test_dir``."""
    import torch
    from torch.utils.data import DataLoader

    samples = scan_dataset(test_dir)
    dist = dict(class_distribution(samples))
    print(f"[eval] test samples: {dist} (total {len(samples)})")

    AudioSpoofDataset = build_dataset_class()
    ds = AudioSpoofDataset(
        samples, bundle.audio_cfg, bundle.feat_cfg, None, None, train=False
    )
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(bundle.device.type == "cuda"),
    )

    all_p, all_y = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(bundle.device, non_blocking=True)
            probs = torch.softmax(bundle.model(x), dim=1)[:, 1]
            all_p.extend(probs.detach().cpu().numpy().tolist())
            all_y.extend(y.numpy().tolist())

    return np.asarray(all_y, dtype=int), np.asarray(all_p, dtype=float), dist


def _write_markdown_report(report: EvalReport, out_path: str, *, model_path: str,
                           test_dir: str, distribution: dict, meta: dict) -> str:
    r = report
    cm = r.confusion_matrix
    passed = r.passes_verification
    status = "✅ PASS" if passed else "❌ FAIL"
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def chk(ok: bool) -> str:
        return "✅" if ok else "❌"

    t = r.thresholds
    primary_rows = [
        ("Overall Accuracy", f"{r.accuracy*100:.2f}%", "≥ 80%",
         chk(r.accuracy >= t["accuracy_min"])),
        ("Equal Error Rate (EER)", f"{r.eer*100:.2f}%", "≤ 12%",
         chk(r.eer <= t["eer_max"])),
    ]
    secondary_rows = [
        ("F1 Score (macro)", f"{r.f1_macro*100:.2f}%", "≥ 80%",
         chk(r.f1_macro >= t["f1_min"])),
        ("Per-Class Accuracy — Genuine", f"{r.per_class_accuracy['Genuine']*100:.2f}%",
         "≥ 75%", chk(r.per_class_accuracy["Genuine"] >= t["per_class_acc_min"])),
        ("Per-Class Accuracy — Deepfake", f"{r.per_class_accuracy['Deepfake']*100:.2f}%",
         "≥ 75%", chk(r.per_class_accuracy["Deepfake"] >= t["per_class_acc_min"])),
    ]

    lines = []
    lines.append("# Performance Report — Deepfake Audio Detection\n")
    lines.append(f"*Generated automatically by `src/evaluate.py` on {now}.*\n")
    lines.append(f"- **Model checkpoint:** `{model_path}`")
    if meta.get("best_epoch") is not None:
        val_eer_pct = (meta.get("val_eer") or 0.0) * 100
        lines.append(f"- **Selected at epoch:** {meta['best_epoch']} "
                     f"(validation EER {val_eer_pct:.2f}%)")
    lines.append(f"- **Test directory:** `{test_dir}`")
    lines.append(f"- **Test samples:** {r.n_samples}  "
                 f"(Genuine: {r.support['Genuine']}, Deepfake: {r.support['Deepfake']})\n")

    lines.append(f"## Verification result: {status}\n")
    lines.append("A submission is valid only if **both** primary thresholds are met "
                 "(§5). The secondary thresholds (§4) are also checked below.\n")

    lines.append("### Primary metrics (§5)\n")
    lines.append("| Metric | Value | Required | Status |")
    lines.append("|---|---|---|---|")
    for name, val, req, ok in primary_rows:
        lines.append(f"| {name} | {val} | {req} | {ok} |")
    lines.append("")

    lines.append("### Secondary metrics (§4)\n")
    lines.append("| Metric | Value | Required | Status |")
    lines.append("|---|---|---|---|")
    for name, val, req, ok in secondary_rows:
        lines.append(f"| {name} | {val} | {req} | {ok} |")
    lines.append("")

    lines.append("### Confusion matrix\n")
    lines.append("Rows = true class, columns = predicted class.\n")
    lines.append("| | Pred: Genuine | Pred: Deepfake |")
    lines.append("|---|---|---|")
    lines.append(f"| **True: Genuine** | {cm[0][0]} | {cm[0][1]} |")
    lines.append(f"| **True: Deepfake** | {cm[1][0]} | {cm[1][1]} |")
    lines.append("")

    lines.append("### Additional metrics\n")
    lines.append(f"- F1 (Genuine): {r.f1_genuine*100:.2f}%")
    lines.append(f"- F1 (Deepfake): {r.f1_deepfake*100:.2f}%")
    lines.append(f"- Precision (macro): {r.precision_macro*100:.2f}%")
    lines.append(f"- Recall (macro): {r.recall_macro*100:.2f}%")
    lines.append(f"- EER operating threshold: {r.eer_threshold:.3f}")
    lines.append(f"- FAR at EER threshold: {r.far_at_eer*100:.2f}%")
    lines.append(f"- FRR at EER threshold: {r.frr_at_eer*100:.2f}%\n")

    lines.append("### Figures\n")
    lines.append("![Confusion matrix](figures/confusion_matrix.png)\n")
    lines.append("![ROC curve with EER](figures/roc_curve.png)\n")

    lines.append("---\n")
    lines.append("Label convention: **0 = Genuine (Human)**, **1 = Deepfake "
                 "(AI-Generated)**. The detection score is the model's deepfake "
                 "probability; EER is computed with *deepfake* as the positive class.")

    text = "\n".join(lines) + "\n"
    ensure_dir(str(Path(out_path).parent))
    Path(out_path).write_text(text, encoding="utf-8")
    return out_path


def evaluate(cfg: Config, model_path: str) -> EvalReport:
    bundle = load_bundle(model_path)
    y_true, p_fake, dist = _infer_over_dir(
        bundle, cfg.paths.test_dir, cfg.train.batch_size, cfg.train.num_workers
    )

    report = evaluate_predictions(y_true, p_fake, threshold=0.5)
    print("\n" + format_report(report) + "\n")

    # Figures
    ensure_dir(cfg.paths.figures_dir)
    cm_path = str(Path(cfg.paths.figures_dir) / "confusion_matrix.png")
    roc_path = str(Path(cfg.paths.figures_dir) / "roc_curve.png")
    try:
        plot_confusion_matrix(report.confusion_matrix, CLASS_NAMES, cm_path,
                              title="Confusion Matrix — Test Set")
        plot_roc_with_eer(y_true, p_fake, report.eer, roc_path,
                          title="ROC Curve — Test Set")
        print(f"[eval] figures -> {cm_path}, {roc_path}")
    except Exception as e:
        print(f"[eval] (warning) could not render figures: {e}")

    # Machine-readable + human-readable reports
    ensure_dir(cfg.paths.reports_dir)
    metrics_json = str(Path(cfg.paths.reports_dir) / "metrics.json")
    save_json(report.to_dict(), metrics_json)

    md_path = str(Path(cfg.paths.reports_dir) / "performance_report.md")
    _write_markdown_report(
        report, md_path, model_path=model_path, test_dir=cfg.paths.test_dir,
        distribution=dist, meta=bundle.meta,
    )
    print(f"[eval] wrote {metrics_json} and {md_path}")

    verdict = "PASS ✅" if report.passes_verification else "FAIL ❌"
    print(f"\n[eval] VERIFICATION: {verdict}  "
          f"(acc={report.accuracy*100:.2f}%, EER={report.eer*100:.2f}%, "
          f"F1={report.f1_macro*100:.2f}%)")
    return report


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Evaluate the detector on a test set.")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--model", default=None, help="Checkpoint path (default from config).")
    p.add_argument("--test-dir", default=None, help="Override paths.test_dir.")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    args = p.parse_args(argv)

    cfg = Config.load(args.config)
    if args.test_dir is not None:
        cfg.paths.test_dir = args.test_dir
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.train.num_workers = args.num_workers
    model_path = args.model or cfg.paths.model_path

    evaluate(cfg, model_path)


if __name__ == "__main__":
    main()

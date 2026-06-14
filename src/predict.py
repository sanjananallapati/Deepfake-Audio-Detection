"""
Inference on new audio — the problem statement's "test new audio" script.

Usage
-----
    python -m src.predict path/to/clip.wav
    python -m src.predict a.wav b.flac c.mp3 --model models/best_model.pt --json

It loads the **self-contained** checkpoint written by :mod:`src.train` (weights
+ the exact audio/feature/model settings + label map), rebuilds the network,
applies the identical preprocessing used in training, and prints a verdict and
confidence for each file. The reusable :func:`predict_file` / :func:`load_bundle`
helpers are also imported by :mod:`src.evaluate` and the Streamlit app, so the
whole project shares one inference code path.

Output convention: label 0 = Genuine (Human), 1 = Deepfake (AI-Generated);
``p_fake`` is the model's deepfake probability and ``confidence`` is the
probability of the predicted class.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from .config import AudioConfig, FeatureConfig, ModelConfig
from .features import audio_path_to_feature
from .metrics import CLASS_NAMES
from .model import build_model


@dataclass
class ModelBundle:
    """Everything needed to run inference, loaded from a checkpoint."""
    model: object                 # torch.nn.Module in eval mode
    audio_cfg: AudioConfig
    feat_cfg: FeatureConfig
    class_names: List[str]
    device: object
    meta: Dict


def load_bundle(model_path: str = "models/best_model.pt", device=None) -> ModelBundle:
    """Load a checkpoint and reconstruct a ready-to-use model + configs."""
    import torch
    from .utils import get_device

    ckpt_path = Path(model_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path.resolve()}\n"
            "Train one first:  python -m src.train"
        )

    device = device or get_device("auto")
    ckpt = torch.load(str(ckpt_path), map_location=device)

    model_cfg = ModelConfig(**ckpt["model_config"])
    audio_cfg = AudioConfig(**ckpt["audio_config"])
    feat_cfg = FeatureConfig(**ckpt["feature_config"])
    class_names = ckpt.get("class_names", CLASS_NAMES)

    model = build_model(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    meta = {
        "best_epoch": ckpt.get("best_epoch"),
        "val_eer": ckpt.get("val_eer"),
        "val_accuracy": ckpt.get("val_accuracy"),
    }
    return ModelBundle(model, audio_cfg, feat_cfg, class_names, device, meta)


def predict_array(bundle: ModelBundle, feat: np.ndarray) -> Dict:
    """Run the model on a single pre-computed ``(n_mels, n_frames)`` feature."""
    import torch

    x = torch.from_numpy(feat).unsqueeze(0).unsqueeze(0).float().to(bundle.device)
    with torch.no_grad():
        probs = torch.softmax(bundle.model(x), dim=1)[0].detach().cpu().numpy()

    p_genuine = float(probs[0])
    p_fake = float(probs[1])
    pred = int(p_fake >= 0.5)
    return {
        "label": pred,
        "label_name": bundle.class_names[pred],
        "p_genuine": p_genuine,
        "p_deepfake": p_fake,
        "confidence": float(max(p_genuine, p_fake)),
    }


def predict_file(bundle: ModelBundle, audio_path: str) -> Dict:
    """Full single-file path -> prediction dict (includes the file path)."""
    feat = audio_path_to_feature(str(audio_path), bundle.audio_cfg, bundle.feat_cfg)
    out = predict_array(bundle, feat)
    out["file"] = str(audio_path)
    return out


def _format_human(res: Dict) -> str:
    verdict = res["label_name"].upper()
    conf = res["confidence"] * 100
    return (f"{Path(res['file']).name:<32}  ->  {verdict:<9}  "
            f"(confidence {conf:5.1f}%  |  p_deepfake={res['p_deepfake']:.3f})")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Classify audio as Genuine or Deepfake.")
    p.add_argument("audio", nargs="+", help="One or more audio files.")
    p.add_argument("--model", default="models/best_model.pt", help="Checkpoint path.")
    p.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    args = p.parse_args(argv)

    bundle = load_bundle(args.model)
    if bundle.meta.get("val_eer") is not None:
        print(f"[predict] loaded {args.model} "
              f"(val EER={bundle.meta['val_eer']*100:.2f}%, "
              f"val acc={bundle.meta['val_accuracy']:.3f})\n")

    results = []
    for path in args.audio:
        try:
            results.append(predict_file(bundle, path))
        except Exception as e:
            results.append({"file": str(path), "error": str(e)})

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            if "error" in r:
                print(f"{Path(r['file']).name:<32}  ->  ERROR: {r['error']}")
            else:
                print(_format_human(r))


if __name__ == "__main__":
    main()

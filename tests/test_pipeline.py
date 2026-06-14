"""
End-to-end smoke test for the deepfake-audio-detection pipeline.

Run with pytest:
    pytest -q

or directly:
    python tests/test_pipeline.py

The torch-free parts (feature extraction, dataset scanning, metric/EER math)
run anywhere. The model / training / inference parts require ``torch``; if it is
not installed those tests are skipped (pytest) or noted (direct run). All tests
use tiny synthetic audio generated on the fly — no dataset download needed.
"""
from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import pytest  # noqa: E402
except Exception:  # noqa: BLE001 — allow standalone `python tests/test_pipeline.py`
    class _Mark:
        @staticmethod
        def skipif(cond, reason=""):
            def deco(fn):
                fn.__skip__ = bool(cond)
                return fn
            return deco

    class _PytestShim:
        mark = _Mark()
    pytest = _PytestShim()  # type: ignore

from src.config import AudioConfig, FeatureConfig, ModelConfig  # noqa: E402
from src.dataset import class_distribution, scan_dataset, stratified_split  # noqa: E402
from src.features import audio_path_to_feature, expected_num_frames  # noqa: E402
from src.metrics import compute_eer, evaluate_predictions  # noqa: E402

# torch is optional: only the model/training/inference tests need it. The
# feature/scan/metric tests below always run.
try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except Exception:  # noqa: BLE001
    torch = None
    HAVE_TORCH = False


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _write_sine_wav(path: Path, freq: float, sr: int = 16000, dur: float = 1.0):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    y = 0.3 * np.sin(2 * np.pi * freq * t)
    pcm = (y * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def _make_dataset(root: Path, n_per_class: int = 4):
    real = root / "real"
    fake = root / "fake"
    real.mkdir(parents=True, exist_ok=True)
    fake.mkdir(parents=True, exist_ok=True)
    for i in range(n_per_class):
        _write_sine_wav(real / f"r{i}.wav", freq=220 + 10 * i)
        _write_sine_wav(fake / f"f{i}.wav", freq=440 + 10 * i)
    return root


# --------------------------------------------------------------------------- #
# Torch-free tests
# --------------------------------------------------------------------------- #
def test_scan_and_labels(tmp_path):
    _make_dataset(tmp_path)
    samples = scan_dataset(str(tmp_path))
    dist = class_distribution(samples)
    assert dist[0] == 4 and dist[1] == 4
    # stratified split preserves both classes
    tr, va = stratified_split(samples, 0.25, seed=0)
    assert len(tr) + len(va) == 8
    assert {l for _, l in va} <= {0, 1}


def test_feature_shape(tmp_path):
    _make_dataset(tmp_path, n_per_class=1)
    a, f = AudioConfig(), FeatureConfig()
    fp = next((tmp_path / "real").glob("*.wav"))
    feat = audio_path_to_feature(str(fp), a, f)
    assert feat.shape == (f.n_mels, expected_num_frames(a, f))
    assert np.isfinite(feat).all()


def test_eer_known_value():
    # Perfectly separable -> EER 0
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    eer, _ = compute_eer(y, p)
    assert eer < 1e-6
    rep = evaluate_predictions(y, p)
    assert rep.accuracy == 1.0
    assert rep.passes_verification


# --------------------------------------------------------------------------- #
# Torch-dependent tests (skipped automatically if torch is missing)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not HAVE_TORCH, reason="torch not installed")
def test_model_forward_shape():
    from src.model import build_model
    a, f = AudioConfig(), FeatureConfig()
    model = build_model(ModelConfig())
    x = torch.randn(3, 1, f.n_mels, expected_num_frames(a, f))
    out = model(x)
    assert out.shape == (3, 2)


@pytest.mark.skipif(not HAVE_TORCH, reason="torch not installed")
def test_one_training_step_changes_loss():
    from src.model import build_model
    a, f = AudioConfig(), FeatureConfig()
    model = build_model(ModelConfig())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = torch.nn.CrossEntropyLoss()
    x = torch.randn(8, 1, f.n_mels, expected_num_frames(a, f))
    y = torch.tensor([0, 1] * 4)
    model.train()
    l0 = crit(model(x), y)
    opt.zero_grad(); l0.backward(); opt.step()
    l1 = crit(model(x), y)
    assert torch.isfinite(l0) and torch.isfinite(l1)


@pytest.mark.skipif(not HAVE_TORCH, reason="torch not installed")
def test_checkpoint_roundtrip_and_predict(tmp_path):
    """Save a checkpoint in the train.py format, then load + predict a file."""
    from dataclasses import asdict
    from src.model import build_model
    from src.predict import load_bundle, predict_file
    from src.metrics import CLASS_NAMES

    a, f, m = AudioConfig(), FeatureConfig(), ModelConfig()
    model = build_model(m)
    ckpt = {
        "format_version": 1,
        "model_state_dict": model.state_dict(),
        "model_config": asdict(m),
        "audio_config": asdict(a),
        "feature_config": asdict(f),
        "class_names": CLASS_NAMES,
        "label_map": {0: CLASS_NAMES[0], 1: CLASS_NAMES[1]},
        "best_epoch": 1, "val_eer": 0.1, "val_accuracy": 0.9,
    }
    model_path = tmp_path / "best_model.pt"
    torch.save(ckpt, str(model_path))

    _make_dataset(tmp_path, n_per_class=1)
    wav = next((tmp_path / "fake").glob("*.wav"))

    bundle = load_bundle(str(model_path))
    res = predict_file(bundle, str(wav))
    assert res["label"] in (0, 1)
    assert 0.0 <= res["confidence"] <= 1.0
    assert abs(res["p_genuine"] + res["p_deepfake"] - 1.0) < 1e-5


# --------------------------------------------------------------------------- #
# Direct-run entry point
# --------------------------------------------------------------------------- #
def _run_directly():
    import tempfile
    d = Path(tempfile.mkdtemp())
    test_scan_and_labels(d / "a")
    test_feature_shape(d / "b")
    test_eer_known_value()
    print("torch-free tests: PASS")
    if HAVE_TORCH:
        test_model_forward_shape()
        test_one_training_step_changes_loss()
        test_checkpoint_roundtrip_and_predict(d / "c")
        print("torch tests: PASS")
    else:
        print("torch tests: SKIPPED (torch not installed)")
    print("OK")


if __name__ == "__main__":
    _run_directly()

"""
Dataset discovery and the PyTorch ``Dataset`` used for training/evaluation.

Folder scanning is deliberately tolerant of the many layouts the Fake-or-Real
and ASVspoof releases ship in. Point it at a directory that contains class
sub-folders (``real``/``fake``, ``genuine``/``spoof``, …, case-insensitive) and
it returns a list of ``(filepath, label)`` pairs, recursing into each class
folder. Labels follow the project convention: 0 = genuine, 1 = deepfake.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .config import (
    AUDIO_EXTENSIONS,
    DEEPFAKE_ALIASES,
    GENUINE_ALIASES,
    AudioConfig,
    FeatureConfig,
    SpecAugmentConfig,
    WaveAugmentConfig,
)
from .features import (
    audio_path_to_feature,
    load_waveform,
    spec_augment,
    wave_augment,
    waveform_to_logmel,
)

Sample = Tuple[str, int]


def _label_from_name(name: str) -> Optional[int]:
    """Map a folder name to a label (0 genuine / 1 deepfake), or None.

    Matching is token-based (split on non-alphanumerics) so that a folder like
    ``training`` is NOT mis-read as deepfake just because it contains the
    letters "ai". A name carrying signals for *both* classes is treated as
    ambiguous and returns None, letting the scanner recurse deeper.
    """
    raw = name.strip().lower()
    norm = re.sub(r"[^a-z0-9]+", "", raw)                 # "bona-fide" -> "bonafide"
    tokens = {t for t in re.split(r"[^a-z0-9]+", raw) if t}

    g_norm = {re.sub(r"[^a-z0-9]+", "", a) for a in GENUINE_ALIASES}
    d_norm = {re.sub(r"[^a-z0-9]+", "", a) for a in DEEPFAKE_ALIASES}
    g_tokens = {a for a in GENUINE_ALIASES if a.isalpha()}
    d_tokens = {a for a in DEEPFAKE_ALIASES if a.isalpha()}

    is_genuine = (norm in g_norm) or bool(tokens & g_tokens)
    is_deepfake = (norm in d_norm) or bool(tokens & d_tokens)

    if is_genuine and not is_deepfake:
        return 0
    if is_deepfake and not is_genuine:
        return 1
    return None  # neither, or ambiguous


def _audio_files(folder: Path) -> List[Path]:
    files: List[Path] = []
    for ext in AUDIO_EXTENSIONS:
        files.extend(folder.rglob(f"*{ext}"))
        files.extend(folder.rglob(f"*{ext.upper()}"))
    # De-dup (case-insensitive globs can overlap on some filesystems) + sort.
    return sorted(set(files))


def scan_dataset(root: str) -> List[Sample]:
    """Return ``[(path, label), …]`` for every audio file under ``root``.

    Looks for immediate sub-directories whose names indicate a class. If none
    are found one level down, searches one level deeper (handles an extra
    wrapper directory). Files whose class cannot be inferred are skipped.
    """
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {root_path.resolve()}\n"
            "Edit config.yaml -> paths, or pass --train-dir/--test-dir."
        )

    samples: List[Sample] = []

    def collect(base: Path) -> int:
        found = 0
        for sub in sorted(p for p in base.iterdir() if p.is_dir()):
            label = _label_from_name(sub.name)
            if label is None:
                continue
            for fp in _audio_files(sub):
                samples.append((str(fp), label))
                found += 1
        return found

    n = collect(root_path)
    if n == 0:  # try one level deeper
        for sub in sorted(p for p in root_path.iterdir() if p.is_dir()):
            collect(sub)

    if not samples:
        raise RuntimeError(
            f"No labelled audio found under {root_path.resolve()}.\n"
            "Expected class sub-folders like real/ and fake/ "
            "(aliases: genuine/spoof/bonafide/…)."
        )
    return samples


def class_distribution(samples: List[Sample]) -> Counter:
    return Counter(lbl for _, lbl in samples)


def class_weights(samples: List[Sample], num_classes: int = 2) -> "np.ndarray":
    """Inverse-frequency weights for a (possibly imbalanced) training set."""
    counts = np.bincount([l for _, l in samples], minlength=num_classes).astype(float)
    counts = np.clip(counts, 1.0, None)
    w = counts.sum() / (num_classes * counts)
    return (w / w.mean()).astype(np.float32)


def stratified_split(
    samples: List[Sample], val_fraction: float, seed: int = 42
) -> Tuple[List[Sample], List[Sample]]:
    """Split ``samples`` into (train, val), preserving class proportions."""
    rng = np.random.default_rng(seed)
    by_label: dict = {}
    for s in samples:
        by_label.setdefault(s[1], []).append(s)
    train: List[Sample] = []
    val: List[Sample] = []
    for label, items in by_label.items():
        items = list(items)
        rng.shuffle(items)
        n_val = int(round(len(items) * val_fraction))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


# --------------------------------------------------------------------------- #
# PyTorch Dataset (imported lazily so the scanner above works without torch)
# --------------------------------------------------------------------------- #
def build_dataset_class():
    """Return the ``AudioSpoofDataset`` class (requires torch at call time)."""
    import torch
    from torch.utils.data import Dataset

    class AudioSpoofDataset(Dataset):
        """Lazily turns ``(path, label)`` pairs into ``(feature_tensor, label)``.

        Features are extracted on access, so memory stays flat regardless of
        dataset size. SpecAugment is applied only when ``train=True``.
        """

        def __init__(
            self,
            samples: List[Sample],
            audio_cfg: AudioConfig,
            feat_cfg: FeatureConfig,
            spec_cfg: Optional[SpecAugmentConfig] = None,
            wave_cfg: Optional[WaveAugmentConfig] = None,
            train: bool = False,
        ):
            self.samples = samples
            self.audio_cfg = audio_cfg
            self.feat_cfg = feat_cfg
            self.spec_cfg = spec_cfg
            self.wave_cfg = wave_cfg
            self.train = train
            self._rng = np.random.default_rng()

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int):
            path, label = self.samples[idx]
            wave_aug_on = (
                self.train and self.wave_cfg is not None and self.wave_cfg.enabled
            )
            try:
                if wave_aug_on:
                    # Augment the raw waveform, then transform — so the model
                    # sees a fresh degraded version of the clip every epoch.
                    y = load_waveform(path, self.audio_cfg)
                    y = wave_augment(y, self.wave_cfg, self._rng)
                    feat = waveform_to_logmel(y, self.audio_cfg, self.feat_cfg)
                else:
                    feat = audio_path_to_feature(path, self.audio_cfg, self.feat_cfg)
            except Exception:
                # A corrupt/unreadable file should not crash a whole epoch.
                T = 1 + self.audio_cfg.num_samples // self.feat_cfg.hop_length
                feat = np.zeros((self.feat_cfg.n_mels, T), dtype=np.float32)

            if self.train and self.spec_cfg is not None and self.spec_cfg.enabled:
                feat = spec_augment(
                    feat,
                    self.spec_cfg.freq_mask_param,
                    self.spec_cfg.time_mask_param,
                    self.spec_cfg.n_freq_masks,
                    self.spec_cfg.n_time_masks,
                    self._rng,
                )

            x = torch.from_numpy(feat).unsqueeze(0)  # (1, n_mels, n_frames)
            return x, torch.tensor(label, dtype=torch.long)

    return AudioSpoofDataset

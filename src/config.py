"""
Central configuration for the deepfake-audio-detection pipeline.

Everything is a plain dataclass with sensible defaults, so the project runs
out of the box. ``config.yaml`` at the repo root can override any field; the
loader merges it on top of these defaults. The exact feature + model settings
used to train a model are stored *inside* the checkpoint, so inference never
depends on an external config being present.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

# Folder names (lower-cased) mapped onto labels when scanning a dataset.
GENUINE_ALIASES = ["genuine", "real", "human", "bonafide", "bona-fide", "bona_fide", "live"]
DEEPFAKE_ALIASES = ["deepfake", "fake", "spoof", "spoofed", "synthetic", "tts", "ai", "generated", "clone"]
AUDIO_EXTENSIONS = [".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".wma"]


@dataclass
class AudioConfig:
    sample_rate: int = 16_000        # Hz; all audio is resampled to this
    duration: float = 4.0            # seconds; clips are padded/truncated to this
    mono: bool = True

    @property
    def num_samples(self) -> int:
        return int(self.sample_rate * self.duration)


@dataclass
class FeatureConfig:
    feature_type: str = "logmel"     # "logmel" (only option implemented)
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    fmin: int = 20
    fmax: int = 8_000                # <= sample_rate / 2
    top_db: float = 80.0
    normalize: str = "instance"      # "instance" (per-clip mean/var) | "none"


@dataclass
class SpecAugmentConfig:
    enabled: bool = True             # train-time only; improves generalisation
    freq_mask_param: int = 16        # max width of a frequency mask (mel bins)
    time_mask_param: int = 24        # max width of a time mask (frames)
    n_freq_masks: int = 2
    n_time_masks: int = 2


@dataclass
class WaveAugmentConfig:
    """Waveform-level augmentation (train only) to close the train->test gap.

    Applied to the raw waveform *before* the log-mel transform, so it perturbs
    exactly the cues a model would otherwise overfit. This simulates the
    recording-condition / loudness / channel differences that make the
    Fake-or-Real *test* split so much harder than training, and is the main
    lever for improving test-set EER (which threshold tuning cannot fix).
    """
    enabled: bool = True
    noise_prob: float = 0.5                                   # add Gaussian noise w/ this prob.
    noise_snr_db: List[float] = field(default_factory=lambda: [5.0, 30.0])  # random SNR range (dB)
    gain_prob: float = 0.5                                    # random loudness change
    gain_db: float = 6.0                                      # +/- this many dB
    shift_prob: float = 0.5                                   # circular time shift
    shift_max_frac: float = 0.25                              # up to this fraction of the clip
    polarity_prob: float = 0.5                                # invert waveform polarity


@dataclass
class ModelConfig:
    name: str = "specnet_cnn"
    channels: List[int] = field(default_factory=lambda: [32, 64, 128, 128])
    fc_dim: int = 64
    dropout: float = 0.3
    num_classes: int = 2


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_split: float = 0.15          # used only if no explicit val_dir is given
    num_workers: int = 4             # parallel feature extraction (CPU-bound; raise on big CPUs)
    early_stopping_patience: int = 7  # epochs without val-EER improvement
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    use_class_weights: bool = True
    label_smoothing: float = 0.0
    seed: int = 42
    amp: bool = True                 # mixed precision (used only when CUDA present)


@dataclass
class PathConfig:
    # Point these at your local copy of the Fake-or-Real dataset.
    # Each directory is expected to contain class sub-folders such as
    # real/ and fake/ (case-insensitive; many aliases are accepted).
    train_dir: str = "data/for-norm/training"
    val_dir: Optional[str] = "data/for-norm/validation"   # set null to auto-split train
    test_dir: str = "data/for-norm/testing"
    model_dir: str = "models"
    model_name: str = "best_model.pt"
    reports_dir: str = "reports"
    figures_dir: str = "reports/figures"

    @property
    def model_path(self) -> str:
        return str(Path(self.model_dir) / self.model_name)


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    specaugment: SpecAugmentConfig = field(default_factory=SpecAugmentConfig)
    waveaugment: WaveAugmentConfig = field(default_factory=WaveAugmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    # ---------------- YAML I/O ---------------- #
    @staticmethod
    def _merge(base: dict, override: dict) -> dict:
        out = dict(base)
        for k, v in (override or {}).items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = Config._merge(out[k], v)
            else:
                out[k] = v
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        d = d or {}
        return cls(
            audio=AudioConfig(**d.get("audio", {})),
            features=FeatureConfig(**d.get("features", {})),
            specaugment=SpecAugmentConfig(**d.get("specaugment", {})),
            waveaugment=WaveAugmentConfig(**d.get("waveaugment", {})),
            model=ModelConfig(**d.get("model", {})),
            train=TrainConfig(**d.get("train", {})),
            paths=PathConfig(**d.get("paths", {})),
        )

    @classmethod
    def load(cls, path: Optional[str] = None) -> "Config":
        """Load defaults, then overlay ``path`` (YAML) if it exists."""
        defaults = cls().to_dict()
        if path and Path(path).exists():
            import yaml
            with open(path, "r") as f:
                user = yaml.safe_load(f) or {}
            merged = cls._merge(defaults, user)
            return cls.from_dict(merged)
        return cls.from_dict(defaults)

    def save(self, path: str) -> None:
        import yaml
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, default_flow_style=False)

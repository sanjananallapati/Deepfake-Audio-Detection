"""
Audio loading and feature extraction.

A single function — :func:`audio_path_to_feature` — turns any audio file on
disk into the exact tensor the model consumes. Training and inference both go
through it, which guarantees the features match. The feature/audio settings are
saved in the model checkpoint, so a served model always reproduces its
training-time preprocessing.

Output feature shape: ``(n_mels, n_frames)`` float32, where ``n_frames`` is
deterministic because every clip is padded/truncated to a fixed sample count.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .config import AudioConfig, FeatureConfig, WaveAugmentConfig


# --------------------------------------------------------------------------- #
# Audio loading
# --------------------------------------------------------------------------- #
def load_waveform(path: str, cfg: AudioConfig) -> np.ndarray:
    """Load an audio file as a mono float32 waveform, fixed to ``cfg.num_samples``.

    Shorter clips are tiled (looped) to fill the window — looping preserves
    voicing characteristics better than zero-padding for very short clips —
    then truncated; longer clips are centre-cropped.
    """
    import librosa

    y, _ = librosa.load(path, sr=cfg.sample_rate, mono=cfg.mono)
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        y = np.zeros(cfg.num_samples, dtype=np.float32)
    return fix_length(y, cfg.num_samples)


def fix_length(y: np.ndarray, num_samples: int) -> np.ndarray:
    """Force a 1-D waveform to exactly ``num_samples`` (loop-pad / centre-crop)."""
    n = y.shape[0]
    if n == num_samples:
        return y
    if n < num_samples:
        reps = int(np.ceil(num_samples / n))
        y = np.tile(y, reps)[:num_samples]
        return y.astype(np.float32)
    # n > num_samples -> centre crop
    start = (n - num_samples) // 2
    return y[start: start + num_samples].astype(np.float32)


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
def waveform_to_logmel(y: np.ndarray, a: AudioConfig, f: FeatureConfig) -> np.ndarray:
    """Convert a fixed-length waveform to a (normalised) log-mel spectrogram."""
    import librosa

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=a.sample_rate,
        n_fft=f.n_fft,
        hop_length=f.hop_length,
        win_length=f.win_length,
        n_mels=f.n_mels,
        fmin=f.fmin,
        fmax=min(f.fmax, a.sample_rate // 2),
        power=2.0,
    )
    logmel = librosa.power_to_db(mel, ref=np.max, top_db=f.top_db)
    logmel = logmel.astype(np.float32)

    if f.normalize == "instance":
        mu, sd = logmel.mean(), logmel.std()
        logmel = (logmel - mu) / (sd + 1e-6)
    return logmel  # shape (n_mels, n_frames)


def expected_num_frames(a: AudioConfig, f: FeatureConfig) -> int:
    """Deterministic time dimension for the configured settings (center=True)."""
    return 1 + a.num_samples // f.hop_length


def audio_path_to_feature(
    path: str, a: AudioConfig, f: FeatureConfig
) -> np.ndarray:
    """End-to-end: file path -> normalised log-mel feature ``(n_mels, n_frames)``."""
    y = load_waveform(path, a)
    return waveform_to_logmel(y, a, f)


# --------------------------------------------------------------------------- #
# Waveform augmentation (train-time, numpy implementation)
# --------------------------------------------------------------------------- #
def wave_augment(
    y: np.ndarray,
    cfg: WaveAugmentConfig,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Perturb a fixed-length waveform to discourage over-fitting clean audio.

    Each transform fires independently with its own probability, so the model
    sees a wide variety of degraded versions of every clip across epochs. All
    transforms are label-preserving (a noisy/quieter/shifted deepfake is still a
    deepfake) and length-preserving. Returns a new array; ``y`` is not mutated.
    """
    rng = rng or np.random.default_rng()
    y = y.astype(np.float32, copy=True)
    n = y.shape[0]

    # Circular time shift — removes any reliance on absolute onset position.
    if n > 1 and rng.random() < cfg.shift_prob:
        max_shift = int(cfg.shift_max_frac * n)
        if max_shift > 0:
            y = np.roll(y, int(rng.integers(-max_shift, max_shift + 1)))

    # Random gain (loudness) in dB — the FoR test split differs in level.
    if rng.random() < cfg.gain_prob:
        gain = 10.0 ** (float(rng.uniform(-cfg.gain_db, cfg.gain_db)) / 20.0)
        y = y * gain

    # Additive white noise at a random SNR — simulates channel/recording noise.
    if rng.random() < cfg.noise_prob:
        sig_power = float(np.mean(y ** 2)) + 1e-12
        snr_db = float(rng.uniform(cfg.noise_snr_db[0], cfg.noise_snr_db[1]))
        noise_power = sig_power / (10.0 ** (snr_db / 10.0))
        y = y + rng.normal(0.0, np.sqrt(noise_power), size=n).astype(np.float32)

    # Polarity inversion — cheap, label-preserving, breaks phase short-cuts.
    if rng.random() < cfg.polarity_prob:
        y = -y

    return np.clip(y, -1.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
# SpecAugment (train-time augmentation, numpy implementation)
# --------------------------------------------------------------------------- #
def spec_augment(
    feat: np.ndarray,
    freq_mask_param: int,
    time_mask_param: int,
    n_freq_masks: int,
    n_time_masks: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Randomly mask horizontal (frequency) and vertical (time) bands.

    Masked regions are set to the feature mean (≈0 after instance norm), which
    encourages the model to rely on distributed cues rather than narrow bands —
    a cheap, effective regulariser for spoof detection.
    """
    rng = rng or np.random.default_rng()
    feat = feat.copy()
    n_mels, n_frames = feat.shape
    fill = float(feat.mean())

    for _ in range(n_freq_masks):
        w = int(rng.integers(0, freq_mask_param + 1))
        if w > 0 and n_mels - w > 0:
            f0 = int(rng.integers(0, n_mels - w))
            feat[f0: f0 + w, :] = fill

    for _ in range(n_time_masks):
        w = int(rng.integers(0, time_mask_param + 1))
        if w > 0 and n_frames - w > 0:
            t0 = int(rng.integers(0, n_frames - w))
            feat[:, t0: t0 + w] = fill

    return feat

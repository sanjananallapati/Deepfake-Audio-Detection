"""
Model architecture: a compact VGG-style 2-D CNN over log-mel spectrograms.

Why this design
---------------
* Convolutions over the (mel x time) image capture the local spectro-temporal
  artefacts that distinguish vocoder/TTS output from natural speech (unnatural
  harmonics, phase/formant inconsistencies, over-smoothed spectra).
* A global ``AdaptiveAvgPool2d(1)`` head makes the classifier input depend only
  on the channel count — never on the input length — so the same weights accept
  clips of any duration at inference and there is no flatten-dimension to get
  wrong.
* Batch-norm + dropout keep it well-regularised and quick to train on CPU/GPU.

Returns 2 logits; ``softmax(logits)[:, 1]`` is the deepfake probability.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """(Conv-BN-ReLU) x2 -> MaxPool(2) -> Dropout2d."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SpecNetCNN(nn.Module):
    """Spectrogram CNN classifier for genuine-vs-deepfake speech."""

    def __init__(
        self,
        channels: List[int] = (32, 64, 128, 128),
        fc_dim: int = 64,
        dropout: float = 0.3,
        num_classes: int = 2,
        in_channels: int = 1,
    ):
        super().__init__()
        chans = list(channels)
        blocks = []
        prev = in_channels
        for c in chans:
            blocks.append(ConvBlock(prev, c, dropout))
            prev = c
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)          # -> (B, C, 1, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prev, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_mels, n_frames)
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)               # (B, num_classes) logits


def build_model(model_cfg) -> SpecNetCNN:
    """Construct a SpecNetCNN from a ModelConfig-like object."""
    return SpecNetCNN(
        channels=model_cfg.channels,
        fc_dim=model_cfg.fc_dim,
        dropout=model_cfg.dropout,
        num_classes=model_cfg.num_classes,
    )


@torch.no_grad()
def predict_proba(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return per-class probabilities for a batch of features."""
    model.eval()
    logits = model(x)
    return torch.softmax(logits, dim=1)

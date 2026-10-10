"""Acoustic features: 128-band log-mel of 48 kHz audio, n_fft 2048 / hop 512 (93.75 frames/s).

The same transform feeds the acoustic model (target), the vocoder (input) and the vocoder's mel
loss, so everything agrees by construction. Runs on any torch device.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


@dataclass
class MelConfig:
    sample_rate: int = 48000
    n_fft: int = 2048
    hop: int = 512
    win: int = 2048
    n_mels: int = 128
    f_min: float = 0.0
    f_max: float = 24000.0
    log_floor: float = 1e-5

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop

    def dict(self) -> dict:
        return asdict(self)


class LogMel(nn.Module):
    def __init__(self, cfg: MelConfig | None = None):
        super().__init__()
        import torchaudio  # training and evaluation only; kept out of inference imports

        self.cfg = cfg = cfg or MelConfig()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate, n_fft=cfg.n_fft, win_length=cfg.win, hop_length=cfg.hop, f_min=cfg.f_min,
            f_max=cfg.f_max, n_mels=cfg.n_mels, power=1.0, center=True, norm="slaney", mel_scale="slaney")

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav (B, T) float in [-1, 1] -> log-mel (B, n_mels, frames), frames = T // hop + 1."""
        return torch.log(torch.clamp(self.mel(wav.float()), min=self.cfg.log_floor))


# Global normalisation of the flow target: mean/std of the log-mel of teacher speech (dry run,
# 2026-10-07); stored in every checkpoint's config, so changing it later only affects new runs.
MEL_MEAN, MEL_STD = -5.9, 2.6

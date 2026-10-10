"""Vocoder: log-mel (93.75 Hz) -> 48 kHz waveform through an inverse-STFT head.

Generator in the style of Vocos (Siuzdak, 2023): a ConvNeXt stack at the mel frame rate predicts the
log-magnitude and phase of an n_fft=2048 / hop=512 STFT, and one iSTFT makes the waveform; no
upsampling convolutions, so it costs a few MFLOPs per frame at any output rate.
Discriminators (training only): HiFi-GAN's multi-period discriminator (Kong et al., 2020) and a
multi-resolution spectrogram discriminator (UnivNet, Jang et al., 2021; as used by Vocos).
Losses: least-squares GAN, feature matching, multi-resolution log-mel L1.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

from antalia_mini.audio import LogMel, MelConfig
from antalia_mini.modules import ConvNeXt1d


@dataclass
class VocoderConfig:
    n_mels: int = 128
    dim: int = 192
    hidden: int = 512
    layers: int = 8
    n_fft: int = 2048
    hop: int = 512
    sample_rate: int = 48000

    def dict(self) -> dict:
        return asdict(self)


class Vocoder(nn.Module):
    def __init__(self, c: VocoderConfig | None = None):
        super().__init__()
        self.c = c = c or VocoderConfig()
        self.inp = nn.Conv1d(c.n_mels, c.dim, 7, padding=3)
        self.norm0 = nn.LayerNorm(c.dim)
        self.blocks = nn.ModuleList([ConvNeXt1d(c.dim, c.hidden, kernel=7, layer_scale=1.0 / c.layers) for _ in range(c.layers)])
        self.norm1 = nn.LayerNorm(c.dim)
        self.head = nn.Linear(c.dim, c.n_fft + 2)
        self.register_buffer("window", torch.hann_window(c.n_fft), persistent=False)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel (B, n_mels, T) -> wav (B, T * hop)."""
        x = self.inp(mel)
        x = self.norm0(x.transpose(1, 2)).transpose(1, 2)
        for b in self.blocks:
            x = b(x)
        x = self.head(self.norm1(x.transpose(1, 2))).transpose(1, 2)
        mag, ph = x.chunk(2, 1)
        mag = torch.exp(mag).clamp(max=1e2)
        spec = torch.polar(mag.float(), ph.float())
        # "same" padding iSTFT as in Vocos: centre=True, frames T -> samples (T - 1) * hop
        wav = torch.istft(spec, self.c.n_fft, self.c.hop, self.c.n_fft, window=self.window, center=True)
        return F.pad(wav, (0, self.c.hop))


# ------------------------------------------------------------------------------------- discriminators
class PeriodDisc(nn.Module):
    def __init__(self, period: int, ch: int = 32):
        super().__init__()
        self.p = period
        chs = [1, ch, ch * 4, ch * 16, ch * 32]
        self.convs = nn.ModuleList([weight_norm(nn.Conv2d(chs[i], chs[i + 1], (5, 1), (3, 1), padding=(2, 0))) for i in range(4)])
        self.convs.append(weight_norm(nn.Conv2d(chs[-1], chs[-1], (5, 1), 1, padding=(2, 0))))
        self.post = weight_norm(nn.Conv2d(chs[-1], 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        b, t = x.shape
        if t % self.p:
            x = F.pad(x, (0, self.p - t % self.p), "reflect")
        x = x.view(b, 1, -1, self.p)
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.1)
            fmap.append(x)
        x = self.post(x)
        fmap.append(x)
        return x.flatten(1), fmap


class SpecDisc(nn.Module):
    def __init__(self, n_fft: int, ch: int = 32):
        super().__init__()
        self.n_fft = n_fft
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, ch, (7, 5), (2, 2), padding=(3, 2))),
            weight_norm(nn.Conv2d(ch, ch, (5, 3), (2, 1), padding=(2, 1))),
            weight_norm(nn.Conv2d(ch, ch, (5, 3), (2, 2), padding=(2, 1))),
            weight_norm(nn.Conv2d(ch, ch, 3, (2, 1), padding=1)),
            weight_norm(nn.Conv2d(ch, ch, 3, (2, 2), padding=1)),
        ])
        self.post = weight_norm(nn.Conv2d(ch, 1, 3, padding=1))

    def forward(self, x):
        s = torch.stft(x.float(), self.n_fft, self.n_fft // 4, window=self.window, return_complex=True).abs()
        x = torch.log(s.clamp(min=1e-5))[:, None]  # (B, 1, F, T)
        fmap = []
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.1)
            fmap.append(x)
        x = self.post(x)
        fmap.append(x)
        return x.flatten(1), fmap


class Discriminators(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11), ffts=(512, 1024, 2048, 4096)):
        super().__init__()
        self.discs = nn.ModuleList([PeriodDisc(p) for p in periods] + [SpecDisc(n) for n in ffts])

    def forward(self, x):
        return [d(x) for d in self.discs]


def d_loss(real_out, fake_out):
    loss = 0.0
    for (r, _), (f, _) in zip(real_out, fake_out):
        loss = loss + torch.mean((1 - r) ** 2) + torch.mean(f ** 2)
    return loss


def g_adv_fm_loss(real_out, fake_out):
    adv, fm = 0.0, 0.0
    for (r, rmap), (f, fmap) in zip(real_out, fake_out):
        adv = adv + torch.mean((1 - f) ** 2)
        for a, b in zip(rmap, fmap):
            fm = fm + torch.mean(torch.abs(a.detach() - b))
    return adv, fm


class MultiResMelLoss(nn.Module):
    def __init__(self, sample_rate: int = 48000):
        super().__init__()
        self.mels = nn.ModuleList([LogMel(MelConfig(sample_rate=sample_rate, n_fft=n, hop=n // 4, win=n, n_mels=m, f_max=sample_rate / 2))
                                   for n, m in ((512, 64), (1024, 96), (2048, 128), (4096, 128))])

    def forward(self, y_hat, y):
        return sum(F.l1_loss(m(y_hat), m(y)) for m in self.mels) / len(self.mels)

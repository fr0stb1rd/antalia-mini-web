"""Output processing after the vocoder: resampling, the default "less bass" EQ, make-up gain, peak limiter.

  48 kHz chunk --resample--> output rate --EQ--> x make-up gain --soft peak limiter--> float32

EQ (designed at the output rate, so 48/24/16/8 kHz all get the same response):
  1. 2nd-order high-pass, 90 Hz, Q 0.707 (RBJ cookbook)
  2. low shelf, 210 Hz, -5 dB, Q 0.707 (RBJ cookbook)
The biquad cascade runs as an exact FIR of its impulse response (truncated where it has decayed below
1e-14 of its energy), applied by FFT convolution with the previous input kept as state, so one stream's
chunks filtered one by one give the same samples as the whole utterance filtered at once, and nothing
beyond numpy is needed.

Make-up gain: one fixed value (EQ_MAKEUP_DB) measured once with scripts/measure_eq_gain.py on the card's
sample sentences so the EQ'd output has about the same integrated loudness (ITU-R BS.1770) as the raw
output; there is no per-utterance loudness normalisation. The limiter is memoryless (a soft knee from
-1 dBFS that never goes above -0.1 dBFS), so it is the same whether audio arrives in chunks or at once.
With the 1.0.0 weights the sample sentences peak below 0.81 after EQ and gain, so the limiter is a
safety net; it matters for louder inputs (an earlier checkpoint reached full scale on a few samples).
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

EQ_HPF_HZ, EQ_HPF_Q = 90.0, 0.707
EQ_SHELF_HZ, EQ_SHELF_DB, EQ_SHELF_Q = 210.0, -5.0, 0.707
# +0.54 dB: BS.1770 loudness lost by the EQ, mean over the 8 card sentences x 3 seeds (spread 0.41-0.62 dB;
# 0.54-0.55 dB at 24/16/8 kHz), measured 2026-10-08 on the 1.0.0 weights (final-ac-main-short bundle) with
# scripts/measure_eq_gain.py; peaks after EQ + gain stayed below 0.81, so the limiter did not engage
EQ_MAKEUP_DB = 0.54
LIMIT_THRESHOLD = 10 ** (-1.0 / 20)  # soft knee starts at -1 dBFS
LIMIT_CEILING = 10 ** (-0.1 / 20)  # and never goes above -0.1 dBFS

Biquad = tuple[tuple[float, float, float], tuple[float, float, float]]  # (b0, b1, b2), (1, a1, a2)


# ------------------------------------------------------------------------------------------ resampling
_KERNELS: dict[tuple[int, int], tuple[torch.Tensor, int, int, int]] = {}


def _sinc_kernel(orig: int, new: int, width: int = 32, rolloff: float = 0.945, beta: float = 14.769656459379492):
    """Kaiser-windowed sinc polyphase kernel (the construction torchaudio.functional.resample uses)."""
    g = math.gcd(orig, new)
    o, n = orig // g, new // g
    base = min(o, n) * rolloff
    w = math.ceil(width * o / base)
    idx = torch.arange(-w, w + o, dtype=torch.float64)[None, None] / o
    t = torch.arange(0, -n, -1, dtype=torch.float64)[:, None, None] / n + idx
    t = (t * base).clamp(-width, width)
    window = torch.i0(beta * torch.sqrt(1 - (t / width) ** 2)) / torch.i0(torch.tensor(beta, dtype=torch.float64))
    t = t * math.pi
    k = torch.where(t == 0, torch.ones_like(t), torch.sin(t) / t) * window * (base / o)
    return k.float(), w, o, n


def resample(audio: np.ndarray | torch.Tensor, orig: int, new: int) -> np.ndarray:
    """Band-limited resampling of a mono signal (no librosa/torchaudio needed)."""
    x = torch.as_tensor(np.asarray(audio, dtype=np.float32))
    if orig == new or x.numel() == 0:
        return x.numpy().astype(np.float32, copy=False)
    if (orig, new) not in _KERNELS:
        _KERNELS[(orig, new)] = _sinc_kernel(orig, new)
    kernel, w, o, n = _KERNELS[(orig, new)]
    T = x.shape[-1]
    y = F.conv1d(F.pad(x.view(1, 1, T), (w, w + o)), kernel, stride=o)  # (1, n, frames)
    y = y.transpose(1, 2).reshape(-1)[: math.ceil(n * T / o)]
    return y.numpy().astype(np.float32, copy=False)


# ------------------------------------------------------------------------------- RBJ cookbook biquads
def _norm(b, a) -> Biquad:
    return (b[0] / a[0], b[1] / a[0], b[2] / a[0]), (1.0, a[1] / a[0], a[2] / a[0])


def highpass(f0: float, q: float, fs: int) -> Biquad:
    w0 = 2 * math.pi * f0 / fs
    alpha, c = math.sin(w0) / (2 * q), math.cos(w0)
    return _norm(((1 + c) / 2, -(1 + c), (1 + c) / 2), (1 + alpha, -2 * c, 1 - alpha))


def lowshelf(f0: float, gain_db: float, q: float, fs: int) -> Biquad:
    A = 10 ** (gain_db / 40)
    w0 = 2 * math.pi * f0 / fs
    alpha, c = math.sin(w0) / (2 * q), math.cos(w0)
    s = 2 * math.sqrt(A) * alpha
    return _norm((A * ((A + 1) - (A - 1) * c + s), 2 * A * ((A - 1) - (A + 1) * c), A * ((A + 1) - (A - 1) * c - s)),
                 ((A + 1) + (A - 1) * c + s, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - s))


def highshelf(f0: float, gain_db: float, q: float, fs: int) -> Biquad:
    A = 10 ** (gain_db / 40)
    w0 = 2 * math.pi * f0 / fs
    alpha, c = math.sin(w0) / (2 * q), math.cos(w0)
    s = 2 * math.sqrt(A) * alpha
    return _norm((A * ((A + 1) + (A - 1) * c + s), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - s)),
                 ((A + 1) - (A - 1) * c + s, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - s))


def eq_sections(fs: int) -> list[Biquad]:
    return [highpass(EQ_HPF_HZ, EQ_HPF_Q, fs), lowshelf(EQ_SHELF_HZ, EQ_SHELF_DB, EQ_SHELF_Q, fs)]


def lfilter(sections: list[Biquad], x: np.ndarray) -> np.ndarray:
    """Direct-form biquad cascade, sample by sample (slow; reference for tests and impulse responses)."""
    y = np.asarray(x, dtype=np.float64).copy()
    for (b0, b1, b2), (_, a1, a2) in sections:
        out = np.empty_like(y)
        x1 = x2 = y1 = y2 = 0.0
        for n, v in enumerate(y):
            o = b0 * v + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            x2, x1, y2, y1 = x1, v, y1, o
            out[n] = o
        y = out
    return y


def impulse_response(sections: list[Biquad], fs: int, max_seconds: float = 1.0, tail: float = 1e-14) -> np.ndarray:
    """The cascade's impulse response, cut where the remaining energy is below `tail` of the total."""
    n = int(max_seconds * fs)
    imp = np.zeros(n)
    imp[0] = 1.0
    h = lfilter(sections, imp)
    e = np.cumsum((h ** 2)[::-1])[::-1]  # energy from n to the end
    keep = int(np.argmax(e < tail * e[0])) or n
    return h[:keep]


@lru_cache(maxsize=16)
def eq_fir(fs: int) -> np.ndarray:
    return impulse_response(eq_sections(fs), fs)


def response_db(h: np.ndarray, freqs, fs: int) -> np.ndarray:
    """Magnitude response of an FIR h at the given frequencies (dB)."""
    n = np.arange(len(h))
    return np.array([20 * np.log10(abs(np.sum(h * np.exp(-2j * np.pi * f * n / fs)))) for f in freqs])


class StreamFIR:
    """Causal FIR applied chunk by chunk with the input history as state (exact linear convolution,
    computed by overlap-save in fixed-size FFT blocks)."""

    def __init__(self, h: np.ndarray):
        self.h = np.asarray(h, dtype=np.float64)
        L = len(self.h)
        self.nfft = max(4096, 1 << (4 * L - 1).bit_length())
        self.block = self.nfft - L + 1
        self.H = np.fft.rfft(self.h, self.nfft)
        self.hist = np.zeros(L - 1)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        n, L = len(x), len(self.h)
        if n == 0:
            return np.zeros(0)
        ext = np.concatenate([self.hist, x])
        nb = -(-n // self.block)
        ext_p = np.pad(ext, (0, nb * self.block + L - 1 - len(ext)))
        frames = np.lib.stride_tricks.sliding_window_view(ext_p, self.nfft)[:: self.block][:nb]
        y = np.fft.irfft(np.fft.rfft(frames, axis=1) * self.H, self.nfft, axis=1)[:, L - 1:]
        self.hist = ext[len(ext) - (L - 1):] if L > 1 else self.hist
        return y.reshape(-1)[:n]


def soft_limit(x: np.ndarray, threshold: float = LIMIT_THRESHOLD, ceiling: float = LIMIT_CEILING) -> np.ndarray:
    """Identity below the threshold; above it a tanh knee that approaches the ceiling (< full scale)."""
    a = np.abs(x)
    over = a > threshold
    if not over.any():
        return x
    room = ceiling - threshold
    y = x.copy()
    y[over] = np.sign(x[over]) * (threshold + room * np.tanh((a[over] - threshold) / room))
    return y


class OutputChain:
    """One utterance's output processing; call it on consecutive 48 kHz chunks."""

    def __init__(self, native_rate: int, out_rate: int, eq: bool = True, makeup_db: float | None = None,
                 tone: tuple[float, float] | None = None):
        """tone: (low-shelf dB at 210 Hz, high-shelf dB at 4 kHz) instead of the default EQ (with eq=True)."""
        self.native_rate, self.out_rate, self.eq = native_rate, out_rate, eq
        if eq and tone:
            secs = [lowshelf(EQ_SHELF_HZ, tone[0], EQ_SHELF_Q, out_rate), highshelf(4000.0, tone[1], EQ_SHELF_Q, out_rate)]
            self.fir = StreamFIR(impulse_response(secs, out_rate))
            makeup_db = 0.0 if makeup_db is None else makeup_db
        else:
            self.fir = StreamFIR(eq_fir(out_rate)) if eq else None
        self.gain = 10 ** ((EQ_MAKEUP_DB if makeup_db is None else makeup_db) / 20)

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        y = resample(chunk, self.native_rate, self.out_rate)
        if self.fir is None:  # raw output: only a hard clip at full scale, as the vocoder output always had
            return np.clip(y, -1.0, 1.0).astype(np.float32, copy=False)
        y = soft_limit(self.fir(y) * self.gain)
        return y.astype(np.float32)


# ------------------------------------------------------------------------------- loudness (measurement)
def loudness(x: np.ndarray, fs: int) -> float:
    """Integrated loudness, ITU-R BS.1770-4 (K-weighting, 400 ms blocks, 75% overlap, absolute -70 LUFS
    and relative -10 LU gates), mono. Used to measure EQ_MAKEUP_DB; not applied at run time."""
    k = [highshelf(1500.0, 4.0, 1 / math.sqrt(2), fs), highpass(38.0, 0.5, fs)]
    y = StreamFIR(impulse_response(k, fs, max_seconds=2.0))(x)
    block, hop = int(0.4 * fs), int(0.1 * fs)
    if len(y) < block:
        return float("-inf")
    ms = np.array([np.mean(y[i: i + block] ** 2) for i in range(0, len(y) - block + 1, hop)])
    lk = -0.691 + 10 * np.log10(np.maximum(ms, 1e-20))
    ms = ms[lk > -70]
    if not len(ms):
        return float("-inf")
    rel = -0.691 + 10 * np.log10(ms.mean()) - 10
    ms = ms[-0.691 + 10 * np.log10(ms) > rel]
    return float(-0.691 + 10 * np.log10(ms.mean()))


def trim_edges(x: np.ndarray, fs: int, lead_ms: float | None, tail_ms: float | None, floor_db: float = 35.0,
               frame_ms: float = 10.0) -> np.ndarray:
    """Cut leading / trailing silence (frames more than floor_db below the loudest frame) down to lead_ms /
    tail_ms (None leaves that edge alone)."""
    frame = max(1, int(fs * frame_ms / 1000))
    n = len(x) // frame
    if n == 0:
        return x
    e = (np.asarray(x[: n * frame], dtype=np.float64).reshape(n, frame) ** 2).mean(1)
    lv = 10 * np.log10(e + 1e-12)
    act = np.flatnonzero(lv > lv.max() - floor_db)
    if not len(act):
        return x
    s = 0 if lead_ms is None else max(0, int(act[0]) * frame - int(lead_ms * fs / 1000))
    t = len(x) if tail_ms is None else min(len(x), (int(act[-1]) + 1) * frame + int(tail_ms * fs / 1000))
    return x[s:t]

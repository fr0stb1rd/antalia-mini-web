"""Building blocks: ConvNeXt-1D, RoPE self-attention, SwiGLU, adaLN-single transformer block.

References: ConvNeXt (Liu et al., 2022) as used for 1-D audio by Vocos (Siuzdak, 2023); rotary position
embeddings (Su et al., 2021); SwiGLU (Shazeer, 2020); adaLN-single, one shared timestep projection for
all blocks plus a learned per-block embedding (PixArt-alpha, Chen et al., 2023).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXt1d(nn.Module):
    def __init__(self, dim: int, hidden: int, kernel: int = 7, dilation: int = 1, layer_scale: float = 1e-2):
        super().__init__()
        self.dw = nn.Conv1d(dim, dim, kernel, padding=dilation * (kernel // 2), dilation=dilation, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, hidden)
        self.pw2 = nn.Linear(hidden, dim)
        self.gamma = nn.Parameter(torch.full((dim,), layer_scale))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x (B, C, T), mask (B, 1, T) of 0/1."""
        r = x
        if mask is not None:
            x = x * mask
        x = self.dw(x).transpose(1, 2)
        x = self.pw2(F.gelu(self.pw1(self.norm(x))))
        x = r + (self.gamma * x).transpose(1, 2)
        return x * mask if mask is not None else x


def rope_cache(n: int, dim: int, device, base: float = 10000.0, pos: torch.Tensor | None = None):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    p = torch.arange(n, device=device).float() if pos is None else pos.float()
    f = p[..., None] * inv  # (..., n, dim/2)
    return f.cos(), f.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x (B, H, T, D); cos/sin (T, D/2) or (B, 1, T, D/2)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, rope, key_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x (B, T, C); key_mask (B, T) bool, True = valid."""
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        cos, sin = rope
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        am = key_mask[:, None, None, :] if key_mask is not None else None
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=am)
        return self.out(y.transpose(1, 2).reshape(B, T, C))


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w12 = nn.Linear(dim, 2 * hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, -1)
        return self.w3(F.silu(a) * b)


class TransformerBlock(nn.Module):
    """Pre-norm block for the text encoder (no conditioning)."""

    def __init__(self, dim: int, heads: int, hidden: int):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.ff = SwiGLU(dim, hidden)

    def forward(self, x, rope, key_mask=None):
        x = x + self.attn(self.n1(x), rope, key_mask)
        return x + self.ff(self.n2(x))


class AdaBlock(nn.Module):
    """Transformer block modulated by adaLN-single: shift/scale/gate come from one shared projection of
    the (time, step-size) embedding plus this block's own learned table (PixArt-alpha)."""

    def __init__(self, dim: int, heads: int, hidden: int):
        super().__init__()
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = Attention(dim, heads)
        self.ff = SwiGLU(dim, hidden)
        self.table = nn.Parameter(torch.zeros(6, dim))  # zero gates at init: every block starts as identity

    def forward(self, x, mod, rope, key_mask=None):
        """mod (B, 6, C) from the shared projection."""
        s1, b1, g1, s2, b2, g2 = (mod + self.table[None]).unbind(1)
        h = self.n1(x) * (1 + s1[:, None]) + b1[:, None]
        x = x + g1[:, None] * self.attn(h, rope, key_mask)
        h = self.n2(x) * (1 + s2[:, None]) + b2[:, None]
        return x + g2[:, None] * self.ff(h)


def sinusoidal(t: torch.Tensor, dim: int, scale: float = 1000.0) -> torch.Tensor:
    half = dim // 2
    f = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
    a = scale * t.float()[:, None] * f[None]
    return torch.cat([a.sin(), a.cos()], -1)


def sequence_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    max_len = max_len or int(lengths.max())
    return torch.arange(max_len, device=lengths.device)[None] < lengths[:, None]

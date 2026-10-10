"""Antalia Mini acoustic model: characters -> durations -> 128-band log-mel at 93.75 Hz.

  text ids --TextEncoder--> h (B, L, d)            ConvNeXt stack + RoPE self-attention
        h --prior--> mu (mel space)                 aligned to frames by MAS during training
        h --DurationPredictor--> log durations      trained on MAS durations (stop-grad)
        h --repeat by durations--> c (B, T, d)      frame conditions
  noise --Generator(c), K steps--> mel              flow matching with shortcut step-size
                                                     conditioning: 1, 2, 4, 8... steps from one model

Alignment needs no external aligner: monotonic alignment search over the Gaussian likelihood of
the (normalised) mel frames under mu, as in Glow-TTS (Kim et al., 2020) and Matcha-TTS (Mehta et al.,
2024). The generator is a DiT-style transformer over pairs of mel frames, modulated by adaLN-single
(PixArt-alpha), trained with the flow-matching objective (Lipman et al., 2023) plus the shortcut
self-consistency objective (Frans et al., 2024, "One Step Diffusion via Shortcut Models"), so a few
steps suffice without a separate distillation stage.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from antalia_mini.audio import MEL_MEAN, MEL_STD
from antalia_mini.modules import AdaBlock, ConvNeXt1d, TransformerBlock, rope_cache, sequence_mask, sinusoidal
from antalia_mini.text import SYMBOLS


@dataclass
class AcousticConfig:
    n_symbols: int = len(SYMBOLS)
    n_mels: int = 128
    d_text: int = 192
    text_convnext: int = 3
    text_hidden: int = 384
    text_attn: int = 1
    heads: int = 4
    d_dur: int = 128
    d_gen: int = 192
    gen_blocks: int = 5
    gen_hidden: int = 512
    d_time: int = 128
    stack: int = 2                  # mel frames per generator token
    cond_drop: float = 0.1          # condition dropout (enables classifier-free guidance)
    shortcut_frac: float = 0.25     # share of the batch trained on the self-consistency target
    shortcut_levels: int = 7        # step sizes 1, 1/2, ..., 1/128
    mel_mean: float = MEL_MEAN
    mel_std: float = MEL_STD
    max_dur_frames: int = 60

    def dict(self) -> dict:
        return asdict(self)


class TextEncoder(nn.Module):
    def __init__(self, c: AcousticConfig):
        super().__init__()
        self.emb = nn.Embedding(c.n_symbols, c.d_text)
        nn.init.normal_(self.emb.weight, std=c.d_text ** -0.5)
        self.convs = nn.ModuleList([ConvNeXt1d(c.d_text, c.text_hidden, kernel=5) for _ in range(c.text_convnext)])
        self.attn = nn.ModuleList([TransformerBlock(c.d_text, c.heads, c.text_hidden) for _ in range(c.text_attn)])
        self.norm = nn.LayerNorm(c.d_text)
        self.heads = c.heads

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """ids (B, L), mask (B, L) bool -> h (B, L, d)."""
        x = self.emb(ids).transpose(1, 2)
        m = mask[:, None].float()
        for blk in self.convs:
            x = blk(x, m)
        x = x.transpose(1, 2)
        rope = rope_cache(x.shape[1], x.shape[2] // self.heads, x.device)
        for blk in self.attn:
            x = blk(x, rope, mask)
        return self.norm(x) * mask[..., None]


class DurationPredictor(nn.Module):
    def __init__(self, c: AcousticConfig):
        super().__init__()
        self.inp = nn.Conv1d(c.d_text, c.d_dur, 1)
        self.convs = nn.ModuleList([nn.Conv1d(c.d_dur, c.d_dur, 3, padding=1) for _ in range(3)])
        self.norms = nn.ModuleList([nn.LayerNorm(c.d_dur) for _ in range(3)])
        self.out = nn.Conv1d(c.d_dur, 1, 1)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """-> log(frames) per symbol (B, L)."""
        m = mask[:, None].float()
        x = self.inp(h.transpose(1, 2)) * m
        for conv, norm in zip(self.convs, self.norms):
            x = F.gelu(norm(conv(x * m).transpose(1, 2)).transpose(1, 2))
        return self.out(x * m).squeeze(1) * mask


class Generator(nn.Module):
    """Velocity model v(x_t, t, d | c) over stacked mel frames."""

    def __init__(self, c: AcousticConfig):
        super().__init__()
        self.c = c
        tok = c.n_mels * c.stack
        self.cond_conv = ConvNeXt1d(c.d_text, c.text_hidden, kernel=7)
        self.inp = nn.Linear(tok + c.d_text, c.d_gen)
        self.t_mlp = nn.Sequential(nn.Linear(2 * c.d_time, c.d_time), nn.SiLU())
        self.mod = nn.Linear(c.d_time, 6 * c.d_gen)  # adaLN-single: one projection for every block
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)
        self.blocks = nn.ModuleList([AdaBlock(c.d_gen, c.heads, c.gen_hidden) for _ in range(c.gen_blocks)])
        self.norm = nn.LayerNorm(c.d_gen, elementwise_affine=False)
        self.final_mod = nn.Parameter(torch.zeros(2, c.d_gen))
        self.out = nn.Linear(c.d_gen, tok)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.null_cond = nn.Parameter(torch.zeros(c.d_text))

    def prep_cond(self, cond: torch.Tensor, fmask: torch.Tensor) -> torch.Tensor:
        """cond (B, T, d_text) per frame -> (B, T/stack, d_text) per token, computed once per utterance."""
        m = fmask[:, None].float()
        x = self.cond_conv(cond.transpose(1, 2) * m, m).transpose(1, 2)
        B, T, D = x.shape
        return x.view(B, T // self.c.stack, self.c.stack, D).mean(2)

    def forward(self, x: torch.Tensor, t: torch.Tensor, d: torch.Tensor, ctok: torch.Tensor, tmask: torch.Tensor) -> torch.Tensor:
        """x (B, N, n_mels*stack) at time t (B,), step size d (B,), ctok (B, N, d_text), tmask (B, N)."""
        te = self.t_mlp(torch.cat([sinusoidal(t, self.c.d_time), sinusoidal(d, self.c.d_time)], -1))
        mod = self.mod(te).view(-1, 6, self.c.d_gen)
        h = self.inp(torch.cat([x, ctok], -1))
        rope = rope_cache(h.shape[1], self.c.d_gen // self.c.heads, h.device)
        for blk in self.blocks:
            h = blk(h, mod, rope, tmask)
        s, b = self.final_mod.unbind(0)
        return self.out(self.norm(h) * (1 + s) + b) * tmask[..., None]


class AcousticModel(nn.Module):
    def __init__(self, c: AcousticConfig | None = None):
        super().__init__()
        self.c = c = c or AcousticConfig()
        self.encoder = TextEncoder(c)
        self.prior = nn.Linear(c.d_text, c.n_mels)
        self.duration = DurationPredictor(c)
        self.generator = Generator(c)

    # ----------------------------------------------------------------- helpers
    def norm_mel(self, mel):
        return (mel - self.c.mel_mean) / self.c.mel_std

    def denorm_mel(self, y):
        return y * self.c.mel_std + self.c.mel_mean

    def _stack(self, y: torch.Tensor) -> torch.Tensor:
        """(B, T, n_mels) -> (B, T/stack, n_mels*stack); T must be a multiple of stack."""
        B, T, M = y.shape
        return y.reshape(B, T // self.c.stack, self.c.stack * M)

    def _unstack(self, z: torch.Tensor) -> torch.Tensor:
        B, N, _ = z.shape
        return z.reshape(B, N * self.c.stack, self.c.n_mels)

    @staticmethod
    def expand(h: torch.Tensor, dur: torch.Tensor, T: int) -> torch.Tensor:
        """Repeat each symbol's vector dur times: (B, L, D), (B, L) int -> (B, T, D)."""
        cum = torch.cumsum(dur, 1)
        frames = torch.arange(T, device=h.device)
        idx = torch.searchsorted(cum, frames[None].expand(h.shape[0], -1).contiguous(), right=True)
        idx = idx.clamp(max=h.shape[1] - 1)
        return torch.gather(h, 1, idx[..., None].expand(-1, -1, h.shape[2]))

    # ----------------------------------------------------------------- training
    def forward(self, ids, id_len, mel, mel_len) -> dict:
        """ids (B, L), mel (B, n_mels, T) log-mel. Returns losses and diagnostics."""
        c = self.c
        B = ids.shape[0]
        xmask = sequence_mask(id_len, ids.shape[1])
        T = (mel.shape[2] // c.stack) * c.stack
        mel_len = (mel_len // c.stack * c.stack).clamp(max=T)
        y = self.norm_mel(mel[:, :, :T]).transpose(1, 2)  # (B, T, M)
        ymask = sequence_mask(mel_len, T)
        h = self.encoder(ids, xmask)
        mu = self.prior(h)  # (B, L, M)

        from antalia_mini.mas import mas  # training only (numba); kept out of inference imports

        # MAS on the unit-variance Gaussian log-likelihood of every frame under every symbol's mu
        with torch.no_grad():
            ll = -0.5 * (torch.cdist(mu.float(), y.float()) ** 2)  # (B, L, T)
            ll = ll.masked_fill(~xmask[:, :, None] | ~ymask[:, None, :], -1e4)
            attn = mas(ll, id_len, mel_len)
            dur = attn.sum(2)  # frames per symbol
        mu_y = torch.bmm(attn.transpose(1, 2).float(), mu)  # (B, T, M)
        prior_loss = (0.5 * ((y - mu_y) ** 2).sum(-1) * ymask).sum() / (ymask.sum() * c.n_mels)

        logd_tgt = torch.log(dur.float().clamp(min=1)) * xmask
        logd = self.duration(h.detach(), xmask)
        dur_loss = ((logd - logd_tgt) ** 2 * xmask).sum() / xmask.sum()

        # generator: flow matching + shortcut self-consistency
        cond = torch.bmm(attn.transpose(1, 2).float(), h)  # (B, T, d_text)
        drop = torch.rand(B, device=ids.device) < c.cond_drop
        cond = torch.where(drop[:, None, None], self.generator.null_cond.expand_as(cond), cond)
        tmask = ymask[:, :: c.stack]
        ctok = self.generator.prep_cond(cond, ymask)
        x1 = self._stack(y * ymask[..., None])
        x0 = torch.randn_like(x1)
        n_sc = int(round(B * c.shortcut_frac))
        # step sizes for the shortcut part: d = 2^-k, t on the grid of d, target from two half steps
        k = torch.randint(1, c.shortcut_levels + 1, (B,), device=ids.device)
        d_half = 2.0 ** (-k.float())
        grid = torch.floor(torch.rand(B, device=ids.device) * (1.0 / (2 * d_half))) * (2 * d_half)
        t = torch.where(torch.arange(B, device=ids.device) < n_sc, grid, torch.rand(B, device=ids.device))
        d = torch.where(torch.arange(B, device=ids.device) < n_sc, 2 * d_half, torch.zeros(B, device=ids.device))
        xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
        v_tgt = x1 - x0
        if n_sc:
            with torch.no_grad():
                g = self.generator
                sl = slice(0, n_sc)
                # the smallest step size bootstraps from the flow-matching velocity (d = 0)
                dq = torch.where(k[sl] == c.shortcut_levels, torch.zeros_like(d_half[sl]), d_half[sl])
                s1 = g(xt[sl], t[sl], dq, ctok[sl], tmask[sl])
                x_mid = xt[sl] + s1 * d_half[sl, None, None]
                s2 = g(x_mid, t[sl] + d_half[sl], dq, ctok[sl], tmask[sl])
                v_tgt = torch.cat([(s1 + s2) / 2, v_tgt[n_sc:]], 0)
        v = self.generator(xt, t, d, ctok, tmask)
        err = ((v - v_tgt) ** 2).mean(-1) * tmask
        w_fm = torch.arange(B, device=ids.device) >= n_sc
        fm_loss = (err[w_fm].sum() / tmask[w_fm].sum().clamp(min=1))
        sc_loss = (err[~w_fm].sum() / tmask[~w_fm].sum().clamp(min=1)) if n_sc else torch.zeros((), device=ids.device)
        loss = fm_loss + sc_loss + prior_loss + dur_loss
        return {"loss": loss, "fm": fm_loss.detach(), "sc": sc_loss.detach(), "prior": prior_loss.detach(),
                "dur": dur_loss.detach(), "frames_per_symbol": (dur.float().sum() / xmask.sum()).detach()}

    # ----------------------------------------------------------------- inference
    @torch.no_grad()
    def synthesize(self, ids, id_len, steps: int = 4, speed: float = 1.0, temperature: float = 1.0, cfg: float = 1.0,
                   generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """-> log-mel (B, n_mels, T) and frame lengths (B,)."""
        c = self.c
        xmask = sequence_mask(id_len, ids.shape[1])
        h = self.encoder(ids, xmask)
        dur = torch.round(torch.exp(self.duration(h, xmask)) / speed).clamp(min=1, max=c.max_dur_frames).long() * xmask
        mel_len = dur.sum(1)
        mel_len = ((mel_len + c.stack - 1) // c.stack) * c.stack
        T = int(mel_len.max())
        ymask = sequence_mask(mel_len, T)
        cond = self.expand(h, dur, T) * ymask[..., None]
        g = self.generator
        ctok = g.prep_cond(cond, ymask)
        tmask = ymask[:, :: c.stack]
        x = torch.randn(ctok.shape[0], ctok.shape[1], c.n_mels * c.stack, device=ids.device, generator=generator) * temperature
        if cfg != 1.0:
            cnull = g.prep_cond(g.null_cond.expand_as(cond) * ymask[..., None], ymask)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((x.shape[0],), i * dt, device=x.device)
            d = torch.full_like(t, dt if steps > 1 else 1.0)
            v = g(x, t, d, ctok, tmask)
            if cfg != 1.0:
                v = v + (cfg - 1.0) * (v - g(x, t, d, cnull, tmask))
            x = x + v * dt
        y = self._unstack(x) * ymask[..., None]
        return self.denorm_mel(y).transpose(1, 2), mel_len


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())

"""Public API: raw Turkish text in, speech out.

  from antalia_mini import Antalia
  tts = Antalia()                                   # downloads the weights once, GPU if there is one
  speech = tts.say("Merhaba, size nasıl yardımcı olabilirim?", path="merhaba.wav", seed=0)
  speech.audio, speech.sample_rate, speech.duration, speech.seed
  for chunk in tts.stream(long_text):               # float32 chunks, first sentence first
      ...

The text goes through the Turkish normaliser (numbers, dates, amounts, initialisms, web addresses,
English and brand words), is split into sentence-sized chunks, and each chunk is synthesised with a
few flow-matching steps and vocoded to 48 kHz. Other output rates (24, 16, 8 kHz) are resampled.
"""
from __future__ import annotations

import math
import os
import platform
import re
import secrets
import sys
import threading
import unicodedata
import urllib.request
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch

from antalia_mini.modules import sequence_mask
from antalia_mini.post import OutputChain, resample, trim_edges
from antalia_mini.text import PAD, clean_model_text, encode, to_model_text, turkish_lower
from antalia_mini.weights import DEFAULTS, FILES, load_release, sha256

DEFAULT_REPO = "cloud0day3/antalia-mini"
OPTION_DEFAULTS = {"prefix": "", "prefix_margin_ms": 0.0, "suffix": "", "suffix_margin_ms": 0.0, "first_stretch": 1.0,
                   "lead_ms": None, "tail_ms": None, "fade_ms": 5.0, "temperature": None, "gain_db": 0.0, "tone": None, "mel_sharpen": 1.0, "prefix_initials": None, "space_stretch": 1.0, "gem_stretch": 1.0}
SAMPLE_RATES = (48000, 24000, 16000, 8000)

__all__ = ["Antalia", "Speech", "model_text", "split_model_text", "resample", "DEFAULT_REPO", "SAMPLE_RATES"]


# ------------------------------------------------------------------------------------------ results
@dataclass
class Speech:
    """Mono float32 audio in [-1, 1]."""

    audio: np.ndarray
    sample_rate: int
    seed: int
    duration: float = field(init=False)

    def __post_init__(self):
        self.duration = len(self.audio) / self.sample_rate

    def save(self, path: str | os.PathLike) -> Path:
        """Write the audio; the format follows the extension (.wav and .flac as 16-bit PCM, .ogg, ...)."""
        import soundfile as sf

        p = Path(path)
        subtype = "PCM_16" if p.suffix.lower() in (".wav", ".flac") else None
        sf.write(str(p), self.audio, self.sample_rate, subtype=subtype)
        return p

    def __array__(self, dtype=None, copy=None):
        return self.audio if dtype is None else self.audio.astype(dtype)


# ------------------------------------------------------------------------------------------ text chunks
_SENTENCE = re.compile(r"(?<=[.?!])\s+")
_CLAUSE = re.compile(r"(?<=[,;:])\s+")
_LETTER = re.compile(r"[^\W\d_]")


def _pack(pieces: list[str], max_chars: int, sep: str = " ") -> list[str]:
    """Greedily join pieces into chunks of at most max_chars (a piece longer than that is cut further)."""
    out, cur = [], ""
    for p in pieces:
        if len(p) > max_chars:
            if cur:
                out.append(cur)
                cur = ""
            if sep == " " and " " in p:
                out.extend(_pack(p.split(" "), max_chars))
            else:
                out.extend(p[i: i + max_chars] for i in range(0, len(p), max_chars))
            continue
        if cur and len(cur) + 1 + len(p) > max_chars:
            out.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}" if cur else p
    if cur:
        out.append(cur)
    return out


def split_model_text(text: str, max_chars: int = 300, first_chars: int | None = 80) -> list[str]:
    """Normalised (model) text -> chunks to synthesise one at a time (say() and stream() use the same).

    Sentences are packed together up to max_chars (~18 s of speech: the model was trained on 1.5-25 s
    clips of one to a few sentences, and packing keeps one-word sentences from standing alone); a longer
    sentence is cut at commas, then at spaces. The first chunk packs only up to first_chars (but always
    holds the whole first sentence), so a stream starts after the first sentence rather than ~18 s in."""
    sentences = [s.strip() for s in _SENTENCE.split(text.strip()) if s.strip()]
    pieces: list[str] = []
    for s in sentences:
        pieces.extend(_pack([c for c in _CLAUSE.split(s) if c], max_chars) if len(s) > max_chars else [s])
    if not pieces:
        return []
    if first_chars:
        head, used = "", 0
        for p in pieces:
            if head and len(head) + 1 + len(p) > first_chars:
                break
            head, used = (f"{head} {p}" if head else p), used + 1
        chunks = [head] + _pack(pieces[used:], max_chars)
    else:
        chunks = _pack(pieces, max_chars)
    return [c for c in chunks if _LETTER.search(c)]


def model_text(raw: str) -> str:
    """Raw text -> model text; if the normaliser fails on some input, fall back to plain lowercasing
    (the reading may be worse, but synthesis never fails on text)."""
    try:
        return to_model_text(raw)
    except Exception as e:  # pragma: no cover - the normaliser has not failed on any test input
        warnings.warn(f"text normaliser failed ({type(e).__name__}: {e}); reading the text as written", RuntimeWarning)
        return clean_model_text(turkish_lower(unicodedata.normalize("NFC", raw)))


# ------------------------------------------------------------------------------------------ weights
def _download(repo: str, revision: str | None, variant: str) -> Path:
    files = ["config.json", *FILES[variant]]
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        hf_hub_download = None
    if hf_hub_download is not None:
        paths = [hf_hub_download(repo, f, revision=revision) for f in files]
        return Path(paths[0]).parent
    # no huggingface_hub: plain HTTPS into ~/.cache/antalia-mini (honours HF_ENDPOINT and HF_TOKEN)
    rev = revision or "main"
    root = Path(os.environ.get("ANTALIA_MINI_CACHE", Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "antalia-mini"))
    d = root / repo.replace("/", "--") / rev
    d.mkdir(parents=True, exist_ok=True)
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    token = os.environ.get("HF_TOKEN")
    for f in files:
        dst = d / f
        if dst.exists():
            continue
        req = urllib.request.Request(f"{endpoint}/{repo}/resolve/{rev}/{f}",
                                     headers={"Authorization": f"Bearer {token}"} if token else {})
        tmp = dst.with_suffix(dst.suffix + f".{os.getpid()}.part")
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as out:
            while block := r.read(1 << 20):
                out.write(block)
        os.replace(tmp, dst)
    return d


def _resolve(model: str | os.PathLike | None, revision: str | None, variant: str) -> Path:
    model = model or os.environ.get("ANTALIA_MINI_MODEL") or DEFAULT_REPO
    p = Path(model).expanduser()
    if p.is_dir():
        return p
    if p.exists() or str(model).startswith((".", "/", "~")):
        raise FileNotFoundError(f"{model}: not a release directory (config.json + *.safetensors)")
    return _download(str(model), revision, variant)


def _verify(d: Path, config: dict, variant: str) -> None:
    for f in FILES[variant]:
        want = config.get("files", {}).get(f, {}).get("sha256")
        if want and sha256(d / f) != want:
            raise ValueError(f"{d / f}: checksum mismatch (incomplete or modified download?)")


# ------------------------------------------------------------------------------------------ the model
class Antalia:
    """Antalia-2 Mini text to speech.

    model:       Hugging Face repo id (default cloud0day3/antalia-mini, or $ANTALIA_MINI_MODEL) or a local
                 release directory with config.json and the safetensors files
    device:      "cuda", "cpu", "mps", ...; default: CUDA if available, else CPU ("mps" works and was
                 ~2.5x faster than the CPU on an Apple M5)
    sample_rate: output rate; 48000 is native, 24000/16000/8000 are resampled from it
    variant:     "fp32" or "fp16" weights file (fp16 halves the download; weights are used as fp32 either way)
    threads:     torch CPU threads (process-wide). "auto": 2 on Apple Silicon CPUs (measured fastest on an
                 M5: the matrix units are shared, more threads only add overhead), torch's default elsewhere;
                 None leaves torch's setting alone
    compile:     torch.compile the generator and vocoder (no gain measured on Apple M5 CPU; try on x86/GPU)
    eq:          the default output EQ, a light bass reduction (90 Hz high-pass, -5 dB low shelf at 210 Hz,
                 fixed make-up gain, peak limiter; see antalia_mini.post); False gives the raw vocoder output.
                 say() and stream() take eq= to override it per call

    Thread-safe: any number of threads may call say()/stream() on one instance; text normalisation runs
    in parallel and the network calls are serialised by a lock.
    """

    def __init__(self, model: str | os.PathLike | None = None, *, device: str | None = None, sample_rate: int = 48000,
                 variant: str = "fp32", revision: str | None = None, threads: int | str | None = "auto", compile: bool = False,
                 eq: bool = True, warmup: bool = True, **options):
        if variant not in FILES:
            raise ValueError(f"variant must be one of {sorted(FILES)}")
        self.path = _resolve(model, revision, variant)
        acoustic, vocoder, config = load_release(self.path, variant)
        _verify(self.path, config, variant)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if threads == "auto":
            threads = 2 if self.device.type == "cpu" and sys.platform == "darwin" and platform.machine() == "arm64" else None
        if threads:
            torch.set_num_threads(int(threads))
        self.acoustic = acoustic.to(self.device).eval().requires_grad_(False)
        self.vocoder = vocoder.to(self.device).eval().requires_grad_(False)
        self.config = config
        self.native_rate = int(config["sample_rate"])
        self.sample_rate = self._check_rate(sample_rate)
        self.eq = bool(eq)
        self.defaults = {**DEFAULTS, **config.get("defaults", {})}
        # synthesis options (fixed per instance; the release defaults live in config.json "defaults"):
        #   prefix / suffix: model text spoken before / after every chunk and cut from the audio again, so
        #     the chunk's first and last words are articulated as mid-utterance words; *_margin_ms moves the
        #     cut (positive keeps more of the prefix / suffix); first_stretch scales the first word's durations;
        #     lead_ms / tail_ms trim edge silence of each chunk down to that much (None leaves it)
        unknown = set(options) - set(OPTION_DEFAULTS)
        if unknown:
            raise TypeError(f"unknown options: {sorted(unknown)}")
        self.options = {**OPTION_DEFAULTS, **config.get("options", {}), **options}
        self._generator = self.acoustic.generator
        self._voc = self.vocoder
        if compile:
            self._generator = torch.compile(self.acoustic.generator, dynamic=True)
            self._voc = torch.compile(self.vocoder, dynamic=True)
        self._lock = threading.Lock()
        if warmup:
            self.say("Merhaba.", seed=0)

    def __repr__(self) -> str:
        p = self.config.get("params", {}).get("total")
        return f"Antalia(path={str(self.path)!r}, device={self.device}, sample_rate={self.sample_rate}, params={p})"

    def _check_rate(self, sr: int) -> int:
        sr = int(sr)
        if sr <= 0:
            raise ValueError("sample_rate must be positive")
        return sr

    # ------------------------------------------------------------------ public
    def say(self, text: str | Sequence[str], path: str | os.PathLike | Sequence[str | os.PathLike] | None = None, *,
            seed: int | None = None, speed: float | None = None, steps: int | None = None, cfg: float | None = None,
            sample_rate: int | None = None, eq: bool | None = None) -> Speech | list[Speech]:
        """Synthesise text (raw: digits, dates, abbreviations and English words are fine).

        A string gives one Speech; a list gives a list (its chunks are batched on GPU). path writes the
        audio (a list of paths for a list of texts). The same text, seed and settings give the same audio.
        Empty or unpronounceable text gives empty audio instead of an error."""
        single = isinstance(text, str)
        texts = [text] if single else list(text)
        paths = [path] if single else (list(path) if path is not None else [None] * len(texts))
        if len(paths) != len(texts):
            raise ValueError("pass one path per text")
        seed = self._seed(seed)
        sr = self._check_rate(sample_rate or self.sample_rate)
        chunked = [split_model_text(model_text(t)) if t else [] for t in texts]
        jobs = [(i, c) for i, cs in enumerate(chunked) for c in cs]
        gens = [torch.Generator().manual_seed(seed) for _ in texts]
        wavs: list[list[tuple[str, np.ndarray]]] = [[] for _ in texts]
        bs = 16 if self.device.type == "cuda" else 1
        for b in range(0, len(jobs), bs):
            batch = jobs[b: b + bs]
            out = self._synth([c for _, c in batch], [gens[i] for i, _ in batch], speed, steps, cfg)
            for (i, c), w in zip(batch, out):
                wavs[i].append((c, w))
        eq = self.eq if eq is None else bool(eq)
        results = []
        for ws, p in zip(wavs, paths):
            sp = Speech(self._render(ws, sr, eq), sr, seed)
            if p is not None:
                sp.save(p)
            results.append(sp)
        return results[0] if single else results

    def stream(self, text: str, *, seed: int | None = None, speed: float | None = None, steps: int | None = None,
               cfg: float | None = None, sample_rate: int | None = None, eq: bool | None = None) -> Iterator[np.ndarray]:
        """Yield float32 audio chunks as they are synthesised: the first sentence (or first few short ones)
        first, then sentence-sized chunks. Joined, the chunks equal say() with the same arguments sample
        for sample (same chunking, same noise, EQ state carried across chunks)."""
        seed = self._seed(seed)
        sr = self._check_rate(sample_rate or self.sample_rate)
        chain = OutputChain(self.native_rate, sr, self.eq if eq is None else bool(eq), tone=self.options["tone"])
        chunks = split_model_text(model_text(text)) if text else []
        gen = torch.Generator().manual_seed(seed)
        for k, c in enumerate(chunks):
            w = self._synth([c], [gen], speed, steps, cfg)[0]
            if k < len(chunks) - 1:
                w = np.concatenate([w, self._pause(c)])
            yield chain(w)

    @staticmethod
    def normalize(text: str) -> str:
        """The exact character string the model reads for this text (spoken form, lowercase)."""
        return model_text(text)

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _seed(seed: int | None) -> int:
        return secrets.randbits(31) if seed is None else int(seed)

    def _pause(self, chunk: str) -> np.ndarray:
        """Silence after a chunk: a little longer after a sentence than after a clause."""
        sec = 0.12 if chunk.rstrip()[-1:] in ".?!" else 0.06
        return np.zeros(int(sec * self.native_rate), dtype=np.float32)

    def _render(self, ws: list[tuple[str, np.ndarray]], sr: int, eq: bool) -> np.ndarray:
        """Chunks of one utterance -> output audio, through the same chain as stream()."""
        if not ws:
            return np.zeros(0, dtype=np.float32)
        chain = OutputChain(self.native_rate, sr, eq, tone=self.options["tone"])
        parts = []
        for k, (c, w) in enumerate(ws):
            parts.append(chain(np.concatenate([w, self._pause(c)]) if k < len(ws) - 1 else w))
        return np.concatenate(parts)

    def _synth(self, chunks: list[str], gens: list[torch.Generator], speed: float, steps: int | None,
               cfg: float | None) -> list[np.ndarray]:
        steps = max(1, int(steps or self.defaults["steps"]))
        cfg = float(self.defaults["cfg"] if cfg is None else cfg)
        speed = float(speed) if speed and speed > 0 else float(self.defaults.get("speed", 1.0))
        with self._lock, torch.inference_mode():
            temp = self.options["temperature"]
            temp = float(self.defaults.get("temperature", 1.0) if temp is None else temp)
            mel, ml, spans = self._acoustic(chunks, gens, steps, cfg, speed, temp)
            hop = self.vocoder.c.hop
            out = []
            for k in range(len(chunks)):
                n = int(ml[k])
                wav = self._voc(mel[k: k + 1, :, :n])[0, : n * hop].float().cpu().numpy()  # unclipped
                out.append(self._edges(wav, spans[k], hop))
        return out

    def _edges(self, wav: np.ndarray, span: tuple[int, int] | None, hop: int) -> np.ndarray:
        """Cut the prefix / suffix audio (with short fades) and trim edge silence, per the options."""
        o, sr = self.options, self.native_rate
        fade = int(o["fade_ms"] * sr / 1000)
        if span is not None:
            a, b, has_pre = span
            s0 = max(0, a * hop - int(o["prefix_margin_ms"] * sr / 1000)) if has_pre else 0
            s1 = min(len(wav), b * hop + int(o["suffix_margin_ms"] * sr / 1000)) if o["suffix"] else len(wav)
            wav = wav[s0:s1].copy()
            if has_pre and fade and len(wav) > 2 * fade:
                wav[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
            if o["suffix"] and fade and len(wav) > 2 * fade:
                wav[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        if o["lead_ms"] is not None or o["tail_ms"] is not None:
            wav = trim_edges(wav, sr, o["lead_ms"], o["tail_ms"])
        if o["gain_db"]:
            wav = wav * np.float32(10 ** (o["gain_db"] / 20))
        return wav

    def _acoustic(self, chunks: list[str], gens: list[torch.Generator], steps: int, cfg: float, speed: float,
                  temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
        """AcousticModel.synthesize with per-chunk noise from each chunk's own (CPU) generator, so the
        result does not depend on the batch or the device's RNG, and with the guided and unguided
        passes of classifier-free guidance run as one batch."""
        am, dev = self.acoustic, self.device
        c = am.c
        o = self.options
        pre, suf = o["prefix"], o["suffix"]
        # prefix_initials: speak the prefix only before chunks whose first letter is one of these (None: all)
        pres = [pre if pre and (o["prefix_initials"] is None or t[:1] in o["prefix_initials"]) else "" for t in chunks]
        npres = [len(encode(p)) - 2 if p else 0 for p in pres]  # symbols of the prefix (encode adds BOS and EOS)
        nsuf = len(encode(suf)) - 2 if suf else 0
        ids = [torch.tensor(encode(f"{p}{t}{suf}"), dtype=torch.long) for p, t in zip(pres, chunks)]
        il = torch.tensor([len(x) for x in ids], device=dev)
        I = torch.full((len(ids), int(il.max())), PAD, dtype=torch.long)
        for k, x in enumerate(ids):
            I[k, : len(x)] = x
        I = I.to(dev)
        xmask = sequence_mask(il, I.shape[1])
        h = am.encoder(I, xmask)
        logd = am.duration(h, xmask)
        if o["first_stretch"] != 1.0:  # the first word of the chunk (after the prefix): symbols up to the first space
            space = encode(" ")[1]
            for k, x in enumerate(ids):
                j = 1 + npres[k]
                while j < len(x) - 1 and int(x[j]) != space:
                    j += 1
                logd[k, 1 + npres[k]: j] += math.log(o["first_stretch"])
        if o["space_stretch"] != 1.0:  # longer word gaps: less merging of neighbouring words
            logd = logd + (I == encode(" ")[1]).float() * math.log(o["space_stretch"])
        if o["gem_stretch"] != 1.0:  # double sounds are long in Turkish: "berraktı" (rr), "alerjim mahvediyor" (m m)
            sp, g = encode(" ")[1], math.log(o["gem_stretch"])
            for k, x in enumerate(ids):
                xs = x.tolist()
                for j in range(2, len(xs) - 1):
                    if xs[j] == xs[j - 1] and xs[j] != sp:
                        logd[k, j] += g  # within a word: the second of the pair
                    elif xs[j - 1] == sp and xs[j] == xs[j - 2]:
                        logd[k, j - 1] += g  # across a word gap: the gap between the pair
        dur = torch.round(torch.exp(logd) / speed).clamp(min=1, max=c.max_dur_frames).long() * xmask
        mel_len = (dur.sum(1) + c.stack - 1) // c.stack * c.stack
        T = int(mel_len.max())
        ymask = sequence_mask(mel_len, T)
        cond = am.expand(h, dur, T) * ymask[..., None]
        g = am.generator
        ctok = g.prep_cond(cond, ymask)
        tmask = ymask[:, :: c.stack]
        width = c.n_mels * c.stack
        x = torch.zeros(len(chunks), ctok.shape[1], width)
        for k, gen in enumerate(gens):
            n = int(mel_len[k]) // c.stack
            x[k, :n] = torch.randn(n, width, generator=gen) * temperature
        x = x.to(dev)
        guided = cfg != 1.0
        if guided:
            cnull = g.prep_cond(g.null_cond.expand_as(cond) * ymask[..., None], ymask)
            ctok2, tmask2 = torch.cat([ctok, cnull]), torch.cat([tmask, tmask])
        dt = 1.0 / steps
        B = x.shape[0]
        for i in range(steps):
            t = torch.full((B,), i * dt, device=dev)
            d = torch.full_like(t, dt if steps > 1 else 1.0)
            if guided:
                vc, vu = self._generator(torch.cat([x, x]), torch.cat([t, t]), torch.cat([d, d]), ctok2, tmask2).chunk(2)
                v = vc + (cfg - 1.0) * (vc - vu)
            else:
                v = self._generator(x, t, d, ctok, tmask)
            x = x + v * dt
        y = am._unstack(x) * ymask[..., None]
        if o["mel_sharpen"] != 1.0:  # per-utterance global-variance boost around each mel bin's mean over time
            m = ymask[..., None].float()
            mu = (y * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp(min=1)
            y = (mu + o["mel_sharpen"] * (y - mu)) * m
        spans = None
        if pre or suf:  # frame span of the chunk's own text: after BOS + prefix, before suffix + EOS
            cum = torch.cumsum(dur, 1).cpu()
            spans = [(int(cum[k, npres[k]]), int(cum[k, len(x) - 2 - nsuf]), bool(pres[k])) for k, x in enumerate(ids)]
        return am.denorm_mel(y).transpose(1, 2), mel_len, spans or [None] * len(chunks)

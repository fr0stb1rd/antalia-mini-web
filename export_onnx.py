"""cloud0day3/antalia-mini PyTorch safetensors -> tarayıcıda çalışan 3 ONNX modeli.

Kullanım:
    pip install -r requirements-export.txt
    python export_onnx.py [--check]

GitHub Actions (.github/workflows/export-onnx.yml) otomatik çalıştırır.

Modeller:
1. text_stage.onnx : ids [B, L], mask [B, L] -> h [B, L, 192], logd [B, L]
2. sound_stage.onnx: cond [B, T, 192], fmask [B, T], noise [B, T/2, 256] -> mel [B, 128, T]
3. decoder.onnx    : mel [B, 128, T] -> audio [B, S] (48 kHz)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from antalia_mini.text import SYMBOLS
from antalia_mini.weights import load_release

REPO = os.environ.get("ANTALIA_MODEL_REPO", "cloud0day3/antalia-mini")
OUT = ROOT / "models"


def do_export(module, dummy_inputs, path, input_names, output_names, dynamic_axes):
    """Legacy TorchScript tabanlı ONNX ihracı (tüm ONNX Runtime sürümleriyle %100 uyumlu)."""
    torch.onnx.export(
        module,
        dummy_inputs,
        str(path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        dynamo=False,
    )
    print(f"ok {path.name}")


# -----------------------------------------------------------------------------
# Wrappers
# -----------------------------------------------------------------------------
class TextStageWrap(nn.Module):
    def __init__(self, acoustic):
        super().__init__()
        self.encoder = acoustic.encoder
        self.duration = acoustic.duration

    def forward(self, ids: torch.Tensor, mask: torch.Tensor):
        h = self.encoder(ids, mask)
        logd = self.duration(h, mask)
        return h, logd


class SoundStageWrap(nn.Module):
    def __init__(self, acoustic, steps: int = 8, cfg: float = 2.0):
        super().__init__()
        self.am = acoustic
        self.g = acoustic.generator
        self.steps = steps
        self.cfg = cfg
        self.c = acoustic.c

    def forward(self, cond: torch.Tensor, fmask: torch.Tensor, noise: torch.Tensor):
        ctok = self.g.prep_cond(cond, fmask)
        tmask = fmask[:, :: self.c.stack]
        if self.cfg != 1.0:
            cnull = self.g.prep_cond(self.g.null_cond.expand_as(cond) * fmask[..., None], fmask)
            ctok2 = torch.cat([ctok, cnull], dim=0)
            tmask2 = torch.cat([tmask, tmask], dim=0)

        x = noise
        dt = 1.0 / self.steps
        B = x.shape[0]
        for i in range(self.steps):
            t = torch.full((B,), i * dt, device=x.device)
            d = torch.full_like(t, dt if self.steps > 1 else 1.0)
            if self.cfg != 1.0:
                vc, vu = self.g(torch.cat([x, x], 0), torch.cat([t, t], 0), torch.cat([d, d], 0), ctok2, tmask2).chunk(2, 0)
                v = vc + (self.cfg - 1.0) * (vc - vu)
            else:
                v = self.g(x, t, d, ctok, tmask)
            x = x + v * dt

        y = self.am._unstack(x) * fmask[..., None]
        mel = self.am.denorm_mel(y).transpose(1, 2)
        return mel


class DecoderWrap(nn.Module):
    def __init__(self, vocoder):
        super().__init__()
        self.voc = vocoder
        c = vocoder.c
        self.n_fft = c.n_fft
        self.hop = c.hop
        win = torch.hann_window(self.n_fft)

        n = torch.arange(self.n_fft).float()
        k = torch.arange(self.n_fft // 2 + 1).float()
        angles = 2 * math.pi * k[:, None] * n[None, :] / self.n_fft

        cos_basis = torch.cos(angles) * (win[None, :] / self.n_fft)
        cos_basis[1:-1, :] *= 2.0
        sin_basis = -torch.sin(angles) * (win[None, :] / self.n_fft)
        sin_basis[1:-1, :] *= 2.0

        weight = torch.cat([cos_basis, sin_basis], dim=0).unsqueeze(1)  # (2050, 1, 2048)
        self.register_buffer("istft_weight", weight, persistent=False)
        self.register_buffer("win_kernel", (win ** 2).view(1, 1, self.n_fft), persistent=False)

    def forward(self, mel: torch.Tensor):
        v = self.voc
        x = v.inp(mel)
        x = v.norm0(x.transpose(1, 2)).transpose(1, 2)
        for b in v.blocks:
            x = b(x)
        x = v.head(v.norm1(x.transpose(1, 2))).transpose(1, 2)
        mag, ph = x.chunk(2, 1)
        mag = torch.exp(mag).clamp(max=1e2)
        real = mag * torch.cos(ph)
        imag = mag * torch.sin(ph)
        feat = torch.cat([real, imag], dim=1)

        ola = F.conv_transpose1d(feat, self.istft_weight, stride=self.hop, padding=0)
        B, _, T = mel.shape
        win_feat = torch.ones(B, 1, T, dtype=mel.dtype, device=mel.device)
        win_env = F.conv_transpose1d(win_feat, self.win_kernel, stride=self.hop, padding=0)

        wav = ola / win_env.clamp(min=1e-8)
        wav = wav[:, 0, self.n_fft // 2 : -(self.n_fft // 2)]
        return F.pad(wav, (0, self.hop))


# -----------------------------------------------------------------------------
# Main export
# -----------------------------------------------------------------------------
def download_hf_release(repo: str) -> Path:
    from huggingface_hub import hf_hub_download

    cache_dir = Path.home() / ".cache" / "antalia-mini" / repo.replace("/", "--")
    cache_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("config.json", "acoustic.safetensors", "vocoder.safetensors"):
        hf_hub_download(repo, fname, local_dir=str(cache_dir))
    return cache_dir


def main(check: bool = False, steps: int = 8, cfg: float = 2.0):
    OUT.mkdir(exist_ok=True)
    print(f"Model yükleniyor: {REPO} ...")
    release_dir = download_hf_release(REPO)
    acoustic, vocoder, config = load_release(release_dir, variant="fp32")

    # vocab ve config kaydet
    (ROOT / "vocab.json").write_text(
        json.dumps(
            {
                "symbols": SYMBOLS,
                "stoi": {s: i for i, s in enumerate(SYMBOLS)},
                "defaults": config.get("defaults", {}),
                "options": config.get("options", {}),
                "sample_rate": config.get("sample_rate", 48000),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    (ROOT / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))

    B, L, T = 1, 16, 32
    d_text = acoustic.c.d_text
    stack = acoustic.c.stack
    n_mels = acoustic.c.n_mels

    # 1. text_stage.onnx
    tw = TextStageWrap(acoustic).eval()
    do_export(
        tw,
        (torch.ones(B, L, dtype=torch.long), torch.ones(B, L, dtype=torch.bool)),
        OUT / "text_stage.onnx",
        ["ids", "mask"],
        ["h", "logd"],
        {
            "ids": {0: "B", 1: "L"},
            "mask": {0: "B", 1: "L"},
            "h": {0: "B", 1: "L"},
            "logd": {0: "B", 1: "L"},
        },
    )

    # 2. sound_stage.onnx
    sw = SoundStageWrap(acoustic, steps=steps, cfg=cfg).eval()
    do_export(
        sw,
        (
            torch.randn(B, T, d_text),
            torch.ones(B, T, dtype=torch.bool),
            torch.randn(B, T // stack, n_mels * stack),
        ),
        OUT / "sound_stage.onnx",
        ["cond", "fmask", "noise"],
        ["mel"],
        {
            "cond": {0: "B", 1: "T"},
            "fmask": {0: "B", 1: "T"},
            "noise": {0: "B", 1: "N"},
            "mel": {0: "B", 2: "T"},
        },
    )

    # 3. decoder.onnx
    dw = DecoderWrap(vocoder).eval()
    do_export(
        dw,
        (torch.randn(B, n_mels, T),),
        OUT / "decoder.onnx",
        ["mel"],
        ["audio"],
        {"mel": {0: "B", 2: "T"}, "audio": {0: "B", 1: "S"}},
    )

    import onnx

    for n in ("text_stage", "sound_stage", "decoder"):
        onnx.checker.check_model(str(OUT / f"{n}.onnx"))
    print("onnx checker ok!")

    # 4. foreign_dict.json (CMU İngilizce + tr_vocab fonetik sözlüğü)
    build_foreign_dictionary(OUT / "foreign_dict.json")

    # 5. tarayıcı normalizer sözlükleri (normalize.js verisi)
    build_web_dicts()

    if check:
        smoke_check(acoustic, vocoder, steps, cfg)
    print(f"Başarıyla tamamlandı. Modeller: {OUT}")


def build_foreign_dictionary(out_file: Path):
    """CMU Pronouncing Dictionary + tr_vocab.json -> 80,000+ kelimelik fonetik çeviri sözlüğü."""
    try:
        import cmudict
        from antalia_mini.text.foreign import is_english, respell, builtin_lexicon

        print("Yabancı sözcük fonetik sözlüğü derleniyor (CMU + tr_vocab)...")
        cmu = cmudict.dict()
        mapping = {}

        # 1. İncelenmiş marka ve terimler (lexicon.json)
        for k, v in builtin_lexicon().items():
            if v:
                mapping[k.lower()] = v.lower()

        # 2. CMU İngilizce kelimeler (tr_vocab'da Türkçe olarak geçmeyenler)
        cmu_count = 0
        for w in cmu:
            if is_english(w):
                r = respell(w)
                if r and r.lower() != w.lower():
                    mapping[w.lower()] = r.lower()
                    cmu_count += 1

        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, separators=(",", ":"))
        print(f"foreign_dict.json üretildi: {len(mapping)} terim ({out_file})")
    except ImportError:
        print("Uyarı: cmudict bulunamadı, foreign_dict.json derlenemedi.")
    except Exception as e:
        print(f"Uyarı: foreign_dict.json oluşturulurken hata: {e}")


def build_web_dicts() -> None:
    """normalize.js tarayıcı sözlükleri (known_words + english_i + foreign_parts).

    scripts/build_web_dicts.py'deki tek kaynağı çağırır; import hatası
    export'u durdurmaz (uyarı verir), CI validate adımı varlığı assert eder.
    """
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from build_web_dicts import main as build_dicts_main

        build_dicts_main()
    except ImportError as e:
        print(f"Uyarı: web sözlükleri derlenemedi (import): {e}")
    except Exception as e:
        print(f"Uyarı: web sözlükleri derlenirken hata: {e}")


def smoke_check(acoustic, vocoder, steps, cfg):
    import numpy as np
    import onnxruntime as ort

    print("Sayısal doğrulama yapılıyor (smoke check)...")
    tw_sess = ort.InferenceSession(str(OUT / "text_stage.onnx"))
    sw_sess = ort.InferenceSession(str(OUT / "sound_stage.onnx"))
    dw_sess = ort.InferenceSession(str(OUT / "decoder.onnx"))

    # --- 1. text_stage ---
    ids = np.array([[1, 12, 17, 34, 22, 12, 13, 12, 2]], dtype=np.int64)  # "merhaba"
    mask = np.ones_like(ids, dtype=bool)

    h_ort, logd_ort = tw_sess.run(None, {"ids": ids, "mask": mask})

    tw_torch = TextStageWrap(acoustic).eval()
    with torch.no_grad():
        h_pt, logd_pt = tw_torch(torch.from_numpy(ids), torch.from_numpy(mask))

    np.testing.assert_allclose(h_ort, h_pt.numpy(), rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(logd_ort, logd_pt.numpy(), rtol=1e-3, atol=1e-4)
    print("text_stage sayısal doğrulama BAŞARILI!")

    # --- 2. sound_stage ---
    L = ids.shape[1]
    d_text = acoustic.c.d_text
    stack = acoustic.c.stack
    n_mels = acoustic.c.n_mels
    # dur hesapla → frame uzunlugu bul
    dur = np.maximum(1, np.minimum(60, np.round(np.exp(logd_ort[0])))).astype(np.int32)
    T_frames = int(dur.sum())
    T_frames = ((T_frames + stack - 1) // stack) * stack  # stack katına yuvarla

    # cond: h_ort'dan expand
    cond = np.zeros((1, T_frames, d_text), dtype=np.float32)
    fi = 0
    for i in range(L):
        d = int(dur[i])
        for k in range(d):
            if fi < T_frames:
                cond[0, fi, :] = h_ort[0, i, :]
                fi += 1
    while fi < T_frames:
        cond[0, fi, :] = h_ort[0, L - 1, :]
        fi += 1

    fmask = np.ones((1, T_frames), dtype=bool)
    np.random.seed(42)
    noise = np.random.randn(1, T_frames // stack, n_mels * stack).astype(np.float32)

    mel_ort = sw_sess.run(None, {"cond": cond, "fmask": fmask, "noise": noise})[0]

    sw_torch = SoundStageWrap(acoustic, steps=steps, cfg=cfg).eval()
    with torch.no_grad():
        mel_pt = sw_torch(
            torch.from_numpy(cond),
            torch.from_numpy(fmask),
            torch.from_numpy(noise),
        ).numpy()

    np.testing.assert_allclose(mel_ort, mel_pt, rtol=1e-2, atol=5e-3)
    print(f"sound_stage sayısal doğrulama BAŞARILI! (mel shape: {mel_ort.shape})")

    # --- 3. decoder ---
    audio_ort = dw_sess.run(None, {"mel": mel_ort})[0]

    dw_torch = DecoderWrap(vocoder).eval()
    with torch.no_grad():
        audio_pt = dw_torch(torch.from_numpy(mel_ort)).numpy()

    np.testing.assert_allclose(audio_ort, audio_pt, rtol=1e-3, atol=1e-4)
    print(f"decoder sayısal doğrulama BAŞARILI! (audio shape: {audio_ort.shape})")

    # --- Bonus: elle yazılmış iSTFT vs torch.istft karşılaştırması ---
    with torch.no_grad():
        mel_t = torch.from_numpy(mel_ort)
        v = vocoder
        x = v.inp(mel_t)
        x = v.norm0(x.transpose(1, 2)).transpose(1, 2)
        for b in v.blocks:
            x = b(x)
        x = v.head(v.norm1(x.transpose(1, 2))).transpose(1, 2)
        mag, ph = x.chunk(2, 1)
        mag = torch.exp(mag).clamp(max=1e2)
        # torch.istft referans
        spec = torch.polar(mag.float(), ph.float())
        wav_ref = torch.istft(spec, v.c.n_fft, v.c.hop, v.c.n_fft,
                              window=v.window, center=True)
        wav_ref = torch.nn.functional.pad(wav_ref, (0, v.c.hop))
    # DecoderWrap çıktısıyla karşılaştır
    min_len = min(audio_pt.shape[1], wav_ref.shape[1])
    np.testing.assert_allclose(
        audio_pt[0, :min_len], wav_ref.numpy()[0, :min_len],
        rtol=5e-3, atol=5e-3,
    )
    print("iSTFT elle yazılmış vs torch.istft karşılaştırma BAŞARILI!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="onnxruntime ile sayısal doğrulama yap")
    parser.add_argument("--steps", type=int, default=8, help="DiT akış adımı (varsayılan: 8)")
    parser.add_argument("--cfg", type=float, default=2.0, help="Classifier-Free Guidance (varsayılan: 2.0)")
    args = parser.parse_args()
    main(check=args.check, steps=args.steps, cfg=args.cfg)

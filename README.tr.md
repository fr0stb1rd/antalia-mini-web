# ⚡ Antalia-2 Mini Web (ONNX)

**Tarayıcınızda çevrimdışı çalışan Türkçe Metinden Sese (TTS) yapay zeka modeli.**

[English](README.md) · **▶️ Canlı Site:** https://fr0stb1rd.github.io/antalia-mini-web/

[![pages](https://github.com/fr0stb1rd/antalia-mini-web/actions/workflows/pages.yml/badge.svg)](https://github.com/fr0stb1rd/antalia-mini-web/actions/workflows/pages.yml)
[![export-onnx](https://github.com/fr0stb1rd/antalia-mini-web/actions/workflows/export-onnx.yml/badge.svg)](https://github.com/fr0stb1rd/antalia-mini-web/actions/workflows/export-onnx.yml)

Türkçe metinden sese web uygulaması — **%100 yerel olarak tarayıcınızda çalışır**.
Sunucu yok, API anahtarı yok, sesiniz veya yazdığınız metinler asla cihazınızdan dışarı çıkmaz.

Model: [cloud0day3/antalia-mini](https://huggingface.co/cloud0day3/antalia-mini) (PatientDesk AI) —
7.62M parametre (~30 MB fp32 / ~15 MB fp16), Apache-2.0 lisansı.
Model ONNX formatına dönüştürülür ve tarayıcıda `onnxruntime-web` + WebGPU/WASM + WebAudio ile çalıştırılır.

## Öne Çıkan Özellikler

- **Dahili Türkçe Metin Normalizer:** Sayıları ("1250" → "bin iki yüz elli"), tarihleri ("15.10.2026"), saatleri ("14:30"), yüzdeleri ("%25"), para birimlerini ("2.500 TL") ve kısaltmaları ("dr.", "km", "SGK") otomatik ve doğal okur.
- **Kesintisiz Akış (Streaming):** Cümleler parça parça üretilir; ilk cümle hemen çalmaya başlarken arkadaki cümleler arka planda üretilmeye devam eder.
- **Akış Eşleştirme + Shortcut Self-Consistency:** 8-adımlı DiT çıkarımı ve CFG 2.0, en iyi kalite-hız dengesi için export aşamasında ONNX modeline gömülmüştür (`--steps 4` veya `--steps 2` ile daha hızlı varyantlar da üretilebilir).
- **Sıfır Bant Genişliği:** Modeller tarayıcı Cache Storage API'sine kaydedilir; sonraki ziyaretlerde internet kotanızdan yemez.
- **Gizlilik:** Tamamen istemci taraflı WebGPU / WASM çalışması.

## Mimari

PyTorch modeli, tarayıcıda akışlı oynatma için 3 aşamaya bölünmüştür:

```mermaid
flowchart LR
    A["Türkçe Metin"] --> B["Normalizer (JS)\nSayı, tarih, kısaltma okuma"]
    B --> C["text_stage.onnx\nHarfler → Vektörler + Süreler"]
    C --> D["Frame Planlama (JS)\nSüreler → Zaman çizelgesi"]
    D --> E["sound_stage.onnx\nDiT Akış Eşleştirme → Mel Spektrogramı"]
    E --> F["decoder.onnx\nConvNeXt Vocoder + iSTFT → 48 kHz Ses"]
    F --> G["WebAudio Oynatma\n.wav İndirme"]
```

## Otomatik İş Akışı (CI/CD)

Her şey GitHub Actions tarafından otomatik yürütülür:

```mermaid
flowchart TD
    A["git push (main)"] --> B["export-onnx.yml\nGitHub Actions Runner"]
    B --> C["export_onnx.py\nHF cloud0day3/antalia-mini → 3 ONNX modeli üret"]
    C --> D["push_hf.py\nModelleri fr0stb1rd/antalia-mini-web-onnx deposuna yükle"]
    D --> E["pages.yml\nSiteyi GitHub Pages'e dağıt"]
    E --> F["Tarayıcı\nHF CDN'den modelleri çekip WebGPU/WASM ile çalıştırır"]
```

### Tek Seferlik Kurulum (Secrets)

1. [huggingface.co/new-model](https://huggingface.co/new-model) adresinde `antalia-mini-web-onnx` adında açık (public) bir model reposu oluşturun.
2. [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) adresinden `write` yetkili bir token alın.
3. GitHub reponuzda **Settings** → **Secrets and variables** → **Actions** kısmına `HF_TOKEN` adıyla ekleyin.

## Lisans

- Model ağırlıkları ve kod: Apache-2.0.
- Web uygulaması: Apache-2.0.

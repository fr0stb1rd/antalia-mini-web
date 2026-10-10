"""models/*.onnx -> HuggingFace Hub.

Neden: tarayıcı, GitHub Releases dosyalarını CORS yüzünden indiremez
(release-assets.githubusercontent.com'da ACAO yok); HF CDN'de
`access-control-allow-origin: *` var, doğrudan fetch ile indirilebilir.
Böylece Pages reposu küçük kalır (ağır ONNX modelleri git geçmişine girmez).

Kullanım (Actions'ta otomatik; yerelde elle):
    export HF_REPO=fr0stb1rd/antalia-mini-web-onnx
    export HF_TOKEN=hf_...   # yazma yetkili token
    python export_onnx.py --check && python push_hf.py
"""
import os
import sys
from pathlib import Path

REPO = os.environ.get("HF_REPO", "fr0stb1rd/antalia-mini-web-onnx")
MODELS = Path(__file__).parent / "models"


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("hata: HF_TOKEN yok. https://huggingface.co/settings/tokens adresinden "
              "yazma yetkili token alıp HF_TOKEN olarak tanımlayın.", file=sys.stderr)
        raise SystemExit(1)
    files = sorted([f for f in MODELS.iterdir() if f.suffix in (".onnx", ".json") and f.is_file()])
    if not files:
        print(f"hata: {MODELS} boş; önce python export_onnx.py çalıştırın.", file=sys.stderr)
        raise SystemExit(1)

    from huggingface_hub import HfApi  # noqa: PLC0415

    api = HfApi(token=token)
    api.create_repo(REPO, exist_ok=True, repo_type="model", private=False)
    for f in files:
        api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name,
                        repo_id=REPO, repo_type="model", commit_message=f"update {f.name}")
        print(f"yüklendi: {REPO}/{f.name}")
    api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md",
                    repo_id=REPO, repo_type="model", commit_message="model card (apache-2.0)")
    print(f"Bitti. Çözüm URL kök: https://huggingface.co/{REPO}/resolve/main")


CARD = """---
license: apache-2.0
language:
- tr
base_model: cloud0day3/antalia-mini
pipeline_tag: text-to-speech
library_name: onnx
tags:
- onnx
- text-to-speech
- tts
- turkish
- onnxruntime-web
- webgpu
- wasm
- antalia-mini
---

# Antalia-2 Mini Web — ONNX weights

ONNX-converted weights of [cloud0day3/antalia-mini](https://huggingface.co/cloud0day3/antalia-mini)
for in-browser inference (produced by [fr0stb1rd/antalia-mini-web](https://github.com/fr0stb1rd/antalia-mini-web)).

- `text_stage.onnx`: text → character features + durations
- `sound_stage.onnx`: DiT Flow-Matching generator → mel frames
- `decoder.onnx`: Vocos ConvNeXt vocoder → 48 kHz waveform
- `foreign_dict.json`: CMU English to Turkish phonetic dictionary (80k+ words)
- `foreign_parts.json`: CamelCase part respellings for brand words (37k entries)
- `known_words.json`: frequent Turkish words (anti-respell shield, 6k entries)
- `english_i.json`: dotted/dotless I decision list for English words (14k entries)

Live: https://fr0stb1rd.github.io/antalia-mini-web/

## Disclaimer

The models in this repository are licensed under the **Apache License 2.0**.
This software comes with no warranty. By using it, you agree that you have
read and accepted the [LICENSE](https://github.com/fr0stb1rd/antalia-mini-web/blob/main/LICENSE).
"""


if __name__ == "__main__":
    main()

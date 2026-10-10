"""Tarayici normalizer (normalize.js) icin sozluk uretimi.

Uretir (models/ altina; CI'da export sirasinda, yerelde tek basina):
  - known_words.json : tr_vocab'da sikligi >=1000 olan kelimeler (known_in_turkish icin)
  - english_i.json   : CMU'da olup Turkce metinde de gecen, 'i' iceren kucuk harf
                       kelimeler (_english_i: buyuk "I" -> "I" karari icin)
  - foreign_parts.json : foreign_dict'te OLMAYAN CMU kelimelerinin parca hecelemesi
                       (respell_part; CamelCase parca cozumu icin, or. Git+Hub)

Kullanim:
    python scripts/build_web_dicts.py [--check]

--check sadece uretmeden dogrular (boyut ve ornek anahtarlar).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

KNOWN_THRESHOLD = 1000  # foreign.py _KNOWN_BRAND ile AYNI olmali


def build_known_words(tr_vocab: dict[str, int]) -> list[str]:
    return sorted(k for k, c in tr_vocab.items() if c >= KNOWN_THRESHOLD)


def build_english_i(cmu: dict[str, list[str]], tr_vocab: dict[str, int]) -> list[str]:
    def tr_lower(w: str) -> str:
        return w.replace("I", "ı").replace("İ", "i").lower()

    out = set()
    for w in cmu:
        if not w.isascii() or not w.isalpha() or "i" not in w:
            continue
        if tr_vocab.get(tr_lower(w), 0) > 0 or tr_vocab.get(w.lower(), 0) > 0:
            out.add(w)
    return sorted(out)


def build_foreign_parts(cmu: dict[str, list[str]], foreign_dict: dict[str, str]) -> dict[str, str]:
    """foreign_dict'te kapsanmayan CMU kelimeleri icin parca hecelemesi.

    foreign_dict yalnizca is_english kelimeleri icerir (Turkce metinde gecmeyen).
    CamelCase markalarin parcalari (Git, Hub, Bay, Star...) sikca Turkce metinde
    gectigi icin disinda kalir; Python bunlari CMU'dan heceler. Bu dosya o boslugu
    kapatir: {kucuk kelime: respell_part} (identity dahil; i/e birlesimi icin gerekli).
    """
    from antalia_mini.text.foreign import respell_part  # noqa: PLC0415

    out = {}
    for w in cmu:
        if not w.isascii() or not w.isalpha():
            continue
        r = respell_part(w)
        if r is None:
            continue
        if foreign_dict.get(w) != r:
            out[w] = r
    return dict(sorted(out.items()))


def main(check: bool = False) -> None:
    from antalia_mini.text.foreign import _cmu, _turkish  # noqa: PLC0415

    tr_vocab = _turkish()
    cmu = _cmu()
    known = build_known_words(tr_vocab)
    english_i = build_english_i(cmu, tr_vocab)

    print(f"tr_vocab: {len(tr_vocab)} kayit")
    print(f"cmu: {len(cmu)} kayit")
    print(f"known_words(>={KNOWN_THRESHOLD}): {len(known)}")
    print(f"english_i: {len(english_i)}")

    # saglik kontrolleri: sik gorulen kelimeler kapsanmali
    for must in ("istanbul", "intel", "iran", "bir", "ve"):
        assert must in tr_vocab, f"tr_vocab'da {must!r} yok!"
    for must in ("istanbul", "intel", "iran"):
        assert must in english_i, f"english_i'de {must!r} yok!"
    assert "merhaba" not in english_i  # CMU'da yok -> I karari gerekmez

    out_dir = ROOT / "models"
    foreign_path = out_dir / "foreign_dict.json"
    foreign_dict: dict[str, str] = (
        json.loads(foreign_path.read_text(encoding="utf-8")) if foreign_path.exists() else {}
    )
    parts = build_foreign_parts(cmu, foreign_dict)
    print(f"foreign_parts: {len(parts)}")
    for must in ("git", "hub", "bay", "star", "p"):
        assert must in parts or must in foreign_dict, f"parca kapsama disinda: {must!r}"

    if check:
        print("check ok")
        return

    out_dir.mkdir(exist_ok=True)
    (out_dir / "known_words.json").write_text(
        json.dumps(known, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    (out_dir / "english_i.json").write_text(
        json.dumps(english_i, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    (out_dir / "foreign_parts.json").write_text(
        json.dumps(parts, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    print(f"yazildi: {out_dir / 'known_words.json'}")
    print(f"yazildi: {out_dir / 'english_i.json'}")
    print(f"yazildi: {out_dir / 'foreign_parts.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    main(check=ap.parse_args().check)

"""Text frontend: raw Turkish text -> spoken-form lowercase characters -> integer ids.

`normalize.py` and `foreign.py` (+ data/) are Patientdesk's serving normaliser (turkish-tts @ 3ed2e60),
shipped here verbatim under this repo's Apache-2.0 licence. The model reads
turkish_lower(normalize_for_cased_model(raw)): numbers, dates, amounts, initialisms, addresses and
English/brand words expanded to how a Turkish speaker says them, then lowercased the Turkish way.
"""
from __future__ import annotations

import re
import unicodedata

from .normalize import normalize_for_cased_model, turkish_lower

PAD, BOS, EOS = 0, 1, 2
LETTERS = "abcçdefgğhıijklmnoöpqrsştuüvwxyz"
PUNCT = " .,?!'-:;"
SYMBOLS = ["<pad>", "<bos>", "<eos>"] + list(PUNCT) + list(LETTERS) + list("âîû")
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}
_FOLD = str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'", "–": "-", "—": "-", "…": ".", " ": " ", "\t": " ", "\n": " "})
_DROP = re.compile(r"[^abcçdefgğhıijklmnoöpqrsştuüvwxyzâîû .,?!'\-:;]")
_SPACES = re.compile(r"\s+")


def to_model_text(raw: str) -> str:
    """Raw text -> the exact character string the model is trained on."""
    t = normalize_for_cased_model(unicodedata.normalize("NFC", raw))
    return clean_model_text(turkish_lower(t))


def clean_model_text(t: str) -> str:
    """Last pass over already-normalised lowercase text: fold quote/dash variants, drop anything
    outside the symbol set, collapse spaces."""
    t = t.translate(_FOLD)
    t = _DROP.sub(" ", t)
    t = re.sub(r"\s+([.,?!:;])", r"\1", t)
    return _SPACES.sub(" ", t).strip()


def encode(model_text: str) -> list[int]:
    return [BOS] + [SYMBOL_TO_ID[c] for c in model_text if c in SYMBOL_TO_ID] + [EOS]


def word_ids(ids: list[int]) -> list[int]:
    """Word index per symbol (spaces and punctuation belong to the preceding word); BOS/EOS get -1."""
    out, w, prev_space = [], 0, False
    space = SYMBOL_TO_ID[" "]
    for i in ids:
        if i in (BOS, EOS, PAD):
            out.append(-1)
            continue
        if i == space:
            prev_space = True
            out.append(w)
            continue
        if prev_space:
            w += 1
            prev_space = False
        out.append(w)
    return out


__all__ = ["SYMBOLS", "SYMBOL_TO_ID", "PAD", "BOS", "EOS", "to_model_text", "clean_model_text", "encode", "word_ids",
           "normalize_for_cased_model", "turkish_lower"]

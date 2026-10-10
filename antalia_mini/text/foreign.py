"""English words and brand names inside Turkish text, written the way a Turkish speaker says them.

Alania-2 reads Turkish orthography. A word spelled the English way ("PatientDesk", "iPhone",
"ChatGPT", "LinkedIn") is read letter by letter as Turkish and comes out garbled, while the same
word respelled in Turkish letters ("Peyşınt Desk", "Ayfon", "Çet Ci Pi Ti", "Linktin") is read
cleanly (measured on production, 2026-10-05: Duyu transcribed the respelled renders back as the
original names). This module finds such words and respells them:

  - `is_english(word)`: ASCII letters only, in the CMU pronouncing dictionary, and not a word that
    Turkish text uses on its own (a frequency list from our Turkish domain text). Turkish words with
    no Turkish letters ("bank", "ten", "son", "ben") stay Turkish because Turkish uses them.
  - `split_camel(word)`: "PatientDesk" -> Patient, Desk; "iPhone" -> i, Phone; "ChatGPT" -> Chat, GPT.
  - `respell(word)`: CMU ARPAbet pronunciation -> Turkish letters (`_PHONES`); an all-caps part is
    read with English letter names (GPT -> ci pi ti); a lone lowercase "i"/"e" prefix is a letter.
  - `respell_foreign(text, lexicon)`: the text with every brand-like or English token replaced;
    `lexicon` (term -> spoken form) wins over the automatic respelling and keeps Turkish suffixes
    ("PatientDesk'e" -> "Peyşınt Desk'e").

Only words the model would otherwise misread are touched: a token is respelled when it has an
internal capital (a brand) or is English by `is_english`; a short all-caps token is left to the
normaliser's initialism rules.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

_DATA = Path(__file__).with_name("data")

# ARPAbet phoneme (stress stripped) -> Turkish letters. AH is "ı" unstressed and "a" stressed;
# ER is "ır" unstressed, "ör" stressed (bird -> börd).
_PHONES = {
    "AA": "a", "AE": "e", "AO": "o", "AW": "av", "AY": "ay", "EH": "e", "EY": "ey", "IH": "i", "IY": "i",
    "OW": "o", "OY": "oy", "UH": "u", "UW": "u",
    "B": "b", "CH": "ç", "D": "d", "DH": "d", "F": "f", "G": "g", "HH": "h", "JH": "c", "K": "k", "L": "l",
    "M": "m", "N": "n", "NG": "ng", "P": "p", "R": "r", "S": "s", "SH": "ş", "T": "t", "TH": "t", "V": "v",
    "W": "v", "Y": "y", "Z": "z", "ZH": "j",
}
# English letter names, in Turkish spelling, for all-caps parts of a brand (GPT, AI, TV).
LETTERS = {
    "A": "ey", "B": "bi", "C": "si", "D": "di", "E": "i", "F": "ef", "G": "ci", "H": "eyç", "I": "ay", "J": "cey",
    "K": "key", "L": "el", "M": "em", "N": "en", "O": "o", "P": "pi", "Q": "kyu", "R": "ar", "S": "es", "T": "ti",
    "U": "yu", "V": "vi", "W": "dabılyu", "X": "eks", "Y": "vay", "Z": "zi",
}

_TR_LETTERS = set("çğıöşüÇĞİÖŞÜ")
_WORD = re.compile(r"[A-Za-zÇĞİIÖŞÜçğıiöşü]+")
# a token and an optional Turkish suffix after an apostrophe: PatientDesk'e, iPhone'un
_TOKEN = re.compile(r"(?<![\w@./])([A-Za-z][A-Za-z0-9]*)(['’][a-zçğıöşü]+)?(?![\w@])")
_CAMEL = re.compile(r"[A-Z]{2,}(?=[A-Z][a-z]|\b|[0-9]|$)|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


@lru_cache(maxsize=1)
def _cmu() -> dict[str, list[str]]:
    import cmudict  # BSD-licensed CMU pronouncing dictionary
    return {w: prons[0] for w, prons in cmudict.dict().items()}


@lru_cache(maxsize=1)
def builtin_lexicon() -> dict[str, str]:
    """Reviewed readings (data/lexicon.json, term -> spoken form). A term mapped to itself is one
    the model already reads right as written and must not be respelled."""
    path = _DATA / "lexicon.json"
    if not path.exists():
        return {}
    return {k: v for k, v in json.loads(path.read_text(encoding="utf-8")).items() if not k.startswith("_")}


@lru_cache(maxsize=1)
def _turkish() -> dict[str, int]:
    path = _DATA / "tr_vocab.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _tr_lower(w: str) -> str:
    return w.replace("I", "ı").replace("İ", "i").lower()


def is_english(word: str) -> bool:
    """English by spelling and dictionary, and never seen in Turkish text. Anything Turkish text uses
    on its own ("okey", "bank", "son", "ben", "Instagram") is left for the model as written."""
    if not word.isascii() or not word.isalpha() or any(c in _TR_LETTERS for c in word):
        return False
    if word.lower() not in _cmu():
        return False
    return _turkish().get(_tr_lower(word), 0) == 0 and _turkish().get(word.lower(), 0) == 0


def looks_english(word: str) -> bool:
    """ASCII and in the English dictionary, whatever Turkish does with it: its capital I is the
    English "i", not the Turkish "ı" (Instagram, Intel, iPhone)."""
    return word.isascii() and word.isalpha() and word.lower() in _cmu()


# A brand Turkish text writes this often is one the model learned as written (WhatsApp, iPhone,
# YouTube, PayPal: measured clean on production 2026-10-05); rarer ones (ChatGPT, GitHub) are not.
_KNOWN_BRAND = 1000


def known_in_turkish(word: str) -> bool:
    return _turkish().get(word.lower(), 0) >= _KNOWN_BRAND


def has_inner_capital(word: str) -> bool:
    return any(c.isupper() for c in word[1:]) and any(c.islower() for c in word)


def split_camel(word: str) -> list[str]:
    return _CAMEL.findall(word)


def _from_phones(phones: list[str], written: str = "") -> str:
    # the American "o" of spot, Dropbox, John (AA) is read "o" in Turkish when the word is written with o
    aa = "o" if "o" in written.lower() and "a" not in written.lower() else "a"
    out = []
    for p in phones:
        base, stress = p.rstrip("012"), p[-1] if p[-1].isdigit() else ""
        if base == "AA":
            out.append(aa)
        elif base == "AH":
            out.append("a" if stress == "1" else "ı")
        elif base == "ER":
            out.append("ör" if stress == "1" else "ır")
        else:
            out.append(_PHONES.get(base, ""))
    s = "".join(out)
    s = re.sub(r"ng(?=[kg])", "n", s)     # "ng" before k/g is the n of "link", "bank"
    return re.sub(r"(.)\1+", r"\1", s)   # doubled letters read as one in Turkish spelling


def respell_part(part: str) -> str | None:
    """One camel part or plain word in Turkish letters, or None when unknown."""
    if part.isdigit():
        return part
    if part.isupper() and len(part) <= 5:
        return " ".join(LETTERS[c].capitalize() if len(part) > 1 else LETTERS[c] for c in part)
    if part in ("i", "e") and len(part) == 1:
        return LETTERS[part.upper()]
    phones = _cmu().get(part.lower())
    return _from_phones(phones, part) if phones else None


def respell(word: str) -> str | None:
    """The whole token respelled, parts joined ("Peyşınt Desk"; "Ayfon" for i+Phone), or None."""
    parts = split_camel(word) if has_inner_capital(word) else [word]
    spoken = []
    for part in parts:
        r = respell_part(part)
        if r is None:
            return None
        spoken.append(r)
    # capitals follow the written word, the Turkish way (i -> İ)
    up = lambda t: ("İ" + t[1:]) if t[:1] == "i" else (t[:1].upper() + t[1:])
    spoken = [up(sp) if (p[:1].isupper() or (k == 0 and word[:1].isupper())) else sp for k, (p, sp) in enumerate(zip(parts, spoken))]
    # a one-letter prefix ("i", "e") fuses with what follows, as people say it: Ayfon, İmeyl;
    # and a part of one or two letters fuses with the part before it: Linked+In -> Linktin
    merged: list[str] = []
    for k, (p, sp) in enumerate(zip(parts, spoken)):
        if merged and (len(p) <= 2 and not p.isupper() or (k == 1 and parts[0] in ("i", "e"))):
            merged[-1] = merged[-1] + _tr_lower(sp)
        else:
            merged.append(sp)
    text = " ".join(merged)
    return up(text) if has_inner_capital(word) or word[:1].isupper() else text


# Spans read by the normaliser's own rules and never respelled: e-mail addresses, URLs, bare domains.
_PROTECTED = re.compile(
    r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"
    r"|(?:https?://|www\.)\S+"
    r"|\b[\w-]+(?:\.[\w-]+)*\.(?:com|net|org|ai|io|co|app|dev|gov|edu|tr|me|info)(?:\.[a-z]{2})?\b",
    re.IGNORECASE,
)


def address_terms(text: str, lexicon: dict[str, str]) -> dict[str, str]:
    """The lexicon entries whose term occurs inside an e-mail or web address of `text`. The
    normaliser spells an address out word by word, so these are applied again afterwards; nothing
    else is, or a spoken form that contains its own term ("Ece" -> "Ece Hanım") would apply twice."""
    spans = [m.group(0) for m in _PROTECTED.finditer(text)]
    if not spans:
        return {}
    return {
        term: say for term, say in lexicon.items()
        if any(re.search(rf"(?<![^\W_]){re.escape(term)}(?![^\W_])", span, re.IGNORECASE) for span in spans)
    }


def respell_foreign(text: str, lexicon: dict[str, str] | None = None, auto: bool = True) -> str:
    """Replace brand-like and English tokens with their spoken Turkish spelling, outside e-mail
    addresses and web addresses. An e-mail address is lowercased the English way (its capital I is
    an English i: ILKER.SAHIN85@... reads "ilker"), then left to the normaliser."""
    out, last = [], 0
    for m in _PROTECTED.finditer(text):
        out.append(_respell_span(text[last:m.start()], lexicon, auto))
        span = m.group(0)
        out.append(span.lower() if "@" in span else span)
        last = m.end()
    out.append(_respell_span(text[last:], lexicon, auto))
    return "".join(out)


def _respell_span(text: str, lexicon: dict[str, str] | None, auto: bool) -> str:
    """`lexicon` maps a written term (case-insensitive, whole token, may be several words) to how it
    is said; a lexicon term is never respelled automatically. A term mapped to itself ("Getir",
    "WhatsApp") is one the model reads right as written: it only shields the term from the automatic
    respelling and leaves the text as it was, so "getir" the verb keeps its case. Turkish suffixes
    after an apostrophe are kept."""
    if not lexicon:
        return _auto(text) if auto else text
    terms = sorted(lexicon, key=len, reverse=True)
    pattern = re.compile(r"(?<!\w)(" + "|".join(re.escape(t) for t in terms) + r")(?=['’]|[^\w]|$)", re.IGNORECASE)
    # The pattern matches ignoring case, which also pairs the dotted and dotless i of both cases
    # ("İnstagram", "ınstagram", "Nıke"), so the lookup key folds them the same way.
    key = lambda t: t.replace("İ", "i").replace("I", "i").replace("ı", "i").lower()
    by_lower = {key(t): lexicon[t] for t in terms}
    out, last = [], 0
    for m in pattern.finditer(text):
        out.append(_auto(text[last:m.start()]) if auto else text[last:m.start()])
        term = m.group(1)
        say = by_lower.get(key(term))
        if say is None:                     # a case pairing the key does not fold: leave the words alone
            out.append(term)
            last = m.end()
            continue
        kept = key(say) == key(term)
        out.append(_english_i(term) if kept and auto else (term if kept else say))
        last = m.end()
    out.append(_auto(text[last:]) if auto else text[last:])
    return "".join(out)


def _english_i(word: str) -> str:
    """The English i of Instagram, Intel: a capital I in an English word is İ, not the Turkish ı."""
    if word.isupper() and len(word) <= 5:
        return word                                # initialisms: the normaliser's rules
    return word.replace("I", "İ") if "I" in word and looks_english(word) else word


def _auto(text: str) -> str:
    def sub(m: re.Match) -> str:
        word, suffix = m.group(1), m.group(2) or ""
        if word.isupper() and len(word) <= 5:
            return m.group(0)                      # initialisms: the normaliser's rules
        if known_in_turkish(word) or not (has_inner_capital(word) or is_english(word)):
            return _english_i(word) + suffix
        r = respell(word)
        return (r if r is not None else _english_i(word)) + suffix

    return _TOKEN.sub(sub, text)

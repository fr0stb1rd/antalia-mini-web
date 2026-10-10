from __future__ import annotations

# ruff: noqa: RUF001, RUF002, RUF003 -- Turkish orthography intentionally uses dotless i
import re
import unicodedata

_WHITESPACE = re.compile(r"\s+")
# A digit glued to a letter ("A4", "mp3", "COVID19", "5kg") escapes every \b-anchored number rule
# below, so the digit reaches the model raw. The v3 vocabulary shows the damage: it holds 0-6 and 9
# only because such leaks occurred in training text, and never saw 7 or 8 at all.
_LETTER_THEN_DIGIT = re.compile(r"(?<=[^\W\d_])(?=\d)")
_DIGIT_THEN_LETTER = re.compile(r"(?<=\d)(?=[^\W\d_])")
# Once split, the letter half of "A4" or "5G" is a lone capital that the 2-6 letter initialism
# rule would leave as a bare grapheme; it is read by name, like the multi-letter case.
_LONE_LETTER_BY_DIGIT = re.compile(r"\b([A-ZÇĞİÖŞÜQWX])(?= \d)|(?<=\d )([A-ZÇĞİÖŞÜQWX])\b")
_COMPARISON_PUNCTUATION = re.compile(r"[^a-zçğıöşü0-9']+")
_CTC_PUNCTUATION = re.compile(r"[^a-zçğıöşüqwx ]+")
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'"})
_QUOTES = str.maketrans({char: None for char in '"“”„«»'})
_TURKISH_CASE = str.maketrans("Iİ", "ıi")
# Addresses are read out piece by piece, so they must be caught before the digit and abbreviation
# rules tear them apart ("mp3.com", "dr.com").
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_URL = re.compile(
    r"\b(?:https?://)?(?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|net|org|gov|edu|info|io|ai|tr)\b(?:/[\w./-]*)?",
    re.IGNORECASE,
)
# Read the way Turkish speakers dictate them; kept in step with the serving normalizer.
_ADDRESS_WORDS = {
    "www": "dabılyu dabılyu dabılyu",
    "com": "kom",
    "net": "net",
    "org": "org",
    "gov": "gov",
    "info": "info",
    "gmail": "cimeyl",
    "hotmail": "hotmeyl",
    "outlook": "autluk",
    "yahoo": "yahu",
    "icloud": "ayklaud",
}
# A hyphen or dash between two numbers is a range, unless it is one link of a longer chain
# (a phone or product number such as 0532-123-45-67).
_RANGE = re.compile(r"(?<![\d\-–])(\d[\d.:,]*)\s*[-–]\s*(\d[\d.:,]*)(?![\d\-–])")
_PHONE = re.compile(r"(?<!\d)(\+90[\s-]?)?(0?)\(?(\d{3})\)?[\s-]?(\d{3})[\s-]?(\d{2})[\s-]?(\d{2})(?!\d)")
_VOWELS = set("aeıioöuüAEIİOÖUÜ")
_PER_UNIT = {"sn": "saniye", "sa": "saat", "dk": "dakika"}
# All-caps words the rules below would get wrong: short words read as words although the pronounceability
# test would spell them, and brand readings. Ordinary initialisms take plain letter names joined into one
# word ("SGK" segeka, "TBMM" tebememe, "ATM" ateme); every entry here needs a native speaker's sign-off.
_CAPS_LEXICON = {
    "MR": "emar", "SMS": "esemes", "NTV": "entivi", "YÖK": "yök", "BİM": "bim", "ŞOK": "şok", "COVID": "kovid",
    "MEB": "meb", "İTÜ": "itü", "IBAN": "iban", "İETT": "ietete", "ALO": "alo", "MİT": "mit", "OK": "okey",
    # H is "he" in initialisms ("THY" teheye), but these two are said with "ha"
    "TSH": "teseha", "MHRS": "meharese", "HES": "hes",
    # English-origin initialisms keep their English letter names
    "CEO": "siio", "PR": "piar", "DJ": "dijey", "TV": "tivi", "FBI": "efbiay", "CNN": "sienen", "VIP": "viaypi",
    "HD": "heyçdi",
}
_CAPS_WORD = re.compile(r"\b[A-ZÇĞİÖŞÜQWX]{2,}\b")
_WORD = re.compile(r"\b[^\W\d_]{2,}\b")
_ONES = ("sıfır", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz")
_TENS = ("", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan")
_MONTHS = {
    1: "ocak",
    2: "şubat",
    3: "mart",
    4: "nisan",
    5: "mayıs",
    6: "haziran",
    7: "temmuz",
    8: "ağustos",
    9: "eylül",
    10: "ekim",
    11: "kasım",
    12: "aralık",
}
_ABBREVIATIONS = {
    "dr.": "doktor",
    "prof.": "profesör",
    "vb.": "ve benzeri",
    "vs.": "vesaire",
    "örn.": "örneğin",
    "no.": "numara",
    "mg": "miligram",
    "kg": "kilogram",
    "gr": "gram",
    "km": "kilometre",
    "cm": "santimetre",
    "mm": "milimetre",
    "lt": "litre",
    "ml": "mililitre",
    "kb": "kilobayt",
    "mb": "megabayt",
    "gb": "gigabayt",
    "tb": "terabayt",
    "tl": "lira",
    # titles, read before a name
    "doç.": "doçent",
    "uzm.": "uzman",
    "op.": "operatör",
    "yrd.": "yardımcı",
    "öğr. üyesi": "öğretim üyesi",
    "öğr. gör.": "öğretim görevlisi",
    "dt.": "diş hekimi",
    "dyt.": "diyetisyen",
    "psk.": "psikolog",
    "fzt.": "fizyoterapist",
    "hemş.": "hemşire",
    "ecz.": "eczacı",
    "av.": "avukat",
    "müh.": "mühendis",
    "hz.": "hazreti",
    "bkz.": "bakınız",
    "vd.": "ve diğerleri",
    # addresses and companies: the abbreviation stands for the possessive form of the name before it
    "mah.": "mahallesi",
    "cad.": "caddesi",
    "cd.": "caddesi",
    "sok.": "sokak",
    "sk.": "sokak",
    "bul.": "bulvarı",
    "blv.": "bulvarı",
    "apt.": "apartmanı",
    "ltd. şti.": "limited şirketi",
    "şti.": "şirketi",
}
# Titles and address words stand before a name or number; at the end of a sentence the same letters are a word
# ("düzene sok.") or something else ("30 DT.").
_BEFORE_A_WORD = {
    "doç.", "uzm.", "op.", "yrd.", "dt.", "dyt.", "psk.", "fzt.", "hemş.", "ecz.", "av.", "müh.", "hz.", "mah.",
    "cad.", "cd.", "sok.", "sk.", "bul.", "blv.", "apt.",
}
# Units read only right after a number, since a bare "m" or "g" is far more often a letter.
_NUMBER_UNITS = {
    "m": "metre", "g": "gram", "cc": "sisi", "mcg": "mikrogram", "μg": "mikrogram", "IU": "ünite",
    "kcal": "kilokalori", "mmHg": "milimetre cıva", "dk": "dakika", "sn": "saniye", "sa": "saat",
    "ms": "milisaniye", "dl": "desilitre",
}
_AREA_UNITS = {"m": "metre", "km": "kilometre", "cm": "santimetre"}
# Capitalised words an ordinal is written before ("2. Dünya Savaşı", "5. Uluslararası Kongre"). Before any
# other capitalised word the full stop ends a sentence: "Sıra numaranız 12. Lütfen bekleyin."
_ORDINAL_HEADS = (
    "Dünya|Lig|Ordu|Kolordu|Tümen|Dönem|Meclis|Kurultay|Kongre|Uluslararası|Ulusal|Olağan|Olağanüstü|Geleneksel"
    "|Yıl|Yüzyıl|Sınıf|Kat|Blok|Etap|Bölge|Asliye|Ağır|Sulh|İcra|Aile|Selim|Murat|Murad|Mehmet|Mehmed|Mahmut"
    "|Mahmud|Ahmet|Ahmed|Osman|Bayezid|Beyazıt|Süleyman|Mustafa|Abdülhamid|Abdülmecid|Louis|Henry|Napolyon"
)
_WEEKDAYS = {
    "Pzt": "pazartesi", "Sal": "salı", "Çar": "çarşamba", "Per": "perşembe", "Cum": "cuma", "Cmt": "cumartesi",
    "Paz": "pazar",
}
_LETTER_NAMES = {
    "A": "a",
    "B": "be",
    "C": "ce",
    "Ç": "çe",
    "D": "de",
    "E": "e",
    "F": "fe",
    "G": "ge",
    "Ğ": "yumuşak ge",
    "H": "he",
    "I": "ı",
    "İ": "i",
    "J": "je",
    "K": "ke",
    "L": "le",
    "M": "me",
    "N": "ne",
    "O": "o",
    "Ö": "ö",
    "P": "pe",
    "R": "re",
    "S": "se",
    "Ş": "şe",
    "T": "te",
    "U": "u",
    "Ü": "ü",
    "V": "ve",
    "Y": "ye",
    "Z": "ze",
    "Q": "kü",
    "W": "dabılyu",
    "X": "iks",
}
# Letter names inside a spelled initialism: K is "ka" there ("KDV" kadeve, "KPSS" kapesese) while H keeps its
# alphabet name ("THY" teheye, "CHP" cehepe). A lone letter takes its alphabet name ("K blok" ke blok) unless it
# is a name initial or a vitamin ("Ahmet K." ka, "K vitamini" ka).
# A plain "I" in an initialism is nearly always a foreign "i" ("ITU", "AIK"); Turkish initialisms that need
# the dotted letter write "İ" ("İTÜ", "İETT").
_INITIALISM_LETTERS = {**_LETTER_NAMES, "K": "ka", "I": "i"}
_ROMAN = "(?:XX|XIX|XVIII|XVII|XVI|XV|XIV|XIII|XII|XI|IX|VIII|VII|VI|IV|III|II)"
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10}
_ORDINAL_EXCEPTIONS = {
    "bir": "birinci",
    "iki": "ikinci",
    "üç": "üçüncü",
    "dört": "dördüncü",
    "beş": "beşinci",
    "altı": "altıncı",
    "yedi": "yedinci",
    "sekiz": "sekizinci",
    "dokuz": "dokuzuncu",
    "on": "onuncu",
    "yirmi": "yirminci",
    "otuz": "otuzuncu",
    "kırk": "kırkıncı",
    "elli": "ellinci",
    "altmış": "altmışıncı",
    "yetmiş": "yetmişinci",
    "seksen": "sekseninci",
    "doksan": "doksanıncı",
    "yüz": "yüzüncü",
    "bin": "bininci",
    "milyon": "milyonuncu",
    "milyar": "milyarıncı",
}
# Words a number, amount or unit expands to. A suffix written after the digits or symbol ("14:30'da",
# "2 GB'lık") is joined to them without the apostrophe, the way it is spoken ("otuzda", "gigabaytlık").
_SPOKEN_STEMS = sorted(
    {
        *_ONES,
        *filter(None, _TENS),
        "yüz",
        "bin",
        "milyon",
        "milyar",
        "trilyon",
        "lira",
        "kuruş",
        "dolar",
        "avro",
        "sent",
        "sterlin",
        "peni",
        "saniye",
        "dakika",
        "saat",
        "buçuk",
        "metrekare",
        "metreküp",
        "kilometrekare",
        "santimetrekare",
        "kilometreküp",
        "santimetreküp",
        *(v.split()[-1] for v in _ABBREVIATIONS.values()),
        *(v.split()[-1] for v in _NUMBER_UNITS.values()),
        *_ORDINAL_EXCEPTIONS.values(),
    },
    key=len,
    reverse=True,
)
_SPOKEN_STEM_SUFFIX = re.compile(
    r"\b(" + "|".join(map(re.escape, _SPOKEN_STEMS)) + r"|\w+(?:inci|ıncı|üncü|uncu|nci|ncı|ncü|ncu))'([a-zçğıöşü]+)\b"
)


def turkish_lower(text: str) -> str:
    # Strip U+0307: lowercasing İ can yield i + combining dot above, and some
    # source transcripts already contain the decomposed form. The dot is silent.
    return text.translate(_TURKISH_CASE).lower().replace("\u0307", "")


def normalize_orthography(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).translate(_APOSTROPHES).translate(_QUOTES)
    normalized = normalized.replace("''", "")  # two apostrophes typed as a quotation mark
    return _WHITESPACE.sub(" ", normalized).strip()


def normalize_for_asr_comparison(text: str) -> str:
    normalized = turkish_lower(normalize_orthography(text))
    normalized = _COMPARISON_PUNCTUATION.sub(" ", normalized)
    return _WHITESPACE.sub(" ", normalized).strip()


def normalize_for_scoring(text: str) -> str:
    """ASR-comparison normalization that also spells out digits Whisper may emit.

    Whisper frequently transcribes spoken Turkish numerals as digits ("3700" for
    "üç bin yedi yüz"), which inflates CER/WER against word-form targets. Scoring
    therefore spells percent signs, decimals, thousands groups, and digit runs
    before comparison. Word-form text passes through unchanged.
    """
    normalized = normalize_orthography(text)
    normalized = normalized.replace("%", " yüzde ")
    normalized = re.sub(r"(?<=\d)\.(?=\d{3}\b)", "", normalized)
    normalized = re.sub(
        r"\d+,\d+",
        lambda match: f" {_decimal_to_words(match[0])} ",
        normalized,
    )
    normalized = re.sub(r"\d+", lambda match: f" {number_to_words(int(match[0]))} ", normalized)
    return normalize_for_asr_comparison(normalized)


def normalize_for_ctc_alignment(text: str) -> str:
    normalized = normalize_for_asr_comparison(text).replace("'", " ")
    normalized = _CTC_PUNCTUATION.sub(" ", normalized)
    return _WHITESPACE.sub(" ", normalized).strip()


def normalize_for_model(text: str) -> str:
    # The typographic minus sign (U+2212) is read like "-": "−5 derece" -> "eksi beş derece" (native-confirmed 2026-10-07).
    normalized = normalize_orthography(text.replace("\u2212", "-"))
    normalized = _lower_if_shouted(normalized)
    normalized = _replace_roman(normalized)
    normalized = re.sub(r"'([A-ZÇĞİÖŞÜ]+)\b", lambda match: "'" + turkish_lower(match[1]), normalized)  # "ABD'DE"
    # "Kimlik No", "takip no: 5" name the number of the thing before them; a bare "No: 12" is just "numara"
    normalized = re.sub(
        r"(?<!\bsokak )(?<!\bsokağı )(?<!\bcadde )(?<!\bcaddesi )(?<!\bbulvarı )(?<!\bmahallesi )(?<!\bapartmanı )"
        r"(?<=[^\W\d_] )no\b\.?:?(?=\s)",
        "numarası",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"\bno\s*[:.]\s*(?=\d)", "numara ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bD\s*:\s*(?=\d)", "daire ", normalized)  # "No: 12 D: 5"
    normalized = re.sub(r"\bTel\s*:", "telefon:", normalized)
    normalized = re.sub(r"\bs\.\s*(?=\d)", "sayfa ", normalized)
    normalized = _EMAIL.sub(_replace_address, normalized)
    normalized = _URL.sub(_replace_address, normalized)
    normalized = _replace_symbols(normalized)
    normalized = re.sub(r"\bTR(\d{2})((?:\s?\d{4}){5})\s?(\d{2})\b", _replace_iban, normalized)
    normalized = re.sub(r"\b(?:[A-ZÇĞİÖŞÜ]\.){2,}", _replace_dotted_initialism, normalized)  # "T.C.", "A.B.D."
    normalized = re.sub(r"\bpH\b", "pe ha", normalized)
    normalized = re.sub(r"\bH2O\b", "haş iki o", normalized)
    normalized = re.sub(r"\bH\.\s*pylori\b", "ha pilori", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bX-?ray\b", "iksrey", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b(hane(?:si|leri)?)\s+(\d{2,6})\b",
                        lambda match: f"{match[1]} {_digit_sequence_to_words(match[2])}", normalized)  # card digits
    normalized = re.sub(r"\bHbA1c\b", "ha be a bir ce", normalized)
    normalized = re.sub(r"\b(AB|A|B|0)\s?Rh\s?\(?([+-])\)?", _replace_blood_group, normalized)
    normalized = re.sub(
        r"\b(\d{2,3})/(\d{2,3})\s*mmHg\b",
        lambda match: f"{_with_dative(number_to_words(int(match[1])))} {number_to_words(int(match[2]))} milimetre cıva",
        normalized,
    )  # blood pressure "120/80" is "yüz yirmiye seksen"
    normalized = re.sub(r"(\bgünde\s+)?\b(\d+)\s?[xX×]\s?(\d+)\b", _replace_times, normalized)
    # rates read the way they are said: "saatte yüz yirmi kilometre", "kilogram başına beş miligram";
    # km/s is the hourly speed as Turkish signs write it: "90 km/s" -> "saatte doksan kilometre" (native-confirmed 2026-10-07)
    normalized = re.sub(
        r"(?<![\d.,])(\d+(?:,\d+)?)\s?km/(?:sa|s|h)\b", lambda match: f"saatte {_numeric_to_words(match[1])} kilometre",
        normalized,
    )
    normalized = re.sub(
        r"(?<![\d.,])(\d+(?:,\d+)?)\s?(mg|mcg|g|ml)/kg\b",
        lambda match: f"kilogram başına {_numeric_to_words(match[1])} {_DOSE_UNITS[match[2]]}",
        normalized,
    )
    normalized = re.sub(
        r"(?<=\d) ?(km|cm|m)([23])(?!\w)",
        lambda match: " " + _AREA_UNITS[match[1]] + ("kare" if match[2] == "2" else "küp"),
        normalized,
    )  # "m²" arrives as "m2" after NFKC
    normalized = re.sub(
        r"(?<=\d) ?(" + "|".join(map(re.escape, _NUMBER_UNITS)) + r")(?![\w/])",
        lambda match: " " + _NUMBER_UNITS[match[1]],
        normalized,
    )
    normalized = re.sub(r"(?<=\d) L\b(?![/'’])(?!\s*\d)", " litre", normalized)  # "2 L su"; not "108L" or "33 L 99"
    normalized = re.sub(
        r"\b(" + "|".join(_WEEKDAYS) + r")\b\.?(?=\s*(?:[-–,/]|\d|$))", lambda match: _WEEKDAYS[match[1]], normalized
    )
    normalized = re.sub(
        r"\b(pazartesi|salı|çarşamba|perşembe|cuma|cumartesi|pazar)\s*[-–]\s*"
        r"(pazartesi|salı|çarşamba|perşembe|cuma|cumartesi|pazar)\b",
        r"\1 ile \2",
        normalized,
        flags=re.IGNORECASE,
    )
    # Amounts, before the code-letter rule below would spell a glued "50TL": "₺12,50" and "12,50 TL" are both
    # "on iki lira elli kuruş".
    normalized = re.sub(r"([₺$€£])\s*((?:\d{1,3}(?:\.\d{3})+)|\d+)(?:[,.](\d{1,2}))?", _replace_currency, normalized)
    normalized = re.sub(
        r"(?<![\d.,])((?:\d{1,3}(?:\.\d{3})+)|\d+)(?:,(\d{1,2}))?\s?(?:([₺$€£])|(TL|USD|EUR|GBP)\b)",
        lambda match: _currency_words(match[3] or _CURRENCY_CODES[match[4]], match[1], match[2]),
        normalized,
    )
    # letters inside a code ("52EB4297", "34KAB123") are read one by one, never joined into a word
    normalized = re.sub(
        r"(?<=\d)[A-ZÇĞİÖŞÜQWX]{1,6}(?=\d|\b)|\b[A-ZÇĞİÖŞÜQWX]{1,6}(?=\d)|(?<=\d )[A-ZÇĞİÖŞÜQWX]{2,6}(?= \d)",
        lambda match: match[0] if match[0] in _CAPS_LEXICON or _is_pronounceable(match[0])
        else " " + " ".join(_INITIALISM_LETTERS[char] for char in match[0]) + " ",
        normalized,
    )
    normalized = _DIGIT_THEN_LETTER.sub(" ", _LETTER_THEN_DIGIT.sub(" ", normalized))
    normalized = re.sub(r"(?<=[^\W\d_])-(?=\d)", " ", normalized)  # "COVID-19"
    normalized = _LONE_LETTER_BY_DIGIT.sub(lambda match: _LETTER_NAMES[match[1] or match[2]], normalized)
    normalized = re.sub(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b", _replace_date, normalized)
    normalized = re.sub(
        r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", lambda match: _replace_date_parts(match[3], match[2], match[1]), normalized
    )
    normalized = re.sub(r"%\s*(\d+(?:[,.]\d+)?)", lambda match: f"yüzde {_numeric_to_words(match[1])}", normalized)
    normalized = re.sub(
        r"(?<![\d.,])(\d+(?:,\d+)?)\s?%", lambda match: f"yüzde {_numeric_to_words(match[1])}", normalized
    )
    normalized = _PHONE.sub(_replace_phone, normalized)
    normalized = _RANGE.sub(_replace_range, normalized)
    normalized = re.sub(r"(?<![\w.,:)\]])-(?=\d)", "eksi ", normalized)
    normalized = re.sub(r"\b(\d{1,2}):([0-5]\d):([0-5]\d)\b", _replace_duration, normalized)
    normalized = re.sub(r"\b([01]?\d|2[0-3])[:.]([0-5]\d)\b", _replace_time, normalized)
    normalized = re.sub(r"\b(\d{1,2})/(\d{1,2})\b", _replace_slash_pair, normalized)
    normalized = re.sub(r"\b(\d+)/(\d+)\b", r"\1 bölü \2", normalized)  # "2026/1847"
    normalized = re.sub(r"\b\d{1,3}(?:\.\d{3})+(?:,\d+)?\b", lambda match: _numeric_to_words(match[0]), normalized)
    normalized = re.sub(r"\b(\d+)[,.](\d+)\b", lambda match: _decimal_to_words(match[0]), normalized)
    # "12'nci" and "2.si": the ordinal suffix is written after the digits, the rest of the word follows.
    normalized = re.sub(
        r"\b(\d+)'?(?:inci|ıncı|üncü|uncu|nci|ncı|ncü|ncu)\b",
        lambda match: _ordinal_to_words(int(match[1])),
        normalized,
    )
    normalized = re.sub(r"\b(\d+)\.(?=[a-zçğıöşü])", lambda match: _ordinal_to_words(int(match[1])), normalized)
    # "3. kat" is an ordinal; a number that ends the text ("Sıra numaranız 12.") is not.
    normalized = re.sub(
        r"\b([1-9]\d*)\.(?=\s+(?:[a-zçğıöşü]|(?:" + _ORDINAL_HEADS + r")\b))",
        lambda match: _ordinal_to_words(int(match[1])),
        normalized,
    )
    normalized = re.sub(r"\b\d{5,}\b", lambda match: _digit_sequence_to_words(match[0]), normalized)
    normalized = re.sub(r"\b0\d+\b", lambda match: _digit_group_to_words(match[0]), normalized)
    normalized = re.sub(r"\b\d+\b", lambda match: number_to_words(int(match[0])), normalized)
    normalized = re.sub(r"\bno\.'?s[ıu]\b", "numarası", normalized, flags=re.IGNORECASE)
    # Rates: "MB/sn", "km/sa". "Sn." is "sayın" only before a name, never after a slash.
    normalized = re.sub(
        r"(?<=\w)/(sn|sa|dk)\b", lambda match: " bölü " + _PER_UNIT[match[1].lower()], normalized, flags=re.I
    )
    normalized = re.sub(r"\b(mg|kg|g|km|m|cm|ml)/(kg|g|m|l|lt|ml|saat|gün)\b", r"\1 bölü \2", normalized)
    normalized = re.sub(r"\bSn\.\s*(?=[A-ZÇĞİÖŞÜ])", "sayın ", normalized)
    for abbreviation, expansion in sorted(_ABBREVIATIONS.items(), key=lambda item: len(item[0]), reverse=True):
        if abbreviation.endswith("."):
            # The full stop already ends the abbreviation, so "Dr.Ayşe" (no space) expands too.
            pattern, replacement = rf"(?<!\w){re.escape(abbreviation)}\s*(?=\w)", f"{expansion} "
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
            if abbreviation in _BEFORE_A_WORD:
                continue  # "Adresimi düzene sok.", "30 DT." end a sentence
            pattern = rf"(?<!\w){re.escape(abbreviation)}"
        else:
            pattern = rf"(?<!\w){re.escape(abbreviation)}(?!\w)"
        normalized = re.sub(pattern, expansion, normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b[A-ZÇĞİÖŞÜQWX]{2,6}\b", _replace_initialism, normalized)
    # A lone capital is a letter ("D vitamini", "C blok", "Ahmet K."); the full stop after an initial goes
    # when the sentence carries on.
    normalized = re.sub(
        r"(?<![\w.'-])([A-ZÇĞİÖŞÜQWX])(\.(?=\s+[a-zçğıöşü]))?(?![\w-])", _replace_lone_letter, normalized
    )
    normalized = _SPOKEN_STEM_SUFFIX.sub(_attach_suffix, normalized)
    normalized = re.sub(r"\s+/\s+", ", ", normalized)  # "Tel: ... / Faks: ..."
    normalized = re.sub(r"(?<=[^\W\d_])/(?=[^\W\d_])", " ", normalized)  # "Kadıköy/İstanbul", "ve/veya"
    normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
    normalized = re.sub(r"^[\s,]+|(?<=[,.;:!?])\s*,", "", normalized)  # a bullet that opened the text
    return _WHITESPACE.sub(" ", normalized).strip()


def normalize_for_cased_model(text: str, lexicon: dict[str, str] | None = None, foreign: bool = True) -> str:
    """Model text for bases that keep letter case (VoxCPM2): expanded, with Turkish I and İ resolved.

    A cased multilingual model lowercases or reads a capital I the English way, so "Iğdır" came out
    as "iğdir". Words left in capitals are lowercased the Turkish way, and every remaining I and İ
    becomes ı and i; other capitals stay, since they carry sentence and name cues.

    First, brand names and English words are written the way they are said (`foreign.py`): the
    caller's `lexicon` (term -> spoken form), then the built-in reviewed lexicon, then, with
    `foreign`, an automatic respelling of mixed-case brands and English words Turkish text never
    uses. An English word kept as written gets the English "i" for its capital I (Instagram).
    """
    from .foreign import address_terms, builtin_lexicon, respell_foreign

    merged = {**builtin_lexicon(), **(lexicon or {})}
    in_addresses = address_terms(text, merged)
    text = respell_foreign(text, merged, auto=foreign)
    normalized = normalize_for_model(text)
    # addresses are spelled out word by word above; readings of the terms inside them apply to
    # those words too ("www.patientdesk.ai" -> "... nokta Peyşınt Desk nokta a i")
    if in_addresses:
        normalized = respell_foreign(normalized, in_addresses, auto=False)
    normalized = _CAPS_WORD.sub(lambda match: turkish_lower(match[0]), normalized)
    return normalized.translate(_TURKISH_CASE)


def number_to_words(number: int) -> str:
    if number < 0:
        return f"eksi {number_to_words(-number)}"
    if number < 10:
        return _ONES[number]
    if number < 100:
        tens, ones = divmod(number, 10)
        return _TENS[tens] if not ones else f"{_TENS[tens]} {_ONES[ones]}"
    if number < 1_000:
        hundreds, remainder = divmod(number, 100)
        prefix = "yüz" if hundreds == 1 else f"{_ONES[hundreds]} yüz"
        return prefix if not remainder else f"{prefix} {number_to_words(remainder)}"
    scales = ((1_000_000_000_000, "trilyon"), (1_000_000_000, "milyar"), (1_000_000, "milyon"), (1_000, "bin"))
    for scale, label in scales:
        if number >= scale:
            count, remainder = divmod(number, scale)
            prefix = label if scale == 1_000 and count == 1 else f"{number_to_words(count)} {label}"
            return prefix if not remainder else f"{prefix} {number_to_words(remainder)}"
    raise ValueError(f"number is too large to normalize: {number}")


def _replace_date(match: re.Match[str]) -> str:
    return _replace_date_parts(match[1], match[2], match[3], original=match[0])


def _replace_date_parts(day_text: str, month_text: str, year_text: str, original: str | None = None) -> str:
    day, month, year = int(day_text), int(month_text), int(year_text)
    if not 1 <= day <= 31 or month not in _MONTHS:
        return original if original is not None else f"{year_text}-{month_text}-{day_text}"
    return f"{number_to_words(day)} {_MONTHS[month]} {number_to_words(year)}"


_CURRENCY_CODES = {"TL": "₺", "USD": "$", "EUR": "€", "GBP": "£"}
_SYMBOLS = {"&": " ve ", "•": " , ", "▪": " , ", "►": " , ", "→": " , ", "⇒": " , "}


def _replace_symbols(text: str) -> str:
    """Drop what is only seen, not said: emoji, markdown emphasis, bullets, a hashtag's #."""
    text = re.sub(r"(?<=\d)\s?°\s?C?(?![^\W\d_])", " derece", text)
    text = re.sub(r"(?<=\d)\s?\+\s?(?=\d)|(?<=\s)\+(?=\s)", " artı ", text)
    text = re.sub(r"(?<=[\d\s])=(?=[\d\s])", " eşittir ", text)
    text = re.sub(r"(?<![\w&])#(?=[^\W\d_])", "heşteg ", text)
    text = re.sub(r"\*+|_{2,}", "", text)
    text = "".join(_SYMBOLS.get(char, char) for char in text)
    text = "".join(char for char in text if unicodedata.category(char) not in {"So", "Cs", "Co"})
    return text.replace("\ufe0f", "").replace("\u200d", "")


def _replace_iban(match: re.Match[str]) -> str:
    groups = [match[1], *match[2].split(), match[3]] if " " in match[2] else [match[1], match[2], match[3]]
    return "te re, " + ", ".join(_digit_sequence_to_words(group) for group in groups)


def _replace_dotted_initialism(match: re.Match[str]) -> str:
    return "".join(_INITIALISM_LETTERS[char] for char in match[0] if char != ".")


def _replace_blood_group(match: re.Match[str]) -> str:
    group = "sıfır" if match[1] == "0" else "".join(_INITIALISM_LETTERS[char] for char in match[1])
    return f"{group} eraj {'pozitif' if match[2] == '+' else 'negatif'}"


def _replace_range(match: re.Match[str]) -> str:
    # "2-1" with the larger number first is a score or a pair, not a range
    start, end = match[1], match[2]
    if re.fullmatch(r"\d{4}", start) and re.fullmatch(r"\d{2}", end) and int(end) == (int(start) + 1) % 100:
        return f"{start} {int(end)}"  # a season, "2003-04": "iki bin üç dört"
    if start.isdigit() and end.isdigit() and int(start) > int(end):
        return f"{start} {end}"
    return f"{start} ile {end}"


def _replace_lone_letter(match: re.Match[str]) -> str:
    letter = match[0][0]
    text, start, end = match.string, match.start(), match.end()
    following = text[end:end + 1]
    # "O" is the pronoun; "A," and "E!" are interjections; a lone "I" is a numeral or English
    if letter in "OI" or (letter in "AE" and following in {",", "!", "?"}):
        return match[0]
    initial = match[2] or following == "."
    vitamin = re.search(r"vitamin\w*\s+(?:\w+\s+ve\s+)?$", text[:start], re.IGNORECASE) or re.match(
        r"\s+vitamin", text[end:], re.IGNORECASE
    )
    return (_INITIALISM_LETTERS if initial or vitamin else _LETTER_NAMES)[letter]


_DOSE_UNITS = {"mg": "miligram", "mcg": "mikrogram", "g": "gram", "ml": "mililitre"}
_DOSE_WORDS = r"tablet|kapsül|şurup|ölçek|damla|doz|puf|kaşık|ampul|draje|saşe|fitil|iğne"


def _replace_times(match: re.Match[str]) -> str:
    """A dose is "günde iki defa bir tablet"; a size is "otuza kırk santimetre"."""
    daily, first, second = match[1], number_to_words(int(match[2])), number_to_words(int(match[3]))
    if daily or re.match(rf"\s*(?:{_DOSE_WORDS})", match.string[match.end():], re.IGNORECASE):
        return f"{daily or ''}{first} defa {second}"
    return f"{_with_dative(first)} {second}"


def _with_dative(words: str) -> str:
    """"iki" -> "ikiye", "otuz" -> "otuza", "dört" -> "dörde"."""
    if words.endswith("dört"):
        return words[:-1] + "de"
    last_vowel = next((char for char in reversed(words) if char in _VOWELS), "e")
    vowel = "a" if last_vowel in "aıou" else "e"
    return f"{words}{'y' if words[-1] in _VOWELS else ''}{vowel}"


def _replace_currency(match: re.Match[str]) -> str:
    symbol, whole, fraction = match.groups()
    return _currency_words(symbol, whole, fraction)


def _currency_words(symbol: str, whole: str, fraction: str | None) -> str:
    unit, subunit = {
        "₺": ("lira", "kuruş"),
        "$": ("dolar", "sent"),
        "€": ("avro", "sent"),
        "£": ("sterlin", "peni"),
    }[symbol]
    value = f"{number_to_words(int(whole.replace('.', '')))} {unit}"
    if fraction and int(fraction):
        value += f" {number_to_words(int(fraction.ljust(2, '0')))} {subunit}"
    return value


def _replace_phone(match: re.Match[str]) -> str:
    # Turkish numbers are read in their written groups, "sıfır beş yüz otuz iki, yüz yirmi üç, ...",
    # and the trunk zero is spoken.
    country, trunk, *groups = match.groups()
    spoken = ["artı doksan"] if country else []
    first = f"sıfır {_digit_group_to_words(groups[0])}" if trunk else _digit_group_to_words(groups[0])
    spoken.append(first)
    spoken.extend(_digit_group_to_words(group) for group in groups[1:])
    return ", ".join(spoken)


def _digit_group_to_words(value: str) -> str:
    """Read a digit group as a number, voicing each leading zero ("045" -> "sıfır kırk beş")."""
    stripped = value.lstrip("0")
    zeros = ["sıfır"] * (len(value) - len(stripped))
    return " ".join([*zeros, number_to_words(int(stripped))] if stripped else zeros)


def _turkish_upper(char: str) -> str:
    return {"i": "İ", "ı": "I"}.get(char, char.upper())


def _replace_slash_pair(match: re.Match[str]) -> str:
    # A proper fraction with a small denominator is read as one ("3/4" -> "dörtte üç"); "7/24" is a pair read
    # number by number; anything else, such as a door number "45/3", is "kırk beş bölü üç".
    numerator, denominator = int(match[1]), int(match[2])
    if 0 < numerator < denominator <= 10:
        return f"{_with_locative(number_to_words(denominator))} {number_to_words(numerator)}"
    if {numerator, denominator} == {7, 24}:
        return f"{number_to_words(numerator)} {number_to_words(denominator)}"
    return f"{number_to_words(numerator)} bölü {number_to_words(denominator)}"


def _with_locative(words: str) -> str:
    last_vowel = next((char for char in reversed(words) if char in _VOWELS), "e")
    vowel = "a" if last_vowel in "aıou" else "e"
    consonant = "t" if words[-1] in "çfhkpsşt" else "d"
    return f"{words}{consonant}{vowel}"


def _replace_address(match: re.Match[str]) -> str:
    address = re.sub(r"^https?://", "", match[0], flags=re.IGNORECASE).rstrip("/")
    spoken: list[str] = []
    for piece in re.split(r"([.@/_-])", address):
        if piece in {".", "@", "/", "_", "-"}:
            spoken.append({".": "nokta", "@": "et", "/": "bölü", "_": "alt tire", "-": "tire"}[piece])
        elif piece:
            lower = piece.lower()
            if lower in _ADDRESS_WORDS:
                spoken.append(_ADDRESS_WORDS[lower])
            elif len(lower) <= 2 and lower.isalpha():
                spoken.append(" ".join(_LETTER_NAMES.get(_turkish_upper(char), char) for char in turkish_lower(lower)))
            else:
                spoken.append(piece)
    return " ".join(spoken)


def _replace_duration(match: re.Match[str]) -> str:
    hours, minutes, seconds = (int(value) for value in match.groups())
    parts = [f"{number_to_words(value)} {unit}" for value, unit in ((hours, "saat"), (minutes, "dakika")) if value]
    if seconds or not parts:
        parts.append(f"{number_to_words(seconds)} saniye")
    return " ".join(parts)


def _attach_suffix(match: re.Match[str]) -> str:
    """Join a written suffix to the words a number or unit became, re-harmonised to the new stem.

    "₺2.350'dir" is written for "elli" but spoken after "lira": "liradır", not "lira'dir".
    """
    stem, suffix = match[1], match[2]
    if stem[-1] not in _VOWELS and len(suffix) > 1 and suffix[0] in "yns" and suffix[1] in _VOWELS:
        suffix = suffix[1:]  # the buffer letter was written for the written form: "mcg'ya" -> "mikrograma"
    if suffix[0] in _VOWELS:  # a vowel softens the final consonant: "dörde", "buçuğu"
        stem = re.sub(r"dört$", "dörd", re.sub(r"buçuk$", "buçuğ", stem))
    vowel = next((char for char in reversed(stem) if char in _VOWELS), "e")
    out = []
    for index, char in enumerate(suffix):
        if char in "ae":
            char = "a" if vowel in "aıou" else "e"
        elif char in "ıiuü":
            char = {(True, False): "ı", (True, True): "u", (False, False): "i", (False, True): "ü"}[
                (vowel in "aıou", vowel in "ouöü")
            ]
        elif index == 0 and char in "dt":
            char = "t" if stem[-1] in "çfhkpsşt" else "d"
        if char in _VOWELS:
            vowel = char
        out.append(char)
    return stem + "".join(out)


def _replace_time(match: re.Match[str]) -> str:
    hour, minute = match.groups()
    hour_words = number_to_words(int(hour))
    if int(minute) == 0:
        return hour_words
    minute_words = _digit_sequence_to_words(minute) if minute.startswith("0") else number_to_words(int(minute))
    return f"{hour_words} {minute_words}"


def _numeric_to_words(value: str) -> str:
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?", value):  # "1.250" or "1.250,75"
        whole, _, fraction = value.replace(".", "").partition(",")
        return _decimal_to_words(f"{whole},{fraction}") if fraction else number_to_words(int(whole))
    return _decimal_to_words(value) if re.search(r"[,.]", value) else number_to_words(int(value))


def _decimal_to_words(value: str) -> str:
    """"2,5" is "iki buçuk", "2,75" "iki virgül yetmiş beş", "1,05" "bir virgül sıfır beş"; longer
    fractions ("3,14159") are read digit by digit."""
    whole, fraction = re.split(r"[,.]", value, maxsplit=1)
    whole_words = number_to_words(int(whole))
    if fraction == "5" and int(whole):
        return f"{whole_words} buçuk"
    fraction_words = _digit_group_to_words(fraction) if len(fraction) <= 2 else _digit_sequence_to_words(fraction)
    return f"{whole_words} virgül {fraction_words}"


def _digit_sequence_to_words(value: str) -> str:
    return " ".join(_ONES[int(digit)] for digit in value)


def _ordinal_to_words(number: int) -> str:
    cardinal = number_to_words(number)
    words = cardinal.split()
    words[-1] = _ORDINAL_EXCEPTIONS.get(words[-1], f"{words[-1]}inci")
    return " ".join(words)


def _roman_to_int(numeral: str) -> int:
    total = 0
    for i, char in enumerate(numeral):
        value = _ROMAN_VALUES[char]
        total += -value if i + 1 < len(numeral) and _ROMAN_VALUES[numeral[i + 1]] > value else value
    return total


def _replace_roman(text: str) -> str:
    """"II. Abdülhamid" -> "ikinci Abdülhamid" (ordinal before a name), "Warcraft III" -> "Warcraft üç"
    (a number after a title). Lone I, V and X stay letters unless written "I." before "Dünya"."""
    text = re.sub(rf"\b({_ROMAN}|I(?=\.\s+Dünya))\.\s+(?=[A-ZÇĞİÖŞÜ])",
                  lambda match: _ordinal_to_words(_roman_to_int(match[1])) + " ", text)
    return re.sub(rf"\b([A-ZÇĞİÖŞÜ][^\W\d_]*\s+)({_ROMAN})\b(?![.'’]?[^\W\d_])",
                  lambda match: match[1] + number_to_words(_roman_to_int(match[2])), text)


def _lower_if_shouted(text: str) -> str:
    """Lowercase text written in capitals throughout, so its words are not spelled as initialisms."""
    words = _WORD.findall(text)
    capitals = [word for word in words if _CAPS_WORD.fullmatch(word)]
    # shouting is capitalised words ("RANDEVUNUZ ONAYLANDI"); a run of initialisms ("SGK, SMS ve TBMM") is not
    spoken_words = sum(_is_pronounceable(word) for word in capitals)
    return turkish_lower(text) if spoken_words >= 3 and len(capitals) >= 0.7 * len(words) else text


def _replace_initialism(match: re.Match[str]) -> str:
    word = match[0]
    if word in _CAPS_LEXICON:
        return _CAPS_LEXICON[word]
    if _is_pronounceable(word):
        return word  # "IŞIK", "NATO": a shouted or acronym word read as a word, lowercased later
    # one word, so it is read fluently ("kadeve"), not as separate syllables ("ke de ve")
    return "".join(_INITIALISM_LETTERS[letter] for letter in word)


def _is_pronounceable(word: str) -> bool:
    """Four or more letters, at least 30% vowels, and no run of three consonants."""
    if len(word) < 4:
        return False
    vowels = sum(char in _VOWELS for char in word)
    return vowels / len(word) >= 0.3 and not re.search(r"[^AEIİOÖUÜ]{3}", word)

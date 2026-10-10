"""Diferansiyel test fixture uretimi: Python to_model_text -> tests/fixtures.json.

Kullanim: python3 tests/make_fixtures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from antalia_mini.text import to_model_text  # noqa: E402

CASES = [
    # temel
    "Merhaba, size nasıl yardımcı olabilirim?",
    "Günaydın, hayırlı sabahlar dilerim.",
    "Evet.", "Hı hı.", "Tamam!", "O.", "A, E!",
    # sayilar
    "0", "5", "12", "100", "101", "1000", "1250", "1.250", "1.250,75", "3,5", "4,2",
    "2,75", "1,05", "3,14159", "1000000", "2500000", "-5 derece", "−5 derece",
    "12'nci", "2.si", "1. gün", "2. Abdülhamit", "3. kat", "Sıra numaranız 12.",
    "2. Dünya Savaşı", "5. Uluslararası Kongre", "XIX. yüzyıl", "II. Abdülhamid",
    "Warcraft III", "I. Dünya", "12345", "007", "045", "A4", "mp3", "5kg", "COVID19",
    "52EB4297", "34KAB123",
    # tarih / saat / sure
    "15.10.2026", "15/10/2026", "2026-10-15", "32.13.2026", "14:30", "09:15",
    "10:00", "15:30'da", "1:02:03", "0:45", "Pzt", "Salı-Cuma", "Pazartesi - Cuma",
    # para / yuzde / olcu
    "%25", "% 18", "18 %", "2.450 TL", "₺12,50", "12,50 TL", "$5", "€10", "100 USD",
    "₺2.350'dir", "5 km", "2 L su", "108L", "m²", "3m2", "90 km/s", "2x1 tablet",
    "günde 2x1", "30x40", "120/80 mmHg", "3/4", "7/24", "45/3", "2026/1847",
    "10 mg/kg", "5 ml", "250 kcal", "MB/sn", "km/sa",
    # telefon / adres / IBAN
    "0212 555 12 34", "+90 532 123 45 67", "0532-123-45-67", "destek@ornek.com",
    "ILKER.SAHIN85@GMAIL.COM", "www.google.com", "https://patientdesk.ai/yol",
    "Kadıköy/İstanbul", "Tel: 0212 / Faks: 0216", "TR12 0001 0000 0000 0000 0000 01",
    "No: 12 D: 5", "Kimlik No 123", "s. 45", "D Blok", "C blok", "K vitamini",
    "Ahmet K.", "D vitamini",
    # kisaltma / unvan
    "Dr. Ayşe Yılmaz", "dr.ayşe", "Prof. Dr. Mehmet", "Doç. Ahmet", "Av. Elif",
    "Mah. Cad. Sok. Apt.", "düzene sok.", "30 DT.", "T.C.", "A.B.D.", "Sn. Müdür",
    "Sn./-", "Hz. Ali", "bkz. sayfa 3", "vb.", "vs.", "örn. test", "no. 5",
    "ltd. şti.", "MB/sn",
    # buyuk harf / initialism
    "SGK", "TBMM", "ATM", "THY", "CHP", "KDV", "KPSS", "TSH", "MHRS", "HES",
    "CEO", "TV", "FBI", "IŞIK", "NATO", "RANDEVUNUZ ONAYLANDI",
    "SGK, SMS ve TBMM", "K blok", "pH dengesi", "H2O", "H. pylori", "X-ray",
    "HbA1c normal", "A Rh(+) kan", "0 Rh(-)", "hane 1234", "hanesi 567890",
    # yabanci / marka (lexicon + CMU)
    "WhatsApp", "iPhone", "ChatGPT", "PatientDesk", "LinkedIn", "YouTube",
    "PatientDesk'e", "iPhone'un", "Instagram", "Intel işlemci", "Istanbul",
    "ISTANBUL", "Iran", "McDonald's", "Coca-Cola", "BluTV", "e-Devlet",
    "Toplantı WhatsApp üzerinden", "GitHub reposu", "Spotify listesi",
    "StarTrekFan", "eBay satıcısı", "OpenAI modeli", "Van Drayv", "Skayp",
    # sembol / noktalama
    "Sıcaklık 37°C", "5 + 3 = 8", "#gündem", "**kalın**", "__alt__",
    "A & B", "• madde", "😀 selam 👋", "a—b", "a–b", "a…b", "''alıntı''",
    "“tırnak”", "‘tek’", "2 GB'lık", "mcg'ya", "dördün", "14:30'daki",
    # ek / uyum
    "liraya", "lira'dan", "dörtte", "ikiye", "otuza", "dörtte üç",
    # cumle / parca
    "Siparişiniz yola çıktı, yarın sabah kapınızda olacak.",
    "Randevunuz 14 Ekim Salı günü saat 15:30'da, Dr. Ayşe Yılmaz ile.",
    "Toplam tutar 1.250,75 TL. Ödemeyi 3 taksitle yapabilirsiniz.",
    "Bize 0212 555 12 34 numarasından veya destek@ornek.com adresinden ulaşabilirsiniz.",
    "Kargonuz yola çıktı; tahmini teslimat tarihi 9 Ekim 2026, Perşembe.",
    "Marketten 5 kg un, 2 lt süt ve 1 düzine yumurta aldım.",
    "Uçuşunuz THY ile saat 09:15'te kalkacaktır, iyi yolculuklar.",
    "Önde gelen isimler toplantıya katıldı, ılık bir karşılama oldu.",
    "Iğdır'dan selamlar, ıspanak aldım.",
    "   ", "", "123", "...", "—",
]


def main() -> None:
    out = []
    for text in CASES:
        try:
            expected = to_model_text(text)
        except Exception as e:  # pragma: no cover
            expected = f"__ERROR__ {type(e).__name__}: {e}"
        out.append({"input": text, "expected": expected})
    dest = ROOT / "tests" / "fixtures.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(out)} fixture -> {dest}")


if __name__ == "__main__":
    main()

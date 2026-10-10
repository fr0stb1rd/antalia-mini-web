"""Ses son-isleme diferansiyel vektorleri: saf-Python referans -> tests/post_vectors.json.

post.py lfilter + soft_limit numpy/torch'suz aynen (float64); Biquad
matematigi ayni oldugu icin Node cikisiyla ~1e-6 icinde eslesmeli.
Tone yolu: lowshelf(210,+6) -> highshelf(4000,-9), gain 1.0.

Kullanim: python3 tests/make_post_vectors.py
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def norm(b, a):
    return (b[0] / a[0], b[1] / a[0], b[2] / a[0]), (1.0, a[1] / a[0], a[2] / a[0])


def lowshelf(f0, gain_db, q, fs):
    A = 10 ** (gain_db / 40)
    w0 = 2 * math.pi * f0 / fs
    alpha, c = math.sin(w0) / (2 * q), math.cos(w0)
    s = 2 * math.sqrt(A) * alpha
    return norm((A * ((A + 1) - (A - 1) * c + s), 2 * A * ((A - 1) - (A + 1) * c), A * ((A + 1) - (A - 1) * c - s)),
                ((A + 1) + (A - 1) * c + s, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - s))


def highshelf(f0, gain_db, q, fs):
    A = 10 ** (gain_db / 40)
    w0 = 2 * math.pi * f0 / fs
    alpha, c = math.sin(w0) / (2 * q), math.cos(w0)
    s = 2 * math.sqrt(A) * alpha
    return norm((A * ((A + 1) + (A - 1) * c + s), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - s)),
                ((A + 1) - (A - 1) * c + s, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - s))


def lfilter(sections, x):
    y = [float(v) for v in x]
    for (b0, b1, b2), (_, a1, a2) in sections:
        out = [0.0] * len(y)
        x1 = x2 = y1 = y2 = 0.0
        for n, v in enumerate(y):
            o = b0 * v + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            x2, x1, y2, y1 = x1, v, y1, o
            out[n] = o
        y = out
    return y


THRESH = 10 ** (-1.0 / 20)
CEIL = 10 ** (-0.1 / 20)


def soft_limit(x):
    room = CEIL - THRESH
    return [v if abs(v) <= THRESH else math.copysign(THRESH + room * math.tanh((abs(v) - THRESH) / room), v)
            for v in x]


def main() -> None:
    low = lowshelf(210.0, 6.0, 0.707, 48000)
    high = highshelf(4000.0, -9.0, 0.707, 48000)
    rng = random.Random(1234)
    cases = []
    # 1: kucuk gurultu (limiter alti) 2: limiteri asan tepe 3: saf sinus (ton cevabi)
    sigs = [
        [rng.gauss(0, 0.1) for _ in range(768)],
        [0.0] * 96 + [1.5] * 8 + [0.0] * 96 + [-1.2] * 8 + [0.0] * 96,
        [0.5 * math.sin(2 * math.pi * 440 * n / 48000) for n in range(768)],
    ]
    for i, sig in enumerate(sigs):
        y = soft_limit([v * 1.0 for v in lfilter([high], lfilter([low], sig))])
        cases.append({"input": [round(v, 7) for v in sig], "expected": [round(v, 7) for v in y]})
    dest = ROOT / "tests" / "post_vectors.json"
    dest.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    print(f"{len(cases)} vektor -> {dest} ({dest.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

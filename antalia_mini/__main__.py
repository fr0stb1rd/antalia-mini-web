"""Command line: python -m antalia_mini "Metin." -o out.wav  (or the `antalia-mini` script)."""
from __future__ import annotations

import argparse
import sys
import time


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="antalia-mini", description="Turkish text to speech (Antalia-2 Mini).")
    ap.add_argument("text", nargs="?", help="text to speak; read from stdin when omitted")
    ap.add_argument("-o", "--out", default="out.wav", help="output file (.wav, .flac, .ogg)")
    ap.add_argument("--model", default=None, help="Hugging Face repo id or local release directory")
    ap.add_argument("--device", default=None)
    ap.add_argument("--sample-rate", type=int, default=48000)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--speed", type=float, default=None, help="default: the release's (0.95)")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--cfg", type=float, default=None)
    ap.add_argument("--no-eq", action="store_true", help="raw vocoder output, without the default bass-reduction EQ")
    ap.add_argument("--normalize", action="store_true", help="print the text the model reads and exit")
    a = ap.parse_args(argv)
    text = a.text if a.text is not None else sys.stdin.read()
    if a.normalize:
        from antalia_mini.text import to_model_text

        print(to_model_text(text))
        return
    from antalia_mini import Antalia

    tts = Antalia(a.model, device=a.device, sample_rate=a.sample_rate, eq=not a.no_eq)
    t0 = time.perf_counter()
    sp = tts.say(text, a.out, seed=a.seed, speed=a.speed, steps=a.steps, cfg=a.cfg)
    dt = time.perf_counter() - t0
    print(f"{a.out}: {sp.duration:.2f} s at {sp.sample_rate} Hz, seed {sp.seed}, {dt:.2f} s to synthesise", file=sys.stderr)


if __name__ == "__main__":
    main()

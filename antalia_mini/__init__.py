"""Antalia-2 Mini: small Turkish text to speech, one synthetic voice.

  from antalia_mini import Antalia
  tts = Antalia()
  tts.say("Merhaba, size nasıl yardımcı olabilirim?", path="merhaba.wav")
"""
__version__ = "1.0.0"

__all__ = ["Antalia", "Speech", "__version__"]


def __getattr__(name):  # lazy, so training code importing submodules does not load the API and the normaliser
    if name in ("Antalia", "Speech"):
        from antalia_mini import api

        return getattr(api, name)
    raise AttributeError(f"module 'antalia_mini' has no attribute {name!r}")

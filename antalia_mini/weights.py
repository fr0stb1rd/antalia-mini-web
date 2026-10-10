"""Release weights: training checkpoints -> inference-only safetensors + config.json, and back to modules.

Release layout (one directory, or one Hugging Face model repo):

  config.json                    acoustic + vocoder configs, symbol table, defaults, provenance, hashes
  acoustic.safetensors           acoustic model without the training-only MAS prior (fp32)
  vocoder.safetensors            vocoder generator (fp32)
  acoustic.fp16.safetensors      the same weights stored as float16 (cast back to fp32 when loaded)
  vocoder.fp16.safetensors

Checkpoints as the training code writes them (antalia_mini/train.py):
  acoustic  {"model", "ema", "opt", "step", "epoch", "config", "kind": "acoustic"}
  vocoder   {"model", "disc", "opt_g", "opt_d", "step", "config", "kind": "vocoder"}
  bundle    {"acoustic": {"config", "model"}, "vocoder": {"config", "model"}}   (infer.py bundle)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from antalia_mini.model import AcousticConfig, AcousticModel
from antalia_mini.text import SYMBOLS
from antalia_mini.vocoder import Vocoder, VocoderConfig

FORMAT_VERSION = 1
TRAINING_ONLY_PREFIXES = ("prior.",)  # the MAS prior: used to align text to mel in training only
DEFAULTS = {"steps": 8, "cfg": 2.0, "speed": 0.95, "temperature": 0.6}
# synthesis options the release ships with (antalia_mini.api OPTION_DEFAULTS documents them): a short "ee." after
# and, before chunks that start with ö or ı, an "ee, " that is spoken and cut away again (the teacher said a
# stray "l" before sentence-initial ö), longer double sounds ("berraktı", "alerjim mahvediyor"), and the output
# tone with eq=True: +6 dB low shelf at 210 Hz, -9 dB high shelf at 4 kHz
OPTIONS = {"suffix": " ee.", "prefix": "ee, ", "prefix_initials": "öı", "gem_stretch": 1.3, "tone": [6.0, -9.0]}
FILES = {"fp32": ("acoustic.safetensors", "vocoder.safetensors"),
         "fp16": ("acoustic.fp16.safetensors", "vocoder.fp16.safetensors")}


def _torch_load(path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def read_checkpoints(acoustic: str | None = None, vocoder: str | None = None, bundle: str | None = None,
                     use_ema: bool = True) -> tuple[dict, dict, dict]:
    """-> (acoustic {"config", "model"}, vocoder {"config", "model"}, provenance)."""
    if bundle:
        b = _torch_load(bundle)
        return b["acoustic"], b["vocoder"], {"bundle": Path(bundle).name}
    if not (acoustic and vocoder):
        raise ValueError("pass a bundle, or both an acoustic and a vocoder checkpoint")
    a, v = _torch_load(acoustic), _torch_load(vocoder)
    if a.get("kind", "acoustic") != "acoustic" or v.get("kind", "vocoder") != "vocoder":
        raise ValueError(f"checkpoint kinds: acoustic={a.get('kind')} vocoder={v.get('kind')}")
    ema = use_ema and "ema" in a
    prov = {"acoustic": {"file": Path(acoustic).name, "step": a.get("step"), "weights": "ema" if ema else "model"},
            "vocoder": {"file": Path(vocoder).name, "step": v.get("step")}}
    return ({"config": a["config"], "model": a["ema"] if ema else a["model"]},
            {"config": v["config"], "model": v["model"]}, prov)


def build_modules(ac: dict, vc: dict) -> tuple[AcousticModel, Vocoder]:
    am = AcousticModel(AcousticConfig(**ac["config"]))
    missing, unexpected = am.load_state_dict(ac["model"], strict=False)
    bad = [k for k in missing if not k.startswith(TRAINING_ONLY_PREFIXES)]
    if bad or unexpected:
        raise ValueError(f"acoustic weights do not match the config: missing {bad[:5]}, unexpected {unexpected[:5]}")
    vm = Vocoder(VocoderConfig(**vc["config"]))
    vm.load_state_dict(vc["model"])
    return am.eval(), vm.eval()


def inference_state(sd: dict) -> dict[str, torch.Tensor]:
    return {k: v.detach().float().contiguous().clone() for k, v in sd.items() if not k.startswith(TRAINING_ONLY_PREFIXES)}


def count(sd: dict) -> int:
    return sum(v.numel() for v in sd.values())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def export(ac: dict, vc: dict, out: str | Path, provenance: dict | None = None, name: str = "antalia-mini",
           version: str | None = None) -> dict:
    """Write the release directory; returns the config dict (also written as config.json)."""
    from safetensors.torch import save_file

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if ac["config"].get("n_symbols", len(SYMBOLS)) != len(SYMBOLS):
        raise ValueError(f"checkpoint has {ac['config']['n_symbols']} symbols, the text frontend {len(SYMBOLS)}")
    build_modules(ac, vc)  # fails early if the weights and configs disagree
    a_sd, v_sd = inference_state(ac["model"]), inference_state(vc["model"])
    meta = {"format": "pt", "model": name}
    for precision, (fa, fv) in FILES.items():
        dt = torch.float16 if precision == "fp16" else torch.float32
        save_file({k: t.to(dt) for k, t in a_sd.items()}, str(out / fa), metadata=meta)
        save_file({k: t.to(dt) for k, t in v_sd.items()}, str(out / fv), metadata=meta)
    for pair in FILES.values():
        for f in pair:
            (out / f).chmod(0o644)  # safetensors writes 0600
    files = {f: {"bytes": (out / f).stat().st_size, "sha256": sha256(out / f)} for pair in FILES.values() for f in pair}
    config = {
        "model_type": name,
        "format_version": FORMAT_VERSION,
        "version": version,
        "sample_rate": VocoderConfig(**vc["config"]).sample_rate,
        "acoustic": AcousticConfig(**ac["config"]).dict(),
        "vocoder": VocoderConfig(**vc["config"]).dict(),
        "symbols": SYMBOLS,
        "defaults": DEFAULTS,
        "options": OPTIONS,
        "params": {"acoustic": count(a_sd), "vocoder": count(v_sd), "total": count(a_sd) + count(v_sd)},
        "source": provenance or {},
        "files": files,
    }
    (out / "config.json").write_text(json.dumps(config, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return config


def load_release(directory: str | Path, variant: str = "fp32") -> tuple[AcousticModel, Vocoder, dict]:
    """Release directory -> (acoustic, vocoder, config); weights are float32 in memory for either variant."""
    from safetensors.torch import load_file

    d = Path(directory)
    config = json.loads((d / "config.json").read_text(encoding="utf-8"))
    if config.get("format_version", 1) > FORMAT_VERSION:
        raise ValueError(f"{d}: format_version {config['format_version']} is newer than this package reads; upgrade antalia-mini")
    if config["symbols"] != SYMBOLS:
        raise ValueError(f"{d}: the weights' symbol table differs from this package's text frontend")
    fa, fv = FILES[variant]
    ac = {"config": config["acoustic"], "model": {k: v.float() for k, v in load_file(str(d / fa)).items()}}
    vc = {"config": config["vocoder"], "model": {k: v.float() for k, v in load_file(str(d / fv)).items()}}
    am, vm = build_modules(ac, vc)
    return am, vm, config


def tiny_checkpoints(seed: int = 0) -> tuple[dict, dict, dict]:
    """A small random model in the training checkpoint format, with the zero-initialised generator
    tensors perturbed so every block does something (tests and pipeline checks only)."""
    torch.manual_seed(seed)
    ac = AcousticConfig(d_text=32, text_hidden=64, d_dur=32, d_gen=32, gen_blocks=2, gen_hidden=64, d_time=32, heads=2)
    vc = VocoderConfig(dim=32, hidden=64, layers=2)
    am, vm = AcousticModel(ac), Vocoder(vc)
    g = am.generator
    with torch.no_grad():
        for p in [g.out.weight, g.out.bias, g.mod.weight, g.mod.bias, g.final_mod, g.null_cond, *(b.table for b in g.blocks)]:
            p.copy_(torch.randn_like(p) * 0.05)
        am.duration.out.bias.fill_(1.5)  # ~4.5 frames per symbol instead of ~1
    return ({"config": ac.dict(), "model": am.state_dict()}, {"config": vc.dict(), "model": vm.state_dict()},
            {"tiny_random": True, "seed": seed})

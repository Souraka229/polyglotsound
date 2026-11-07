"""Singing Voice Conversion with Seed-VC (zero-shot, f0-conditioned 44.1 kHz model).

Why Seed-VC for this MVP (see docs/BENCHMARK.md for the comparison):
- zero-shot from a 1-30 s reference: no per-artist GPU training (RVC needs
  10+ min of clean vocals and hours of GPU training, impossible on this CPU-only machine);
- dedicated singing model with F0 conditioning, so the guide's melody is kept;
- Amphion / Vevo2: strong research results but heavy (GPU-oriented), evaluated later.

Licensing / isolation: Seed-VC is GPL-3.0 and pins old dependencies. It runs in its own
virtual environment (third_party/seed-vc/.venv) and is called as a separate process,
so PolyglotSound does not link GPL code and keeps compatible dependency sets.

Consent: conversion is refused unless the registry holds an active
'voice_conversion' consent for the artist whose voice is the target.
"""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from app.core.device import torch_device

ROOT = Path(__file__).resolve().parents[2]
SEED_DIR = Path(os.environ.get("POLYGLOT_SEEDVC_DIR", ROOT / "third_party" / "seed-vc"))
SEED_PY = SEED_DIR / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


@dataclass
class ConversionResult:
    output: Path
    seconds: float
    command: list[str]
    log_tail: str


def available() -> tuple[bool, str]:
    if not (SEED_DIR / "inference.py").exists():
        return False, f"Seed-VC absent ({SEED_DIR}). Lancez scripts/setup_seed_vc.ps1 (Windows) ou .sh (Linux/Colab)"
    if not SEED_PY.exists():
        return False, "Environnement Seed-VC absent. Lancez scripts/setup_seed_vc.ps1 (Windows) ou .sh (Linux/Colab)"
    return True, "ok"


def convert(guide_vocal: str | Path, artist_reference: str | Path, out_dir: str | Path, *,
            registry=None, artist_id: int | None = None, diffusion_steps: int = 25,
            semitone_shift: int = 0, auto_f0_adjust: bool = False, timeout: int = 3600) -> ConversionResult:
    if registry is None or artist_id is None:
        raise PermissionError("Conversion refusee : artiste et registre de consentement requis.")
    registry.require_consent(artist_id, "voice_conversion")
    ok, msg = available()
    if not ok:
        raise RuntimeError(msg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(SEED_PY), "inference.py", "--source", str(Path(guide_vocal).resolve()),
           "--target", str(Path(artist_reference).resolve()), "--output", str(out_dir.resolve()),
           "--diffusion-steps", str(diffusion_steps), "--length-adjust", "1.0", "--inference-cfg-rate", "0.7",
           "--f0-condition", "True", "--auto-f0-adjust", str(auto_f0_adjust), "--semi-tone-shift",
           str(semitone_shift), "--fp16", str(torch_device() == "cuda")]
    t0 = time.perf_counter()
    before = set(out_dir.glob("*.wav"))
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run(cmd, cwd=SEED_DIR, capture_output=True, text=True, timeout=timeout, env=env,
                          encoding="utf-8", errors="replace")
    tail = (proc.stdout + proc.stderr)[-2000:]
    if proc.returncode != 0:
        raise RuntimeError(f"Seed-VC a echoue (code {proc.returncode}) :\n{tail}")
    new = sorted(set(out_dir.glob("*.wav")) - before, key=lambda p: p.stat().st_mtime)
    if not new:
        raise RuntimeError(f"Seed-VC n'a produit aucun fichier.\n{tail}")
    return ConversionResult(new[-1], round(time.perf_counter() - t0, 1), cmd, tail)

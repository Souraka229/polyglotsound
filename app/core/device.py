"""Device selection. GPU for the PyTorch models (Demucs, AudioSeal, MMS-TTS, Seed-VC) when available.

NLLB and Whisper (CTranslate2) stay on CPU on purpose: their CUDA build needs a cuDNN
version that differs from the one bundled with PyTorch, a classic source of crashes
on Colab. On CPU they take ~1-3 s per line, which is not the bottleneck.
Override with POLYGLOT_DEVICE=cpu|cuda.
"""
from __future__ import annotations

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def torch_device() -> str:
    forced = os.environ.get("POLYGLOT_DEVICE")
    if forced:
        return forced
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

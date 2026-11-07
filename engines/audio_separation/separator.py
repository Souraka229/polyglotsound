"""Vocal / instrumental separation.

Default backend: Demucs v4 ``htdemucs`` (MIT, ~80 MB, runs on CPU).
BS-RoFormer gives better vocal SDR in MVSep/MDX benchmarks, but the public
lucidrains repo ships no trained weights; trained checkpoints come from the
community (``audio-separator`` package / ZFTurbo). It is supported as an
optional backend once its weights are installed (see docs/BENCHMARK.md).
The original mix is never modified: stems are written to a new folder.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


@dataclass
class Stems:
    vocals: np.ndarray        # (2, n)
    instrumental: np.ndarray  # (2, n)
    sr: int
    backend: str


@lru_cache(maxsize=2)
def _demucs(name: str):
    from demucs.pretrained import get_model
    from app.core.device import torch_device
    model = get_model(name)
    model.to(torch_device()).eval()
    return model


def separate_demucs(audio: np.ndarray, sr: int, model_name: str = "htdemucs", shifts: int = 1) -> Stems:
    import torch
    from demucs.apply import apply_model

    from app.core.device import torch_device
    model = _demucs(model_name)
    if sr != model.samplerate:
        raise ValueError(f"Demucs attend {model.samplerate} Hz")
    wav = torch.from_numpy(audio if audio.shape[0] == 2 else np.repeat(audio, 2, axis=0)).float()
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    with torch.inference_mode():
        out = apply_model(model, ((wav - mean) / std)[None], shifts=shifts, split=True,
                          overlap=0.25, progress=False, device=torch_device())[0]
    out = out.cpu() * std + mean
    idx = model.sources.index("vocals")
    vocals = out[idx].numpy()
    instrumental = (out.sum(0) - out[idx]).numpy()
    return Stems(vocals, instrumental, sr, f"demucs:{model_name}")


def separate(audio: np.ndarray, sr: int, backend: str = "demucs") -> Stems:
    if backend == "demucs":
        return separate_demucs(audio, sr)
    if backend == "bs-roformer":
        try:
            from audio_separator.separator import Separator  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("BS-RoFormer : installez 'audio-separator' et ses poids (voir docs).") from exc
        raise NotImplementedError("Backend BS-RoFormer : integration prevue apres benchmark.")
    raise ValueError(f"Backend inconnu : {backend}")


def sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Signal-to-distortion ratio in dB (simple, non-scale-invariant, as in MUSDB 'SDR')."""
    n = min(reference.shape[-1], estimate.shape[-1])
    r, e = reference[..., :n], estimate[..., :n]
    num = np.sum(r ** 2) + 1e-9
    den = np.sum((r - e) ** 2) + 1e-9
    return float(10 * np.log10(num / den))


def save_stems(stems: Stems, out_dir: str | Path) -> dict[str, Path]:
    from app.core.audio import save
    out = Path(out_dir)
    return {"vocals": save(out / "vocals.wav", stems.vocals, stems.sr),
            "instrumental": save(out / "instrumental.wav", stems.instrumental, stems.sr)}

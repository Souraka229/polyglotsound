"""Audio I/O. FFmpeg decodes anything (MP3/WAV/FLAC...), soundfile writes WAV."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

SUPPORTED = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


def ffmpeg_bin() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("FFmpeg introuvable dans le PATH (winget install Gyan.FFmpeg).")
    return exe


def load(path: str | Path, sr: int = 44100, mono: bool = False) -> np.ndarray:
    """Return float32 array shaped (channels, samples)."""
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED:
        raise ValueError(f"Format non pris en charge : {path.suffix}")
    ch = 1 if mono else 2
    cmd = [ffmpeg_bin(), "-v", "error", "-i", str(path), "-f", "f32le", "-acodec", "pcm_f32le",
           "-ac", str(ch), "-ar", str(sr), "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    audio = np.frombuffer(raw, dtype=np.float32).reshape(-1, ch).T.copy()
    if audio.size == 0:
        raise ValueError(f"Fichier audio vide ou illisible : {path}")
    return audio


def save(path: str | Path, audio: np.ndarray, sr: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = audio.T if audio.ndim == 2 else audio
    sf.write(str(path), data, sr, subtype="PCM_24")
    return path


def to_mp3(wav: str | Path, mp3: str | Path, bitrate: str = "320k") -> Path:
    subprocess.run([ffmpeg_bin(), "-v", "error", "-y", "-i", str(wav), "-b:a", bitrate, str(mp3)], check=True)
    return Path(mp3)


def describe(audio: np.ndarray, sr: int) -> dict:
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    peak = float(np.max(np.abs(mono))) or 1e-9
    rms = float(np.sqrt(np.mean(mono ** 2))) or 1e-9
    return {"duration_s": round(mono.size / sr, 2), "sample_rate": sr,
            "channels": audio.shape[0] if audio.ndim == 2 else 1,
            "peak_dbfs": round(20 * np.log10(peak), 2), "rms_dbfs": round(20 * np.log10(rms), 2)}

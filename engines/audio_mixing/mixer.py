"""Mix the new vocal over the untouched original instrumental."""
from __future__ import annotations

import numpy as np


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) + 1e-12


def match_length(x: np.ndarray, n: int) -> np.ndarray:
    x = np.atleast_2d(x)
    if x.shape[-1] >= n:
        return x[..., :n]
    return np.pad(x, ((0, 0), (0, n - x.shape[-1])))


def to_stereo(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    return np.repeat(x, 2, axis=0) if x.shape[0] == 1 else x[:2]


def align_offset(reference: np.ndarray, vocal: np.ndarray, sr: int, max_shift_s: float = 0.5) -> int:
    """Lag (samples) that best aligns the new vocal energy envelope with the original vocal."""
    hop = 512
    def env(x):
        m = np.atleast_2d(x).mean(0)
        n = len(m) // hop
        return np.sqrt(np.mean(m[: n * hop].reshape(n, hop) ** 2, axis=1))
    a, b = env(reference), env(vocal)
    n = min(len(a), len(b))
    a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    max_lag = int(max_shift_s * sr / hop)
    best, best_lag = -np.inf, 0
    for lag in range(-max_lag, max_lag + 1):
        c = np.dot(a[max(0, lag):n + min(0, lag)], b[max(0, -lag):n - max(0, lag)])
        if c > best:
            best, best_lag = c, lag
    return best_lag * hop


def mix(instrumental: np.ndarray, new_vocal: np.ndarray, original_vocal: np.ndarray | None = None,
        sr: int = 44100, vocal_gain_db: float = 0.0, auto_align: bool = True) -> tuple[np.ndarray, dict]:
    inst = to_stereo(instrumental)
    voc = to_stereo(new_vocal)
    info: dict = {}
    if original_vocal is not None and auto_align:
        lag = align_offset(original_vocal, voc, sr)
        voc = np.roll(voc, lag, axis=-1) if lag else voc
        info["alignment_offset_ms"] = round(1000 * lag / sr, 1)
    voc = match_length(voc, inst.shape[-1])
    if original_vocal is not None:
        target = rms(to_stereo(original_vocal))           # keep the original vocal/instrumental balance
        voc = voc * (target / rms(voc))
    voc = voc * 10 ** (vocal_gain_db / 20)
    out = inst + voc
    peak = float(np.max(np.abs(out)))
    if peak > 0.98:                                       # transparent peak safety, no limiter colouring
        out = out * (0.98 / peak)
        info["normalized_from_peak"] = round(peak, 3)
    return out.astype(np.float32), info

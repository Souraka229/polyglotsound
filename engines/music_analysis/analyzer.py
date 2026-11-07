"""Tempo, beats, key, F0 contour, voiced segments and note segmentation.

F0: librosa pYIN (no download, CPU). RMVPE is more robust on polyphonic
residue and is the planned upgrade (its weights are already pulled by Seed-VC).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


@dataclass
class Note:
    start: float
    end: float
    midi: float


@dataclass
class MusicAnalysis:
    tempo_bpm: float
    beats: list[float]
    key: str
    key_confidence: float
    f0_times: np.ndarray
    f0_hz: np.ndarray
    voiced_segments: list[tuple[float, float]]
    notes: list[Note] = field(default_factory=list)

    def summary(self) -> dict:
        return {"tempo_bpm": round(self.tempo_bpm, 1), "beats": len(self.beats), "key": self.key,
                "key_confidence": round(self.key_confidence, 2), "voiced_segments": len(self.voiced_segments),
                "notes": len(self.notes)}


def estimate_key(y: np.ndarray, sr: int) -> tuple[str, float]:
    import librosa
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
    best, best_r = "C major", -1.0
    scores = []
    for i in range(12):
        for prof, mode in ((_MAJOR, "major"), (_MINOR, "minor")):
            r = float(np.corrcoef(chroma, np.roll(prof, i))[0, 1])
            scores.append(r)
            if r > best_r:
                best, best_r = f"{KEYS[i]} {mode}", r
    scores.sort(reverse=True)
    return best, max(0.0, scores[0] - scores[1]) * 10   # margin over 2nd best, rough confidence


F0_SR = 16000


def track_f0(vocals_mono: np.ndarray, sr: int, hop: int = 320, fmax: float = 1050.0):
    """pYIN at 16 kHz, 20 ms hop: same accuracy as 22 kHz / 11.6 ms on the synthetic set, ~7x faster."""
    import librosa
    if sr != F0_SR:
        vocals_mono = librosa.resample(vocals_mono, orig_sr=sr, target_sr=F0_SR)
        sr = F0_SR
    f0, voiced, _ = librosa.pyin(vocals_mono, fmin=65.0, fmax=fmax, sr=sr, hop_length=hop,
                                 frame_length=1024)
    times = librosa.times_like(f0, sr=sr, hop_length=hop)
    f0 = np.where(voiced, f0, np.nan)
    return times, f0


def voiced_segments(times: np.ndarray, f0: np.ndarray, min_gap: float = 0.25, min_len: float = 0.2):
    voiced = np.isfinite(f0)
    segs, start = [], None
    for t, v in zip(times, voiced):
        if v and start is None:
            start = t
        elif not v and start is not None:
            segs.append([start, t])
            start = None
    if start is not None:
        segs.append([start, times[-1]])
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] < min_gap:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return [(round(a, 3), round(b, 3)) for a, b in merged if b - a >= min_len]


def segment_notes(times: np.ndarray, f0: np.ndarray, jump: float = 0.8, min_len: float = 0.08) -> list[Note]:
    midi = 69 + 12 * np.log2(np.where(np.isfinite(f0), f0, np.nan) / 440.0)
    notes, cur = [], []
    for t, m in zip(times, midi):
        if not np.isfinite(m):
            if cur:
                notes.append(cur)
            cur = []
            continue
        if cur and abs(m - np.median([x for _, x in cur])) > jump:
            notes.append(cur)
            cur = []
        cur.append((t, m))
    if cur:
        notes.append(cur)
    out = [Note(c[0][0], c[-1][0], float(np.median([m for _, m in c]))) for c in notes]
    return [n for n in out if n.end - n.start >= min_len]


def track_beats(y: np.ndarray, sr: int) -> tuple[float, list[float]]:
    """Beat tracking with half-period phase correction.

    librosa often locks onto off-beats (hi-hats). We compare the found phase with the
    phase shifted by half a period and keep the one carrying more low-band onset
    energy (kick / bass usually mark the beat). Heuristic, measured in docs/BENCHMARK.md.
    """
    import librosa
    hop = 512
    tempo, frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
    if len(frames) < 4:
        return float(np.atleast_1d(tempo)[0]), librosa.frames_to_time(frames, sr=sr, hop_length=hop).tolist()
    low = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, fmax=200, n_mels=32)
    half = int(round(np.median(np.diff(frames)) / 2))
    shifted = frames + half
    shifted = shifted[shifted < len(low)]
    def energy(fr):
        return float(np.mean([low[max(0, f - 2):f + 3].max() for f in fr]))
    if energy(shifted) > 1.15 * energy(frames):
        frames = np.concatenate([[frames[0] - half] if frames[0] - half >= 0 else [], shifted]).astype(int)
    return float(np.atleast_1d(tempo)[0]), librosa.frames_to_time(frames, sr=sr, hop_length=hop).tolist()


def analyze(mix: np.ndarray, vocals: np.ndarray | None, sr: int, analysis_sr: int = 22050,
            instrumental: np.ndarray | None = None) -> MusicAnalysis:
    import librosa
    src = instrumental if instrumental is not None else mix
    mono = src.mean(axis=0) if src.ndim == 2 else src
    y = librosa.resample(mono, orig_sr=sr, target_sr=analysis_sr)
    tempo, beats = track_beats(y, analysis_sr)
    key, kc = estimate_key(y, analysis_sr)
    v = vocals if vocals is not None else mix
    vm = v.mean(axis=0) if v.ndim == 2 else v
    vy = librosa.resample(vm, orig_sr=sr, target_sr=analysis_sr)
    times, f0 = track_f0(vy, analysis_sr)
    return MusicAnalysis(tempo, beats, key, kc, times, f0,
                         voiced_segments(times, f0), segment_notes(times, f0))

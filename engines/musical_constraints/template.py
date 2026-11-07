"""Build the structured musical representation (LineTemplate) of original lines."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from app.schemas.models import Confidence, LineTemplate, LyricLine, MelodySlot
from engines.syllable_engine import syllabify

BEAT_TOLERANCE = 0.07  # seconds


def hz_to_midi(hz: float) -> float:
    return 69.0 + 12.0 * np.log2(hz / 440.0)


def beat_strength(t: float, beats: Sequence[float], meter: int = 4) -> float:
    if beats is None or len(beats) == 0:
        return 0.5
    beats = np.asarray(beats)
    k = int(np.argmin(np.abs(beats - t)))
    if abs(beats[k] - t) > BEAT_TOLERANCE:
        return 0.3
    return 1.0 if k % meter == 0 else 0.6


def median_midi(start: float, end: float, f0_times, f0_hz) -> Optional[float]:
    if f0_times is None or len(f0_times) == 0:
        return None
    t = np.asarray(f0_times)
    f = np.asarray(f0_hz)
    sel = (t >= start) & (t < end) & np.isfinite(f) & (f > 0)
    if sel.sum() < 2:
        return None
    return float(np.round(hz_to_midi(float(np.median(f[sel]))), 2))


def _syllable_spans(line: LyricLine, n_syl: int, word_syllables: list[int]) -> tuple[list[tuple[float, float]], Confidence]:
    if line.words and all(w.start is not None and w.end is not None for w in line.words) \
            and len(line.words) == len(word_syllables):
        spans = []
        for w, k in zip(line.words, word_syllables):
            step = (w.end - w.start) / max(k, 1)
            spans += [(w.start + i * step, w.start + (i + 1) * step) for i in range(k)]
        return spans, Confidence.MEDIUM
    if line.start is not None and line.end is not None and n_syl:
        step = (line.end - line.start) / n_syl
        return [(line.start + i * step, line.start + (i + 1) * step) for i in range(n_syl)], Confidence.LOW
    return [(i * 0.3, (i + 1) * 0.3) for i in range(n_syl)], Confidence.UNKNOWN


def build_template(line: LyricLine, f0_times=None, f0_hz=None, beats=None, meter: int = 4) -> LineTemplate:
    from engines.syllable_engine.base import tokenize
    syl_line = syllabify(line.text, line.language)
    # syllables actually sung in the original: optional schwas are counted as sung
    syllables = syl_line.syllables
    word_counts = [syllabify(w, line.language).count for w in tokenize(line.text)]
    spans, conf = _syllable_spans(line, len(syllables), word_counts)
    slots = []
    for (s, e), syl in zip(spans, syllables):
        syl.start, syl.end = s, e
        slots.append(MelodySlot(start=s, end=e, midi=median_midi(s, e, f0_times, f0_hz),
                                beat_strength=beat_strength(s, beats, meter)))
    notes = list(syl_line.warnings)
    if conf in (Confidence.LOW, Confidence.UNKNOWN):
        notes.append("Durees des syllabes estimees (pas d'alignement force) : contraintes temporelles indicatives.")
    return LineTemplate(line=line, syllables=syllables, slots=slots, rhyme_key=syl_line.rhyme_key,
                        timing_confidence=conf, notes=notes)


def assign_rhyme_scheme(templates: list[LineTemplate]) -> None:
    """Label lines A, B, C... by identical rhyme keys (vowel+coda, plus tone for fon)."""
    letters: dict[str, str] = {}
    counts: dict[str, int] = {}
    for t in templates:
        counts[t.rhyme_key] = counts.get(t.rhyme_key, 0) + 1
    for t in templates:
        if not t.rhyme_key or counts[t.rhyme_key] < 2:
            t.rhyme_group = None
            continue
        if t.rhyme_key not in letters:
            letters[t.rhyme_key] = chr(ord("A") + len(letters))
        t.rhyme_group = letters[t.rhyme_key]

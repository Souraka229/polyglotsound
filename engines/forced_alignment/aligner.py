"""Multilingual Lyric Alignment: place corrected lyrics on the sung segments.

MVP method (language-independent, no model): lines are mapped to voiced
segments; inside a line, words get time proportional to their syllable count.
Confidence is LOW and stated as such. Planned upgrade: CTC forced alignment
with torchaudio MMS_FA (romanised, 1100+ languages) for word/syllable accuracy.
"""
from __future__ import annotations

from app.schemas.models import Confidence, LyricLine, Word
from engines.syllable_engine import syllabify
from engines.syllable_engine.base import tokenize


def distribute_words(line: LyricLine) -> LyricLine:
    if line.start is None or line.end is None or not line.text.strip():
        return line
    words = tokenize(line.text)
    counts = [max(1, syllabify(w, line.language).count) for w in words]
    total = sum(counts) or 1
    t, dur, out = line.start, line.end - line.start, []
    for w, c in zip(words, counts):
        d = dur * c / total
        out.append(Word(text=w, start=round(t, 3), end=round(t + d, 3)))
        t += d
    line.words = out
    return line


def align_lyrics(texts: list[str], language: str, segments: list[tuple[float, float]],
                 existing: list[LyricLine] | None = None) -> list[LyricLine]:
    """Combine user-corrected texts with timings (existing ASR lines first, else voiced segments)."""
    texts = [t.strip() for t in texts if t.strip()]
    lines: list[LyricLine] = []
    if existing and len(existing) == len(texts):
        for i, (t, l) in enumerate(zip(texts, existing)):
            changed = t != l.text
            new = LyricLine(index=i, text=t, language=language, start=l.start, end=l.end,
                            words=[] if changed else l.words, human_verified=True,
                            confidence=Confidence.LOW if changed else l.confidence)
            lines.append(distribute_words(new) if changed or not new.words else new)
        return lines
    segs = list(segments)
    if segs and len(segs) != len(texts):
        # merge or split segments so there is one span per line, proportional to syllables
        start, end = segs[0][0], segs[-1][1]
        weights = [max(1, syllabify(t, language).count) for t in texts]
        tot, acc, segs = sum(weights), start, []
        for w in weights:
            d = (end - start) * w / tot
            segs.append((acc, acc + d))
            acc += d
    for i, t in enumerate(texts):
        s, e = segs[i] if i < len(segs) else (None, None)
        lines.append(distribute_words(LyricLine(index=i, text=t, language=language, start=s, end=e,
                                                human_verified=True, confidence=Confidence.LOW)))
    return lines

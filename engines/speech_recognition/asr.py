"""Speech recognition on the separated vocals.

- French / English / Yoruba: faster-whisper (CTranslate2 Whisper, MIT) with word timestamps.
- Fon: Whisper does NOT support Fon. Omnilingual ASR (Apache-2.0) lists 1600+
  languages but depends on fairseq2, which has no native Windows build (WSL2
  only). On native Windows the Fon lyrics are typed or corrected by the artist,
  and the voiced segments found in the vocals give the line timings.
A transcription is never presented as certain: every line carries a confidence.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from app.core.languages import get_language
from app.schemas.models import Confidence, LyricLine, Word

WHISPER_SIZE = "small"


@lru_cache(maxsize=1)
def _whisper(size: str = WHISPER_SIZE):
    from faster_whisper import WhisperModel
    return WhisperModel(size, device="cpu", compute_type="int8")


def _conf(p: float) -> Confidence:
    return Confidence.HIGH if p >= 0.85 else Confidence.MEDIUM if p >= 0.6 else Confidence.LOW


def detect_language(vocals16k: np.ndarray) -> dict[str, float]:
    model = _whisper()
    _, info = model.transcribe(vocals16k, beam_size=1, without_timestamps=True)
    probs = dict(sorted(info.all_language_probs or [], key=lambda x: -x[1])[:5])
    return probs


def transcribe(vocals16k: np.ndarray, language: str,
               voiced_segments: list[tuple[float, float]] | None = None) -> tuple[list[LyricLine], list[str]]:
    spec = get_language(language)
    notes: list[str] = []
    if spec.whisper_code is None:
        notes.append(f"Reconnaissance vocale indisponible pour le {spec.name} sur Windows natif : "
                     "saisissez les paroles, les segments chantes detectes servent de reperes temporels.")
        lines = [LyricLine(index=i, text="", language=language, start=s, end=e, confidence=Confidence.UNKNOWN)
                 for i, (s, e) in enumerate(voiced_segments or [])]
        return lines, notes
    model = _whisper()
    segments, _ = model.transcribe(vocals16k, language=spec.whisper_code, beam_size=5, word_timestamps=True,
                                   vad_filter=True, condition_on_previous_text=False)
    lines = []
    for i, seg in enumerate(segments):
        words = [Word(text=w.word.strip(), start=w.start, end=w.end, probability=w.probability)
                 for w in (seg.words or []) if w.word.strip()]
        p = float(np.mean([w.probability for w in words])) if words else 0.0
        lines.append(LyricLine(index=i, text=seg.text.strip(), language=language, start=seg.start, end=seg.end,
                               words=words, confidence=_conf(p)))
    if any(l.confidence != Confidence.HIGH for l in lines):
        notes.append("Certaines lignes ont une confiance moyenne ou faible : relecture conseillee.")
    notes.append("Whisper est entraine surtout sur la parole : sur le chant, verifiez toujours les paroles.")
    return lines, notes

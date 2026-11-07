"""Fon (Fɔ̀ngbè) syllabification and tone extraction.

Fon is a tonal language: the tone carried by each vowel can change the meaning
of a word. The tone-bearing unit is the vowel (or a syllabic nasal), so every
vowel is a syllable nucleus. Standard orthography marks tones with diacritics:
  acute  ´  high (H)      grave  `  low (L)
  caron  ˇ  rising (LH)   circumflex ^ falling (HL)
Nasal vowels are written vowel + n (an, ɛn, in, ɔn, un) when no vowel follows.

Limits (stated, not hidden): tones are very often omitted in everyday writing
and in ASR output. Unmarked vowels get ``Tone.UNMARKED``, and every check that
depends on tones is then reported as *not measurable* and sent to human review.
"""
from __future__ import annotations

import re
import unicodedata

from app.schemas.models import Syllable, Tone
from engines.syllable_engine.base import Syllabifier, register

BASE_VOWELS = "aeɛiouɔ"
TONE_MARKS = {"́": Tone.HIGH, "̀": Tone.LOW, "̌": Tone.RISING,
              "̂": Tone.FALLING, "̄": Tone.MID}
CONSONANT_DIGRAPHS = ("gb", "kp", "ny", "xw", "hw", "gw", "kw")
LIQUIDS = "lrw"


def _segments(word: str) -> list[tuple[str, str]]:
    """Split a word into (kind, text) segments: 'V' (vowel+marks) or 'C'."""
    decomposed = unicodedata.normalize("NFD", word)
    segs: list[tuple[str, str]] = []
    i = 0
    while i < len(decomposed):
        ch = decomposed[i]
        if ch in BASE_VOWELS:
            j = i + 1
            while j < len(decomposed) and unicodedata.combining(decomposed[j]):
                j += 1
            segs.append(("V", decomposed[i:j]))
            i = j
            continue
        if decomposed[i:i + 2] in CONSONANT_DIGRAPHS:
            segs.append(("C", decomposed[i:i + 2]))
            i += 2
            continue
        j = i + 1
        while j < len(decomposed) and unicodedata.combining(decomposed[j]):
            j += 1          # tone on a syllabic nasal (ń, ǹ)
        segs.append(("C", decomposed[i:j]))
        i = j
    return segs


def tone_of(segment: str) -> Tone:
    for mark, tone in TONE_MARKS.items():
        if mark in segment:
            return tone
    return Tone.UNMARKED


class FonSyllabifier(Syllabifier):
    language = "fon"

    def __init__(self) -> None:
        self._warnings: list[str] = []

    def syllabify_word(self, word, next_word, line_final):
        segs = _segments(word)
        # nasal vowel: V + n/m not followed by a vowel
        merged: list[tuple[str, str]] = []
        for k, (kind, text) in enumerate(segs):
            nxt = segs[k + 1] if k + 1 < len(segs) else None
            if kind == "C" and text[0] in "nm" and merged and merged[-1][0] == "V" \
                    and (nxt is None or nxt[0] == "C") and not unicodedata.combining(text[-1]):
                merged[-1] = ("V", merged[-1][1] + text)
            else:
                merged.append((kind, text))
        # syllabic nasal: n/m with no adjacent vowel inside the word (e.g. "n", "ń", "nkɔ")
        syls: list[Syllable] = []
        onset = ""
        for k, (kind, text) in enumerate(merged):
            nxt = merged[k + 1] if k + 1 < len(merged) else None
            if kind == "V":
                vowel = unicodedata.normalize("NFC", text)
                syls.append(Syllable(text=unicodedata.normalize("NFC", onset) + vowel,
                                     nucleus=vowel, tone=tone_of(text), open=True))
                onset = ""
            elif text[0] in "nm" and (nxt is None or nxt[0] == "C") and onset == "":
                syls.append(Syllable(text=unicodedata.normalize("NFC", text), nucleus=text[0],
                                     tone=tone_of(text), open=True))
            else:
                onset += text
        if onset:                                        # stray final consonant(s)
            if syls:
                syls[-1].text += unicodedata.normalize("NFC", onset)
                syls[-1].open = False
            else:
                syls.append(Syllable(text=word, nucleus="", open=False, tone=Tone.UNMARKED))
            self._warnings.append(f"'{word}' : consonne finale inhabituelle en fon, a verifier.")
        return syls

    def syllabify(self, text):
        line = super().syllabify(text)
        unmarked = sum(1 for s in line.syllables if s.tone == Tone.UNMARKED)
        if line.syllables and unmarked:
            line.warnings.append(
                f"{unmarked}/{len(line.syllables)} syllabes sans ton ecrit : "
                "verification tonale incomplete, validation humaine requise.")
        return line

    def rhyme_key(self, syllables):
        if not syllables:
            return ""
        nucleus = unicodedata.normalize("NFD", syllables[-1].nucleus)
        base = "".join(c for c in nucleus if not unicodedata.combining(c) or c == "̃")
        tone = syllables[-1].tone.value if syllables[-1].tone else "?"
        return f"{base}/{tone}"


register("fon", FonSyllabifier())

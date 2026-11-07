"""English syllabification with lexical stress.

Uses the CMU Pronouncing Dictionary data file (BSD-2, models/cmudict/, read by our own
loader: the ``cmudict`` PyPI wrapper is GPL-3.0 and is deliberately NOT used):
syllable count = number of vowel phones, stress = phones marked ``1``.
Falls back to orthographic rules (flagged as less reliable).
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from app.schemas.models import Syllable
from engines.syllable_engine.base import Syllabifier, register, split_by_nuclei

ONSETS = {"bl", "br", "cl", "cr", "dr", "fl", "fr", "gl", "gr", "pl", "pr", "tr", "sc", "sk", "sl",
          "sm", "sn", "sp", "st", "sw", "th", "sh", "ch", "wh", "ph", "wr", "kn", "qu", "str", "spr",
          "scr", "thr", "shr", "squ"}


CMU_PATH = Path(__file__).resolve().parents[2] / "models" / "cmudict" / "cmudict.dict"


@lru_cache(maxsize=1)
def _cmu() -> dict | None:
    if not CMU_PATH.exists():
        return None
    d: dict[str, list[list[str]]] = {}
    with open(CMU_PATH, encoding="utf-8") as f:
        for line in f:
            parts = line.split("#")[0].split()
            if len(parts) < 2:
                continue
            word = re.sub(r"\(\d+\)$", "", parts[0])
            d.setdefault(word, []).append(parts[1:])
    return d


def _ortho_nuclei(word: str) -> list[tuple[int, int]]:
    spans = [(m.start(), m.end()) for m in re.finditer(r"[aeiouy]+", word)]
    if word.startswith("y") and spans and spans[0][0] == 0 and len(word) > 1 and word[1] in "aeiou":
        spans[0] = (1, spans[0][1])
    if len(spans) > 1:
        s, e = spans[-1]
        tail = word[s:]
        silent_e = tail in ("e", "es", "ed") and not re.search(r"[^aeiouy]le$", word)
        if tail == "es" and re.search(r"(s|x|z|ch|sh|ge|ce)es$", word):
            silent_e = False
        if tail == "ed" and re.search(r"[td]ed$", word):
            silent_e = False
        if silent_e:
            spans.pop()
    return spans or [(0, len(word))]


class EnglishSyllabifier(Syllabifier):
    language = "eng"

    def __init__(self) -> None:
        self._warnings: list[str] = []

    def phones(self, word: str) -> list[str] | None:
        d = _cmu()
        if d is None:
            return None
        entries = d.get(word.replace("'", "")) or d.get(word)
        return entries[0] if entries else None

    def syllabify_word(self, word, next_word, line_final):
        spans = _ortho_nuclei(word)
        parts = split_by_nuclei(word, spans, ONSETS)
        phones = self.phones(word)
        stresses: list[int] = []
        if phones:
            stresses = [int(p[-1]) for p in phones if p[-1].isdigit()]
            while len(parts) > len(stresses) > 0:      # merge to match the dictionary
                parts[-2:] = [parts[-2] + parts[-1]]
            while len(parts) < len(stresses):
                parts.append("")                        # dictionary hears an extra vowel
        else:
            self._warnings.append(f"'{word}' absent du dictionnaire CMU : comptage orthographique.")
        syls = []
        for i, p in enumerate(parts):
            st = (stresses[i] == 1) if stresses else (i == 0 and len(parts) > 1)
            syls.append(Syllable(text=p or "·", nucleus=p, stressed=st,
                                 open=not re.search(r"[^aeiouy]+$", p)))
        if len(syls) == 1 and not phones:
            syls[0].stressed = len(word) > 3            # function words usually unstressed
        elif len(syls) == 1 and phones:
            syls[0].stressed = stresses[0] >= 1 and word not in _FUNCTION_WORDS
        return syls

    def rhyme_key(self, syllables):
        # true rhyme needs phones of the last word; approximate from text of last syllable
        for syl in reversed(syllables):
            m = re.search(r"[aeiouy].*$", syl.text)
            if m:
                return re.sub(r"e$", "", m.group(0)) or m.group(0)
        return ""

    def syllabify(self, text):
        from engines.syllable_engine.base import tokenize
        line = super().syllabify(text)
        words = tokenize(text)
        if words:
            phon = self.phonetic_rhyme(words[-1])
            if phon:
                line.rhyme_key = phon
        return line

    def phonetic_rhyme(self, last_word: str) -> str | None:
        phones = self.phones(last_word.lower())
        if not phones:
            return None
        idx = max((i for i, p in enumerate(phones) if p[-1] in "12"), default=None)
        if idx is None:
            return None
        return " ".join(re.sub(r"\d", "", p) for p in phones[idx:])


_FUNCTION_WORDS = {"a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at", "for", "with",
                   "is", "are", "was", "be", "my", "your", "his", "her", "its", "our", "their", "i",
                   "you", "he", "she", "it", "we", "they", "me", "him", "us", "them", "as", "so"}

register("english", EnglishSyllabifier())

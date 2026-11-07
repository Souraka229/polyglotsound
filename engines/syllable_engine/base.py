"""Shared syllabification machinery (onset maximisation over vowel nuclei)."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from app.schemas.models import Syllable


@dataclass
class SyllabifiedLine:
    syllables: list[Syllable]
    rhyme_key: str
    warnings: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.syllables)

    @property
    def min_count(self) -> int:
        return sum(1 for s in self.syllables if not s.optional)

    @property
    def max_count(self) -> int:
        return len(self.syllables)


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def tokenize(text: str) -> list[str]:
    """Words, keeping apostrophe elisions attached (l'amour -> l'amour)."""
    text = nfc(text.lower()).replace("’", "'")
    letter = r"(?:[^\W\d_]|[̀-ͯ])"   # keep combining tone marks (ɔ̌, ɛ̀...)
    return re.findall(rf"{letter}+(?:['-]{letter}+)*", text)


def split_by_nuclei(word: str, spans: list[tuple[int, int]], onsets: set[str]) -> list[str]:
    """Cut a word into syllables given nucleus spans [(start, end), ...].

    Consonants between two nuclei go to the next syllable as long as they form a
    legal onset (longest legal onset wins), the rest closes the previous syllable.
    """
    if not spans:
        return [word] if word else []
    cuts = [0]
    for (s1, e1), (s2, _e2) in zip(spans, spans[1:]):
        cluster = word[e1:s2]
        cut = s2
        for k in range(len(cluster), -1, -1):
            onset = cluster[len(cluster) - k:]
            if k == 0 or k == 1 or onset in onsets:
                cut = s2 - k
                break
        cuts.append(cut)
    cuts.append(len(word))
    return [word[a:b] for a, b in zip(cuts, cuts[1:]) if word[a:b]]


class Syllabifier:
    language = "generic"
    vowels = set("aeiouy")

    def syllabify_word(self, word: str, next_word: str | None, line_final: bool) -> list[Syllable]:
        spans = [(m.start(), m.end()) for m in re.finditer(f"[{''.join(self.vowels)}]+", word)]
        parts = split_by_nuclei(word, spans, set())
        return [Syllable(text=p, nucleus=p) for p in parts] or [Syllable(text=word, nucleus=word)]

    def rhyme_key(self, syllables: list[Syllable]) -> str:
        for syl in reversed(syllables):
            if not syl.optional:
                m = re.search(f"[{''.join(self.vowels)}].*$", syl.text)
                return m.group(0) if m else syl.text
        return ""

    def syllabify(self, text: str) -> SyllabifiedLine:
        words = tokenize(text)
        out: list[Syllable] = []
        warnings: list[str] = []
        for i, w in enumerate(words):
            nxt = words[i + 1] if i + 1 < len(words) else None
            out.extend(self.syllabify_word(w, nxt, nxt is None))
        warnings.extend(getattr(self, "_warnings", []))
        self._warnings = []
        return SyllabifiedLine(out, self.rhyme_key(out), warnings)


_REGISTRY: dict[str, Syllabifier] = {}


def register(name: str, syllabifier: Syllabifier) -> None:
    _REGISTRY[name] = syllabifier


def get_syllabifier(name: str) -> Syllabifier:
    if not _REGISTRY:
        from engines.syllable_engine import english, fon, french  # noqa: F401  (self-registering)
    return _REGISTRY.get(name) or _REGISTRY.setdefault("generic", Syllabifier())


def syllabify(text: str, language: str) -> SyllabifiedLine:
    from app.core.languages import get_language
    return get_syllabifier(get_language(language).syllabifier).syllabify(text)

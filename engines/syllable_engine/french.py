"""French syllabification for sung lyrics.

Key point for singing: the final mute e (e muet / schwa) is *optional*. It is
elided before a vowel, usually dropped in modern pop, but can be sung on a note
(classical chanson). We therefore return syllables flagged ``optional`` and the
constraint engine works with a syllable-count *range* instead of one number.
"""
from __future__ import annotations

import re

from app.schemas.models import Syllable
from engines.syllable_engine.base import Syllabifier, register, split_by_nuclei

VOWELS = "aeiouyàâäéèêëîïôöûùüœæ"
FULL_VOWELS = VOWELS.replace("e", "")
ONSETS = {"bl", "br", "cl", "cr", "dr", "fl", "fr", "gl", "gr", "pl", "pr", "tr", "vr",
          "ch", "ph", "th", "gn", "qu", "gu", "chr", "phr", "phl"}
# monosyllables whose only vowel is "e": never mute
_HIATUS_BREAK = re.compile(r"(?<=[éèêë])(?=[aeiouàâîôû])|(?<=[aeiouàâ])(?=[ïü])|(?<=[aoe])(?=é)")
_NASAL_ENT = {"moment", "souvent", "vent", "dent", "lent", "cent", "gent", "argent", "content",
              "comment", "vraiment", "seulement", "tellement", "doucement", "parent", "enfant",
              "client", "accent", "talent", "absent", "present", "présent", "sent", "ment"}


def _is_vowel_initial(word: str | None) -> bool:
    if not word:
        return False
    w = word.lstrip("'")
    return bool(w) and (w[0] in VOWELS or (w[0] == "h" and len(w) > 1 and w[1] in VOWELS))


def _nuclei(word: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for m in re.finditer(f"[{VOWELS}]+", word):
        start, group = m.start(), m.group(0)
        # "qu" / "gu" + vowel: the u is part of the consonant
        if group[0] == "u" and start > 0 and word[start - 1] in "qg" and len(group) > 1:
            start, group = start + 1, group[1:]
        # split true hiatus (cré-er, na-ïf, po-ète)
        pos = start
        for piece in _HIATUS_BREAK.split(group):
            if piece:
                spans.append((pos, pos + len(piece)))
                pos += len(piece)
    return spans


class FrenchSyllabifier(Syllabifier):
    language = "fra"

    def __init__(self) -> None:
        self._warnings: list[str] = []

    def syllabify_word(self, word, next_word, line_final):
        core = word.split("'")[-1] if "'" in word else word
        prefix = word[: len(word) - len(core)]
        spans = _nuclei(core)
        if not spans:
            return [Syllable(text=word, nucleus="", open=False)] if not prefix else []
        parts = split_by_nuclei(core, spans, ONSETS)
        if prefix:
            parts[0] = prefix + parts[0]
        syls = []
        for i, p in enumerate(parts):
            s, e = spans[i]
            nucleus = core[s:e]
            syls.append(Syllable(text=p, nucleus=nucleus, open=not re.search(r"[^aeiouyàâéèêëîïôûùœ]+$", p)))

        last = syls[-1]
        mute = (len(spans) > 1 and last.nucleus == "e"
                and re.fullmatch(r"[^aeiouy]*e(s|nt)?", last.text) is not None
                and not (core.endswith("ent") and (core in _NASAL_ENT or core.endswith("ment") and len(core) > 6)))
        if mute:
            if core.endswith("ent"):
                self._warnings.append(f"'{word}' : terminaison -ent supposee muette (verbe), a verifier.")
            if _is_vowel_initial(next_word) and not last.text.endswith(("s", "nt")):
                syls.pop()                       # elision: chant(e) une -> merged
                syls[-1].text += last.text
            else:
                last.optional = True
        # lexical-group stress: last full syllable of the word
        for syl in reversed(syls):
            if not syl.optional:
                syl.stressed = True
                break
        return syls

    def rhyme_key(self, syllables):
        for syl in reversed(syllables):
            if syl.optional:
                continue
            t = syl.text.split("'")[-1]
            t = re.sub(r"(?<=[aeiouéèêàâôûy])(s|t|d|x|z|ts|ds)$", "", t)
            t = re.sub(r"(er|ez|et)$", "é", t) if len(t) > 2 else t
            for a, b in (("eaux", "o"), ("eau", "o"), ("au", "o"), ("ô", "o"), ("ai", "è"), ("ei", "è"),
                         ("ê", "è"), ("ain", "in"), ("ein", "in"), ("un", "in"), ("am", "an"),
                         ("em", "an"), ("en", "an"), ("om", "on"), ("œu", "eu"), ("ou", "U")):
                t = t.replace(a, b)
            m = re.search(r"[aeiouyéèàUœ].*$", t)
            return m.group(0) if m else t
        return ""


register("french", FrenchSyllabifier())

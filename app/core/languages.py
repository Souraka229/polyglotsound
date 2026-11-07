"""Language registry: one place to declare what each model really supports.

Written translation support (NLLB) and speech recognition support (Whisper,
Omnilingual ASR) are tracked separately on purpose: they are not the same thing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class LanguageSpec:
    code: str                         # internal ISO 639-3
    name: str
    nllb_code: Optional[str]          # FLORES-200 code used by NLLB, None if absent
    whisper_code: Optional[str]       # None = Whisper cannot transcribe it
    omnilingual_code: Optional[str]   # Omnilingual ASR code (needs fairseq2 / WSL2 on Windows)
    tonal: bool = False
    syllabifier: str = "generic"
    stress_model: str = "none"        # "lexical" (en), "phrase_final" (fr), "tone" (fon)
    notes: list[str] = field(default_factory=list)


LANGUAGES: dict[str, LanguageSpec] = {
    "fon": LanguageSpec(
        "fon", "Fon", "fon_Latn", None, "fon_Latn", tonal=True,
        syllabifier="fon", stress_model="tone",
        notes=[
            "Whisper ne prend pas en charge le fon : la transcription doit etre fournie ou corrigee par un humain.",
            "Les tons ne sont souvent pas ecrits : sans diacritiques, la verification tonale est impossible.",
        ],
    ),
    "fra": LanguageSpec("fra", "Francais", "fra_Latn", "fr", "fra_Latn",
                        syllabifier="french", stress_model="phrase_final"),
    "eng": LanguageSpec("eng", "Anglais", "eng_Latn", "en", "eng_Latn",
                        syllabifier="english", stress_model="lexical"),
    "yor": LanguageSpec("yor", "Yoruba", "yor_Latn", "yo", "yor_Latn", tonal=True,
                        syllabifier="fon", stress_model="tone",
                        notes=["Syllabation fon reutilisee (structure CV proche) : a valider."]),
    "guw": LanguageSpec("guw", "Goun", None, None, None, tonal=True,
                        syllabifier="fon", stress_model="tone",
                        notes=["Absent de NLLB-200 : traduction automatique non disponible."]),
    "ddn": LanguageSpec("ddn", "Dendi", None, None, None, tonal=True,
                        syllabifier="generic", stress_model="none",
                        notes=["Absent de NLLB-200 et de Whisper."]),
}

REQUIRED_PAIRS = [("fon", "fra"), ("fon", "eng"), ("fra", "fon"),
                  ("eng", "fon"), ("fra", "eng"), ("eng", "fra")]


def get_language(code: str) -> LanguageSpec:
    try:
        return LANGUAGES[code]
    except KeyError as exc:
        raise ValueError(f"Langue inconnue : {code}. Disponibles : {sorted(LANGUAGES)}") from exc


def can_translate(src: str, tgt: str) -> bool:
    return bool(get_language(src).nllb_code and get_language(tgt).nllb_code)

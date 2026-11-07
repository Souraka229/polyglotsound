"""Meaning-preservation estimate, usable in every direction (including Fon).

No reliable sentence-embedding model exists for Fon, so we use round-trip
translation: the candidate is back-translated into the source language and
compared with the original line (chrF). The result is calibrated per line
against the round trip of the literal NLLB translation, so 1.0 means "as close
to the original as the literal translation". Added content is penalised
(LLMs tend to add images that are not in the song).
This is a *proxy*: role inversions ("the sun lifts us" vs "the sun rises for us")
can still pass, so the criterion is always flagged for artist validation.
"""
from __future__ import annotations

from engines.musical_constraints.metrics import chrf

ADDITION_RATIO = 1.35


def round_trip(text: str, src: str, tgt: str, translator) -> str:
    return translator.translate(text, tgt, src, n_best=1, beam=4)[0]


def semantic_score(candidate: str, original: str, literal_ref: str, src: str, tgt: str,
                   translator=None, literal_rt: float | None = None) -> tuple[float, str]:
    direct = chrf(candidate, literal_ref)
    if translator is None:
        return direct, "chrF vs traduction litterale"
    try:
        back = round_trip(candidate, src, tgt, translator)
    except Exception:
        return direct, "chrF vs traduction litterale"
    rt = chrf(back, original, beta=1.0)
    if literal_rt is None:
        literal_rt = chrf(round_trip(literal_ref, src, tgt, translator), original, beta=1.0)
    rel = min(1.0, rt / literal_rt) if literal_rt > 0 else rt
    a = 0.4 if tgt != "fon" else 0.25          # literal NLLB Fon is weak, trust the round trip more
    score = a * direct + (1 - a) * rel
    notes = [f"aller {direct:.2f}", f"retour « {back} » {rt:.2f} (litteral {literal_rt:.2f})"]
    ratio = len(back.split()) / max(1, len(original.split()))
    if ratio > ADDITION_RATIO:
        score *= 0.7
        notes.append(f"contenu probablement ajoute (x{ratio:.1f} mots)")
    return round(score, 3), "chrF " + ", ".join(notes)

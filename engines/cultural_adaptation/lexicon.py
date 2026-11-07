"""Cultural Meaning Preservation: detect cultural expressions and check they survive adaptation."""
from __future__ import annotations

import json
import unicodedata
from functools import lru_cache
from pathlib import Path

from app.schemas.models import CriterionScore

LEXICON_PATH = Path(__file__).resolve().parents[2] / "configs" / "cultural_lexicon.json"


def _fold(text: str) -> str:
    """lowercase + strip diacritics, so 'Vodún' matches 'vodun'."""
    d = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in d if not unicodedata.combining(c)).replace("ɛ", "e").replace("ɔ", "o")


@lru_cache(maxsize=4)
def load_lexicon(path: str = str(LEXICON_PATH)) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["entries"]


def detect(text: str, language: str, extra: list[dict] | None = None) -> list[dict]:
    folded = f" {_fold(text)} "
    found = []
    for e in load_lexicon() + (extra or []):
        if e["language"] in (language, "*") and f"{_fold(e['term'])}" in folded:
            found.append(e)
    return found


def cultural_score(original: str, src: str, candidate: str, tgt: str, weight: float,
                   extra: list[dict] | None = None) -> CriterionScore | None:
    found = detect(original, src, extra)
    if not found:
        return None
    cand = f" {_fold(candidate)} "
    kept, lost = [], []
    for e in found:
        options = e["renderings"].get(tgt) or e["renderings"].get("*") or [e["term"]]
        (kept if any(_fold(o) in cand for o in options) else lost).append(e["term"])
    unvalidated = any(not e.get("validated") for e in found)
    expl = f"{len(kept)}/{len(found)} expression(s) culturelle(s) preservee(s)"
    if lost:
        expl += f" ; perdue(s) : {', '.join(lost)}"
    return CriterionScore(name="culture", score=round(len(kept) / len(found), 3), weight=weight,
                          reliable=not unvalidated, explanation=expl + ".")


def prompt_hints(original: str, src: str, tgt: str) -> list[str]:
    hints = []
    for e in detect(original, src):
        options = e["renderings"].get(tgt) or e["renderings"].get("*") or [e["term"]]
        hints.append(f"'{e['term']}' ({e['meaning']}) : garder ou rendre par {', '.join(options)}")
    return hints

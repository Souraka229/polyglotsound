"""Musical Constraint Engine: score, reject and compare lyric adaptations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.core.languages import get_language
from app.schemas.models import AdaptationCandidate, CriterionScore, LineTemplate
from engines.musical_constraints import criteria as C
from engines.musical_constraints.alignment import align
from engines.syllable_engine import syllabify

DEFAULT_WEIGHTS = {"sens": 0.25, "syllabes": 0.18, "temps": 0.14, "accents": 0.10, "tons": 0.10,
                   "phonetique": 0.07, "rimes": 0.10, "culture": 0.08, "fluidite": 0.08}
MAX_SYLLABLE_GAP_RATIO = 0.25     # reject if gap > max(2, 25% of the notes)
MIN_SEMANTIC = 0.35


@dataclass
class ScoringContext:
    template: LineTemplate
    target_language: str
    semantic: Optional[float] = None          # 0..1 from the translation engine
    semantic_method: str = "non mesure"
    cultural: Optional[CriterionScore] = None
    partner_rhyme_keys: list[str] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))


def score_candidate(cand: AdaptationCandidate, ctx: ScoringContext) -> AdaptationCandidate:
    spec = get_language(ctx.target_language)
    w = ctx.weights
    line = syllabify(cand.text, ctx.target_language)
    cand.syllables = line.syllables
    al = align(line.syllables, ctx.template.slots)
    cand.alignment = al.mapping

    crit: list[CriterionScore] = []
    if ctx.semantic is None:
        crit.append(CriterionScore(name="sens", score=None, weight=w["sens"], reliable=False,
                                   explanation="Fidelite du sens non mesuree : validation humaine requise."))
    else:
        crit.append(CriterionScore(name="sens", score=round(ctx.semantic, 3), weight=w["sens"],
                                   reliable=False, explanation=f"Proximite du sens ({ctx.semantic_method})."))
    crit.append(C.syllabic(ctx.template, line.syllables, w["syllabes"]))
    crit.append(C.temporal(ctx.template, line.syllables, al, w["temps"]))
    crit.append(C.prosody(ctx.template, line.syllables, al, spec.tonal,
                          w["tons"] if spec.tonal else w["accents"]))
    crit.append(C.singability(ctx.template, line.syllables, al, w["phonetique"]))
    crit.append(C.rhyme(ctx.template, line.rhyme_key, ctx.partner_rhyme_keys, w["rimes"]))
    if ctx.cultural is not None:
        crit.append(ctx.cultural)
    crit.append(C.fluency(cand.text, w["fluidite"]))
    cand.criteria = crit

    measured = [c for c in crit if c.score is not None]
    total_w = sum(c.weight for c in measured)
    cand.total = round(sum(c.score * c.weight for c in measured) / total_w, 3) if total_w else None

    # hard constraints
    cand.rejection_reasons = []
    target = len(ctx.template.slots)
    lo, hi = line.min_count, line.max_count
    gap = 0 if lo <= target <= hi else min(abs(target - lo), abs(target - hi))
    if gap > max(2, round(MAX_SYLLABLE_GAP_RATIO * target)):
        cand.rejection_reasons.append(f"ecart de {gap} syllabes avec la melodie ({target} notes)")
    if not al.feasible:
        cand.rejection_reasons.append("impossible a chanter sur la melodie")
    if ctx.semantic is not None and ctx.semantic < MIN_SEMANTIC:
        cand.rejection_reasons.append("sens trop eloigne de l'original")
    cand.rejected = bool(cand.rejection_reasons)
    cand.needs_human_review = [c.name for c in crit if c.score is None or not c.reliable] + line.warnings
    return cand


def rank(cands: list[AdaptationCandidate]) -> list[AdaptationCandidate]:
    return sorted(cands, key=lambda c: (c.rejected, -(c.total or 0.0)))


def explain_comparison(cands: list[AdaptationCandidate]) -> str:
    """Human-readable explanation of why the best candidate wins."""
    ranked = rank(cands)
    if not ranked:
        return "Aucune proposition."
    best = ranked[0]
    lines = [f"Meilleure proposition : « {best.text} » (score {best.total})."]
    if best.rejected:
        lines.append("Attention : toutes les propositions violent une contrainte dure ; "
                     "regeneration ou correction manuelle recommandee.")
    b = {c.name: c.score for c in best.criteria}
    for other in ranked[1:]:
        o = {c.name: c.score for c in other.criteria}
        wins = [n for n in b if b[n] is not None and o.get(n) is not None and b[n] - o[n] >= 0.1]
        loses = [n for n in b if b[n] is not None and o.get(n) is not None and o[n] - b[n] >= 0.1]
        msg = f"Face a « {other.text} » ({other.total})"
        if other.rejected:
            msg += f" [rejetee : {'; '.join(other.rejection_reasons)}]"
        if wins:
            msg += f" : meilleure en {', '.join(wins)}"
        if loses:
            msg += f" ; moins bonne en {', '.join(loses)}"
        lines.append(msg + ".")
    unmeasured = sorted({n for c in ranked for n in c.needs_human_review if " " not in n})
    if unmeasured:
        lines.append(f"Criteres a valider par l'artiste ou un locuteur natif : {', '.join(unmeasured)}.")
    return "\n".join(lines)

"""Individual, explainable scoring criteria. Each returns a CriterionScore (0..1 or None)."""
from __future__ import annotations

import math
import re

from app.schemas.models import Confidence, CriterionScore, LineTemplate, Syllable, Tone
from engines.musical_constraints.alignment import MIN_SINGABLE, Alignment
from engines.musical_constraints.metrics import rhyme_similarity

TONE_LEVEL = {Tone.HIGH: (2, 2), Tone.MID: (1, 1), Tone.LOW: (0, 0),
              Tone.RISING: (0, 2), Tone.FALLING: (2, 0)}
OPEN_VOWELS = set("aoɔɛeàâôèêé")


def syllabic(template: LineTemplate, syls: list[Syllable], w: float) -> CriterionScore:
    target = len(template.slots)
    lo = sum(1 for s in syls if not s.optional)
    hi = len(syls)
    diff = 0 if lo <= target <= hi else min(abs(target - lo), abs(target - hi))
    score = math.exp(-0.5 * diff)
    rng = f"{lo}" if lo == hi else f"{lo} a {hi}"
    expl = (f"{rng} syllabes pour {target} notes : compatible." if diff == 0
            else f"{rng} syllabes pour {target} notes : ecart de {diff}.")
    return CriterionScore(name="syllabes", score=round(score, 3), weight=w, explanation=expl)


def temporal(template: LineTemplate, syls: list[Syllable], al: Alignment, w: float) -> CriterionScore:
    reliable = template.timing_confidence in (Confidence.HIGH, Confidence.MEDIUM)
    if not al.feasible:
        return CriterionScore(name="temps", score=0.0, weight=w, reliable=reliable,
                              explanation="Impossible de placer les syllabes sur la melodie.")
    share: dict[int, int] = {}
    for idx in al.mapping:
        for j in idx:
            share[j] = share.get(j, 0) + 1
    durs = [d for d, idx in zip(al.durations(template.slots, share), al.mapping) if idx]
    too_short = sum(1 for d in durs if d < MIN_SINGABLE)
    sung = max(len(durs), 1)
    score = math.exp(-al.cost / sung) * (1 - too_short / sung)
    parts = [f"cout d'alignement {al.cost:.2f}"]
    if too_short:
        parts.append(f"{too_short} syllabe(s) sous {int(MIN_SINGABLE * 1000)} ms")
    if al.operations:
        parts.append("; ".join(al.operations[:3]))
    if not reliable:
        parts.append("durees estimees, pas mesurees")
    return CriterionScore(name="temps", score=round(score, 3), weight=w, reliable=reliable,
                          explanation=", ".join(parts) + ".")


def singability(template: LineTemplate, syls: list[Syllable], al: Alignment, w: float) -> CriterionScore:
    long_slots = {j for j, s in enumerate(template.slots) if s.duration >= 0.4}
    checks = []
    for syl, idx in zip(syls, al.mapping):
        if idx and any(j in long_slots for j in idx):
            vowel = re.sub(r"[^\w]", "", syl.nucleus)[:1]
            good = syl.open and (vowel in OPEN_VOWELS or not vowel)
            checks.append(1.0 if good else (0.6 if syl.open else 0.2))
    if not checks:
        return CriterionScore(name="phonetique", score=None, weight=w,
                              explanation="Aucune note longue : critere non applicable.")
    score = sum(checks) / len(checks)
    return CriterionScore(name="phonetique", score=round(score, 3), weight=w, reliable=False,
                          explanation=f"{len(checks)} note(s) longue(s) ; voyelles ouvertes et syllabes "
                                      f"ouvertes privilegiees (heuristique).")


def prosody(template: LineTemplate, syls: list[Syllable], al: Alignment, tonal: bool, w: float) -> CriterionScore:
    if tonal:
        return _tone_melody(template, syls, al, w)
    pairs = [(syl, template.slots[idx[0]]) for syl, idx in zip(syls, al.mapping) if idx]
    stressed = [(s, sl) for s, sl in pairs if s.stressed]
    if not stressed:
        return CriterionScore(name="accents", score=None, weight=w,
                              explanation="Aucune syllabe accentuee detectee.")
    on_strong = sum(1 for _, sl in stressed if sl.beat_strength >= 0.6)
    score = on_strong / len(stressed)
    reliable = template.timing_confidence in (Confidence.HIGH, Confidence.MEDIUM)
    return CriterionScore(name="accents", score=round(score, 3), weight=w, reliable=reliable,
                          explanation=f"{on_strong}/{len(stressed)} accents sur un temps fort.")


def _tone_melody(template, syls, al, w) -> CriterionScore:
    """Tonal languages: melodic contour should follow tone contour (H higher than L)."""
    seq = []
    for syl, idx in zip(syls, al.mapping):
        if idx and syl.tone in TONE_LEVEL and template.slots[idx[0]].midi is not None:
            seq.append((TONE_LEVEL[syl.tone], template.slots[idx[0]].midi, template.slots[idx[-1]].midi))
    agree = total = 0
    for (t1, _p1s, p1e), (t2, p2s, _p2e) in zip(seq, seq[1:]):
        dt = t2[0] - t1[1]
        dp = (p2s or 0) - (p1e or 0)
        total += 1
        if (dt == 0 and abs(dp) <= 1.0) or (dt > 0 and dp > 0.5) or (dt < 0 and dp < -0.5):
            agree += 1
    if total < 2:
        return CriterionScore(name="tons", score=None, weight=w, reliable=False,
                              explanation="Tons non ecrits ou hauteurs inconnues : un locuteur natif doit "
                                          "verifier que la melodie ne change pas le sens des mots.")
    return CriterionScore(name="tons", score=round(agree / total, 3), weight=w, reliable=False,
                          explanation=f"{agree}/{total} transitions tonales suivent la melodie "
                                      f"(a confirmer par un locuteur natif).")


def rhyme(template: LineTemplate, cand_key: str, partner_keys: list[str], w: float) -> CriterionScore:
    if template.rhyme_group is None or not partner_keys:
        return CriterionScore(name="rimes", score=None, weight=w,
                              explanation="Ligne sans rime dans l'original : critere non applicable.")
    best = max(rhyme_similarity(cand_key, k) for k in partner_keys)
    return CriterionScore(name="rimes", score=best, weight=w,
                          explanation=f"Rime '{cand_key}' vs groupe {template.rhyme_group} "
                                      f"({', '.join(partner_keys)}) : {best:.1f}.")


def fluency(text: str, w: float) -> CriterionScore:
    words = re.findall(r"\w+", text.lower())
    if not words:
        return CriterionScore(name="fluidite", score=0.0, weight=w, explanation="Texte vide.")
    penalties = []
    rep = sum(1 for a, b in zip(words, words[1:]) if a == b)
    if rep:
        penalties.append((0.25 * rep, f"{rep} repetition(s) immediate(s)"))
    if re.search(r"[\[\]{}<>#*_|]|\d", text):
        penalties.append((0.4, "symboles ou chiffres parasites"))
    if len(words) > 3 and len(set(words)) / len(words) < 0.5:
        penalties.append((0.3, "vocabulaire tres repetitif"))
    score = max(0.0, 1.0 - sum(p for p, _ in penalties))
    expl = "; ".join(r for _, r in penalties) or "aucun defaut de surface detecte"
    return CriterionScore(name="fluidite", score=round(score, 3), weight=w, reliable=False,
                          explanation=f"Heuristique : {expl}.")

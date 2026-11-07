"""Generate -> verify -> regenerate loop producing the 3 best singable adaptations of a line."""
from __future__ import annotations

from dataclasses import dataclass, field

from app.schemas.models import AdaptationCandidate, LineTemplate
from engines.cultural_adaptation.lexicon import cultural_score, prompt_hints
from engines.musical_constraints import DEFAULT_WEIGHTS, ScoringContext, explain_comparison, rank, score_candidate
from engines.translation import llm_adapter
from engines.musical_constraints.metrics import chrf
from engines.translation.semantic import round_trip, semantic_score


@dataclass
class LineAdaptation:
    template: LineTemplate
    literal: str
    candidates: list[AdaptationCandidate]
    explanation: str
    log: list[str] = field(default_factory=list)

    @property
    def top3(self) -> list[AdaptationCandidate]:
        return rank(self.candidates)[:3]


def _score(text: str, source: str, tpl: LineTemplate, src: str, tgt: str, literal: str,
           translator, partner_keys: list[str], literal_rt: float | None = None) -> AdaptationCandidate:
    sem, method = semantic_score(text, tpl.line.text, literal, src, tgt, translator, literal_rt)
    ctx = ScoringContext(template=tpl, target_language=tgt, semantic=sem, semantic_method=method,
                         cultural=cultural_score(tpl.line.text, src, text, tgt, DEFAULT_WEIGHTS["culture"]),
                         partner_rhyme_keys=partner_keys)
    return score_candidate(AdaptationCandidate(text=text, language=tgt, source=source), ctx)


def adapt_line(tpl: LineTemplate, src: str, tgt: str, translator, partner_keys: list[str] | None = None,
               manual: list[str] | None = None, llm_model: str | None = None, max_rounds: int = 2,
               use_llm: bool = True, fast: bool = False) -> LineAdaptation:
    """fast=True: ONE NLLB call per line (3 hypotheses, no back-translation). The meaning criterion then
    falls back to chrF against the literal translation. Used for the instant proposal in the studio."""
    partner_keys = partner_keys or []
    log: list[str] = []
    nbest = translator.translate(tpl.line.text, src, tgt, n_best=3 if fast else 5, beam=3 if fast else 6)
    literal = nbest[0]
    if fast:
        translator, use_llm, literal_rt = None, False, None
        log.append("mode rapide : sens estime sans retraduction")
    else:
        literal_rt = chrf(round_trip(literal, src, tgt, translator), tpl.line.text, beta=1.0)
    log.append(f"NLLB {src}->{tgt} : {len(nbest)} hypotheses")
    seen: set[str] = set()
    cands: list[AdaptationCandidate] = []

    def add(text: str, source: str):
        key = text.strip().lower()
        if key and key not in seen:
            seen.add(key)
            cands.append(_score(text, source, tpl, src, tgt, literal, translator, partner_keys, literal_rt))

    for i, t in enumerate(nbest):
        add(t, "nllb" if i == 0 else "nllb-nbest")
    for t in manual or []:
        add(t, "manual")

    llm_allowed = use_llm and tgt in llm_adapter.LLM_TARGETS
    models = llm_adapter.ollama_available() if llm_allowed else []
    if not llm_allowed or not models:
        model = None
    elif llm_model in models:
        model = llm_model
    else:
        model = llm_adapter.DEFAULT_MODEL if llm_adapter.DEFAULT_MODEL in models else models[0]
    if use_llm and tgt in llm_adapter.LLM_TARGETS and not model:
        log.append("Ollama indisponible : adaptations artistiques LLM desactivees.")
    if use_llm and tgt not in llm_adapter.LLM_TARGETS:
        log.append(f"LLM non utilise pour '{tgt}' (qualite non fiable) : propositions NLLB + humaines.")

    feedback: list[str] = []
    stress = [i + 1 for i, s in enumerate(tpl.slots) if s.beat_strength >= 0.6]
    for rnd in range(max_rounds if model else 0):
        good = [c for c in cands if not c.rejected and (c.total or 0) >= 0.6]
        if len(good) >= 3 and rnd > 0:
            break
        prompt = llm_adapter.build_prompt(tpl.line.text, src, literal, tgt, len(tpl.slots), stress,
                                          partner_keys[0] if partner_keys else None,
                                          prompt_hints(tpl.line.text, src, tgt), 4, feedback)
        try:
            props = llm_adapter.generate(prompt, model=model)
        except Exception as exc:
            log.append(f"LLM erreur : {exc}")
            break
        log.append(f"tour {rnd + 1} : {len(props)} propositions de {model}")
        before = len(cands)
        for p in props:
            add(p, f"llm:{model}")
        feedback = [f"« {c.text} » : {len(c.syllables)} syllabes, "
                    + ("; ".join(c.rejection_reasons) or f"score {c.total}") for c in cands[before:]]
    return LineAdaptation(tpl, literal, cands, explain_comparison(cands), log)

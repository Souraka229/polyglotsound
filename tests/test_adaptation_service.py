"""Adaptation loop rules, tested with a fake translator and a fake LLM (no model needed)."""
import numpy as np
import pytest

from app.schemas.models import LyricLine
from app.services import adaptation as AD
from engines.musical_constraints import build_template
from engines.translation import llm_adapter
from engines.translation.semantic import semantic_score


class FakeTranslator:
    table = {("Je chante pour toi ce soir", "fra", "eng"): ["I sing for you tonight", "I am singing for you tonight"],
             ("I sing for you tonight", "eng", "fra"): ["Je chante pour toi ce soir"],
             ("I am singing for you tonight", "eng", "fra"): ["Je chante pour toi ce soir"],
             ("I sing tonight for you my love when the night is calm", "eng", "fra"):
                 ["Je chante ce soir pour toi mon amour quand la nuit est calme"],
             ("Tonight I sing for you", "eng", "fra"): ["Ce soir je chante pour toi"]}

    def translate(self, text, src, tgt, n_best=1, beam=5, sampling=False):
        return self.table.get((text, src, tgt), [text])[:n_best]


@pytest.fixture
def tpl():
    line = LyricLine(index=0, text="Je chante pour toi ce soir", language="fra", start=0, end=2.1)
    return build_template(line, beats=np.arange(0, 3, 0.3))


@pytest.fixture
def fake_llm(monkeypatch):
    calls = []
    monkeypatch.setattr(llm_adapter, "ollama_available", lambda timeout=2.0: ["llama3.2:latest"])
    def gen(prompt, model="x", **kw):
        calls.append(prompt)
        return ["Tonight I sing for you", "I sing tonight for you my love when the night is calm"]
    monkeypatch.setattr(llm_adapter, "generate", gen)
    return calls


def test_llm_disabled_means_no_llm_even_with_model_name(tpl, fake_llm):
    res = AD.adapt_line(tpl, "fra", "eng", FakeTranslator(), use_llm=False, llm_model="llama3.2:latest")
    assert not fake_llm and all(not c.source.startswith("llm") for c in res.candidates)


def test_llm_never_used_for_fon_target(tpl, fake_llm):
    res = AD.adapt_line(tpl, "fra", "fon", FakeTranslator(), use_llm=True, llm_model="llama3.2:latest")
    assert not fake_llm and any("non utilise" in l for l in res.log)


def test_llm_prompt_carries_constraints_and_added_content_is_penalised(tpl, fake_llm):
    res = AD.adapt_line(tpl, "fra", "eng", FakeTranslator(), use_llm=True, max_rounds=1)
    assert fake_llm and f"exactly {len(tpl.slots)} sung syllables" in fake_llm[0]
    assert "Do NOT add new ideas" in fake_llm[0]
    added = next(c for c in res.candidates if c.text.startswith("I sing tonight for you my love"))
    faithful = next(c for c in res.candidates if c.text == "I sing for you tonight")
    sem = lambda c: next(k.score for k in c.criteria if k.name == "sens")  # noqa: E731
    assert sem(added) < sem(faithful)
    assert added.rejected                                    # too many syllables for the melody


def test_semantic_score_flags_additions():
    tr = FakeTranslator()
    s, expl = semantic_score("I sing tonight for you my love when the night is calm", "Je chante pour toi ce soir",
                             "I sing for you tonight", "fra", "eng", tr)
    assert "ajoute" in expl and s < 0.7

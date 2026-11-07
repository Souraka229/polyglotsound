import numpy as np

from app.schemas.models import AdaptationCandidate, Confidence, LyricLine, MelodySlot, Syllable, Tone, Word
from engines.musical_constraints import ScoringContext, assign_rhyme_scheme, build_template, explain_comparison, rank, score_candidate
from engines.musical_constraints.alignment import align
from engines.musical_constraints.metrics import chrf, rhyme_similarity


def slots(n, d=0.3, strengths=None):
    strengths = strengths or [0.5] * n
    return [MelodySlot(start=i * d, end=(i + 1) * d, midi=60 + i, beat_strength=strengths[i]) for i in range(n)]


def test_align_one_to_one():
    al = align([Syllable(text=x) for x in "abc"], slots(3))
    assert al.feasible and al.mapping == [[0], [1], [2]] and al.cost == 0


def test_align_melisma_when_fewer_syllables():
    al = align([Syllable(text="a"), Syllable(text="o")], slots(3, d=0.5))
    assert al.feasible and sum(len(m) for m in al.mapping) == 3


def test_align_elides_optional_schwa():
    syls = [Syllable(text="chan"), Syllable(text="te", optional=True), Syllable(text="toi")]
    al = align(syls, slots(2))
    assert al.feasible and al.mapping[1] == [] and al.cost < 0.1


def test_align_crowding_costs_more_than_one_to_one():
    four = align([Syllable(text=x) for x in "abcd"], slots(3))
    three = align([Syllable(text=x) for x in "abc"], slots(3))
    assert four.feasible and four.cost > three.cost


def test_align_infeasible():
    assert not align([Syllable(text="a")], slots(5)).feasible


def test_stress_on_weak_beat_is_penalised():
    good = align([Syllable(text="a", stressed=True)], slots(1, strengths=[1.0]))
    bad = align([Syllable(text="a", stressed=True)], slots(1, strengths=[0.3]))
    assert bad.cost > good.cost


def test_chrf_identity_and_disjoint():
    assert chrf("je t'aime", "je t'aime") > 0.99
    assert chrf("abc", "xyz") == 0.0


def test_rhyme_similarity_fon_tone_sensitive():
    assert rhyme_similarity("an/LH", "an/LH") == 1.0
    assert rhyme_similarity("an/LH", "an/H") == 0.6
    assert rhyme_similarity("an/H", "o/H") == 0.0


def _template(text="Je chante pour toi ce soir"):
    line = LyricLine(index=0, text=text, language="fra", start=0.0, end=2.1)
    return build_template(line, beats=np.arange(0, 3, 0.3))


def test_template_uses_word_timings_when_available():
    line = LyricLine(index=0, text="bonjour toi", language="fra", start=0, end=1,
                     words=[Word(text="bonjour", start=0, end=0.6), Word(text="toi", start=0.6, end=1.0)])
    t = build_template(line)
    assert t.timing_confidence == Confidence.MEDIUM and len(t.slots) == 3
    assert abs(t.slots[2].start - 0.6) < 1e-9


def test_scoring_prefers_singable_candidate_and_rejects_overlong():
    t = _template()
    mk = lambda s, sem: score_candidate(AdaptationCandidate(text=s, language="eng", source="t"),  # noqa: E731
                                        ScoringContext(t, "eng", semantic=sem))
    a = mk("I sing for you tonight", 0.8)
    b = mk("Tonight I sing this song for you my love", 0.8)
    assert not a.rejected and b.rejected
    assert rank([b, a])[0] is a
    expl = explain_comparison([a, b])
    assert "I sing for you tonight" in expl.splitlines()[0]


def test_unmeasured_semantic_goes_to_human_review():
    c = score_candidate(AdaptationCandidate(text="I sing for you tonight", language="eng", source="t"),
                        ScoringContext(_template(), "eng", semantic=None))
    assert "sens" in c.needs_human_review


def test_fon_target_uses_tone_criterion():
    line = LyricLine(index=0, text="I love you", language="eng", start=0, end=0.9)
    t = build_template(line)
    for i, s in enumerate(t.slots):
        s.midi = [60, 67, 62][i]          # low, high, lower
    c = score_candidate(AdaptationCandidate(text="ǹ yí wǎn", language="fon", source="t"),
                        ScoringContext(t, "fon", semantic=0.7))
    names = {k.name: k for k in c.criteria}
    assert "tons" in names and "accents" not in names
    assert names["tons"].reliable is False


def test_rhyme_scheme_detection():
    ts = [build_template(LyricLine(index=i, text=x, language="fra"))
          for i, x in enumerate(["Je pense à toi", "Tu es mon roi", "Le jour se lève"])]
    assign_rhyme_scheme(ts)
    assert ts[0].rhyme_group == ts[1].rhyme_group == "A" and ts[2].rhyme_group is None

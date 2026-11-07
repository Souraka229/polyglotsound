import pytest

from app.schemas.models import Tone
from engines.syllable_engine import syllabify


def texts(line):
    return [s.text for s in line.syllables]


# ---------- French ----------
@pytest.mark.parametrize("text,lo,hi", [
    ("Mon amour est parti ce matin", 9, 9),
    ("Je chante pour toi ce soir", 6, 7),         # chant(e) : schwa optionnel
    ("Il pleure dans mon coeur", 5, 6),
    ("La vie en rose", 4, 5),
    ("créer", 2, 2),                               # hiatus é-e
    ("les enfants", 3, 3),                         # "les" monosyllabe : e non muet
])
def test_french_counts(text, lo, hi):
    line = syllabify(text, "fra")
    assert (line.min_count, line.max_count) == (lo, hi), texts(line)


def test_french_elision_before_vowel():
    line = syllabify("Elle aime", "fra")           # el-l(e) ai-m(e) : 2 a 3
    assert line.min_count == 2


def test_french_rhyme_keys_match():
    a = syllabify("Je pense à toi", "fra").rhyme_key
    b = syllabify("Tu es mon roi", "fra").rhyme_key
    assert a == b


# ---------- English ----------
@pytest.mark.parametrize("text,n", [
    ("I will always love you", 6),
    ("Tonight", 2),
    ("Beautiful morning", 5),
    ("The table", 3),
])
def test_english_counts(text, n):
    assert syllabify(text, "eng").count == n


def test_english_lexical_stress():
    line = syllabify("tonight", "eng")
    assert [s.stressed for s in line.syllables] == [False, True]


def test_english_phonetic_rhyme():
    assert syllabify("in the night", "eng").rhyme_key == syllabify("burning bright", "eng").rhyme_key


# ---------- Fon ----------
def test_fon_tones_extracted_with_combining_marks():
    line = syllabify("Nyɔ̌nu", "fon")
    assert texts(line) == ["nyɔ̌", "nu"]
    assert line.syllables[0].tone == Tone.RISING
    assert line.syllables[1].tone == Tone.UNMARKED


def test_fon_syllabic_nasal_and_nasal_vowel():
    line = syllabify("ń yí wǎn", "fon")
    assert texts(line) == ["ń", "yí", "wǎn"]
    assert [s.tone for s in line.syllables] == [Tone.HIGH, Tone.HIGH, Tone.RISING]


def test_fon_digraph_consonants():
    assert texts(syllabify("gbe kpɔ́n", "fon")) == ["gbe", "kpɔ́n"]


def test_fon_warns_when_tones_missing():
    line = syllabify("Mawu ni kpé we", "fon")
    assert line.count == 5
    assert any("sans ton" in w for w in line.warnings)

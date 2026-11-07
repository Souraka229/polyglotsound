import numpy as np
import pytest

from app.core.registry import ConsentError, Registry
from engines.audio_mixing.mixer import align_offset, mix
from engines.cultural_adaptation.lexicon import cultural_score, detect
from engines.forced_alignment.aligner import align_lyrics

STATEMENT = "J'autorise PolyglotSound a creer un modele de ma voix pour adapter mes chansons."


@pytest.fixture
def reg(tmp_path):
    return Registry(tmp_path / "r.db")


def test_consent_required_and_revocable(reg, tmp_path):
    a = reg.add_artist("Artiste Test")
    with pytest.raises(ConsentError):
        reg.require_consent(a, "voice_model")
    reg.grant_consent(a, "voice_model", STATEMENT)
    reg.require_consent(a, "voice_model")
    reg.revoke_consent(a, "voice_model")
    assert not reg.has_consent(a, "voice_model")


def test_consent_statement_must_be_explicit(reg):
    with pytest.raises(ValueError):
        reg.grant_consent(reg.add_artist("X"), "voice_model", "ok")


def test_generation_versions_and_watermark_lookup(reg, tmp_path):
    a = reg.add_artist("A")
    f = tmp_path / "song.wav"
    f.write_bytes(b"fake-audio-bytes")
    w = reg.add_work(a, f, "Chanson", "fon")
    g1, v1, m1 = reg.add_generation(w, a, "fon", "fra", "studio", [{"t": "x"}], {"nllb": "600M"})
    g2, v2, _ = reg.add_generation(w, a, "fon", "fra", "studio", [], {})
    assert (v1, v2) == (1, 2)
    assert reg.lookup_watermark(m1)[0]["title"] == "Chanson"
    reg.log_validation("lyrics", "approved", "artiste", work_id=w)
    assert reg.history(w)["validations"][0]["decision"] == "approved"


def test_mixer_restores_balance_and_alignment():
    sr = 8000
    rng = np.random.default_rng(0)
    inst = 0.1 * rng.standard_normal((2, sr * 2)).astype(np.float32)
    orig = np.zeros((2, sr * 2), np.float32)
    orig[:, 4000:8000] = 0.3 * rng.standard_normal((2, 4000))
    new = np.roll(orig, 1024, axis=-1) * 0.1                # late and too quiet
    assert abs(align_offset(orig, new, sr) + 1024) <= 512
    out, info = mix(inst, new, orig, sr)
    assert out.shape == inst.shape and np.max(np.abs(out)) <= 0.98 + 1e-6
    assert abs(info["alignment_offset_ms"] + 128) <= 64


def test_cultural_detection_and_score():
    assert detect("Mawu ni kpé we", "fon")
    good = cultural_score("Mawu ni kpé we", "fon", "Que Dieu te garde", "fra", 0.1)
    bad = cultural_score("Mawu ni kpé we", "fon", "Bonne chance a toi", "fra", 0.1)
    assert good.score == 1.0 and bad.score == 0.0 and good.reliable is False
    assert cultural_score("bonjour", "fra", "hello", "eng", 0.1) is None


def test_align_lyrics_on_segments():
    lines = align_lyrics(["Mawu ni kpé we", "ń yí wǎn nú we"], "fon", [(0.0, 2.0), (2.5, 4.0)])
    assert [l.start for l in lines] == [0.0, 2.5]
    assert lines[0].words[0].text == "mawu" and lines[0].human_verified

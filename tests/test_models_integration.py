"""Integration tests on the real downloaded models. Run: pytest -m models"""
import json
from pathlib import Path

import numpy as np
import pytest

from app.core import audio as A
from app.core.languages import REQUIRED_PAIRS

ROOT = Path(__file__).resolve().parents[1]
SYN = ROOT / "datasets" / "synthetic" / "fon_refrain_01"
pytestmark = pytest.mark.models


@pytest.fixture(scope="module")
def translator():
    from engines.translation.nllb import get_translator
    return get_translator()


@pytest.mark.parametrize("src,tgt", REQUIRED_PAIRS)
def test_nllb_all_required_directions(translator, src, tgt):
    text = {"fon": "Un yí wǎn nú we", "fra": "Je t'aime", "eng": "I love you"}[src]
    out = translator.translate(text, src, tgt, n_best=3)
    assert out and all(o.strip() for o in out)
    if tgt == "fra":
        assert "aime" in out[0].lower()
    if tgt == "eng":
        assert "love" in out[0].lower()


def test_nllb_fon_output_contains_tone_marks(translator):
    out = translator.translate("I love you", "eng", "fon")[0]
    assert any(c in out for c in "áàǎâéèěíìǐóòǒúùǔɔ́ɛ́")


def test_audioseal_roundtrip_and_false_positive():
    from engines.watermarking.audioseal_wm import detect, embed
    sr = 44100
    t = np.arange(sr * 3) / sr
    x = (0.3 * np.sin(2 * np.pi * 220 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t)))[None].astype(np.float32)
    assert not detect(x, sr).detected
    d = detect(embed(x, sr, 1234), sr)
    assert d.detected and d.message == 1234


@pytest.mark.skipif(not (SYN / "mix.wav").exists(), reason="run scripts/make_synthetic_song.py first")
def test_demucs_and_analysis_on_synthetic_refrain():
    from engines.audio_separation.separator import sdr, separate
    from engines.music_analysis.analyzer import analyze
    sr = 44100
    mix = A.load(SYN / "mix.wav", sr)[:, : 6 * sr]
    true_v = A.load(SYN / "vocals_true.wav", sr)[:, : 6 * sr]
    stems = separate(mix, sr)
    assert sdr(true_v, stems.vocals) > 15
    meta = json.loads((SYN / "meta.json").read_text(encoding="utf-8"))
    ana = analyze(mix, stems.vocals, sr, instrumental=stems.instrumental)
    assert abs(ana.tempo_bpm - meta["bpm"]) < 3

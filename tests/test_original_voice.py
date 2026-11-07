"""Original-voice preview: consent gate and reference extraction (Seed-VC mocked, no model)."""
import shutil
from pathlib import Path

import numpy as np
import pytest

from app.core import audio as A
from app.core.registry import ConsentError, Registry
from app.services import pipeline as P
from engines.singing_voice_conversion import seed_vc


@pytest.fixture
def project(tmp_path, monkeypatch):
    sr = P.SR
    t = np.arange(sr * 6) / sr
    voc = (0.3 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 0.5 * t) > 0))[None].repeat(2, 0)
    p = P.Project("t", str(tmp_path / "mix.wav"), "fon", "fra", tmp_path)
    A.save(tmp_path / "mix.wav", voc, sr)
    A.save(p.vocals_path, voc, sr)
    A.save(p.instrumental_path, voc * 0.1, sr)
    monkeypatch.setattr(P, "synth_guide", lambda proj: A.save(tmp_path / "voice" / "guide.wav", voc, sr))
    calls = []
    def fake_convert(guide, ref, out_dir, registry=None, artist_id=None, **kw):
        registry.require_consent(artist_id, "voice_conversion")
        calls.append(Path(ref))
        return seed_vc.ConversionResult(Path(guide), 0.1, [], "")
    monkeypatch.setattr(seed_vc, "convert", fake_convert)
    finals = []
    monkeypatch.setattr(P, "step_finalize", lambda proj, vocal, reg, a, title, mode, *x: finals.append(mode) or {"mode": mode})
    return p, calls, finals


def test_original_voice_requires_consent(project, tmp_path):
    p, calls, _ = project
    reg = Registry(tmp_path / "r.db")
    a = reg.add_artist("X")
    with pytest.raises(ConsentError):
        P.step_preview(p, reg, a, "t", original_voice=True)
    assert not calls


def test_original_voice_uses_separated_vocal_reference(project, tmp_path):
    p, calls, finals = project
    reg = Registry(tmp_path / "r.db")
    a = reg.add_artist("X")
    reg.grant_consent(a, "voice_conversion", "Je declare que c'est ma voix et j'autorise la conversion.")
    P.step_preview(p, reg, a, "t", original_voice=True)
    assert calls and calls[0].name == "reference_originale.wav"
    ref = A.load(calls[0], P.SR)
    assert 2.0 < ref.shape[1] / P.SR <= 25.0          # silences removed, capped at 25 s
    assert finals == ["auto-voix-originale"]


def test_synthetic_preview_needs_no_consent(project, tmp_path):
    p, calls, finals = project
    reg = Registry(tmp_path / "r.db")
    P.step_preview(p, reg, reg.add_artist("X"), "t")
    assert not calls and finals == ["apercu-voix-synthese"]

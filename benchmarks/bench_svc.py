"""Measured Seed-VC benchmark (CPU): conversion time and melody preservation.

Source: synthetic Fon guide (known melody). Reference timbre: synthetic MMS-TTS English
voice (belongs to no real person). A synthetic test artist is registered with an explicit
consent so the consent gate is exercised exactly as in production.
python benchmarks/bench_svc.py [diffusion_steps]
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from app.core import audio as A  # noqa: E402
from app.core.registry import ConsentError, Registry  # noqa: E402
from bench_synthetic import f0_metrics  # noqa: E402
from engines.music_analysis.analyzer import track_f0  # noqa: E402
from engines.singing_voice_conversion import seed_vc  # noqa: E402

DATA = ROOT / "datasets" / "synthetic"


def main(steps: int = 25):
    import librosa
    meta = json.loads((DATA / "fon_refrain_01" / "meta.json").read_text(encoding="utf-8"))
    reg = Registry(Path(tempfile.mkdtemp()) / "bench.db")
    artist = reg.add_artist("Artiste synthetique de test")
    src, ref = DATA / "fon_refrain_01" / "vocals_true.wav", DATA / "reference_voice_synthetic_eng.wav"
    res = {"steps": steps, "source": str(src.name), "reference": str(ref.name)}
    try:
        seed_vc.convert(src, ref, ROOT / "benchmarks" / "results" / "svc", registry=reg, artist_id=artist)
        res["consent_gate"] = "ECHEC : conversion acceptee sans consentement"
    except ConsentError:
        res["consent_gate"] = "ok : conversion refusee sans consentement"
    reg.grant_consent(artist, "voice_conversion", "Consentement de test pour une voix synthetique sans proprietaire.")
    out = seed_vc.convert(src, ref, ROOT / "benchmarks" / "results" / "svc", registry=reg, artist_id=artist,
                          diffusion_steps=steps)
    y = A.load(out.output, 22050, mono=True)[0]
    src_y = A.load(src, 22050, mono=True)[0]
    res.update({"seconds": out.seconds, "audio_seconds": round(len(src_y) / 22050, 1),
                "realtime_factor": round(out.seconds / (len(src_y) / 22050), 1), "output": out.output.name})
    t, f = track_f0(y, 22050)
    res["melody_after_conversion"] = f0_metrics(t, f, meta["notes"])
    ts, fs = track_f0(src_y, 22050)
    res["melody_source"] = f0_metrics(ts, fs, meta["notes"])
    (ROOT / "benchmarks" / "results" / f"svc_seedvc_{steps}steps.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))

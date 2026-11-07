"""Build a SYNTHETIC Fon refrain with known ground truth (stems, melody, beats, lyrics).

Purpose: measurable end-to-end tests (separation SDR, F0 error, beat error, watermark)
without any third-party recording. The 'singer' is MMS-TTS Fon forced onto a melody by
WORLD, so it sounds robotic: results on it are NOT representative of real singing.
Usage: python scripts/make_synthetic_song.py [out_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.audio import save  # noqa: E402
from app.schemas.models import LyricLine  # noqa: E402
from engines.musical_constraints import build_template  # noqa: E402
from engines.singing_voice_conversion.guide_synth import WSR, build_guide, speak  # noqa: E402

SR = 44100
BPM = 96
BEAT = 60 / BPM
# Lyrics: "Un yí wǎn nú we" is the common Fon "I love you"; line 2 is NLLB output for
# "Je t'aime de tout mon coeur" (machine Fon, to be checked by a native speaker).
LINES = [("Un yí wǎn nú we", [64, 67, 69, 67, 64], 1.0),
         ("Un yí wǎn nú we kpó ayi ce bǐ kpó", [64, 67, 69, 67, 64, 62, 64, 65, 64, 62, 60], 5.0)]


def instrumental(total: float, rng) -> np.ndarray:
    n = int(total * SR)
    out = np.zeros(n)
    t = np.arange(int(0.25 * SR)) / SR
    kick = np.sin(2 * np.pi * (50 + 80 * np.exp(-t * 30)) * t) * np.exp(-t * 12)
    hat = rng.standard_normal(int(0.05 * SR)) * np.exp(-np.arange(int(0.05 * SR)) / (0.008 * SR)) * 0.25
    beats = np.arange(0, total - 0.3, BEAT)
    for i, b in enumerate(beats):
        a = int(b * SR)
        out[a:a + len(kick)] += kick[: n - a] * (1.0 if i % 4 == 0 else 0.6)
        h = int((b + BEAT / 2) * SR)
        out[h:h + len(hat)] += hat[: max(0, n - h)]
    chords = [[48, 52, 55], [53, 57, 60], [55, 59, 62], [48, 52, 55]]
    bar = 4 * BEAT
    tt = np.arange(n) / SR
    for k in range(int(total / bar) + 1):
        a, e = int(k * bar * SR), min(n, int((k + 1) * bar * SR))
        for m in chords[k % 4]:
            f = 440 * 2 ** ((m - 69) / 12)
            env = np.minimum(1, (tt[a:e] - tt[a]) * 8) * np.exp(-(tt[a:e] - tt[a]) * 0.6)
            out[a:e] += 0.06 * np.sin(2 * np.pi * f * tt[a:e]) * env
    return np.stack([out, out]) * 0.8, beats


def main(out_dir: str = "datasets/synthetic/fon_refrain_01") -> None:
    out = Path(out_dir)
    rng = np.random.default_rng(7)
    total = 12.0
    rows, notes, f0_t = [], [], np.arange(0, total, 0.005)
    f0 = np.full_like(f0_t, np.nan)
    templates = []
    for text, melody, start in LINES:
        dur = BEAT  # one syllable per beat
        line = LyricLine(index=len(rows), text=text, language="fon", start=start, end=start + dur * len(melody))
        tpl = build_template(line, beats=np.arange(0, total, BEAT))
        assert len(tpl.slots) == len(melody), (text, len(tpl.slots))
        for slot, m in zip(tpl.slots, melody):
            slot.midi = float(m)
            sel = (f0_t >= slot.start) & (f0_t < slot.end - 0.03)
            vib = 0.25 * np.sin(2 * np.pi * 5.5 * f0_t[sel])
            f0[sel] = 440 * 2 ** ((m + vib - 69) / 12)
            notes.append({"start": slot.start, "end": slot.end, "midi": m})
        rows.append((text, tpl, [[i] for i in range(len(melody))]))
        templates.append({"text": text, "start": line.start, "end": line.end})
    vocal = build_guide(rows, "fon", total, f0_t, f0, SR)
    vocal = np.repeat(vocal, 2, axis=0)[:, : int(total * SR)]
    vocal = np.pad(vocal, ((0, 0), (0, int(total * SR) - vocal.shape[1])))
    inst, beats = instrumental(total, rng)
    mix = 0.7 * vocal + inst
    peak = np.max(np.abs(mix))
    scale = 0.9 / peak
    save(out / "vocals_true.wav", vocal * 0.7 * scale, SR)
    save(out / "instrumental_true.wav", inst * scale, SR)
    save(out / "mix.wav", mix * scale, SR)
    meta = {"synthetic": True, "sr": SR, "bpm": BPM, "beats": beats.tolist(), "lines": templates, "notes": notes,
            "warning": "Voix synthetique (MMS-TTS + WORLD). Ne represente pas un chant reel."}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    # reference timbre for SVC tests: a synthetic MMS-TTS English voice (belongs to no real person)
    import librosa
    ref = speak("This is a synthetic reference voice used only for testing the conversion pipeline. "
                "It belongs to no real person.", "eng").astype(np.float32)
    save(out.parent / "reference_voice_synthetic_eng.wav", librosa.resample(ref, orig_sr=WSR, target_sr=SR)[None], SR)
    print("ok", out)


if __name__ == "__main__":
    main(*sys.argv[1:])

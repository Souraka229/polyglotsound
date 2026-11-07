"""Measured benchmark on the synthetic refrain (ground truth known). Writes benchmarks/results/*.json.

python benchmarks/bench_synthetic.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core import audio as A  # noqa: E402
from engines.audio_separation.separator import sdr, separate  # noqa: E402
from engines.music_analysis.analyzer import analyze  # noqa: E402

DATA = ROOT / "datasets" / "synthetic" / "fon_refrain_01"
OUT = ROOT / "benchmarks" / "results"
SR = 44100


def f0_metrics(times, f0, notes):
    errs, voiced_hits, total = [], 0, 0
    for n in notes:
        sel = (times >= n["start"] + 0.05) & (times < n["end"] - 0.08)
        total += int(sel.sum())
        est = f0[sel]
        ok = np.isfinite(est)
        voiced_hits += int(ok.sum())
        if ok.any():
            midi = 69 + 12 * np.log2(est[ok] / 440)
            errs += list(np.abs(midi - n["midi"]))
    errs = np.array(errs)
    return {"voicing_recall": round(voiced_hits / max(total, 1), 3),
            "median_abs_err_semitones": round(float(np.median(errs)), 3) if errs.size else None,
            "raw_pitch_accuracy_50c": round(float(np.mean(errs <= 0.5)), 3) if errs.size else None}


def beat_f(est, ref, tol=0.07):
    est, ref = np.asarray(est), np.asarray(ref)
    if not len(est) or not len(ref):
        return 0.0
    hits = sum(1 for r in ref if np.min(np.abs(est - r)) <= tol)
    p, r = hits / len(est), hits / len(ref)
    return round(2 * p * r / (p + r), 3) if p + r else 0.0


def main():
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    mix = A.load(DATA / "mix.wav", SR)
    v_true = A.load(DATA / "vocals_true.wav", SR)
    i_true = A.load(DATA / "instrumental_true.wav", SR)
    res = {"dataset": "synthetic/fon_refrain_01 (voix MMS-TTS+WORLD, NON representative du chant reel)",
           "machine": "Intel i5-1135G7, 4 coeurs, 16 Go RAM, CPU seul", "date": time.strftime("%Y-%m-%d %H:%M")}
    t0 = time.perf_counter()
    stems = separate(mix, SR, "demucs")
    res["separation"] = {"backend": stems.backend, "seconds": round(time.perf_counter() - t0, 1),
                         "audio_seconds": round(mix.shape[1] / SR, 1),
                         "vocals_sdr_db": round(sdr(v_true, stems.vocals), 2),
                         "instrumental_sdr_db": round(sdr(i_true, stems.instrumental), 2),
                         "baseline_mix_as_vocals_sdr_db": round(sdr(v_true, mix), 2)}
    t1 = time.perf_counter()
    ana = analyze(mix, stems.vocals, SR, instrumental=stems.instrumental)
    res["analysis_seconds"] = round(time.perf_counter() - t1, 1)
    res["tempo"] = {"estimated_bpm": round(ana.tempo_bpm, 1), "true_bpm": meta["bpm"],
                    "beat_f_measure_70ms": beat_f(ana.beats, meta["beats"])}
    res["f0_on_separated_vocals"] = f0_metrics(ana.f0_times, ana.f0_hz, meta["notes"])
    from engines.music_analysis.analyzer import track_f0
    import librosa
    vt = librosa.resample(v_true.mean(0), orig_sr=SR, target_sr=22050)
    tt, ft = track_f0(vt, 22050)
    res["f0_on_true_vocals"] = f0_metrics(tt, ft, meta["notes"])
    res["key"] = {"estimated": ana.key, "note": "accords C-F-G-C : C major attendu"}

    from engines.watermarking.audioseal_wm import detect, embed
    msg = 0xBEEF
    t2 = time.perf_counter()
    wm = embed(stems.vocals, SR, msg)
    emb_s = round(time.perf_counter() - t2, 2)
    snr = 10 * np.log10(np.sum(stems.vocals ** 2) / (np.sum((wm - stems.vocals) ** 2) + 1e-12))
    mixed = wm + stems.instrumental
    A.save(OUT / "_tmp_wm.wav", mixed, SR)
    A.to_mp3(OUT / "_tmp_wm.wav", OUT / "_tmp_wm.mp3", "128k")
    after_mp3 = A.load(OUT / "_tmp_wm.mp3", SR)
    checks = {"clean_vocal_unmarked": detect(stems.vocals, SR), "vocal_marked": detect(wm, SR),
              "mix_marked": detect(mixed, SR), "mix_marked_mp3_128k": detect(after_mp3, SR),
              "original_mix_unmarked": detect(mix, SR)}
    res["watermark"] = {"embed_seconds": emb_s, "snr_db": round(float(snr), 1), "message": msg,
                        **{k: {"p": round(v.probability, 3), "message_ok": v.message == msg} for k, v in checks.items()}}
    for f in ("_tmp_wm.wav", "_tmp_wm.mp3"):
        (OUT / f).unlink(missing_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "synthetic_pipeline.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

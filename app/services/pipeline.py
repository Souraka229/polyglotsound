"""End-to-end PolyglotSound pipeline, step by step, with artist validation between steps.

Step 1 analyse   : load -> separate -> analyse music -> ASR (or manual lyrics for Fon)
Step 2 lyrics    : artist corrects lyrics -> alignment -> musical templates
Step 3 adapt     : 3 scored adaptations per line, artist picks one per line
Step 4 voice     : Studio (guide vocal -> artist timbre) or Auto (experimental)
Step 5 finalise  : watermark -> mix over original instrumental -> export + registry
The original file is never modified; everything is written to the project folder.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.core import audio as A
from app.schemas.models import LineTemplate, LyricLine

ROOT = Path(__file__).resolve().parents[2]
PROJECTS = ROOT / "datasets" / "projects"
SR = 44100


@dataclass
class Project:
    id: str
    source_path: str
    source_lang: str
    target_lang: str
    dir: Path
    analysis: dict = field(default_factory=dict)
    asr_notes: list[str] = field(default_factory=list)
    lines: list[LyricLine] = field(default_factory=list)
    templates: list[LineTemplate] = field(default_factory=list)
    adaptations: list[dict] = field(default_factory=list)      # per line: literal, top3, explanation
    chosen: list[str] = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    beats: list[float] = field(default_factory=list)

    @property
    def vocals_path(self) -> Path:
        return self.dir / "stems" / "vocals.wav"

    @property
    def instrumental_path(self) -> Path:
        return self.dir / "stems" / "instrumental.wav"

    def save(self) -> None:
        data = {"id": self.id, "source_path": self.source_path, "source_lang": self.source_lang,
                "target_lang": self.target_lang, "analysis": self.analysis, "asr_notes": self.asr_notes,
                "lines": [l.model_dump() for l in self.lines], "templates": [t.model_dump() for t in self.templates],
                "adaptations": self.adaptations, "chosen": self.chosen, "timings": self.timings, "beats": self.beats}
        (self.dir / "project.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def new_project(source_path: str, source_lang: str, target_lang: str) -> Project:
    pid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    d = PROJECTS / pid
    d.mkdir(parents=True, exist_ok=True)
    return Project(pid, str(source_path), source_lang, target_lang, d)


def step_analyse(p: Project, separation_backend: str = "demucs", run_asr: bool = True) -> Project:
    from engines.audio_separation.separator import save_stems, separate
    from engines.music_analysis.analyzer import analyze
    from engines.speech_recognition.asr import transcribe
    t0 = time.perf_counter()
    mix = A.load(p.source_path, SR)
    p.analysis["input"] = A.describe(mix, SR)
    stems = separate(mix, SR, separation_backend)
    save_stems(stems, p.dir / "stems")
    p.timings["separation_s"] = round(time.perf_counter() - t0, 1)
    t1 = time.perf_counter()
    ana = analyze(mix, stems.vocals, SR, instrumental=stems.instrumental)
    p.analysis.update(ana.summary())
    p.analysis["separation_backend"] = stems.backend
    p.beats = ana.beats
    np.savez(p.dir / "f0.npz", times=ana.f0_times, f0=ana.f0_hz)
    p.analysis["voiced_segments_list"] = ana.voiced_segments
    p.timings["analysis_s"] = round(time.perf_counter() - t1, 1)
    if run_asr:
        t2 = time.perf_counter()
        import librosa
        v16 = librosa.resample(stems.vocals.mean(0), orig_sr=SR, target_sr=16000)
        p.lines, p.asr_notes = transcribe(v16, p.source_lang, ana.voiced_segments)
        p.timings["asr_s"] = round(time.perf_counter() - t2, 1)
    p.save()
    return p


def load_f0(p: Project):
    f = p.dir / "f0.npz"
    if not f.exists():
        return None, None
    z = np.load(f)
    return z["times"], z["f0"]


def step_lyrics(p: Project, corrected_text: str) -> Project:
    from engines.forced_alignment.aligner import align_lyrics
    from engines.musical_constraints import assign_rhyme_scheme, build_template
    texts = [t for t in corrected_text.splitlines() if t.strip()]
    segs = [tuple(s) for s in p.analysis.get("voiced_segments_list", [])]
    p.lines = align_lyrics(texts, p.source_lang, segs, existing=p.lines or None)
    times, f0 = load_f0(p)
    p.templates = [build_template(l, times, f0, p.beats) for l in p.lines]
    assign_rhyme_scheme(p.templates)
    p.save()
    return p


def step_adapt(p: Project, use_llm: bool = True, llm_model: str | None = None) -> Project:
    from app.services.adaptation import adapt_line
    from engines.translation.nllb import get_translator
    tr = get_translator()
    p.adaptations, chosen_keys = [], {}
    t0 = time.perf_counter()
    for tpl in p.templates:
        partners = [chosen_keys[tpl.rhyme_group]] if tpl.rhyme_group in chosen_keys else []
        res = adapt_line(tpl, p.source_lang, p.target_lang, tr, partner_keys=partners,
                         use_llm=use_llm, llm_model=llm_model)
        top = res.top3
        if tpl.rhyme_group and top and tpl.rhyme_group not in chosen_keys:
            from engines.syllable_engine import syllabify
            chosen_keys[tpl.rhyme_group] = syllabify(top[0].text, p.target_lang).rhyme_key
        p.adaptations.append({"original": tpl.line.text, "literal": res.literal, "explanation": res.explanation,
                              "log": res.log, "top3": [c.model_dump() for c in top],
                              "n_candidates": len(res.candidates)})
    p.chosen = [a["top3"][0]["text"] if a["top3"] else "" for a in p.adaptations]
    p.timings["adaptation_s"] = round(time.perf_counter() - t0, 1)
    p.save()
    return p


def step_voice(p: Project, mode: str, artist_reference: str, registry, artist_id: int,
               guide_vocal: str | None = None, diffusion_steps: int = 25) -> Path:
    """mode 'studio': convert a recorded guide vocal; mode 'auto' (experimental): synthesise the guide first."""
    from engines.singing_voice_conversion import seed_vc
    registry.require_consent(artist_id, "voice_conversion")
    vdir = p.dir / "voice"
    vdir.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    if mode == "auto":
        from engines.singing_voice_conversion.guide_synth import build_guide
        times, f0 = load_f0(p)
        rows = []
        for i, tpl in enumerate(p.templates):
            text = p.chosen[i] if i < len(p.chosen) else ""
            cand = next((c for c in p.adaptations[i]["top3"] if c["text"] == text), None) if p.adaptations else None
            mapping = cand["alignment"] if cand else []
            if text:
                rows.append((text, tpl, mapping))
        total = p.analysis.get("input", {}).get("duration_s", 30.0)
        guide = build_guide(rows, p.target_lang, total, times, f0, SR)
        guide_vocal = str(A.save(vdir / "guide_auto.wav", guide, SR))
        p.timings["guide_synth_s"] = round(time.perf_counter() - t0, 1)
    elif not guide_vocal:
        raise ValueError("Mode Studio : la voix guide enregistree est obligatoire.")
    res = seed_vc.convert(guide_vocal, artist_reference, vdir, registry=registry, artist_id=artist_id,
                          diffusion_steps=diffusion_steps)
    p.timings["voice_conversion_s"] = res.seconds
    p.save()
    return res.output


def step_finalize(p: Project, new_vocal: str, registry, artist_id: int, title: str, mode: str,
                  vocal_gain_db: float = 0.0) -> dict:
    from engines.audio_mixing.mixer import mix
    from engines.watermarking.audioseal_wm import detect, embed
    work_id = registry.add_work(artist_id, p.source_path, title, p.source_lang)
    lyrics = [{"original": l.text, "adapted": p.chosen[i] if i < len(p.chosen) else ""} for i, l in enumerate(p.lines)]
    models = {"separation": p.analysis.get("separation_backend"), "translation": "nllb-200-distilled-600M-ct2-int8",
              "svc": "seed-vc DiT_seed_v2_uvit_whisper_base_f0_44k", "watermark": "audioseal_wm_16bits"}
    gid, version, message = registry.add_generation(work_id, artist_id, p.source_lang, p.target_lang, mode,
                                                    lyrics, models)
    voc = A.load(new_vocal, SR)
    voc_wm = embed(voc, SR, message)                     # watermark the generated vocal itself
    inst = A.load(p.instrumental_path, SR)
    orig_voc = A.load(p.vocals_path, SR)
    out, info = mix(inst, voc_wm, orig_voc, SR, vocal_gain_db)
    # the vocal mark is diluted by the instrumental (p=0.82 measured), so the mix is marked too
    out = np.clip(embed(out, SR, message), -1.0, 1.0).astype(np.float32)
    out_dir = p.dir / "export"
    wav = A.save(out_dir / f"{p.id}_{p.target_lang}_v{version}.wav", out, SR)
    mp3 = A.to_mp3(wav, wav.with_suffix(".mp3"))
    A.save(out_dir / f"{p.id}_{p.target_lang}_v{version}_vocal.wav", voc_wm, SR)
    registry.finalize_generation(gid, wav)
    check_vocal = detect(voc_wm, SR)
    check_mix = detect(out, SR)
    registry.log_validation("final_render", "generated", "systeme", f"watermark={message}", gid, work_id)
    result = {"generation_id": gid, "version": version, "watermark_message": message, "wav": str(wav),
              "mp3": str(mp3), "mix_info": info,
              "watermark_check_vocal": vars(check_vocal), "watermark_check_mix": vars(check_mix)}
    (out_dir / "provenance.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    return result

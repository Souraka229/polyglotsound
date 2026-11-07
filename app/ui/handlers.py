"""UI callbacks: thin layer between Gradio and app.services.pipeline."""
from __future__ import annotations

import json

import gradio as gr

from app.core import audio as A
from app.core.languages import LANGUAGES
from app.core.registry import ConsentError, Registry
from app.services import pipeline as P
from app.ui.render import adaptation_cards, timeline_figure

REG = Registry()
_PROJECTS: dict[str, P.Project] = {}
LANG_CHOICES = [(f"{v.name}", k) for k, v in LANGUAGES.items() if v.nllb_code]


def _proj(pid: str) -> P.Project:
    if not pid or pid not in _PROJECTS:
        raise gr.Error("Commencez par importer et analyser un morceau (etape 1).")
    return _PROJECTS[pid]


def _fig(p: P.Project):
    times, f0 = P.load_f0(p)
    voc = A.load(p.vocals_path, P.SR)
    return timeline_figure(voc, P.SR, times, f0, p.beats, p.templates)


def analyse(file, src, tgt, progress=gr.Progress()):
    if not file:
        raise gr.Error("Importez un fichier MP3, WAV ou FLAC.")
    if src == tgt:
        raise gr.Error("Choisissez une langue cible differente de la langue originale.")
    progress(0.05, desc="Separation voix / instrumental (Demucs, CPU : environ 1 min pour 30 s)")
    p = P.new_project(file, src, tgt)
    P.step_analyse(p)
    _PROJECTS[p.id] = p
    a = p.analysis
    info = (f"**Duree** {a['input']['duration_s']} s · **Tempo** {a['tempo_bpm']} BPM · **Tonalite (indicative, non utilisee dans les scores)** {a['key']} "
            f"· **Segments chantes** {a['voiced_segments']} · **Temps de calcul** {p.timings}")
    notes = "\n".join(f"- {n}" for n in p.asr_notes)
    lyrics = "\n".join(l.text for l in p.lines if l.text)
    conf = "\n".join(f"Ligne {l.index + 1} : confiance {l.confidence.value}" for l in p.lines if l.text)
    hint = notes + ("\n\n" + conf if conf else "")
    return (p.id, str(p.vocals_path), str(p.instrumental_path), info, hint, lyrics, _fig(p))


def validate_lyrics(pid, text, artist):
    p = _proj(pid)
    if not text.strip():
        raise gr.Error("Saisissez ou corrigez les paroles (une ligne chantee par ligne).")
    P.step_lyrics(p, text)
    a_id = REG.add_artist(artist or "Artiste")
    w_id = REG.add_work(a_id, p.source_path, "", p.source_lang)
    REG.log_validation("lyrics", "approved", artist or "artiste", f"{len(p.lines)} lignes", work_id=w_id)
    rows = []
    for t in p.templates:
        syl = " · ".join(s.text for s in t.syllables)
        rows.append([t.line.index + 1, t.line.text, len(t.slots), syl, t.rhyme_group or "-", t.timing_confidence.value])
    return rows, _fig(p)


def adapt(pid, use_llm, model, progress=gr.Progress()):
    p = _proj(pid)
    if not p.templates:
        raise gr.Error("Validez d'abord les paroles (etape 2).")
    progress(0.1, desc="Traduction NLLB + adaptations + verification musicale")
    P.step_adapt(p, use_llm=use_llm, llm_model=model or None)
    table = [[i + 1, a["original"], p.chosen[i]] for i, a in enumerate(p.adaptations)]
    logs = "\n".join(f"Ligne {i + 1} : " + " | ".join(a["log"]) for i, a in enumerate(p.adaptations))
    return adaptation_cards(p.adaptations), table, logs


def choose(pid, table, artist):
    p = _proj(pid)
    rows = table.values.tolist() if hasattr(table, "values") else table
    p.chosen = [str(r[2]).strip() for r in rows]
    p.save()
    a_id = REG.add_artist(artist or "Artiste")
    w_id = REG.add_work(a_id, p.source_path, "", p.source_lang)
    REG.log_validation("adaptation_choice", "approved", artist or "artiste",
                       json.dumps(p.chosen, ensure_ascii=False), work_id=w_id)
    return "Choix enregistres : " + " / ".join(p.chosen)


def grant(artist, statement, agree):
    if not agree:
        raise gr.Error("Le consentement explicite de l'artiste est obligatoire.")
    a_id = REG.add_artist(artist or "Artiste")
    REG.grant_consent(a_id, "voice_model", statement)
    REG.grant_consent(a_id, "voice_conversion", statement)
    return f"Consentement enregistre pour {artist} (portee : modele vocal + conversion)."


def convert(pid, artist, mode, reference, guide, steps, progress=gr.Progress()):
    p = _proj(pid)
    if not reference:
        raise gr.Error("Ajoutez un extrait de 5 a 30 s de la voix de l'artiste (reference de timbre).")
    a_id = REG.add_artist(artist or "Artiste")
    progress(0.1, desc="Conversion de voix (Seed-VC, CPU : plusieurs minutes)")
    try:
        out = P.step_voice(p, "auto" if mode.startswith("Auto") else "studio", reference, REG, a_id,
                           guide_vocal=guide, diffusion_steps=int(steps))
    except ConsentError as exc:
        raise gr.Error(str(exc))
    return str(out), str(out), f"Temps : {p.timings}"


def finalize(pid, artist, title, mode, vocal, gain):
    p = _proj(pid)
    if not vocal:
        raise gr.Error("Aucune voix convertie : terminez l'etape 4.")
    a_id = REG.add_artist(artist or "Artiste")
    res = P.step_finalize(p, vocal, REG, a_id, title or "Sans titre",
                          "auto" if str(mode).startswith("Auto") else "studio", gain)
    REG.log_validation("final_render", "approved", artist or "artiste", res["wav"], res["generation_id"])
    return p.source_path, res["wav"], [res["wav"], res["mp3"]], json.dumps(res, indent=1, ensure_ascii=False)

"""UI callbacks: thin layer between Gradio and app.services.pipeline."""
from __future__ import annotations

import json
from pathlib import Path

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


NEXT_AFTER_ANALYSE = ("**Etape suivante** : ouvrez l'onglet **2 · Paroles**, corrigez les paroles (une ligne chantee "
                      "par ligne), puis cliquez sur **Creer la nouvelle chanson**.")


def analyse(file, src, tgt, start_s, dur_s, progress=gr.Progress()):
    if not file:
        raise gr.Error("Importez un fichier MP3, WAV ou FLAC.")
    if src == tgt:
        raise gr.Error("Choisissez une langue cible differente de la langue originale.")
    p = P.new_project(file, src, tgt)
    if Path(file).suffix.lower() not in A.SUPPORTED:
        raise gr.Error(f"Format {Path(file).suffix} non pris en charge. Formats : {', '.join(sorted(A.SUPPORTED))}")
    is_video = Path(file).suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".3gp"}
    if (dur_s and dur_s > 0) or is_video:
        progress(0.02, desc="Decoupe de l'extrait")
        p.source_path = str(P.make_excerpt(file, float(start_s or 0), float(dur_s), p.dir / "extrait.wav"))
        p.analysis["original_file"] = str(file)
        p.analysis["excerpt"] = {"start_s": float(start_s or 0), "duration_s": float(dur_s)}
    progress(0.05, desc="Separation voix / instrumental (Demucs). CPU : environ 1 min pour 12 s d'audio")
    P.step_analyse(p)
    _PROJECTS[p.id] = p
    a = p.analysis
    info = (f"**Duree traitee** {a['input']['duration_s']} s · **Tempo** {a['tempo_bpm']} BPM · "
            f"**Tonalite (indicative)** {a['key']} · **Segments chantes** {a['voiced_segments']} · "
            f"**Temps de calcul** {p.timings}\n\n{NEXT_AFTER_ANALYSE}")
    notes = "\n".join(f"- {n}" for n in p.asr_notes)
    lyrics = "\n".join(l.text for l in p.lines if l.text)
    conf = "\n".join(f"Ligne {l.index + 1} : confiance {l.confidence.value}" for l in p.lines if l.text)
    hint = notes + ("\n\n" + conf if conf else "")
    return (p.id, str(p.vocals_path), str(p.instrumental_path), info, hint, lyrics, _fig(p))


OWN_VOICE_STATEMENT = ("Je declare que la voix chantee dans ce morceau est la mienne et j'autorise PolyglotSound "
                       "a la convertir pour chanter l'adaptation de ma propre oeuvre.")


def quick_song(pid, text, artist, title, use_llm, model, own_voice=False, progress=gr.Progress()):
    """One click: lyrics -> 3 scored adaptations -> best chosen -> sung by a synthetic voice -> mix -> player."""
    p = _proj(pid)
    if not text.strip():
        raise gr.Error("Saisissez ou corrigez les paroles (une ligne chantee par ligne).")
    progress(0.05, desc="Placement des syllabes sur la melodie")
    rows, fig = validate_lyrics(pid, text, artist)
    progress(0.2, desc=f"Adaptations ({len(p.templates)} lignes, environ 3 s par ligne sans IA, 15 s avec)")
    P.step_adapt(p, use_llm=use_llm, llm_model=model or None)
    a_id = REG.add_artist(artist or "Artiste")
    if own_voice:
        from engines.singing_voice_conversion import seed_vc
        ok, msg = seed_vc.available()
        if not ok:
            raise gr.Error(f"Voix d'origine indisponible : {msg}")
        REG.grant_consent(a_id, "voice_conversion", OWN_VOICE_STATEMENT)
        progress(0.6, desc="Voix guide puis conversion au timbre d'origine (Seed-VC ; GPU : ~1 min, CPU : tres long)")
    else:
        progress(0.7, desc="La voix de synthese chante les nouvelles paroles sur la melodie originale")
    res = P.step_preview(p, REG, a_id, title or "Sans titre", original_voice=bool(own_voice))
    table = [[i + 1, a["original"], p.chosen[i]] for i, a in enumerate(p.adaptations)]
    voice = ("chantee avec la voix d'origine (Seed-VC, consentement enregistre)" if own_voice else
             "voix de synthese experimentale, la voix de personne n'est clonee. Pour la voix d'origine, cochez "
             "la case au-dessus du bouton (uniquement si c'est VOTRE voix)")
    status = (f"**Nouvelle chanson prete** (version {res['version']}, filigrane n° {res['watermark_message']}) : "
              f"{voice}.\n\nVous pouvez remplacer une ligne dans l'onglet 3 puis cliquer sur *Reecouter avec mes choix*.")
    return (rows, fig, adaptation_cards(p.adaptations), table, p.source_path, res["wav"],
            [res["wav"], res["mp3"]], status)


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


def preview_choice(pid, table, artist, title, progress=gr.Progress()):
    """Re-sing the song with the lines chosen or edited by the artist."""
    p = _proj(pid)
    choose(pid, table, artist)
    for i, text in enumerate(p.chosen):                  # score edited lines so their alignment is known
        if p.adaptations and i < len(p.adaptations) and not any(c["text"] == text for c in p.adaptations[i]["top3"]):
            from app.schemas.models import AdaptationCandidate
            from engines.musical_constraints import ScoringContext, score_candidate
            c = score_candidate(AdaptationCandidate(text=text, language=p.target_lang, source="manual"),
                                ScoringContext(p.templates[i], p.target_lang))
            p.adaptations[i]["top3"].append(c.model_dump())
    progress(0.3, desc="La voix de synthese chante vos choix")
    res = P.step_preview(p, REG, REG.add_artist(artist or "Artiste"), title or "Sans titre")
    return p.source_path, res["wav"], [res["wav"], res["mp3"]], (
        f"**Nouvelle version {res['version']} prete** avec vos choix : ecoutez-la dans l'onglet **2 · Paroles**.")


def translate_text(text, src, tgt, use_llm, model, progress=gr.Progress()):
    """Lyrics only, no audio: faithful translation + best singable adaptation per line (same syllable count)."""
    import numpy as np

    from app.schemas.models import LyricLine
    from app.services.adaptation import adapt_line
    from engines.musical_constraints import assign_rhyme_scheme, build_template
    from engines.syllable_engine import syllabify
    from engines.translation.nllb import get_translator
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    if not lines:
        raise gr.Error("Collez des paroles : une ligne chantee par ligne.")
    if src == tgt:
        raise gr.Error("Choisissez une langue cible differente de la langue originale.")
    if len(lines) > 60:
        raise gr.Error("60 lignes maximum a la fois (environ 3 s par ligne).")
    step = 0.35                                            # even synthetic timing: one syllable per 0.35 s
    templates = []
    for i, l in enumerate(lines):
        n = max(1, syllabify(l, src).count)
        templates.append(build_template(LyricLine(index=i, text=l, language=src, start=0.0, end=n * step),
                                        beats=np.arange(0, n * step + 1, 2 * step)))
    assign_rhyme_scheme(templates)
    tr = get_translator()
    rows, adaptations, rhyme_keys = [], [], {}
    for i, tpl in enumerate(templates):
        progress((i + 0.5) / len(templates), desc=f"Ligne {i + 1}/{len(templates)}")
        partners = [rhyme_keys[tpl.rhyme_group]] if tpl.rhyme_group in rhyme_keys else []
        res = adapt_line(tpl, src, tgt, tr, partner_keys=partners, use_llm=use_llm, llm_model=model or None)
        best = res.top3[0]
        if tpl.rhyme_group and tpl.rhyme_group not in rhyme_keys:
            rhyme_keys[tpl.rhyme_group] = syllabify(best.text, tgt).rhyme_key
        rows.append([i + 1, tpl.line.text, res.literal, best.text,
                     f"{len(tpl.slots)} -> {syllabify(best.text, tgt).count}",
                     "?" if best.total is None else int(round(100 * best.total))])
        adaptations.append({"original": tpl.line.text, "literal": res.literal, "explanation": res.explanation,
                            "log": res.log, "top3": [c.model_dump() for c in res.top3]})
    import tempfile
    out = Path(tempfile.mkdtemp()) / f"traduction_{src}_{tgt}.txt"
    out.write_text("\n".join(f"{r[1]}\n  fidele   : {r[2]}\n  chantable: {r[3]}\n" for r in rows), encoding="utf-8")
    return rows, adaptation_cards(adaptations), str(out)


def auto_translate(text, src, tgt, progress=gr.Progress()):
    """Run right after the analysis: propose the lyrics in the chosen language (fast, no LLM)."""
    if not (text or "").strip():
        msg = ("Aucune parole reconnue automatiquement"
               + (" (Whisper ne reconnait pas le fon)" if src == "fon" else "")
               + " : tapez les paroles dans la case ci-dessus puis cliquez sur **Traduire ces paroles**.")
        return [], "", None, msg
    rows, cards, path = translate_text(text, src, tgt, False, None, progress=progress)
    return rows, cards, path, (f"**Proposition en {LANGUAGES[tgt].name}** pour {len(rows)} ligne(s). "
                               "Corrigez les paroles si besoin puis cliquez sur *Traduire ces paroles*, ou allez a "
                               "l'onglet **2 · Paroles** pour creer la nouvelle chanson.")


def auto_song(pid, text, artist, title, progress=gr.Progress()):
    """After the translation: sing the new version (synthetic voice) so it can be heard right away."""
    if not pid or pid not in _PROJECTS:
        return None, None, "Importez et analysez d'abord un morceau pour entendre la version traduite."
    if not (text or "").strip():
        return None, None, "Tapez les paroles puis cliquez sur **Traduire ces paroles** pour entendre la version traduite."
    out = quick_song(pid, text, artist, title, False, None, False, progress=progress)
    return out[5], out[6], ("**Version traduite chantee** (voix de synthese experimentale) : ecoutez-la ci-dessus. "
                            "Pour la voix d'origine ou pour choisir d'autres propositions : onglets **2** et **3**.")


def translation_audio(table, tgt, progress=gr.Progress()):
    """Read the singable translation aloud with the synthetic voice of the target language (no song needed)."""
    import tempfile

    import numpy as np

    from engines.singing_voice_conversion.guide_synth import WSR, speak
    rows = table.values.tolist() if hasattr(table, "values") else (table or [])
    lines = [str(r[3]).strip() for r in rows if len(r) > 3 and str(r[3]).strip()]
    if not lines:
        return None, "Traduisez d'abord des paroles."
    parts, pause = [], np.zeros(int(0.6 * WSR), np.float32)
    for i, l in enumerate(lines):
        progress((i + 0.5) / len(lines), desc=f"Lecture de la ligne {i + 1}/{len(lines)}")
        parts += [speak(l, tgt).astype(np.float32), pause]
    wav = np.concatenate(parts)
    wav = 0.9 * wav / (np.max(np.abs(wav)) or 1.0)
    out = Path(tempfile.mkdtemp()) / f"traduction_{tgt}.wav"
    A.save(out, wav[None], WSR)
    return str(out), (f"**Traduction en audio** ({LANGUAGES[tgt].name}, voix de synthese lue). Pour l'entendre "
                      "chantee sur votre musique : onglet **1 · Morceau**.")

"""PolyglotSound HTTP API (FastAPI). Same pipeline as the UI, for integration in other tools.

uvicorn app.api.main:app --port 8000   ->  http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.core.languages import LANGUAGES, can_translate
from app.core.registry import ConsentError, Registry
from app.services import pipeline as P

app = FastAPI(title="PolyglotSound API", version="0.1.0")
REG = Registry()
PROJECTS: dict[str, P.Project] = {}


class LyricsIn(BaseModel):
    text: str


class AdaptIn(BaseModel):
    use_llm: bool = True
    llm_model: str | None = None


class ChoiceIn(BaseModel):
    chosen: list[str]
    artist: str


class ConsentIn(BaseModel):
    artist: str
    statement: str
    scopes: list[str] = ["voice_model", "voice_conversion"]


class LineScoreIn(BaseModel):
    original: str
    source_lang: str
    target_lang: str
    candidates: list[str]
    duration_s: float | None = None


def _get(pid: str) -> P.Project:
    if pid not in PROJECTS:
        raise HTTPException(404, "Projet inconnu")
    return PROJECTS[pid]


@app.get("/languages")
def languages():
    return {k: {"name": v.name, "translation": bool(v.nllb_code), "asr_windows": bool(v.whisper_code),
                "tonal": v.tonal, "notes": v.notes} for k, v in LANGUAGES.items()}


@app.post("/projects")
def create(file: UploadFile = File(...), source_lang: str = Form(...), target_lang: str = Form(...)):
    if not can_translate(source_lang, target_lang):
        raise HTTPException(400, f"Paire {source_lang}->{target_lang} non prise en charge")
    tmp = P.PROJECTS / "_uploads"
    tmp.mkdir(parents=True, exist_ok=True)
    path = tmp / Path(file.filename or "upload.wav").name
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    p = P.new_project(str(path), source_lang, target_lang)
    P.step_analyse(p)
    PROJECTS[p.id] = p
    return {"id": p.id, "analysis": p.analysis, "asr_notes": p.asr_notes,
            "lines": [l.model_dump() for l in p.lines], "timings": p.timings}


@app.post("/projects/{pid}/lyrics")
def lyrics(pid: str, body: LyricsIn):
    p = P.step_lyrics(_get(pid), body.text)
    return {"templates": [t.model_dump() for t in p.templates]}


@app.post("/projects/{pid}/adapt")
def adapt(pid: str, body: AdaptIn):
    p = P.step_adapt(_get(pid), use_llm=body.use_llm, llm_model=body.llm_model)
    return {"adaptations": p.adaptations, "chosen": p.chosen}


@app.post("/projects/{pid}/choice")
def choice(pid: str, body: ChoiceIn):
    p = _get(pid)
    p.chosen = body.chosen
    p.save()
    a = REG.add_artist(body.artist)
    w = REG.add_work(a, p.source_path, "", p.source_lang)
    REG.log_validation("adaptation_choice", "approved", body.artist, " / ".join(body.chosen), work_id=w)
    return {"ok": True}


@app.post("/consents")
def consent(body: ConsentIn):
    a = REG.add_artist(body.artist)
    try:
        ids = [REG.grant_consent(a, s, body.statement) for s in body.scopes]
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"artist_id": a, "consent_ids": ids}


@app.post("/score")
def score(body: LineScoreIn):
    """Score candidate adaptations of one line without audio (synthetic even timing)."""
    from app.schemas.models import AdaptationCandidate, LyricLine
    from engines.musical_constraints import ScoringContext, build_template, explain_comparison, score_candidate
    from engines.syllable_engine import syllabify
    n = syllabify(body.original, body.source_lang).count
    line = LyricLine(index=0, text=body.original, language=body.source_lang, start=0.0,
                     end=body.duration_s or n * 0.35)
    tpl = build_template(line)
    cands = [score_candidate(AdaptationCandidate(text=c, language=body.target_lang, source="api"),
                             ScoringContext(tpl, body.target_lang)) for c in body.candidates]
    return {"candidates": [c.model_dump() for c in cands], "explanation": explain_comparison(cands)}


@app.get("/watermark/{message}")
def watermark_lookup(message: int):
    return REG.lookup_watermark(message)


@app.exception_handler(ConsentError)
def consent_error(_, exc: ConsentError):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=403, content={"detail": str(exc)})

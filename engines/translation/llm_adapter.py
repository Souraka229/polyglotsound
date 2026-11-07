"""Constrained artistic adaptation with a local LLM (Ollama).

The LLM never writes freely: it receives the literal translation, the target
syllable count, the stress pattern and the rhyme to reach, and must answer in
JSON. Every proposal is then *verified* by the Musical Constraint Engine; the
ones that fail are sent back with the measured errors (regeneration loop).

Known limit: small local LLMs (llama3.2 3B, gemma) write French and English
well but do NOT write reliable Fon. For Fon targets the candidates come from
NLLB n-best and the artist / a native speaker, and the LLM is not used.
"""
from __future__ import annotations

import json
import os
import re

import requests

OLLAMA_URL = os.environ.get("POLYGLOT_OLLAMA_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.environ.get("POLYGLOT_LLM", "llama3.2:latest")
LANG_NAMES = {"fra": "French", "eng": "English", "fon": "Fon", "yor": "Yoruba"}
LLM_TARGETS = {"fra", "eng"}


def ollama_available(timeout: float = 2.0) -> list[str]:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=timeout)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []


def build_prompt(original: str, src: str, literal: str, tgt: str, syllables: int, stress_slots: list[int],
                 rhyme_with: str | None, cultural_hints: list[str], n: int, feedback: list[str]) -> str:
    lines = [
        f"You adapt song lyrics from {LANG_NAMES.get(src, src)} to {LANG_NAMES.get(tgt, tgt)} so they can be SUNG on the original melody.",
        f"Original line: {original}",
        f"Literal translation (meaning to keep): {literal}",
        f"HARD CONSTRAINT: each proposal must have exactly {syllables} sung syllables.",
    ]
    if stress_slots:
        lines.append(f"Stressed syllables should fall on syllable positions {stress_slots} (1-based).")
    if rhyme_with:
        lines.append(f"The line should end with a sound that rhymes with: {rhyme_with}")
    for h in cultural_hints:
        lines.append(f"Cultural term to preserve: {h}")
    if feedback:
        lines.append("Previous proposals were rejected by the verifier:")
        lines += [f"- {f}" for f in feedback]
    lines += [
        "Keep the meaning, emotion and imagery of the original. Do NOT add new ideas, images or words that are not in the original; do NOT change who does what. Prefer shorter synonyms, reordering or contractions to reach the syllable count. No explanations.",
        f'Answer ONLY with JSON: {{"proposals": ["...", "..."]}} containing {n} different proposals in {LANG_NAMES.get(tgt, tgt)}.',
    ]
    return "\n".join(lines)


def generate(prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0.8, timeout: float = 180.0) -> list[str]:
    r = requests.post(f"{OLLAMA_URL}/api/generate", timeout=timeout, json={
        "model": model, "prompt": prompt, "stream": False, "format": "json",
        "options": {"temperature": temperature, "num_predict": 400}})
    r.raise_for_status()
    raw = r.json().get("response", "")
    try:
        data = json.loads(raw)
        props = data.get("proposals") or next((v for v in data.values() if isinstance(v, list)), [])
    except json.JSONDecodeError:
        props = re.findall(r'"([^"]{3,200})"', raw)
    return [p.strip() for p in props if isinstance(p, str) and p.strip()]

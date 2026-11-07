"""AUTO mode (EXPERIMENTAL): build a sung guide vocal from adapted lyrics.

A speech TTS does not sing. So we:
1. synthesise each word with Meta MMS-TTS (VITS; fon, fra, eng... CC-BY-NC-4.0),
2. analyse it with the WORLD vocoder (f0 / spectral envelope / aperiodicity),
3. stretch each word onto the time span of the notes it was aligned to by the
   Musical Constraint Engine, and replace its pitch by the ORIGINAL sung F0,
4. resynthesise. The result is a robotic but in-tune, in-time guide that Seed-VC
   then converts to the artist's (consented) timbre.
Quality is clearly below a human guide: the UI labels this mode experimental.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from app.schemas.models import LineTemplate
from engines.syllable_engine import syllabify
from engines.syllable_engine.base import tokenize

MMS_CODES = {"fon": "fon", "fra": "fra", "eng": "eng", "yor": "yor"}
WSR = 16000          # MMS-TTS output rate
FRAME_MS = 5.0


@lru_cache(maxsize=3)
def _tts(lang: str):
    import transformers
    from transformers import AutoTokenizer, VitsModel
    # transformers reports weight_g/weight_v as "not used": spurious, the weight-norm
    # parametrisation is rebuilt correctly (verified: max diff 1.5e-8, see docs/BENCHMARK.md)
    transformers.logging.set_verbosity_error()
    name = f"facebook/mms-tts-{MMS_CODES[lang]}"
    from app.core.device import torch_device
    return AutoTokenizer.from_pretrained(name), VitsModel.from_pretrained(name).to(torch_device()).eval()


def speak(text: str, lang: str, seed: int = 0) -> np.ndarray:
    import torch
    tok, model = _tts(lang)
    torch.manual_seed(seed)
    inputs = tok(text, return_tensors="pt")
    if inputs["input_ids"].shape[-1] == 0:
        return np.zeros(int(0.1 * WSR), np.float64)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        wav = model(**inputs).waveform[0].cpu().numpy().astype(np.float64)
    return wav


def _trim(x: np.ndarray, thresh_db: float = -35.0) -> np.ndarray:
    hop = 160
    n = max(1, len(x) // hop)
    e = 20 * np.log10(np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1)) + 1e-9)
    on = np.where(e > e.max() + thresh_db)[0]
    return x[on[0] * hop:(on[-1] + 1) * hop] if len(on) else x


def _stretch_frames(arr: np.ndarray, n_out: int) -> np.ndarray:
    n_in = arr.shape[0]
    pos = np.linspace(0, n_in - 1, n_out)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, n_in - 1)
    w = (pos - lo)[:, None] if arr.ndim == 2 else pos - lo
    return arr[lo] * (1 - w) + arr[hi] * w


def sing_word(wav: np.ndarray, duration: float, f0_target: np.ndarray) -> np.ndarray:
    import pyworld as pw
    x = np.ascontiguousarray(_trim(wav))
    f0, t = pw.harvest(x, WSR, frame_period=FRAME_MS)
    sp = pw.cheaptrick(x, f0, t, WSR)
    ap = pw.d4c(x, f0, t, WSR)
    n_out = max(2, int(round(duration * 1000 / FRAME_MS)))
    sp2, ap2 = _stretch_frames(sp, n_out), _stretch_frames(ap, n_out)
    voiced = _stretch_frames((f0 > 0).astype(float), n_out) > 0.5
    tgt = _stretch_frames(np.nan_to_num(f0_target, nan=0.0), n_out) if len(f0_target) else np.zeros(n_out)
    if not np.any(tgt > 0):                     # melody unknown here: keep spoken pitch
        tgt = _stretch_frames(f0, n_out)
    else:
        fill = np.median(tgt[tgt > 0])
        tgt = np.where(tgt > 0, tgt, fill)
    f0_new = np.where(voiced, tgt, 0.0)
    y = pw.synthesize(np.ascontiguousarray(f0_new), np.ascontiguousarray(sp2), np.ascontiguousarray(ap2),
                      WSR, frame_period=FRAME_MS)
    return y


def _word_spans(text: str, lang: str, tpl: LineTemplate, mapping: list[list[int]]):
    words = tokenize(text)
    counts = [max(1, syllabify(w, lang).count) for w in words]
    spans, k = [], 0
    if sum(counts) != len(mapping) or not tpl.slots:
        start = tpl.slots[0].start if tpl.slots else 0.0
        end = tpl.slots[-1].end if tpl.slots else 0.3 * len(words)
        tot, acc = sum(counts), start
        for w, c in zip(words, counts):
            d = (end - start) * c / tot
            spans.append((w, acc, acc + d))
            acc += d
        return spans
    for w, c in zip(words, counts):
        idx = [j for m in mapping[k:k + c] for j in m]
        k += c
        if idx:
            spans.append((w, tpl.slots[min(idx)].start, tpl.slots[max(idx)].end))
    return spans


def build_guide(lines: list[tuple[str, LineTemplate, list[list[int]]]], lang: str, total_seconds: float,
                f0_times: np.ndarray | None, f0_hz: np.ndarray | None, out_sr: int = 44100) -> np.ndarray:
    import librosa
    buf = np.zeros(int(total_seconds * WSR) + WSR, np.float64)
    for text, tpl, mapping in lines:
        for word, s, e in _word_spans(text, lang, tpl, mapping):
            if e - s < 0.05:
                continue
            if f0_times is not None and len(f0_times):
                sel = (f0_times >= s) & (f0_times < e)
                f0_seg = f0_hz[sel]
            else:
                f0_seg = np.array([])
            y = sing_word(speak(word, lang), e - s, f0_seg)
            a = int(s * WSR)
            y = y[: max(0, len(buf) - a)]
            fade = min(80, len(y) // 4)
            if fade:
                ramp = np.linspace(0, 1, fade)
                y[:fade] *= ramp
                y[-fade:] *= ramp[::-1]
            buf[a:a + len(y)] += y
    peak = np.max(np.abs(buf)) or 1.0
    buf = 0.9 * buf / peak
    return librosa.resample(buf.astype(np.float32), orig_sr=WSR, target_sr=out_sr)[None, :]

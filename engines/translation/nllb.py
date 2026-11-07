"""NLLB-200 translation through CTranslate2 (CPU, int8).

License warning: NLLB-200 weights are CC-BY-NC-4.0 and Meta states the model is
"not released for production deployment". Fine for research / demo, NOT for a
commercial product. See docs/LICENSES.md for the commercial alternatives to evaluate.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

from app.core.languages import get_language

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = Path(os.environ.get("POLYGLOT_NLLB_DIR", ROOT / "models" / "nllb-200-distilled-600M-ct2-int8"))


class NLLBTranslator:
    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR, threads: int | None = None):
        import ctranslate2
        import sentencepiece as spm
        if not (Path(model_dir) / "model.bin").exists():
            raise FileNotFoundError(f"Modele NLLB absent : {model_dir}. Lancez scripts/download_models.py")
        self.translator = ctranslate2.Translator(str(model_dir), device="cpu", compute_type="int8",
                                                 intra_threads=threads or max(1, os.cpu_count() or 1))
        self.sp = spm.SentencePieceProcessor(model_file=str(Path(model_dir) / "sentencepiece.bpe.model"))

    def translate(self, text: str, src: str, tgt: str, n_best: int = 1, sampling: bool = False,
                  beam: int = 5) -> list[str]:
        """Return up to ``n_best`` translations (beam n-best, or sampled variants)."""
        s, t = get_language(src).nllb_code, get_language(tgt).nllb_code
        if not s or not t:
            raise ValueError(f"Paire {src}->{tgt} non prise en charge par NLLB-200.")
        tokens = [s] + self.sp.encode(text, out_type=str) + ["</s>"]
        kwargs = dict(target_prefix=[[t]], max_decoding_length=256)
        if sampling:
            kwargs.update(sampling_topk=20, sampling_temperature=0.9, num_hypotheses=n_best, beam_size=1)
            batch = [tokens] * 1
        else:
            kwargs.update(beam_size=max(beam, n_best), num_hypotheses=n_best)
            batch = [tokens]
        res = self.translator.translate_batch(batch, **kwargs)[0]
        outs = []
        for hyp in res.hypotheses:
            toks = [x for x in hyp if x != t]
            outs.append(self.sp.decode(toks).strip())
        seen, unique = set(), []
        for o in outs:
            o = clean(o)
            key = re.sub(r"[\W_]+", "", o.lower())
            if key and key not in seen:
                seen.add(key)
                unique.append(o)
        return unique


def clean(text: str) -> str:
    """Lyrics style: no tokenizer spacing artefacts, no final full stop."""
    text = re.sub(r"\s+'\s*", "'", text) if re.search(r"\w\s+'(ll|s|re|ve|d|m|t)\b", text) else text
    text = re.sub(r"\s+([,.!?;:])", r"\1", text).strip()
    return re.sub(r"[.]+$", "", text).strip()


@lru_cache(maxsize=1)
def get_translator() -> NLLBTranslator:
    return NLLBTranslator()

"""Language-independent text metrics (no model download needed)."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", text)).strip()


def _ngrams(text: str, n: int) -> Counter:
    s = text.replace(" ", "")
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def chrf(hypothesis: str, reference: str, max_n: int = 6, beta: float = 2.0) -> float:
    """chrF (Popovic 2015), 0..1. Character n-gram F-score, robust for morphologically rich text."""
    hyp, ref = _norm(hypothesis), _norm(reference)
    if not hyp or not ref:
        return 0.0
    precisions, recalls = [], []
    for n in range(1, max_n + 1):
        h, r = _ngrams(hyp, n), _ngrams(ref, n)
        if not h or not r:
            continue
        overlap = sum((h & r).values())
        precisions.append(overlap / sum(h.values()))
        recalls.append(overlap / sum(r.values()))
    if not precisions:
        return 0.0
    p, r = sum(precisions) / len(precisions), sum(recalls) / len(recalls)
    if p == 0 and r == 0:
        return 0.0
    return (1 + beta ** 2) * p * r / (beta ** 2 * p + r)


def rhyme_similarity(a: str, b: str) -> float:
    """1 = full rhyme, 0.5 = same vowel only (assonance), 0.6 = fon same vowel other tone, 0 = none."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if "/" in a and "/" in b:                       # fon keys: "an/LH"
        va, ta = a.split("/", 1)
        vb, tb = b.split("/", 1)
        if va == vb:
            return 0.6 if "?" not in (ta, tb) else 0.8
        return 0.0
    if " " in a or " " in b:                        # CMU phone keys: "AH V"
        pa, pb = a.split(), b.split()
        return 0.5 if pa[0] == pb[0] else 0.0
    va = re.match(r"[aeiouyéèàâêîôûUœ]+", a)
    vb = re.match(r"[aeiouyéèàâêîôûUœ]+", b)
    return 0.5 if va and vb and va.group(0) == vb.group(0) else 0.0

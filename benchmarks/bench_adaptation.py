"""Measured adaptation benchmark on the 6 mandatory directions.

For each line: literal NLLB translation vs. best candidate of the full engine
(NLLB n-best + constrained local LLM + Musical Constraint Engine).
Metrics: syllable gap to the melody, total score, rejection rate, time.
Timing is synthetic (one syllable per 0.35 s, beats every 0.35 s), so the
temporal / accent criteria are indicative. Meaning is a round-trip chrF proxy.
python benchmarks/bench_adaptation.py [--no-llm] [--model llama3.2:latest]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.languages import REQUIRED_PAIRS  # noqa: E402
from app.schemas.models import LyricLine  # noqa: E402
from app.services.adaptation import adapt_line  # noqa: E402
from engines.musical_constraints import build_template  # noqa: E402
from engines.syllable_engine import syllabify  # noqa: E402
from engines.translation.nllb import get_translator  # noqa: E402

STEP = 0.35


def gap(text, lang, n):
    s = syllabify(text, lang)
    return 0 if s.min_count <= n <= s.max_count else min(abs(n - s.min_count), abs(n - s.max_count))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--model", default="llama3.2:latest")
    args = ap.parse_args()
    data = json.loads((ROOT / "datasets" / "lyrics_test_set.json").read_text(encoding="utf-8"))["lines"]
    tr = get_translator()
    rows = []
    for src, tgt in REQUIRED_PAIRS:
        for item in data[src]:
            text = item["text"]
            n = syllabify(text, src).count
            line = LyricLine(index=0, text=text, language=src, start=0.0, end=n * STEP)
            tpl = build_template(line, beats=np.arange(0, n * STEP + 1, STEP * 2))
            t0 = time.perf_counter()
            res = adapt_line(tpl, src, tgt, tr, use_llm=not args.no_llm, llm_model=args.model)
            dt = time.perf_counter() - t0
            best = res.top3[0]
            lit = next(c for c in res.candidates if c.source == "nllb")
            rows.append({"pair": f"{src}->{tgt}", "original": text, "notes": len(tpl.slots),
                         "literal": res.literal, "literal_gap": gap(res.literal, tgt, len(tpl.slots)),
                         "literal_score": lit.total, "best": best.text, "best_source": best.source,
                         "best_gap": gap(best.text, tgt, len(tpl.slots)), "best_score": best.total,
                         "best_rejected": best.rejected, "n_candidates": len(res.candidates),
                         "n_rejected": sum(c.rejected for c in res.candidates), "seconds": round(dt, 1),
                         "criteria": {c.name: c.score for c in best.criteria}})
            r = rows[-1]
            print(f"{r['pair']} | {text} | lit[{r['literal_gap']}] {r['literal']} | best[{r['best_gap']}] "
                  f"{r['best']} ({r['best_source']}, {r['best_score']}) | {r['seconds']}s", flush=True)
    summary = {}
    for pair in {r["pair"] for r in rows}:
        rs = [r for r in rows if r["pair"] == pair]
        summary[pair] = {"lines": len(rs),
                         "mean_literal_gap": round(float(np.mean([r["literal_gap"] for r in rs])), 2),
                         "mean_best_gap": round(float(np.mean([r["best_gap"] for r in rs])), 2),
                         "mean_literal_score": round(float(np.mean([r["literal_score"] or 0 for r in rs])), 3),
                         "mean_best_score": round(float(np.mean([r["best_score"] or 0 for r in rs])), 3),
                         "mean_seconds": round(float(np.mean([r["seconds"] for r in rs])), 1)}
    tag = "nollm" if args.no_llm else args.model.replace(":", "_")
    out = ROOT / "benchmarks" / "results" / f"adaptation_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()

"""End-to-end MVP run (the 11 steps) on the synthetic Fon refrain, headless.

1 import  2 separate  3 transcribe (Fon: manual lyrics)  4 correct  5 translate  6 three adaptations
7 scores  8 artist choice  9 voice conversion (Auto mode, experimental)  10 watermark  11 mix + export
python scripts/demo_e2e.py [target_lang] [--use-llm]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.registry import Registry  # noqa: E402
from app.services import pipeline as P  # noqa: E402

SYN = ROOT / "datasets" / "synthetic"


def main(target: str = "fra", use_llm: bool = False) -> None:
    t0 = time.perf_counter()
    meta = json.loads((SYN / "fon_refrain_01" / "meta.json").read_text(encoding="utf-8"))
    reg = Registry(ROOT / "benchmarks" / "results" / "_demo_registry.db")
    artist = reg.add_artist("Artiste synthetique de demonstration")
    reg.grant_consent(artist, "voice_conversion",
                      "Consentement de demonstration : voix synthetique sans proprietaire reel.")
    p = P.new_project(str(SYN / "fon_refrain_01" / "mix.wav"), "fon", target)
    P.step_analyse(p)                                                   # 1-3
    print("analyse", p.analysis | {"voiced_segments_list": len(p.analysis["voiced_segments_list"])}, p.asr_notes)
    lyrics = "\n".join(l["text"] for l in meta["lines"])               # 4 (artist types Fon lyrics)
    P.step_lyrics(p, lyrics)
    for t in p.templates:
        print("ligne", t.line.text, "| notes", len(t.slots), "|", t.line.start, "->", t.line.end,
              "| midi", [s.midi for s in t.slots])
    P.step_adapt(p, use_llm=use_llm)                                    # 5-7
    for a in p.adaptations:
        print("\n", a["original"], "->", a["literal"])
        for c in a["top3"]:
            print("   ", c["total"], "REJ" if c["rejected"] else "   ", c["text"], "|", c["source"])
        print("   ", a["explanation"].splitlines()[0])
    reg.log_validation("adaptation_choice", "approved", "demo", " / ".join(p.chosen))   # 8
    vocal = P.step_voice(p, "auto", str(SYN / "reference_voice_synthetic_eng.wav"), reg, artist)  # 9
    res = P.step_finalize(p, str(vocal), reg, artist, "Refrain synthetique", "auto")    # 10-11
    res["timings"] = p.timings
    res["total_seconds"] = round(time.perf_counter() - t0, 1)
    res["chosen"] = p.chosen
    out = ROOT / "benchmarks" / "results" / f"demo_e2e_fon_{target}.json"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(*(args or ["fra"]), use_llm="--use-llm" in sys.argv)

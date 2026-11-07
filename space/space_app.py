"""Hugging Face Space entry point (free CPU hardware).

Downloads the models on first start, then launches the artist studio with a 20 s excerpt by
default (CPU). No local LLM (Ollama) and no Seed-VC here: the preview is sung by the
synthetic voice; the 'original voice' option explains that it needs the GPU / Colab version.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("POLYGLOT_EXCERPT_DUR", "20")
os.environ["MPLBACKEND"] = "Agg"

if not (ROOT / "models" / "nllb-200-distilled-600M-ct2-int8" / "model.bin").exists():
    import subprocess
    subprocess.run([sys.executable, str(ROOT / "scripts" / "download_models.py")], check=True)

from app.ui.gradio_app import build  # noqa: E402

build().queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=7860)

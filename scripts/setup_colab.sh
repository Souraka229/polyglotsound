#!/usr/bin/env bash
# Installation complete en une commande (Google Colab ou Linux Ubuntu).
#   WITH_LLM=1 bash scripts/setup_colab.sh      (WITH_LLM=0 pour sauter Ollama)
# Les versions sont celles testees en local : environnement isole .venv (Python 3.11),
# independant des paquets preinstalles de Colab (evite les conflits numpy/torch).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
WITH_LLM="${WITH_LLM:-1}"
WITH_SEEDVC="${WITH_SEEDVC:-1}"
step() { echo; echo "===== $* ====="; }

step "1/6 outils systeme"
command -v ffmpeg >/dev/null || (apt-get -qq update && apt-get -qq install -y ffmpeg)
command -v zstd >/dev/null || (apt-get -qq update && apt-get -qq install -y zstd)
command -v uv >/dev/null || pip -q install uv

step "2/6 environnement Python 3.11 + PyTorch"
[ -x .venv/bin/python ] || uv venv --python 3.11 .venv
PY="$ROOT/.venv/bin/python"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then IDX=cu121; else IDX=cpu; fi
echo "PyTorch : $IDX"
uv pip install --python "$PY" torch==2.5.1 torchaudio==2.5.1 --index-url "https://download.pytorch.org/whl/$IDX"
uv pip install --python "$PY" -r requirements.txt httpx
if [ "$IDX" != "cpu" ]; then
  # ctranslate2 4.4 attend cuDNN 8, PyTorch cu121 apporte cuDNN 9 : 4.5+ gere cuDNN 9 (GPU pour NLLB et Whisper)
  uv pip install --python "$PY" "ctranslate2==4.5.0"
fi

step "3/6 modeles (NLLB int8, Demucs, AudioSeal, Whisper small)"
"$PY" scripts/download_models.py

if [ "$WITH_SEEDVC" = "1" ]; then
  step "4/6 Seed-VC (voix chantee, environnement isole)"
  bash scripts/setup_seed_vc.sh
else
  step "4/6 Seed-VC ignore (WITH_SEEDVC=0) : apercu en voix de synthese seulement"
fi

if [ "$WITH_LLM" = "1" ]; then
  step "5/6 Ollama + llama3.2 (adaptations artistiques)"
  command -v ollama >/dev/null || (curl -fsSL https://ollama.com/install.sh | sh)
  if ! curl -s http://127.0.0.1:11434/api/tags >/dev/null; then
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    for i in $(seq 1 30); do curl -s http://127.0.0.1:11434/api/tags >/dev/null && break; sleep 1; done
  fi
  ollama pull llama3.2
else
  step "5/6 Ollama ignore (WITH_LLM=0)"
fi

step "6/6 tests"
"$PY" -m pytest -q
"$PY" -c "import torch; print('GPU :', torch.cuda.is_available())"
echo; echo "Installation terminee."

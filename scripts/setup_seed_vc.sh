#!/usr/bin/env bash
# Installe Seed-VC (GPL-3.0) dans un environnement isole (Linux / Colab). GPU CUDA si present.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SEED="$ROOT/third_party/seed-vc"
if [ ! -f "$SEED/inference.py" ]; then
  git clone --depth 1 https://github.com/Plachtaa/seed-vc "$SEED"
fi
cd "$SEED"
[ -x .venv/bin/python ] || uv venv --python 3.11 .venv
PY="$SEED/.venv/bin/python"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then IDX=cu121; else IDX=cpu; fi
# 2.4.1 : version pinnee par Seed-VC (2.4.0 + correctif)
uv pip install --python "$PY" torch==2.4.1 torchaudio==2.4.1 --index-url "https://download.pytorch.org/whl/$IDX"
uv pip install --python "$PY" "numpy==1.26.4" "scipy==1.13.1" "librosa==0.10.2" "huggingface-hub>=0.28.1" \
  "munch==4.0.0" "einops==0.8.0" "descript-audio-codec==1.0.0" "pydub==0.25.1" "transformers==4.46.3" \
  "soundfile==0.12.1" "hydra-core==1.3.2" pyyaml accelerate matplotlib "setuptools<81"
"$PY" -c "import torch; print('Seed-VC pret, torch', torch.__version__, 'cuda', torch.cuda.is_available())"

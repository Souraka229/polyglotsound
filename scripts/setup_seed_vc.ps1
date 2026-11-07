# Installe Seed-VC (GPL-3.0) dans un environnement isole, appele en sous-processus par PolyglotSound.
# Usage : powershell -ExecutionPolicy Bypass -File scripts\setup_seed_vc.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$seed = Join-Path $root "third_party\seed-vc"
if (-not (Test-Path (Join-Path $seed "inference.py"))) {
    git clone --depth 1 https://github.com/Plachtaa/seed-vc $seed
}
Set-Location $seed
uv venv --python 3.11 .venv
$py = Join-Path $seed ".venv\Scripts\python.exe"
# CPU build of the torch version pinned by Seed-VC (2.4.1 : 2.4.0 a un bug fbgemm.dll sous Windows)
uv pip install --python $py torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu
# Inference-only subset of Seed-VC requirements (GUI, ASR eval and training extras skipped)
uv pip install --python $py "numpy==1.26.4" "scipy==1.13.1" "librosa==0.10.2" "huggingface-hub>=0.28.1" `
    "munch==4.0.0" "einops==0.8.0" "descript-audio-codec==1.0.0" "pydub==0.25.1" "transformers==4.46.3" `
    "soundfile==0.12.1" "hydra-core==1.3.2" pyyaml accelerate matplotlib "setuptools<81"
Write-Host "Seed-VC pret : $py"

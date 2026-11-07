"""Download the models of the main environment (about 1 GB). Checks free disk space first."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NEEDED_GB = 2.0


def main() -> None:
    free = shutil.disk_usage(ROOT).free / 1e9
    if free < NEEDED_GB:
        sys.exit(f"Espace disque insuffisant : {free:.1f} Go libres, {NEEDED_GB} Go requis.")
    from huggingface_hub import snapshot_download
    print("NLLB-200 600M int8 (CTranslate2, ~620 Mo, CC-BY-NC-4.0)...")
    snapshot_download("JustFrederik/nllb-200-distilled-600M-ct2-int8",
                      local_dir=ROOT / "models" / "nllb-200-distilled-600M-ct2-int8")
    print("Demucs htdemucs (~80 Mo, MIT)...")
    from demucs.pretrained import get_model
    get_model("htdemucs")
    print("AudioSeal 16 bits (~90 Mo, MIT)...")
    from audioseal import AudioSeal
    AudioSeal.load_generator("audioseal_wm_16bits")
    AudioSeal.load_detector("audioseal_detector_16bits")
    print("Whisper small (faster-whisper, ~480 Mo, MIT)...")
    from faster_whisper import WhisperModel
    WhisperModel("small", device="cpu", compute_type="int8")
    cmu = ROOT / "models" / "cmudict" / "cmudict.dict"
    if not cmu.exists():
        import urllib.request
        cmu.parent.mkdir(parents=True, exist_ok=True)
        base = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/"
        urllib.request.urlretrieve(base + "cmudict.dict", cmu)
        urllib.request.urlretrieve(base + "LICENSE", cmu.parent / "LICENSE")
    print("ok. MMS-TTS (mode Auto) se telecharge a la premiere utilisation (~145 Mo par langue).")


if __name__ == "__main__":
    main()

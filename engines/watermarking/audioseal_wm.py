"""AudioSeal watermark (MIT, code + weights). 16-bit payload identifies the generation.

The 16-bit message is the low 16 bits of the generation's registry id, so a
detected watermark can be looked up in the provenance registry. A watermark is
evidence of origin when detectable; it is NOT legal proof of consent (the
consent registry is the record of consent).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

WM_SR = 16000


@lru_cache(maxsize=1)
def _models():
    from audioseal import AudioSeal

    from app.core.device import torch_device
    dev = torch_device()
    return (AudioSeal.load_generator("audioseal_wm_16bits").to(dev).eval(),
            AudioSeal.load_detector("audioseal_detector_16bits").to(dev).eval())


def _bits(value: int) -> "torch.Tensor":
    import torch
    return torch.tensor([[(value >> i) & 1 for i in range(16)]], dtype=torch.int32)


def _value(bits) -> int:
    return int(sum(int(b) << i for i, b in enumerate(bits)))


def embed(audio: np.ndarray, sr: int, message: int, alpha: float = 1.0) -> np.ndarray:
    """Watermark each channel. The watermark is computed at 16 kHz and added back at sr."""
    import torch
    import torchaudio.functional as AF
    gen, _ = _models()
    x = torch.from_numpy(np.atleast_2d(audio)).float()
    x16 = AF.resample(x, sr, WM_SR)
    dev = next(gen.parameters()).device
    with torch.inference_mode():
        wm16 = gen.get_watermark(x16[:, None, :].to(dev), WM_SR,
                                 message=_bits(message).repeat(x16.shape[0], 1).to(dev)).cpu()
    wm = AF.resample(wm16[:, 0, :], WM_SR, sr)[:, : x.shape[-1]]
    if wm.shape[-1] < x.shape[-1]:
        wm = torch.nn.functional.pad(wm, (0, x.shape[-1] - wm.shape[-1]))
    return (x + alpha * wm).numpy().reshape(audio.shape)


@dataclass
class Detection:
    probability: float
    message: int
    detected: bool


def detect(audio: np.ndarray, sr: int, threshold: float = 0.5) -> Detection:
    import torch
    import torchaudio.functional as AF
    _, det = _models()
    x = torch.from_numpy(np.atleast_2d(audio)).float().mean(0, keepdim=True)
    x16 = AF.resample(x, sr, WM_SR)
    with torch.inference_mode():
        prob, msg = det.detect_watermark(x16[None].to(next(det.parameters()).device), WM_SR)
    msg = msg.cpu()
    prob = float(prob)
    return Detection(prob, _value(msg[0].round().int().tolist()), prob >= threshold)
